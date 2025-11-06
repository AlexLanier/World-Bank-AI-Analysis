"""
Script to train and save the CatBoost model for the Flask app.
This extracts the model training logic from the Jupyter notebook.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle

print("📚 Loading and preprocessing data...")

# Load the parquet file
df = pd.read_parquet("worldbank_loans.parquet")

print("✅ Data loaded")

# Load GDP data
gdp_total = pd.read_csv("gdp_total.csv")
gdp_per_capita = pd.read_csv("gdp_per_capita.csv")

# Merge GDP datasets
gdp = gdp_total.merge(
    gdp_per_capita,
    on=["Country", "Year"],
    how="outer",
    suffixes=("_total", "_per_capita")
)

print("✅ GDP data loaded and merged")

# Clean country codes in GDP data
import pycountry
valid_iso3 = {c.alpha_3 for c in pycountry.countries}
gdp = gdp[gdp["Country"].isin(valid_iso3) | gdp["Country"].eq("XKX")]

iso3_to_iso2 = {c.alpha_3: c.alpha_2 for c in pycountry.countries}
gdp["country_code"] = gdp["Country"].map(iso3_to_iso2)

manual_fixes = {
    "XKX": "XK", "SRB": "YU", "SCG": "RS", 
    "TLS": "TL", "COD": "ZR", "TWN": "TW"
}

gdp["country_code"] = gdp.apply(
    lambda row: manual_fixes.get(row["Country"], row["country_code"]),
    axis=1
)

# Clean country codes in loans data
df["country_code"] = df["country_code"].replace("YF", "YU")
df["country_code"] = df["country_code"].replace("TP", "TL")

invalid_codes = {None, "3W", "3E", "3S", "6R", "TW"}
df = df[~df["country_code"].isin(invalid_codes)].copy()

print("✅ Country codes standardized")

# Select features
features_to_keep = [
    "loan_number", "country", "country_code", "region",
    "original_principal_amount", "disbursed_amount", "cancelled_amount",
    "loan_type", "loan_status", "interest_rate",
    "borrowers_obligation", "due_to_ibrd", "loans_held",
    "board_approval_date", "end_of_period",
    "project_name_", "borrower",
]

df_clean = df[features_to_keep].copy()

# Extract year
df_clean["Year"] = pd.to_datetime(df_clean["board_approval_date"], errors="coerce").dt.year

# Merge with GDP data
df_merged = pd.merge(
    df_clean,
    gdp,
    how="left",
    on=["country_code", "Year"],
    indicator=True
)

df = df_merged.drop(columns=["_merge", "Country"]).copy()

print("✅ Data merged with GDP")

# Drop rows with zero principal amount
df_clean = df[df["original_principal_amount"] != 0].copy()

# Calculate ratios
df_clean["disbursement_ratio"] = np.where(
    df_clean["original_principal_amount"] > 0,
    df_clean["disbursed_amount"] / df_clean["original_principal_amount"],
    np.nan
)

df_clean["cancellation_ratio"] = np.where(
    df_clean["original_principal_amount"] > 0,
    df_clean["cancelled_amount"] / df_clean["original_principal_amount"],
    np.nan
)

# Clip ratios
df_clean["disbursement_ratio"] = df_clean["disbursement_ratio"].clip(0, 1)
df_clean["cancellation_ratio"] = df_clean["cancellation_ratio"].clip(0, 1)

# Create target variable
def categorize(row):
    if row["cancellation_ratio"] == 0:
        return "Fully Disbursed"
    elif row["cancellation_ratio"] < 0.2:
        return "Minor Cancellation"
    else:
        return "Major Cancellation"

df_clean["loan_outcome"] = df_clean.apply(categorize, axis=1)

# Encode target
le = LabelEncoder()
df_clean["loan_outcome_encoded"] = le.fit_transform(df_clean["loan_outcome"])

print("✅ Target variable created")

# Prepare features
cat_cols = ["loan_type", "region"]
num_cols = [
    "interest_rate",
    "log_original_principal_amount",
    "log_borrowers_obligation",
    "log_due_to_ibrd",
    "log_gdp_total", 
    "log_gdp_per_capita"
]

# Clip negative values
cols_to_clip = ["borrowers_obligation", "due_to_ibrd"]
for col in cols_to_clip:
    df_clean[col] = df_clean[col].clip(lower=0)

# Log-transform
log_cols = ["original_principal_amount", "borrowers_obligation",
            "due_to_ibrd", "gdp_total", "gdp_per_capita"]

for col in log_cols:
    df_clean[f"log_{col}"] = np.log1p(df_clean[col])

# Drop rows with missing features
df_model = df_clean.dropna(subset=cat_cols + num_cols + ["loan_outcome_encoded"]).copy()

# Split features and target
X = df_model[cat_cols + num_cols]
y = df_model["loan_outcome_encoded"]

print(f"✅ Features prepared: {len(X)} samples")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Get categorical indices
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

# Build pools
train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool = Pool(X_test, y_test, cat_features=cat_idx)

print("✅ Data pools created")

# Train CatBoost model
print("🚀 Training CatBoost model...")
model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    od_type="Iter",
    od_wait=40,
    verbose=50
)

model.fit(train_pool, eval_set=test_pool)
print("✅ Model trained")

# Save the model
model.save_model('trained_model.cbm')
print("💾 Model saved as 'trained_model.cbm'")

# Also save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("💾 Label encoder saved as 'label_encoder.pkl'")

print("\n🎉 Training complete! You can now run the Flask app with 'python app.py'")

