# World Bank Loan Prediction Flask App

A Flask web application for predicting World Bank loan cancellation and disbursement outcomes using machine learning.

## Features

- 🌍 Predict loan outcomes (Fully Disbursed, Major Cancellation, Minor Cancellation)
- 📊 Beautiful, modern UI with gradient design
- 🤖 Uses CatBoost machine learning model
- 📈 Displays prediction probabilities
- 🎨 Responsive design

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and Save the Model

First, you need to train the model and save it. Run this in your Jupyter notebook or as a separate Python script:

```python
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import pickle

# Load and preprocess your data (following the notebook)
# ... your data loading and preprocessing code ...

# Train the model
model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    verbose=50
)

# Train on your preprocessed data
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# Save the model
model.save_model('trained_model.cbm')
print("Model saved successfully!")
```

### 3. Run the Flask App

```bash
python app.py
```

The app will start on `http://localhost:5000`

## Usage

1. Open your web browser and navigate to `http://localhost:5000`
2. Fill in the loan details:
   - Loan Type
   - Region
   - Interest Rate
   - Original Principal Amount
   - Borrower's Obligation
   - Due to IBRD
   - GDP Total
   - GDP Per Capita
3. Click "Predict Loan Outcome"
4. View the prediction and probabilities

## API Endpoints

### POST /predict
Make a prediction for loan outcome.

**Request Body:**
```json
{
  "loan_type": "SCP USD",
  "region": "EAST ASIA AND PACIFIC",
  "interest_rate": 4.25,
  "original_principal_amount": 50000000,
  "borrowers_obligation": 50000000,
  "due_to_ibrd": 50000000,
  "gdp_total": 1000000000000,
  "gdp_per_capita": 5000
}
```

**Response:**
```json
{
  "prediction": "Fully Disbursed",
  "probabilities": {
    "Fully Disbursed": 0.85,
    "Major Cancellation": 0.05,
    "Minor Cancellation": 0.10
  }
}
```

### GET /health
Check if the app and model are loaded correctly.

## Features Used by the Model

### Categorical Features
- `loan_type`: Type of loan
- `region`: Geographic region

### Numerical Features (log-transformed)
- `interest_rate`: Loan interest rate
- `log_original_principal_amount`: Log of principal amount
- `log_borrowers_obligation`: Log of borrower's obligation
- `log_due_to_ibrd`: Log of amount due to IBRD
- `log_gdp_total`: Log of total GDP
- `log_gdp_per_capita`: Log of GDP per capita

## Model Performance

- Accuracy: ~98%
- Precision: 0.98 (macro avg)
- Recall: 0.98 (macro avg)
- F1-score: 0.98 (macro avg)

## Directory Structure

```
.
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Web UI
├── requirements.txt      # Python dependencies
├── trained_model.cbm     # Trained CatBoost model (you'll need to generate this)
└── README.md             # This file
```

## Notes

- The app expects a trained CatBoost model saved as `trained_model.cbm`
- All numerical features are automatically log-transformed (log1p) before prediction
- The model predicts one of three outcomes: Fully Disbursed, Major Cancellation, or Minor Cancellation

## License

MIT License

