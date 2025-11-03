import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from catboost import CatBoostClassifier
import pickle
import os

app = Flask(__name__)

# Global variables
model = None
scaler = None
label_encoder = None
model_features = None

# Define categorical and numerical features
CAT_COLS = ["loan_type", "region"]
NUM_COLS = [
    "interest_rate",
    "log_original_principal_amount",
    "log_borrowers_obligation",
    "log_due_to_ibrd",
    "log_gdp_total",
    "log_gdp_per_capita"
]

# Loan type options (based on notebook analysis)
LOAN_TYPES = [
    "SCP USD", "SCPD", "POOL LOAN", "CPL", "FSL", 
    "SNGL CRNCY", "SCL", "NON POOL", "NPL", "SCP EUR",
    "SCPM", "SCPY", "SCP JPY", "GURB"
]

# Region options
REGIONS = [
    "EAST ASIA AND PACIFIC",
    "EUROPE AND CENTRAL ASIA",
    "SOUTH ASIA",
    "MIDDLE EAST AND NORTH AFRICA",
    "LATIN AMERICA AND CARIBBEAN",
    "EASTERN AND SOUTHERN AFRICA",
    "WESTERN AND CENTRAL AFRICA",
    "AFRICA EAST",
    "AFRICA WEST",
    "MID EAST,NORTH AFRICA,AFG,PAK"
]

def load_model():
    """Load the trained model and preprocessors"""
    global model, scaler, label_encoder
    
    # Check if model files exist
    model_file = 'trained_model.cbm'
    if os.path.exists(model_file):
        model = CatBoostClassifier()
        model.load_model(model_file)
        print("✅ Model loaded successfully")
    else:
        print("⚠️  Model file not found. Please run train_model.py first to train and save the model.")
        return False
    
    # Load label encoder if it exists
    label_encoder_file = 'label_encoder.pkl'
    if os.path.exists(label_encoder_file):
        with open(label_encoder_file, 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Label encoder loaded successfully")
    
    return True

def preprocess_input(data):
    """Preprocess input data for prediction"""
    # Create a DataFrame from input
    df = pd.DataFrame([data])
    
    # Apply log transformation to numerical features
    for col in NUM_COLS:
        if col in df.columns and not col.startswith('log_'):
            original_col = col.replace('log_', '')
            df[f'log_{original_col}'] = np.log1p(df[col])
    
    # Ensure all required features are present
    required_features = CAT_COLS + NUM_COLS
    for feature in required_features:
        if feature not in df.columns:
            # Set default values for missing features
            if feature in CAT_COLS:
                df[feature] = CAT_COLS[0] if feature == 'loan_type' else REGIONS[0]
            else:
                df[feature] = 0.0
    
    return df[required_features]

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', 
                         loan_types=LOAN_TYPES, 
                         regions=REGIONS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.json
        
        # Validate input
        required_fields = [
            'interest_rate', 'original_principal_amount', 
            'borrowers_obligation', 'due_to_ibrd',
            'gdp_total', 'gdp_per_capita', 'loan_type', 'region'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Preprocess input
        processed_data = preprocess_input(data)
        
        # Make prediction
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        prediction = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[0]
        
        # Map prediction to class name
        class_names = ["Fully Disbursed", "Major Cancellation", "Minor Cancellation"]
        predicted_class = class_names[int(prediction[0])]
        
        # Format probabilities
        prob_dict = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        return jsonify({
            'prediction': predicted_class,
            'probabilities': prob_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    if load_model():
        print("✅ Ready to serve predictions!")
    else:
        print("⚠️  Running without pre-trained model")
    
    app.run(debug=True, host='0.0.0.0', port=9000)

