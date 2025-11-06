"""
Simple test script to verify the Flask API is working correctly
"""

import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict():
    """Test the predict endpoint"""
    print("Testing /predict endpoint...")
    
    # Sample data
    data = {
        "loan_type": "SCP USD",
        "region": "EAST ASIA AND PACIFIC",
        "interest_rate": 4.25,
        "original_principal_amount": 50000000,
        "borrowers_obligation": 50000000,
        "due_to_ibrd": 50000000,
        "gdp_total": 1000000000000,
        "gdp_per_capita": 5000
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("🧪 Testing World Bank Loan Prediction API")
    print("=" * 50)
    print()
    
    try:
        # Test health endpoint
        test_health()
        
        # Test predict endpoint
        test_predict()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the server.")
        print("Make sure the Flask app is running: python app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

