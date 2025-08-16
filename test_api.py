# Testing the Flask API

import requests
import json

# Test data for API
test_claim = {
    "claim_id": "TEST-001",
    "product_type": "Electronics", 
    "claim_date": "2024-01-15",
    "claim_amount": 850.0,
    "customer_age": 35,
    "issue_code": "MAL",
    "claim_severity": "Medium",
    "customer_history": "Good",
    "processing_time": 5
}

# Test the API endpoint
if __name__ == "__main__":
    # Start the Flask app first, then run this test
    url = "http://localhost:5000/predict"
    
    try:
        response = requests.post(url, json=test_claim)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except requests.exceptions.ConnectionError:
        print("API server is not running. Start with: python src/api/app.py")
    except Exception as e:
        print(f"Error: {e}")
