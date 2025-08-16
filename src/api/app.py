"""
Warranty Claims Fraud Detection - Flask API
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import yaml
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.train_model import ModelPredictor
from data.preprocessing import DataPreprocessor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
predictor = None
config = None

def load_config():
    """Load configuration"""
    global config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file_handler']),
            logging.StreamHandler() if config['logging']['console_handler'] else None
        ]
    )
    return logging.getLogger(__name__)

def initialize_predictor():
    """Initialize the model predictor"""
    global predictor
    try:
        predictor = ModelPredictor()
        predictor.load_model()
        logger.info("Model predictor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        return False

def validate_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input data for prediction"""
    required_features = [
        'Region', 'State', 'Area', 'City', 'Consumer_profile',
        'Product_category', 'Product_type', 'Claim_Value',
        'Service_Centre', 'Product_Age', 'Purchased_from',
        'Call_details', 'Purpose'
    ]
    
    # Check required features
    missing_features = []
    for feature in required_features:
        if feature not in data:
            missing_features.append(feature)
    
    if missing_features:
        return {
            'valid': False,
            'error': f"Missing required features: {missing_features}"
        }
    
    # Validate data types and ranges
    validation_errors = []
    
    # Numerical validations
    try:
        claim_value = float(data['Claim_Value'])
        if claim_value < 0 or claim_value > config['business_rules']['max_claim_amount']:
            validation_errors.append(f"Claim_Value must be between 0 and {config['business_rules']['max_claim_amount']}")
    except (ValueError, TypeError):
        validation_errors.append("Claim_Value must be a valid number")
    
    try:
        product_age = int(data['Product_Age'])
        if product_age < config['business_rules']['min_product_age'] or product_age > config['business_rules']['max_product_age']:
            validation_errors.append(f"Product_Age must be between {config['business_rules']['min_product_age']} and {config['business_rules']['max_product_age']}")
    except (ValueError, TypeError):
        validation_errors.append("Product_Age must be a valid integer")
    
    try:
        service_centre = int(data['Service_Centre'])
        if service_centre < 10 or service_centre > 20:  # Based on data analysis
            validation_errors.append("Service_Centre must be between 10 and 20")
    except (ValueError, TypeError):
        validation_errors.append("Service_Centre must be a valid integer")
    
    try:
        call_details = float(data['Call_details'])
        if call_details < 0 or call_details > 30:  # Based on data analysis
            validation_errors.append("Call_details must be between 0 and 30")
    except (ValueError, TypeError):
        validation_errors.append("Call_details must be a valid number")
    
    # Categorical validations
    valid_regions = ['North', 'South', 'East', 'West', 'North East']
    if data.get('Region') not in valid_regions:
        validation_errors.append(f"Region must be one of: {valid_regions}")
    
    valid_areas = ['Urban', 'Rural']
    if data.get('Area') not in valid_areas:
        validation_errors.append(f"Area must be one of: {valid_areas}")
    
    valid_consumer_profiles = ['Business', 'Personal']
    if data.get('Consumer_profile') not in valid_consumer_profiles:
        validation_errors.append(f"Consumer_profile must be one of: {valid_consumer_profiles}")
    
    valid_product_categories = ['Entertainment', 'Household']
    if data.get('Product_category') not in valid_product_categories:
        validation_errors.append(f"Product_category must be one of: {valid_product_categories}")
    
    valid_product_types = ['TV', 'AC']
    if data.get('Product_type') not in valid_product_types:
        validation_errors.append(f"Product_type must be one of: {valid_product_types}")
    
    valid_purchased_from = ['Manufacturer', 'Dealer', 'Online']
    if data.get('Purchased_from') not in valid_purchased_from:
        validation_errors.append(f"Purchased_from must be one of: {valid_purchased_from}")
    
    valid_purposes = ['Claim', 'Complaint', 'Inquiry']
    if data.get('Purpose') not in valid_purposes:
        validation_errors.append(f"Purpose must be one of: {valid_purposes}")
    
    if validation_errors:
        return {
            'valid': False,
            'error': f"Validation errors: {'; '.join(validation_errors)}"
        }
    
    return {'valid': True}

# API Routes

@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Warranty Claims Fraud Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
            .endpoint { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
            .method { color: #28a745; font-weight: bold; }
            .url { color: #007bff; font-family: monospace; }
            code { background-color: #e9ecef; padding: 2px 4px; border-radius: 3px; }
            .example { background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Warranty Claims Fraud Detection API</h1>
            <p>Real-time fraud detection for warranty claims using machine learning</p>
        </div>
        
        <h2>📋 Available Endpoints</h2>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> <span class="url">/</span></h3>
            <p>This documentation page</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> <span class="url">/health</span></h3>
            <p>API health check</p>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> <span class="url">/predict</span></h3>
            <p>Predict fraud probability for a single claim</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre><code>{
    "Region": "South",
    "State": "Karnataka", 
    "Area": "Urban",
    "City": "Bangalore",
    "Consumer_profile": "Business",
    "Product_category": "Entertainment",
    "Product_type": "TV",
    "Claim_Value": 15000.0,
    "Service_Centre": 10,
    "Product_Age": 60,
    "Purchased_from": "Manufacturer",
    "Call_details": 0.5,
    "Purpose": "Complaint"
}</code></pre>
            </div>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> <span class="url">/predict_batch</span></h3>
            <p>Predict fraud probability for multiple claims</p>
            <div class="example">
                <strong>Request Body:</strong>
                <pre><code>{
    "claims": [
        {...claim1_data...},
        {...claim2_data...}
    ]
}</code></pre>
            </div>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> <span class="url">/model_info</span></h3>
            <p>Get information about the loaded model</p>
        </div>
        
        <h2>📊 Response Format</h2>
        <div class="example">
            <strong>Prediction Response:</strong>
            <pre><code>{
    "success": true,
    "prediction": {
        "is_fraud": 0,
        "fraud_probability": 0.123,
        "risk_level": "Low",
        "confidence": 0.887
    },
    "input_data": {...},
    "timestamp": "2023-01-01T12:00:00Z"
}</code></pre>
        </div>
        
        <h2>🔧 Risk Levels</h2>
        <ul>
            <li><strong>Low Risk:</strong> Fraud probability < 0.5</li>
            <li><strong>Medium Risk:</strong> Fraud probability 0.5 - 0.7</li>
            <li><strong>High Risk:</strong> Fraud probability > 0.7</li>
        </ul>
        
        <h2>📝 Required Fields</h2>
        <p>All prediction requests must include the following fields:</p>
        <ul>
            <li>Region, State, Area, City</li>
            <li>Consumer_profile, Product_category, Product_type</li>
            <li>Claim_Value, Service_Centre, Product_Age</li>
            <li>Purchased_from, Call_details, Purpose</li>
        </ul>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/health', methods=['GET'])
def health_check():
    """API health check"""
    status = {
        'status': 'healthy' if predictor is not None else 'unhealthy',
        'model_loaded': predictor is not None,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    return jsonify(status)

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        # Load model metadata if available
        models_dir = Path(config['paths']['models_dir'])
        metadata_path = models_dir / "best_model_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'model_name': 'Unknown', 'best_score': 'N/A'}
        
        info = {
            'success': True,
            'model_info': {
                'model_name': metadata.get('model_name', 'Unknown'),
                'best_score': metadata.get('best_score', 'N/A'),
                'primary_metric': metadata.get('primary_metric', 'N/A'),
                'feature_count': len(metadata.get('feature_names', [])),
                'loaded_at': pd.Timestamp.now().isoformat()
            }
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict_single():
    """Predict fraud for a single claim"""
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate input data
        validation_result = validate_input_data(data)
        if not validation_result['valid']:
            return jsonify({
                'success': False,
                'error': validation_result['error']
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Add placeholder columns for issue features (set to 'No Issue' for new predictions)
        issue_columns = ['AC_1001_Issue', 'AC_1002_Issue', 'AC_1003_Issue', 
                        'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue']
        for col in issue_columns:
            df[col] = 'No Issue'
        
        # Make prediction with confidence
        result = predictor.predict_with_confidence(df)
        
        # Extract results
        is_fraud = int(result['predictions'][0])
        fraud_probability = float(result['fraud_probability'][0])
        confidence = float(result['confidence'][0])
        
        # Determine risk level
        if fraud_probability < config['business_rules']['medium_risk_threshold']:
            risk_level = "Low"
        elif fraud_probability < config['business_rules']['high_risk_threshold']:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        response = {
            'success': True,
            'prediction': {
                'is_fraud': is_fraud,
                'fraud_probability': round(fraud_probability, 4),
                'risk_level': risk_level,
                'confidence': round(confidence, 4)
            },
            'input_data': data,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Prediction made: fraud_probability={fraud_probability:.4f}, risk_level={risk_level}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict fraud for multiple claims"""
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        # Get request data
        request_data = request.get_json()
        
        if not request_data or 'claims' not in request_data:
            return jsonify({
                'success': False,
                'error': 'No claims data provided'
            }), 400
        
        claims = request_data['claims']
        
        if not isinstance(claims, list) or len(claims) == 0:
            return jsonify({
                'success': False,
                'error': 'Claims must be a non-empty list'
            }), 400
        
        # Validate all claims
        validation_errors = []
        for i, claim in enumerate(claims):
            validation_result = validate_input_data(claim)
            if not validation_result['valid']:
                validation_errors.append(f"Claim {i+1}: {validation_result['error']}")
        
        if validation_errors:
            return jsonify({
                'success': False,
                'error': f"Validation errors: {'; '.join(validation_errors)}"
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(claims)
        
        # Add placeholder columns for issue features
        issue_columns = ['AC_1001_Issue', 'AC_1002_Issue', 'AC_1003_Issue', 
                        'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue']
        for col in issue_columns:
            df[col] = 'No Issue'
        
        # Make predictions
        result = predictor.predict_with_confidence(df)
        
        # Process results
        predictions = []
        for i in range(len(claims)):
            is_fraud = int(result['predictions'][i])
            fraud_probability = float(result['fraud_probability'][i])
            confidence = float(result['confidence'][i])
            
            # Determine risk level
            if fraud_probability < config['business_rules']['medium_risk_threshold']:
                risk_level = "Low"
            elif fraud_probability < config['business_rules']['high_risk_threshold']:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            predictions.append({
                'claim_index': i + 1,
                'is_fraud': is_fraud,
                'fraud_probability': round(fraud_probability, 4),
                'risk_level': risk_level,
                'confidence': round(confidence, 4)
            })
        
        # Summary statistics
        fraud_count = sum(1 for p in predictions if p['is_fraud'] == 1)
        high_risk_count = sum(1 for p in predictions if p['risk_level'] == 'High')
        avg_fraud_probability = np.mean([p['fraud_probability'] for p in predictions])
        
        response = {
            'success': True,
            'predictions': predictions,
            'summary': {
                'total_claims': len(claims),
                'fraud_detected': fraud_count,
                'high_risk_claims': high_risk_count,
                'average_fraud_probability': round(avg_fraud_probability, 4)
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Batch prediction completed: {len(claims)} claims, {fraud_count} fraud detected")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Load configuration
    load_config()
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize predictor
    if not initialize_predictor():
        logger.error("Failed to initialize predictor. Exiting...")
        exit(1)
    
    logger.info("Starting Flask API server...")
    
    # Get API configuration
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', False)
    
    # Start the Flask app
    app.run(host=host, port=port, debug=debug)
