# 🔍 Warranty Claims Fraud Detection System

A comprehensive machine learning pipeline for detecting fraudulent warranty claims using advanced ML techniques, real-time API, and interactive dashboard.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline to detect fraudulent warranty claims. It uses multiple ML algorithms, comprehensive data preprocessing, and provides both API and dashboard interfaces for real-time fraud detection.

### Key Capabilities
- **Multi-algorithm ML Pipeline**: Logistic Regression, Random Forest, XGBoost, Gradient Boosting
- **Real-time Fraud Detection**: REST API with input validation and confidence scores
- **Interactive Dashboard**: Streamlit-based visualization and prediction interface
- **Comprehensive Analytics**: EDA, model performance analysis, and business insights
- **Production Ready**: Docker containerization, logging, monitoring, and CI/CD support

## ✨ Features

### 🤖 Machine Learning
- **Advanced Preprocessing**: Data cleaning, feature engineering, SMOTE for class imbalance
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Model Evaluation**: ROC-AUC, Precision-Recall, Feature importance analysis
- **Class Imbalance Handling**: SMOTE oversampling with stratified splits

### 🌐 API & Web Interface
- **Flask REST API**: Real-time predictions with input validation
- **Streamlit Dashboard**: Interactive EDA, model performance, and prediction interface
- **Batch Processing**: Support for multiple claims prediction
- **Health Monitoring**: API health checks and status endpoints

### 🐳 Deployment & DevOps
- **Docker Support**: Multi-service containerization
- **Docker Compose**: Complete stack deployment
- **Logging & Monitoring**: Comprehensive logging with configurable levels
- **Configuration Management**: YAML-based configuration system

## 📁 Project Structure

```
Warranty-Claims-Fraud-Prediction/
├── 📊 data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed datasets
├── 🔧 src/
│   ├── api/                    # Flask API application
│   │   └── app.py
│   ├── data/                   # Data processing modules
│   │   └── preprocessing.py
│   ├── models/                 # ML model modules
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   └── visualization/          # Dashboard and plots
│       └── dashboard.py
├── ⚙️ config/
│   └── config.yaml            # Configuration file
├── 🤖 models/                 # Trained model artifacts
├── 📊 notebooks/              # Jupyter notebooks
├── 📈 plots/                  # Generated visualizations
├── 📋 scripts/                # Utility scripts
├── 🧪 tests/                  # Unit tests
├── 📝 logs/                   # Application logs
├── 🐳 Docker files            # Containerization
├── 📚 requirements.txt        # Python dependencies
└── 📖 README.md              # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Git
- Docker (optional, for containerized deployment)

### Option 1: Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/Warranty-Claims-Fraud-Prediction.git
cd Warranty-Claims-Fraud-Prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional)
```bash
export PYTHONPATH=$PWD
```

### Option 2: Docker Installation

1. **Clone and build**
```bash
git clone https://github.com/your-username/Warranty-Claims-Fraud-Prediction.git
cd Warranty-Claims-Fraud-Prediction
docker-compose up --build
```

## 🚀 Quick Start

### 1. Run Complete Pipeline
```bash
# Execute the full ML pipeline
python scripts/run_pipeline.py
```

### 2. Start API Server
```bash
# Start the Flask API
python src/api/app.py
```
API will be available at: `http://localhost:5000`

### 3. Launch Dashboard
```bash
# Start Streamlit dashboard
streamlit run src/visualization/dashboard.py
```
Dashboard will be available at: `http://localhost:8501`

### 4. Docker Deployment
```bash
# Deploy complete stack
docker-compose up
```
- API: `http://localhost:5000`
- Dashboard: `http://localhost:8501`

## 📖 Usage

### Data Preprocessing
```python
from src.data.preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Run complete preprocessing pipeline
X_train, X_val, X_test, y_train, y_val, y_test, quality_report = preprocessor.preprocess_pipeline()
```

### Model Training
```python
from src.models.train_model import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Train all models with hyperparameter tuning
results = trainer.full_training_pipeline()
print(f"Best model: {results['best_model_name']}")
```

### Model Evaluation
```python
from src.models.evaluate_model import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Comprehensive evaluation
evaluation_results = evaluator.comprehensive_evaluation(X_test, y_test)
```

### Making Predictions
```python
from src.models.train_model import ModelPredictor

# Initialize predictor
predictor = ModelPredictor()

# Load model
predictor.load_model()

# Make prediction
prediction = predictor.predict_with_confidence(new_data)
```

## 🔌 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### Health Check
```http
GET /health
```
**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Single Prediction
```http
POST /predict
Content-Type: application/json

{
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
}
```

**Response:**
```json
{
    "success": true,
    "prediction": {
        "is_fraud": 0,
        "fraud_probability": 0.234,
        "risk_level": "Low",
        "confidence": 0.876
    },
    "input_data": {...},
    "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Batch Prediction
```http
POST /predict_batch
Content-Type: application/json

{
    "claims": [
        {...claim1_data...},
        {...claim2_data...}
    ]
}
```

#### Model Information
```http
GET /model_info
```

### API Client Example
```python
import requests

# Single prediction
data = {
    "Region": "South",
    "Claim_Value": 15000.0,
    # ... other required fields
}

response = requests.post('http://localhost:5000/predict', json=data)
result = response.json()

print(f"Fraud Probability: {result['prediction']['fraud_probability']:.3f}")
print(f"Risk Level: {result['prediction']['risk_level']}")
```

## 🐳 Deployment

### Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Services
```bash
# Build and run API
docker build -t fraud-api .
docker run -p 5000:5000 fraud-api

# Build and run Dashboard
docker build -f Dockerfile.streamlit -t fraud-dashboard .
docker run -p 8501:8501 fraud-dashboard
```

### Environment Variables
```bash
# Production settings
FLASK_ENV=production
PYTHONPATH=/app
LOG_LEVEL=INFO

# Database (optional)
DATABASE_URL=postgresql://user:pass@host:5432/db

# Redis (optional)
REDIS_URL=redis://host:6379/0
```

## 📊 Model Performance

### Dataset Statistics
- **Total Claims**: 358
- **Fraudulent Claims**: 35 (9.8%)
- **Features**: 20 (after preprocessing)
- **Class Balance**: Handled with SMOTE

### Key Insights
- **High-Risk Service Centers**: Centers 10, 15, 16 show elevated fraud rates
- **Product Risk**: TV products show higher fraud probability than AC
- **Regional Patterns**: South region has highest fraud concentration
- **Claim Value**: Fraudulent claims average 15% higher value

## 🚀 Quick Commands

```bash
# Complete pipeline
python scripts/run_pipeline.py

# Start API
python src/api/app.py

# Launch dashboard
streamlit run src/visualization/dashboard.py

# Docker deployment
docker-compose up

# Check health
curl http://localhost:5000/health
```

**Built with ❤️ for fraud detection and prevention**
