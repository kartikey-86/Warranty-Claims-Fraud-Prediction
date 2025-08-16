#!/usr/bin/env python3
"""
Main Pipeline Execution Script
Runs the complete fraud detection pipeline from data preprocessing to model training and evaluation
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessing import DataPreprocessor
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_complete_pipeline():
    """Run the complete ML pipeline"""
    logger = setup_logging()
    
    try:
        logger.info("🚀 Starting Warranty Claims Fraud Detection Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Data Preprocessing
        logger.info("📊 Step 1: Data Preprocessing")
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test, quality_report = preprocessor.preprocess_pipeline()
        logger.info("✅ Data preprocessing completed successfully")
        
        # Step 2: Model Training
        logger.info("🤖 Step 2: Model Training and Hyperparameter Tuning")
        trainer = ModelTrainer()
        training_results = trainer.full_training_pipeline()
        logger.info(f"✅ Model training completed. Best model: {training_results['best_model_name']}")
        
        # Step 3: Model Evaluation
        logger.info("📈 Step 3: Model Evaluation and Visualization")
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.comprehensive_evaluation(X_test, y_test)
        logger.info("✅ Model evaluation completed successfully")
        
        # Pipeline Summary
        logger.info("=" * 60)
        logger.info("🎉 Pipeline Execution Summary")
        logger.info(f"   • Data shape: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        logger.info(f"   • Features: {X_train.shape[1]} features")
        logger.info(f"   • Best model: {training_results['best_model_name']}")
        logger.info(f"   • Best score: {training_results['best_score']:.4f}")
        logger.info(f"   • Models evaluated: {len(evaluation_results['evaluation_results'])}")
        logger.info("   • Plots and reports saved to respective directories")
        logger.info("=" * 60)
        
        # Next Steps
        logger.info("🔧 Next Steps:")
        logger.info("   • Start API server: python src/api/app.py")
        logger.info("   • Launch dashboard: streamlit run src/visualization/dashboard.py")
        logger.info("   • Deploy with Docker: docker-compose up")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_complete_pipeline()
    sys.exit(0 if success else 1)
