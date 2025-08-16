"""
Warranty Claims Fraud Detection - Model Training Pipeline
"""

import pandas as pd
import numpy as np
import yaml
import logging
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import xgboost as xgb

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.preprocessing import DataPreprocessor

class ModelTrainer:
    """
    Comprehensive model training pipeline for warranty claims fraud detection
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the model trainer with configuration"""
        self.config = self._load_config(config_path)
        self.models = {}
        self.best_model = None
        self.best_score = -float('inf')
        self.cv_results = {}
        self.feature_names = None
        
        # Set up logging
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(self.config['logging']['file_handler']),
                logging.StreamHandler() if self.config['logging']['console_handler'] else None
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all ML models based on configuration"""
        self.logger.info("Initializing ML models...")
        
        models = {}
        model_config = self.config['models']
        
        # Logistic Regression
        if model_config['logistic_regression']['enabled']:
            models['logistic_regression'] = LogisticRegression(
                random_state=self.config['data']['random_state'],
                max_iter=model_config['logistic_regression']['params']['max_iter']
            )
        
        # Random Forest
        if model_config['random_forest']['enabled']:
            models['random_forest'] = RandomForestClassifier(
                random_state=self.config['data']['random_state'],
                n_jobs=-1
            )
        
        # XGBoost
        if model_config['xgboost']['enabled']:
            models['xgboost'] = xgb.XGBClassifier(
                random_state=self.config['data']['random_state'],
                eval_metric='logloss',
                n_jobs=-1
            )
        
        # Gradient Boosting
        if model_config['gradient_boosting']['enabled']:
            models['gradient_boosting'] = GradientBoostingClassifier(
                random_state=self.config['data']['random_state']
            )
        
        self.models = models
        self.logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
        return models
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """Get parameter grids for hyperparameter tuning"""
        param_grids = {}
        model_config = self.config['models']
        
        # Logistic Regression parameters
        if 'logistic_regression' in self.models:
            param_grids['logistic_regression'] = {
                'C': model_config['logistic_regression']['params']['C'],
                'penalty': model_config['logistic_regression']['params']['penalty'],
                'solver': model_config['logistic_regression']['params']['solver']
            }
        
        # Random Forest parameters
        if 'random_forest' in self.models:
            param_grids['random_forest'] = {
                'n_estimators': model_config['random_forest']['params']['n_estimators'],
                'max_depth': model_config['random_forest']['params']['max_depth'],
                'min_samples_split': model_config['random_forest']['params']['min_samples_split'],
                'min_samples_leaf': model_config['random_forest']['params']['min_samples_leaf'],
                'class_weight': model_config['random_forest']['params']['class_weight']
            }
        
        # XGBoost parameters
        if 'xgboost' in self.models:
            param_grids['xgboost'] = {
                'n_estimators': model_config['xgboost']['params']['n_estimators'],
                'max_depth': model_config['xgboost']['params']['max_depth'],
                'learning_rate': model_config['xgboost']['params']['learning_rate'],
                'subsample': model_config['xgboost']['params']['subsample'],
                'colsample_bytree': model_config['xgboost']['params']['colsample_bytree'],
                'scale_pos_weight': model_config['xgboost']['params']['scale_pos_weight']
            }
        
        # Gradient Boosting parameters
        if 'gradient_boosting' in self.models:
            param_grids['gradient_boosting'] = {
                'n_estimators': model_config['gradient_boosting']['params']['n_estimators'],
                'max_depth': model_config['gradient_boosting']['params']['max_depth'],
                'learning_rate': model_config['gradient_boosting']['params']['learning_rate'],
                'subsample': model_config['gradient_boosting']['params']['subsample']
            }
        
        return param_grids
    
    def perform_cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Dict]:
        """Perform cross-validation on all models"""
        self.logger.info("Starting cross-validation...")
        
        cv_config = self.config['cv']
        scoring = cv_config['scoring']
        n_folds = cv_config['n_folds']
        stratify = cv_config['stratify']
        
        # Set up cross-validation strategy
        if stratify:
            cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                                        random_state=self.config['data']['random_state'])
        else:
            cv_strategy = n_folds
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Performing CV for {model_name}...")
            
            try:
                scores = cross_validate(
                    model, X_train, y_train,
                    cv=cv_strategy,
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=True
                )
                
                # Calculate mean and std for each metric
                cv_results[model_name] = {}
                for metric in scoring:
                    test_scores = scores[f'test_{metric}']
                    train_scores = scores[f'train_{metric}']
                    
                    cv_results[model_name][f'{metric}_test_mean'] = np.mean(test_scores)
                    cv_results[model_name][f'{metric}_test_std'] = np.std(test_scores)
                    cv_results[model_name][f'{metric}_train_mean'] = np.mean(train_scores)
                    cv_results[model_name][f'{metric}_train_std'] = np.std(train_scores)
                
                self.logger.info(f"CV completed for {model_name}")
                
            except Exception as e:
                self.logger.error(f"Error in CV for {model_name}: {str(e)}")
                continue
        
        self.cv_results = cv_results
        self.logger.info("Cross-validation completed for all models")
        return cv_results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Perform hyperparameter tuning using GridSearchCV"""
        self.logger.info("Starting hyperparameter tuning...")
        
        param_grids = self.get_param_grids()
        tuned_models = {}
        
        cv_config = self.config['cv']
        primary_metric = self.config['model_selection']['primary_metric']
        
        # Set up cross-validation strategy
        cv_strategy = StratifiedKFold(
            n_splits=cv_config['n_folds'], 
            shuffle=True, 
            random_state=self.config['data']['random_state']
        )
        
        for model_name, model in self.models.items():
            if model_name in param_grids:
                self.logger.info(f"Tuning hyperparameters for {model_name}...")
                
                try:
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grids[model_name],
                        scoring=primary_metric,
                        cv=cv_strategy,
                        n_jobs=-1,
                        verbose=1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    
                    tuned_models[model_name] = {
                        'model': grid_search.best_estimator_,
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_,
                        'cv_results': grid_search.cv_results_
                    }
                    
                    self.logger.info(f"Best score for {model_name}: {grid_search.best_score_:.4f}")
                    self.logger.info(f"Best params for {model_name}: {grid_search.best_params_}")
                    
                except Exception as e:
                    self.logger.error(f"Error in hyperparameter tuning for {model_name}: {str(e)}")
                    # Use default model if tuning fails
                    tuned_models[model_name] = {
                        'model': model,
                        'best_params': {},
                        'best_score': 0.0,
                        'cv_results': {}
                    }
        
        self.models = {name: result['model'] for name, result in tuned_models.items()}
        self.logger.info("Hyperparameter tuning completed")
        return tuned_models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train all models with best hyperparameters"""
        self.logger.info("Training final models...")
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                model.fit(X_train, y_train)
                trained_models[model_name] = model
                self.logger.info(f"{model_name} training completed")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.models = trained_models
        self.logger.info("All models trained successfully")
        return trained_models
    
    def evaluate_models(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict]:
        """Evaluate all trained models on validation set"""
        self.logger.info("Evaluating models on validation set...")
        
        evaluation_results = {}
        primary_metric = self.config['model_selection']['primary_metric']
        
        for model_name, model in self.models.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, average='weighted'),
                    'recall': recall_score(y_val, y_pred, average='weighted'),
                    'f1': f1_score(y_val, y_pred, average='weighted'),
                }
                
                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)
                
                # Store additional evaluation data
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'predictions_proba': y_pred_proba,
                    'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
                    'classification_report': classification_report(y_val, y_pred, output_dict=True)
                }
                
                # Check if this is the best model
                if metrics[primary_metric] > self.best_score:
                    self.best_score = metrics[primary_metric]
                    self.best_model = model
                    self.best_model_name = model_name
                
                self.logger.info(f"{model_name} - {primary_metric}: {metrics[primary_metric]:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        self.logger.info(f"Best model: {self.best_model_name} with {primary_metric}: {self.best_score:.4f}")
        return evaluation_results
    
    def get_feature_importance(self, model, model_name: str) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        feature_importance = {}
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return feature_importance
        
        if self.feature_names is not None:
            feature_importance = dict(zip(self.feature_names, importance))
        
        return feature_importance
    
    def save_models(self, save_dir: str = None):
        """Save trained models and results"""
        if save_dir is None:
            save_dir = self.config['paths']['models_dir']
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Save all models if configured
            if self.config['model_selection']['save_all_models']:
                for model_name, model in self.models.items():
                    model_path = Path(save_dir) / f"{model_name}_model.pkl"
                    joblib.dump(model, model_path)
                    self.logger.info(f"Saved {model_name} model")
            
            # Save best model
            if self.config['model_selection']['save_best_model'] and self.best_model is not None:
                best_model_path = Path(save_dir) / "best_model.pkl"
                joblib.dump(self.best_model, best_model_path)
                
                # Save model metadata
                metadata = {
                    'model_name': self.best_model_name,
                    'best_score': self.best_score,
                    'primary_metric': self.config['model_selection']['primary_metric'],
                    'feature_names': self.feature_names
                }
                
                with open(Path(save_dir) / "best_model_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.logger.info(f"Saved best model: {self.best_model_name}")
            
            # Save CV results
            if self.cv_results:
                with open(Path(save_dir) / "cv_results.json", 'w') as f:
                    json.dump(self.cv_results, f, indent=2)
                self.logger.info("Saved cross-validation results")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_best_model(self, load_dir: str = None):
        """Load the best trained model"""
        if load_dir is None:
            load_dir = self.config['paths']['models_dir']
        
        try:
            # Load best model
            best_model_path = Path(load_dir) / "best_model.pkl"
            if best_model_path.exists():
                self.best_model = joblib.load(best_model_path)
                
                # Load metadata
                metadata_path = Path(load_dir) / "best_model_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    self.best_model_name = metadata['model_name']
                    self.best_score = metadata['best_score']
                    self.feature_names = metadata['feature_names']
                
                self.logger.info(f"Loaded best model: {self.best_model_name}")
                return self.best_model
            else:
                self.logger.warning("Best model file not found")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading best model: {str(e)}")
            raise
    
    def full_training_pipeline(self) -> Dict[str, Any]:
        """Complete model training pipeline"""
        self.logger.info("Starting full training pipeline...")
        
        # Load preprocessed data
        preprocessor = DataPreprocessor()
        processed_data_dir = Path(self.config['data']['processed_data_path'])
        
        try:
            # Load training and validation data
            X_train = pd.read_csv(processed_data_dir / "X_train.csv")
            X_val = pd.read_csv(processed_data_dir / "X_val.csv")
            y_train = pd.read_csv(processed_data_dir / "y_train.csv").iloc[:, 0]
            y_val = pd.read_csv(processed_data_dir / "y_val.csv").iloc[:, 0]
            
            self.feature_names = X_train.columns.tolist()
            self.logger.info(f"Loaded data - Train: {X_train.shape}, Val: {X_val.shape}")
            
        except Exception as e:
            self.logger.error(f"Error loading processed data: {str(e)}")
            # If processed data not available, run preprocessing
            self.logger.info("Running preprocessing pipeline...")
            X_train, X_val, X_test, y_train, y_val, y_test, quality_report = preprocessor.preprocess_pipeline()
            self.feature_names = X_train.columns.tolist()
        
        # Initialize models
        self.initialize_models()
        
        # Perform cross-validation
        cv_results = self.perform_cross_validation(X_train, y_train)
        
        # Hyperparameter tuning
        tuning_results = self.hyperparameter_tuning(X_train, y_train)
        
        # Train final models
        trained_models = self.train_models(X_train, y_train)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(X_val, y_val)
        
        # Save models and results
        self.save_models()
        
        pipeline_results = {
            'cv_results': cv_results,
            'tuning_results': tuning_results,
            'evaluation_results': evaluation_results,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score
        }
        
        self.logger.info("Full training pipeline completed successfully")
        return pipeline_results


class ModelPredictor:
    """
    Model predictor for making predictions on new data
    """
    
    def __init__(self, config_path: str = None, model_path: str = None):
        """Initialize the predictor"""
        self.config = self._load_config(config_path)
        self.model = None
        self.preprocessor = None
        
        # Set up logging
        self._setup_logging()
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_path: str = None):
        """Load trained model and preprocessor"""
        if model_path is None:
            model_path = self.config['paths']['models_dir']
        
        try:
            # Load model
            if Path(model_path).is_file():
                self.model = joblib.load(model_path)
            else:
                best_model_path = Path(model_path) / "best_model.pkl"
                self.model = joblib.load(best_model_path)
            
            # Load preprocessor
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessor(model_path)
            
            self.logger.info("Model and preprocessor loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded first")
        
        try:
            # Preprocess data
            X_processed = self.preprocessor.preprocess_new_data(data)
            
            # Make predictions
            if return_proba and hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(X_processed)[:, 1]
            else:
                predictions = self.model.predict(X_processed)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_with_confidence(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions with confidence scores"""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded first")
        
        try:
            # Preprocess data
            X_processed = self.preprocessor.preprocess_new_data(data)
            
            # Make predictions
            predictions = self.model.predict(X_processed)
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_processed)
                confidence = np.max(probabilities, axis=1)
                fraud_probability = probabilities[:, 1]
            else:
                confidence = np.ones(len(predictions))
                fraud_probability = predictions.astype(float)
            
            return {
                'predictions': predictions,
                'fraud_probability': fraud_probability,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error making predictions with confidence: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    results = trainer.full_training_pipeline()
    
    print("Training pipeline completed!")
    print(f"Best model: {results['best_model_name']}")
    print(f"Best score: {results['best_score']:.4f}")
