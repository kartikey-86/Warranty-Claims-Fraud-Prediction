"""
Warranty Claims Fraud Detection - Model Evaluation Framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML evaluation libraries
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
import joblib

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization framework
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the model evaluator"""
        self.config = self._load_config(config_path)
        self.evaluation_results = {}
        self.models = {}
        self.plots_dir = Path(self.config['paths']['plots_dir'])
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
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
        self.logger = logging.getLogger(__name__)
    
    def load_models(self, models_dir: str = None) -> Dict[str, Any]:
        """Load all trained models"""
        if models_dir is None:
            models_dir = self.config['paths']['models_dir']
        
        models_path = Path(models_dir)
        loaded_models = {}
        
        try:
            # Load best model
            best_model_path = models_path / "best_model.pkl"
            if best_model_path.exists():
                loaded_models['best_model'] = joblib.load(best_model_path)
                
                # Load metadata
                metadata_path = models_path / "best_model_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    loaded_models['best_model_metadata'] = metadata
            
            # Load individual models
            for model_file in models_path.glob("*_model.pkl"):
                model_name = model_file.stem.replace('_model', '')
                loaded_models[model_name] = joblib.load(model_file)
            
            self.models = loaded_models
            self.logger.info(f"Loaded {len(loaded_models)} models")
            return loaded_models
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Class-specific metrics
        metrics['precision_class_0'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics['precision_class_1'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['recall_class_0'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics['recall_class_1'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['f1_class_0'] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        metrics['f1_class_1'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
            except Exception as e:
                self.logger.warning(f"Could not calculate probability-based metrics: {str(e)}")
        
        return metrics
    
    def evaluate_model_performance(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                                 model_name: str = "Model") -> Dict[str, Any]:
        """Evaluate a single model's performance"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # ROC and Precision-Recall curves
            roc_data = None
            pr_data = None
            
            if y_pred_proba is not None:
                fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
                precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
                
                roc_data = {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds}
                pr_data = {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
            
            results = {
                'model_name': model_name,
                'metrics': metrics,
                'confusion_matrix': cm,
                'classification_report': class_report,
                'predictions': y_pred,
                'predictions_proba': y_pred_proba,
                'roc_data': roc_data,
                'pr_data': pr_data
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, 
                            class_names: List[str] = None) -> go.Figure:
        """Create an interactive confusion matrix plot"""
        if class_names is None:
            class_names = ['Legitimate', 'Fraudulent']
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotation text
        annotations = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                annotations.append(
                    f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)"
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=np.array(annotations).reshape(cm.shape),
            texttemplate="%{text}",
            textfont={"size": 12},
            showscale=True
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            width=500,
            height=400
        )
        
        return fig
    
    def plot_roc_curve(self, roc_data_dict: Dict[str, Dict]) -> go.Figure:
        """Create ROC curve comparison plot"""
        fig = go.Figure()
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier'
        ))
        
        # Add ROC curves for each model
        for model_name, roc_data in roc_data_dict.items():
            if roc_data is not None:
                auc_score = roc_auc_score(roc_data.get('y_true', []), roc_data.get('y_scores', []))
                fig.add_trace(go.Scatter(
                    x=roc_data['fpr'],
                    y=roc_data['tpr'],
                    mode='lines',
                    name=f'{model_name} (AUC = {auc_score:.3f})'
                ))
        
        fig.update_layout(
            title='ROC Curve Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_precision_recall_curve(self, pr_data_dict: Dict[str, Dict]) -> go.Figure:
        """Create Precision-Recall curve comparison plot"""
        fig = go.Figure()
        
        # Add PR curves for each model
        for model_name, pr_data in pr_data_dict.items():
            if pr_data is not None:
                avg_precision = average_precision_score(
                    pr_data.get('y_true', []), 
                    pr_data.get('y_scores', [])
                )
                fig.add_trace(go.Scatter(
                    x=pr_data['recall'],
                    y=pr_data['precision'],
                    mode='lines',
                    name=f'{model_name} (AP = {avg_precision:.3f})'
                ))
        
        fig.update_layout(
            title='Precision-Recall Curve Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                               model_name: str, top_n: int = 20) -> go.Figure:
        """Create feature importance plot"""
        try:
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                return None
            
            # Create feature importance DataFrame
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Create bar plot
            fig = go.Figure(data=go.Bar(
                x=feature_imp_df['importance'],
                y=feature_imp_df['feature'],
                orientation='h'
            ))
            
            fig.update_layout(
                title=f'Top {top_n} Feature Importance - {model_name}',
                xaxis_title='Importance',
                yaxis_title='Features',
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            self.logger.warning(f"Could not create feature importance plot: {str(e)}")
            return None
    
    def plot_learning_curves(self, model, X: pd.DataFrame, y: pd.Series, 
                           model_name: str, cv: int = 5) -> go.Figure:
        """Create learning curves plot"""
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=cv, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                random_state=42
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            fig = go.Figure()
            
            # Training score
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode='lines+markers',
                name='Training Score',
                line=dict(color='blue'),
                error_y=dict(type='data', array=train_std, visible=True)
            ))
            
            # Validation score
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=val_mean,
                mode='lines+markers',
                name='Validation Score',
                line=dict(color='red'),
                error_y=dict(type='data', array=val_std, visible=True)
            ))
            
            fig.update_layout(
                title=f'Learning Curves - {model_name}',
                xaxis_title='Training Set Size',
                yaxis_title='Score',
                width=600,
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.logger.warning(f"Could not create learning curves: {str(e)}")
            return None
    
    def create_metrics_comparison_plot(self, evaluation_results: Dict[str, Dict]) -> go.Figure:
        """Create model metrics comparison plot"""
        models = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        auc_scores = []
        
        for model_name, results in evaluation_results.items():
            metrics = results.get('metrics', {})
            models.append(model_name)
            accuracy_scores.append(metrics.get('accuracy', 0))
            precision_scores.append(metrics.get('precision', 0))
            recall_scores.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))
            auc_scores.append(metrics.get('roc_auc', 0))
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Summary'],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add bar charts
        fig.add_trace(go.Bar(x=models, y=accuracy_scores, name='Accuracy'), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=precision_scores, name='Precision'), row=1, col=2)
        fig.add_trace(go.Bar(x=models, y=recall_scores, name='Recall'), row=1, col=3)
        fig.add_trace(go.Bar(x=models, y=f1_scores, name='F1-Score'), row=2, col=1)
        fig.add_trace(go.Bar(x=models, y=auc_scores, name='ROC AUC'), row=2, col=2)
        
        # Add summary radar chart
        metrics_df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracy_scores,
            'Precision': precision_scores,
            'Recall': recall_scores,
            'F1-Score': f1_scores,
            'ROC AUC': auc_scores
        })
        
        # Create radar chart for best performing models
        best_model_idx = np.argmax(f1_scores)
        best_model_metrics = [accuracy_scores[best_model_idx], precision_scores[best_model_idx], 
                            recall_scores[best_model_idx], f1_scores[best_model_idx], 
                            auc_scores[best_model_idx]]
        
        fig.add_trace(go.Scatterpolar(
            r=best_model_metrics,
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
            fill='toself',
            name=f'Best Model ({models[best_model_idx]})'
        ), row=2, col=3)
        
        fig.update_layout(
            title='Model Performance Comparison',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Dict], 
                                 save_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        report = {
            'summary': {},
            'detailed_results': evaluation_results,
            'recommendations': []
        }
        
        # Find best performing model
        best_model = None
        best_score = -1
        primary_metric = self.config['model_selection']['primary_metric']
        
        for model_name, results in evaluation_results.items():
            score = results.get('metrics', {}).get(primary_metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        report['summary']['best_model'] = best_model
        report['summary']['best_score'] = best_score
        report['summary']['primary_metric'] = primary_metric
        
        # Model comparison
        comparison_data = []
        for model_name, results in evaluation_results.items():
            metrics = results.get('metrics', {})
            comparison_data.append({
                'model': model_name,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'roc_auc': metrics.get('roc_auc', 0)
            })
        
        report['summary']['model_comparison'] = comparison_data
        
        # Business recommendations
        if best_model and best_score > 0.8:
            report['recommendations'].append("Model performance is good and ready for deployment.")
        elif best_model and best_score > 0.7:
            report['recommendations'].append("Model performance is acceptable but could benefit from further tuning.")
        else:
            report['recommendations'].append("Model performance needs improvement before deployment.")
        
        # Class imbalance analysis
        for model_name, results in evaluation_results.items():
            metrics = results.get('metrics', {})
            recall_fraud = metrics.get('recall_class_1', 0)
            precision_fraud = metrics.get('precision_class_1', 0)
            
            if recall_fraud < 0.7:
                report['recommendations'].append(
                    f"{model_name}: Low fraud recall ({recall_fraud:.3f}). Consider adjusting decision threshold or addressing class imbalance."
                )
            
            if precision_fraud < 0.7:
                report['recommendations'].append(
                    f"{model_name}: Low fraud precision ({precision_fraud:.3f}). Model may generate too many false positives."
                )
        
        # Save report
        if save_path is None:
            save_path = self.config['paths']['reports_dir']
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        with open(Path(save_path) / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation report saved to {save_path}")
        return report
    
    def save_plots(self, figures: Dict[str, go.Figure], format: str = 'html'):
        """Save all plots to files"""
        for plot_name, fig in figures.items():
            if fig is not None:
                if format == 'html':
                    fig.write_html(self.plots_dir / f"{plot_name}.html")
                elif format == 'png':
                    fig.write_image(self.plots_dir / f"{plot_name}.png")
                elif format == 'pdf':
                    fig.write_image(self.plots_dir / f"{plot_name}.pdf")
        
        self.logger.info(f"Plots saved to {self.plots_dir}")
    
    def comprehensive_evaluation(self, X_test: pd.DataFrame, y_test: pd.Series, 
                               feature_names: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation on all models"""
        self.logger.info("Starting comprehensive model evaluation...")
        
        if not self.models:
            self.load_models()
        
        if feature_names is None:
            feature_names = X_test.columns.tolist()
        
        evaluation_results = {}
        all_figures = {}
        
        # Evaluate each model
        for model_name, model in self.models.items():
            if model_name == 'best_model_metadata':
                continue
                
            self.logger.info(f"Evaluating {model_name}...")
            
            try:
                results = self.evaluate_model_performance(model, X_test, y_test, model_name)
                evaluation_results[model_name] = results
                
                # Create confusion matrix plot
                cm_fig = self.plot_confusion_matrix(results['confusion_matrix'], model_name)
                all_figures[f'confusion_matrix_{model_name}'] = cm_fig
                
                # Create feature importance plot
                fi_fig = self.plot_feature_importance(model, feature_names, model_name)
                if fi_fig:
                    all_figures[f'feature_importance_{model_name}'] = fi_fig
                
                # Create learning curves (if we have training data)
                # lc_fig = self.plot_learning_curves(model, X_test, y_test, model_name)
                # if lc_fig:
                #     all_figures[f'learning_curves_{model_name}'] = lc_fig
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        # Create comparison plots
        if len(evaluation_results) > 1:
            comparison_fig = self.create_metrics_comparison_plot(evaluation_results)
            all_figures['metrics_comparison'] = comparison_fig
            
            # Collect ROC and PR data for comparison
            roc_data_dict = {}
            pr_data_dict = {}
            
            for model_name, results in evaluation_results.items():
                if results['roc_data']:
                    roc_data_dict[model_name] = results['roc_data']
                    roc_data_dict[model_name]['y_true'] = y_test
                    roc_data_dict[model_name]['y_scores'] = results['predictions_proba']
                
                if results['pr_data']:
                    pr_data_dict[model_name] = results['pr_data']
                    pr_data_dict[model_name]['y_true'] = y_test
                    pr_data_dict[model_name]['y_scores'] = results['predictions_proba']
            
            if roc_data_dict:
                roc_fig = self.plot_roc_curve(roc_data_dict)
                all_figures['roc_comparison'] = roc_fig
            
            if pr_data_dict:
                pr_fig = self.plot_precision_recall_curve(pr_data_dict)
                all_figures['pr_comparison'] = pr_fig
        
        # Generate evaluation report
        report = self.generate_evaluation_report(evaluation_results)
        
        # Save plots
        self.save_plots(all_figures)
        
        self.evaluation_results = evaluation_results
        
        self.logger.info("Comprehensive evaluation completed")
        
        return {
            'evaluation_results': evaluation_results,
            'figures': all_figures,
            'report': report
        }


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Load test data (this would be your actual test data)
    # X_test = pd.read_csv("data/processed/X_test.csv")
    # y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]
    
    # Run comprehensive evaluation
    # results = evaluator.comprehensive_evaluation(X_test, y_test)
    
    print("Model evaluation framework ready!")
