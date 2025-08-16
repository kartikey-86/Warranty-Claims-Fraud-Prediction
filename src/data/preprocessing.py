"""
Warranty Claims Fraud Detection - Data Preprocessing Pipeline
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import joblib
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for warranty claims fraud detection
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the preprocessor with configuration"""
        self.config = self._load_config(config_path)
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.target_column = self.config['features']['target_column']
        
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
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load raw data from CSV file"""
        if file_path is None:
            file_path = self.config['data']['raw_data_path']
        
        try:
            self.logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Remove index column if exists
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)
            
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def data_quality_check(self, df: pd.DataFrame) -> dict:
        """Perform comprehensive data quality assessment"""
        self.logger.info("Performing data quality checks...")
        
        quality_report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'target_distribution': df[self.target_column].value_counts().to_dict() if self.target_column in df.columns else None
        }
        
        # Check for anomalies in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        quality_report['outliers'] = {}
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            quality_report['outliers'][col] = len(outliers)
        
        self.logger.info("Data quality check completed")
        return quality_report
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset"""
        self.logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # Remove duplicates
        initial_shape = df_clean.shape
        df_clean = df_clean.drop_duplicates()
        if df_clean.shape[0] != initial_shape[0]:
            self.logger.info(f"Removed {initial_shape[0] - df_clean.shape[0]} duplicate rows")
        
        # Map issue codes to meaningful labels (as seen in original notebook)
        issue_mapping = {0: 'No Issue', 1: 'repair', 2: 'replacement'}
        issue_columns = ['AC_1001_Issue', 'AC_1002_Issue', 'AC_1003_Issue', 
                        'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue']
        
        for col in issue_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map(issue_mapping)
        
        # Handle missing values
        self._handle_missing_values(df_clean)
        
        # Business rule validations
        self._apply_business_rules(df_clean)
        
        self.logger.info("Data cleaning completed")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame):
        """Handle missing values in the dataset"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Handle numerical missing values
        if len(numerical_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
        
        # Handle categorical missing values
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    def _apply_business_rules(self, df: pd.DataFrame):
        """Apply business rule validations"""
        rules = self.config['business_rules']
        
        # Validate claim amounts
        df['Claim_Value'] = df['Claim_Value'].clip(0, rules['max_claim_amount'])
        
        # Validate product age
        if 'Product_Age' in df.columns:
            df['Product_Age'] = df['Product_Age'].clip(rules['min_product_age'], rules['max_product_age'])
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features and transform existing ones"""
        self.logger.info("Starting feature engineering...")
        df_features = df.copy()
        
        if self.config['features']['create_derived_features']:
            # Create claim value categories
            df_features['Claim_Value_Category'] = pd.cut(
                df_features['Claim_Value'], 
                bins=[0, 5000, 15000, 30000, float('inf')], 
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
            
            # Create product age categories
            if 'Product_Age' in df_features.columns:
                df_features['Product_Age_Category'] = pd.cut(
                    df_features['Product_Age'], 
                    bins=[0, 30, 365, 1095, float('inf')], 
                    labels=['New', 'Recent', 'Old', 'Very_Old']
                )
            
            # Create service center risk score (simplified)
            service_center_fraud_rate = df_features.groupby('Service_Centre')[self.target_column].mean()
            df_features['Service_Centre_Risk'] = df_features['Service_Centre'].map(service_center_fraud_rate)
            
            # Create region-product interaction
            df_features['Region_Product'] = df_features['Region'] + '_' + df_features['Product_type']
            
            # Create claim ratio feature
            df_features['Claim_Product_Ratio'] = df_features['Claim_Value'] / (df_features['Product_Age'] + 1)
        
        self.logger.info("Feature engineering completed")
        return df_features
    
    def encode_features(self, df: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        self.logger.info("Encoding categorical features...")
        df_encoded = df.copy()
        
        categorical_features = self.config['features']['categorical_features']
        
        # Add derived categorical features if they exist
        if 'Claim_Value_Category' in df_encoded.columns:
            categorical_features.append('Claim_Value_Category')
        if 'Product_Age_Category' in df_encoded.columns:
            categorical_features.append('Product_Age_Category')
        if 'Region_Product' in df_encoded.columns:
            categorical_features.append('Region_Product')
        
        for feature in categorical_features:
            if feature in df_encoded.columns:
                if fit_encoders:
                    encoder = LabelEncoder()
                    df_encoded[feature] = encoder.fit_transform(df_encoded[feature].astype(str))
                    self.label_encoders[feature] = encoder
                else:
                    if feature in self.label_encoders:
                        # Handle unseen categories
                        try:
                            df_encoded[feature] = self.label_encoders[feature].transform(df_encoded[feature].astype(str))
                        except ValueError:
                            # For unseen categories, use most frequent class
                            most_frequent = self.label_encoders[feature].classes_[0]
                            df_encoded[feature] = df_encoded[feature].astype(str).apply(
                                lambda x: x if x in self.label_encoders[feature].classes_ else most_frequent
                            )
                            df_encoded[feature] = self.label_encoders[feature].transform(df_encoded[feature])
        
        self.logger.info("Categorical encoding completed")
        return df_encoded
    
    def scale_features(self, X: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        self.logger.info("Scaling numerical features...")
        
        scaling_method = self.config['features']['scaling_method']
        
        if fit_scaler:
            if scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaling_method == 'robust':
                self.scaler = RobustScaler()
            
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            if self.scaler is not None:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                self.logger.warning("Scaler not fitted yet")
                X_scaled = X
        
        self.logger.info("Feature scaling completed")
        return X_scaled
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Handle class imbalance using various techniques"""
        if not self.config['features']['handle_class_imbalance']:
            return X, y
        
        self.logger.info("Handling class imbalance...")
        
        # Check initial class distribution
        initial_distribution = y.value_counts()
        self.logger.info(f"Initial class distribution: {initial_distribution.to_dict()}")
        
        # Use SMOTE for oversampling
        smote = SMOTE(random_state=self.config['data']['random_state'])
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        final_distribution = pd.Series(y_resampled).value_counts()
        self.logger.info(f"Final class distribution: {final_distribution.to_dict()}")
        
        return X_resampled, y_resampled
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Split data into train, validation, and test sets"""
        self.logger.info("Splitting data...")
        
        test_size = self.config['data']['test_size']
        val_size = self.config['data']['validation_size']
        random_state = self.config['data']['random_state']
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        self.logger.info(f"Data split completed - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, save_dir: str = None):
        """Save the preprocessor artifacts"""
        if save_dir is None:
            save_dir = self.config['paths']['models_dir']
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, Path(save_dir) / "scaler.pkl")
            self.logger.info("Scaler saved successfully")
        
        # Save label encoders
        if self.label_encoders:
            joblib.dump(self.label_encoders, Path(save_dir) / "label_encoders.pkl")
            self.logger.info("Label encoders saved successfully")
        
        # Save feature names
        if self.feature_names is not None:
            joblib.dump(self.feature_names, Path(save_dir) / "feature_names.pkl")
            self.logger.info("Feature names saved successfully")
    
    def load_preprocessor(self, load_dir: str = None):
        """Load the preprocessor artifacts"""
        if load_dir is None:
            load_dir = self.config['paths']['models_dir']
        
        try:
            # Load scaler
            scaler_path = Path(load_dir) / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info("Scaler loaded successfully")
            
            # Load label encoders
            encoders_path = Path(load_dir) / "label_encoders.pkl"
            if encoders_path.exists():
                self.label_encoders = joblib.load(encoders_path)
                self.logger.info("Label encoders loaded successfully")
            
            # Load feature names
            features_path = Path(load_dir) / "feature_names.pkl"
            if features_path.exists():
                self.feature_names = joblib.load(features_path)
                self.logger.info("Feature names loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Error loading preprocessor artifacts: {str(e)}")
            raise
    
    def preprocess_pipeline(self, df: pd.DataFrame = None, save_artifacts: bool = True) -> tuple:
        """Complete preprocessing pipeline"""
        self.logger.info("Starting complete preprocessing pipeline...")
        
        # Load data if not provided
        if df is None:
            df = self.load_data()
        
        # Data quality check
        quality_report = self.data_quality_check(df)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Feature engineering
        df_features = self.feature_engineering(df_clean)
        
        # Separate features and target
        X = df_features.drop(columns=[self.target_column])
        y = df_features[self.target_column]
        
        # Encode categorical features
        X_encoded = self.encode_features(X, fit_encoders=True)
        
        # Store feature names
        self.feature_names = X_encoded.columns.tolist()
        
        # Scale features
        X_scaled = self.scale_features(X_encoded, fit_scaler=True)
        
        # Handle class imbalance
        X_resampled, y_resampled = self.handle_class_imbalance(X_scaled, y)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_resampled, y_resampled)
        
        # Save preprocessing artifacts
        if save_artifacts:
            self.save_preprocessor()
            
            # Save processed data
            processed_data_dir = Path(self.config['data']['processed_data_path'])
            processed_data_dir.mkdir(parents=True, exist_ok=True)
            
            X_train.to_csv(processed_data_dir / "X_train.csv", index=False)
            X_val.to_csv(processed_data_dir / "X_val.csv", index=False)
            X_test.to_csv(processed_data_dir / "X_test.csv", index=False)
            pd.Series(y_train).to_csv(processed_data_dir / "y_train.csv", index=False)
            pd.Series(y_val).to_csv(processed_data_dir / "y_val.csv", index=False)
            pd.Series(y_test).to_csv(processed_data_dir / "y_test.csv", index=False)
            
            self.logger.info("Processed data saved successfully")
        
        self.logger.info("Preprocessing pipeline completed successfully")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, quality_report
    
    def preprocess_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess new data using fitted transformers"""
        self.logger.info("Preprocessing new data...")
        
        # Clean data (without target)
        df_clean = df.copy()
        
        # Apply same cleaning steps
        issue_mapping = {0: 'No Issue', 1: 'repair', 2: 'replacement'}
        issue_columns = ['AC_1001_Issue', 'AC_1002_Issue', 'AC_1003_Issue', 
                        'TV_2001_Issue', 'TV_2002_Issue', 'TV_2003_Issue']
        
        for col in issue_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map(issue_mapping)
        
        # Handle missing values
        self._handle_missing_values(df_clean)
        
        # Apply business rules
        self._apply_business_rules(df_clean)
        
        # Feature engineering
        df_features = self.feature_engineering(df_clean)
        
        # Remove target column if present
        if self.target_column in df_features.columns:
            df_features = df_features.drop(columns=[self.target_column])
        
        # Encode features
        X_encoded = self.encode_features(df_features, fit_encoders=False)
        
        # Scale features
        X_scaled = self.scale_features(X_encoded, fit_scaler=False)
        
        self.logger.info("New data preprocessing completed")
        return X_scaled


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, quality_report = preprocessor.preprocess_pipeline()
    
    print("Preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
