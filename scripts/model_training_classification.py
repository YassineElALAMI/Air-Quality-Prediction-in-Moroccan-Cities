import pandas as pd
import numpy as np
import os
import joblib
import logging
import warnings
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, validation_curve
)
from sklearn.preprocessing import (
    RobustScaler, StandardScaler, MinMaxScaler, 
    LabelEncoder, label_binarize
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE, chi2
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_score, recall_score, roc_auc_score, roc_curve, 
    precision_recall_curve, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from scipy import stats
import itertools

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ClassificationConfig:
    """Configuration class for classification model training.
    
    Attributes:
        test_size: Fraction of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility
        cv_folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs (-1 to use all cores)
        max_features: Maximum number of features to select
        outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
        scaling_method: Method for feature scaling ('standard', 'minmax', 'robust')
        feature_selection: Feature selection method ('selectk', 'rfe', 'both')
        handle_class_imbalance: Whether to handle class imbalance
        use_ensemble: Whether to create an ensemble of models
        scoring_metric: Metric for model evaluation
        early_stopping_rounds: Rounds to wait before early stopping (for XGBoost)
        verbosity: Level of logging verbosity (0-3)
    """
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1
    max_features: int = 15
    outlier_method: str = 'iqr'
    scaling_method: str = 'robust'
    feature_selection: str = 'selectk'
    handle_class_imbalance: bool = True
    use_ensemble: bool = True
    scoring_metric: str = 'f1_weighted'
    early_stopping_rounds: int = 10
    verbosity: int = 1  # 'accuracy', 'f1_weighted', 'f1_macro'

class ClassificationPipeline:
    """Enhanced Classification Pipeline for Air Quality Index prediction."""
    
    def __init__(self, config: ClassificationConfig = None):
        self.config = config or ClassificationConfig()
        self.setup_logging()
        self.setup_directories()
        self.results = []
        self.best_model = None
        self.selected_features = None
        self.label_encoder = None
        self.class_names = None
        self.class_weights = None
        
    def setup_logging(self):
        """Configure enhanced logging."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/classification_training_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories."""
        dirs = ["logs", "output", "models", "plots", "data"]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
            
    def load_and_validate_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load and validate input data for classification.
        
        Args:
            file_path: Path to the input CSV file
            
        Returns:
            pd.DataFrame: Loaded and validated DataFrame
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the data is empty or missing required columns
            Exception: For other data loading/validation errors
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")
                
            self.logger.info(f"Loading data from {file_path}")
            
            # Read CSV with optimized parameters
            df = pd.read_csv(
                file_path,
                parse_dates=['timestamp'],
                infer_datetime_format=True,
                low_memory=False
            )
            
            # Basic validation
            if df.empty:
                raise ValueError("Dataset is empty")
                
            # Check for required columns
            required_columns = ['pm2.5', 'temperature', 'humidity', 'pressure', 'wind_speed']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Check for target column (aqi or aqi_class)
            target_cols = ['aqi', 'aqi_class']
            available_targets = [col for col in target_cols if col in df.columns]
            if not available_targets:
                raise ValueError(f"No target column found. Expected one of: {target_cols}")
            
            # Basic data quality checks
            self._perform_data_quality_checks(df)
            
            self.logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            self.logger.info(f"Available target columns: {available_targets}")
            self.logger.info(f"Missing values per column:\n{df.isnull().sum()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in load_and_validate_data: {str(e)}")
            self.logger.exception("Stack trace:")
            raise
            
    def _perform_data_quality_checks(self, df: pd.DataFrame) -> None:
        """Perform data quality checks on the loaded DataFrame.
        
        Args:
            df: Input DataFrame to check
            
        Raises:
            ValueError: If data quality issues are found
        """
        # Check for duplicate rows
        if df.duplicated().sum() > 0:
            self.logger.warning(f"Found {df.duplicated().sum()} duplicate rows. Keeping first occurrence.")
            df.drop_duplicates(keep='first', inplace=True)
            
        # Check for constant columns
        constant_columns = [col for col in df.columns if df[col].nunique() == 1]
        if constant_columns:
            self.logger.warning(f"Found constant columns: {constant_columns}")
            
        # Check for high percentage of missing values
        missing_pct = df.isnull().mean() * 100
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            self.logger.warning(f"Columns with >50% missing values: {high_missing}")
            
        # Check numeric ranges
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if col.startswith('aqi'):
                continue
                
            min_val = df[col].min()
            max_val = df[col].max()
            
            if col == 'temperature':
                if min_val < -50 or max_val > 60:  # Reasonable temperature range in Celsius
                    self.logger.warning(f"Suspicious temperature range: {min_val} to {max_val}°C")
            elif col == 'humidity':
                if min_val < 0 or max_val > 100:  # Humidity should be 0-100%
                    self.logger.warning(f"Suspicious humidity range: {min_val} to {max_val}%")
            elif col == 'pressure':
                if min_val < 800 or max_val > 1100:  # Reasonable atmospheric pressure in hPa
                    self.logger.warning(f"Suspicious pressure range: {min_val} to {max_val} hPa")
            elif col == 'pm2.5':
                if min_val < 0 or max_val > 1000:  # Reasonable PM2.5 range
                    self.logger.warning(f"Suspicious PM2.5 range: {min_val} to {max_val} µg/m³")
            
    def _handle_single_class(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Handle single class scenario by creating meaningful classes from features.
        
        Args:
            X: Feature matrix
            y: Target series with single class
            
        Returns:
            Series with multiple classes
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        self.logger.warning("Only one class found. Attempting to create meaningful classes...")
        
        # Try to use AQI if available
        if 'aqi' in X.columns:
            aqi = X['aqi']
            if aqi.nunique() > 1:  # If we have varying AQI values
                bins = [float('-inf'), 50, 100, 150, 200, 300, float('inf')]
                labels = ['Good', 'Moderate', 'Unhealthy for Sensitive', 
                         'Unhealthy', 'Very Unhealthy', 'Hazardous']
                
                # If values are normalized, scale them to typical AQI range
                if aqi.between(-1, 1).mean() > 0.9:  # If most values are between -1 and 1
                    aqi = (aqi - aqi.min()) / (aqi.max() - aqi.min()) * 500  # Scale to 0-500
                
                y_new = pd.cut(aqi, bins=bins, labels=labels, include_lowest=True)
                self.logger.info("Created classes from AQI values:")
                self.logger.info(y_new.value_counts())
                return y_new
        
        # Try using PM2.5 if available
        if 'pm2.5' in X.columns:
            pm25 = X['pm2.5']
            if pm25.nunique() > 1:  # If we have varying PM2.5 values
                q1 = pm25.quantile(0.33)
                q2 = pm25.quantile(0.67)
                
                conditions = [
                    (pm25 <= q1),
                    (pm25 > q1) & (pm25 <= q2),
                    (pm25 > q2)
                ]
                
                classes = ['Low PM2.5', 'Medium PM2.5', 'High PM2.5']
                y_new = pd.Series(np.select(conditions, classes, default='Unknown'), index=X.index)
                
                self.logger.info("Created classes from PM2.5 levels:")
                self.logger.info(y_new.value_counts())
                return y_new
        
        # Fallback to KMeans clustering on all numeric features
        numeric_cols = X.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X[numeric_cols])
            
            # Determine optimal number of clusters (2-4)
            n_clusters = min(4, max(2, len(X) // 20))  # At least 20 samples per cluster
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            y_new = pd.Series([f'Cluster {c+1}' for c in clusters], index=X.index)
            
            self.logger.info(f"Created {n_clusters} clusters using KMeans:")
            self.logger.info(y_new.value_counts())
            return y_new
        
        # If all else fails, create two artificial classes
        self.logger.warning("Could not create meaningful classes. Splitting data in half.")
        y_new = pd.Series(['Class 1'] * len(y), index=y.index)
        y_new.iloc[len(y)//2:] = 'Class 2'
        return y_new

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the data for classification with improved single-class handling.
        
        Args:
            df: Input DataFrame containing the data
            
        Returns:
            Tuple of (features, target) where features is a DataFrame and target is a Series
            
        Raises:
            ValueError: If target column is not found or if there's an issue with the data
        """
        try:
            self.logger.info("Preprocessing data for classification...")
            
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Handle missing values
            self.logger.info("Handling missing values...")
            data = self.handle_missing_values(data)
            
            # Only remove outliers if we have enough data
            if self.config.outlier_method != 'none' and len(data) > 100:
                self.logger.info(f"Removing outliers using {self.config.outlier_method} method...")
                data, _ = self.detect_outliers(data, method=self.config.outlier_method)
            
            # Feature engineering
            self.logger.info("Engineering features...")
            data = self.feature_engineering(data)
            
            # Find target column
            target_col = self.config.target_column if hasattr(self.config, 'target_column') else 'aqi_class'
            if target_col not in data.columns:
                # Try to find a suitable target column
                possible_targets = [col for col in data.columns if 'aqi' in col.lower() or 'class' in col.lower()]
                if possible_targets:
                    target_col = possible_targets[0]
                    self.logger.warning(f"Using '{target_col}' as target column")
                else:
                    raise ValueError("No suitable target column found. Please specify target_column in config.")
            
            X = data.drop(columns=[target_col])
            y = data[target_col].astype(str)  # Ensure target is string
            
            # Log class distribution
            self.logger.info("Class distribution:")
            self.logger.info(y.value_counts())
            
            # Handle single class case
            if len(y.unique()) == 1:
                y = self._handle_single_class(X, y)
            
            # Final validation
            if len(y.unique()) < 2:
                raise ValueError(f"At least 2 classes are required for classification. "
                               f"Found only {len(y.unique())} class(es).")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing data: {str(e)}")
            raise
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced missing value handling."""
        self.logger.info("Handling missing values...")
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Handle numeric missing values
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            
        # Handle categorical missing values
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
            
        self.logger.info(f"Missing values after imputation: {df.isnull().sum().sum()}")
        return df
        
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced outlier removal with multiple methods."""
        self.logger.info(f"Removing outliers using {self.config.outlier_method} method...")
        initial_shape = df.shape[0]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude target columns from outlier removal
        exclude_cols = ['aqi', 'aqi_class', 'pm2.5']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols and df[col].nunique() > 2]
        
        if self.config.outlier_method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                
        elif self.config.outlier_method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < 3]
                
        removed_rows = initial_shape - df.shape[0]
        self.logger.info(f"Removed {removed_rows} outlier rows ({removed_rows/initial_shape*100:.2f}%)")
        
        return df
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for air quality classification."""
        self.logger.info("Engineering features...")
        
        # Time-based features if timestamp exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                           3: 1, 4: 1, 5: 1,   # Spring
                                           6: 2, 7: 2, 8: 2,   # Summer
                                           9: 3, 10: 3, 11: 3}) # Fall
            
        # Air quality specific features
        numeric_features = df.select_dtypes(include=[np.number]).columns
        pollutant_cols = [col for col in numeric_features if col not in ['aqi', 'aqi_class', 'pm2.5']]
        
        # Pollutant ratios and interactions
        if 'no2' in df.columns and 'o3' in df.columns:
            df['no2_o3_ratio'] = df['no2'] / (df['o3'] + 1e-8)
            df['no2_o3_interaction'] = df['no2'] * df['o3']
            
        if 'pm2.5' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm2.5'] / (df['pm10'] + 1e-8)
            
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_index'] = df['temperature'] / (df['humidity'] + 1e-8)
            
        # Create pollution severity categories
        if 'pm2.5' in df.columns:
            df['pm25_category'] = pd.cut(df['pm2.5'], 
                                       bins=[0, 12, 35, 55, 150, float('inf')],
                                       labels=['Good', 'Moderate', 'Unhealthy_Sensitive', 'Unhealthy', 'Very_Unhealthy'])
            # One-hot encode the categories
            pm25_dummies = pd.get_dummies(df['pm25_category'], prefix='pm25')
            df = pd.concat([df, pm25_dummies], axis=1)
            df.drop('pm25_category', axis=1, inplace=True)
            
        # Aggregate pollution index
        pollutant_numeric = [col for col in pollutant_cols if col in df.columns]
        if len(pollutant_numeric) > 2:
            df['pollution_index'] = df[pollutant_numeric].mean(axis=1)
            df['pollution_std'] = df[pollutant_numeric].std(axis=1)
            
        self.logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df.dropna()
        
    def prepare_target_variable(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:
        """Prepare and encode target variable."""
        # Determine target column
        if 'aqi_class' in df.columns:
            target_col = 'aqi_class'
            y = df[target_col]
        elif 'aqi' in df.columns:
            target_col = 'aqi'
            # Convert numeric AQI to categories if needed
            if df[target_col].dtype in ['int64', 'float64']:
                # Create AQI categories based on EPA standards
                y = pd.cut(df[target_col], 
                          bins=[0, 50, 100, 150, 200, 300, float('inf')],
                          labels=['Good', 'Moderate', 'Unhealthy_Sensitive', 'Unhealthy', 'Very_Unhealthy', 'Hazardous'])
            else:
                y = df[target_col]
        else:
            raise ValueError("No suitable target column found")
            
        # Ensure we have categorical values
        if pd.api.types.is_numeric_dtype(y):
            raise ValueError("Target variable must be categorical for classification")
            
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y.astype(str))
        self.class_names = self.label_encoder.classes_.tolist()
        
        self.logger.info(f"Target variable: {target_col}")
        self.logger.info(f"Classes: {self.class_names}")
        self.logger.info(f"Class distribution:\n{pd.Series(y_encoded).value_counts().sort_index()}")
        
        return df, pd.Series(y_encoded), target_col
        
    def handle_class_imbalance(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights to handle imbalanced data."""
        if self.config.handle_class_imbalance:
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            self.class_weights = dict(zip(classes, class_weights))
            self.logger.info(f"Class weights: {self.class_weights}")
            return self.class_weights
        return None
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Enhanced feature selection for classification."""
        self.logger.info(f"Selecting features using {self.config.feature_selection} method...")
        
        # Separate numeric and categorical features
        numeric_features = X.select_dtypes(include=['number']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Handle numeric features
        X_numeric = X[numeric_features]
        
        if self.config.feature_selection == 'selectk':
            if len(numeric_features) > 0:
                selector = SelectKBest(
                    score_func=f_classif, 
                    k=min(self.config.max_features, len(numeric_features))
                )
                X_selected = selector.fit_transform(X_numeric, y)
                selected_numeric = X_numeric.columns[selector.get_support()].tolist()
            else:
                selected_numeric = []
            
        elif self.config.feature_selection == 'rfe':
            if len(numeric_features) > 0:
                estimator = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
                selector = RFE(
                    estimator=estimator, 
                    n_features_to_select=min(self.config.max_features, len(numeric_features))
                )
                X_selected = selector.fit_transform(X_numeric, y)
                selected_numeric = X_numeric.columns[selector.get_support()].tolist()
            else:
                selected_numeric = []
            
        elif self.config.feature_selection == 'both':
            if len(numeric_features) > 0:
                # First use SelectKBest
                selector1 = SelectKBest(score_func=f_classif, k=min(20, len(numeric_features)))
                X_temp = selector1.fit_transform(X_numeric, y)
                temp_features = X_numeric.columns[selector1.get_support()]
                
                # Then use RFE on selected features
                estimator = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
                selector2 = RFE(estimator=estimator, 
                               n_features_to_select=min(self.config.max_features, len(temp_features)))
                X_selected = selector2.fit_transform(X_temp, y)
                selected_numeric = temp_features[selector2.get_support()].tolist()
            else:
                selected_numeric = []
            
        # Combine selected numeric features with all categorical features
        selected_features = list(selected_numeric) + list(categorical_features)
        self.selected_features = selected_features
        
        self.logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        return X[selected_features], selected_features
        
    def get_scaler(self, X=None):
        """Get the appropriate scaler based on configuration."""
        # Determine which features are numeric and categorical
        if X is not None and hasattr(X, 'dtypes'):
            numeric_features = X.select_dtypes(include=['number']).columns.tolist()
            categorical_features = [f for f in X.columns if f not in numeric_features]
        elif hasattr(self, 'X_train') and hasattr(self.X_train, 'dtypes'):
            numeric_features = self.X_train.select_dtypes(include=['number']).columns.tolist()
            categorical_features = [f for f in self.X_train.columns if f not in numeric_features]
        else:
            raise ValueError("No data provided to determine feature types")
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
        
    def get_models(self) -> Dict[str, Any]:
        """Get enhanced model dictionary with class weight handling.
        
        Returns:
            Dictionary containing three optimized classification models:
            - Logistic Regression: Good baseline model with elasticnet regularization
            - Random Forest: Robust ensemble method with balanced class weights
            - XGBoost: High-performance gradient boosting with optimized parameters
        """
        # Create a pipeline that includes the preprocessor and the model
        def make_pipeline(estimator):
            return Pipeline([
                ('preprocessor', self.get_scaler()),
                ('classifier', estimator)
            ])
            
        # Handle class weights for XGBoost
        scale_pos_weight = None
        if self.class_weights is not None and len(self.class_weights) > 1:
            # Calculate scale_pos_weight for binary classification
            scale_pos_weight = self.class_weights[1] / self.class_weights[0] \
                if len(self.class_weights) == 2 else 1.0
        
        models = {
            # 1. Logistic Regression with elasticnet regularization
            "Logistic Regression": make_pipeline(
                LogisticRegression(
                    class_weight=self.class_weights,
                    max_iter=1000,
                    random_state=self.config.random_state,
                    solver='saga',
                    penalty='elasticnet',
                    l1_ratio=0.5,
                    n_jobs=self.config.n_jobs,
                    verbose=self.config.verbosity
                )
            ),
            
            # 2. Random Forest with balanced class weights
            "Random Forest": make_pipeline(
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight='balanced_subsample' if self.config.handle_class_imbalance else 'balanced',
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    verbose=self.config.verbosity,
                    bootstrap=True,
                    oob_score=True,
                    warm_start=True,
                    max_features='sqrt'
                )
            ),
            
            # 3. XGBoost with optimized parameters and early stopping
            "XGBoost": make_pipeline(
                XGBClassifier(
                    n_estimators=1000,  # Will use early stopping
                    learning_rate=0.1,
                    max_depth=7,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    scale_pos_weight=scale_pos_weight,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    verbosity=self.config.verbosity,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    tree_method='hist',  # Faster training
                    enable_categorical=False
                )
            )
        }
        return models

        
    def evaluate_classification(self, model: Any, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, Any]:
        """Comprehensive classification evaluation with cross-validation.
        
        Args:
            model: The trained model to evaluate
            X: Features for evaluation
            y: True labels
            model_name: Name of the model for logging
            
        Returns:
            Dictionary containing evaluation metrics and predictions
        """
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Calculate base metrics
        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y, y_pred),
            "MCC": matthews_corrcoef(y, y_pred),
            "Cohen's Kappa": cohen_kappa_score(y, y_pred)
        }
        
        # Add per-class and weighted metrics for multi-class
        if len(np.unique(y)) > 2:
            metrics.update({
                "Precision (weighted)": precision_score(y, y_pred, average='weighted', zero_division=0),
                "Recall (weighted)": recall_score(y, y_pred, average='weighted', zero_division=0),
                "F1 Score (weighted)": f1_score(y, y_pred, average='weighted', zero_division=0),
                "Precision (macro)": precision_score(y, y_pred, average='macro', zero_division=0),
                "Recall (macro)": recall_score(y, y_pred, average='macro', zero_division=0),
                "F1 Score (macro)": f1_score(y, y_pred, average='macro', zero_division=0)
            })
        else:
            metrics.update({
                "Precision": precision_score(y, y_pred, average='binary', zero_division=0),
                "Recall": recall_score(y, y_pred, average='binary', zero_division=0),
                "F1 Score": f1_score(y, y_pred, average='binary', zero_division=0)
            })
        
        # Add ROC AUC if probability estimates are available
        if y_pred_proba is not None:
            if len(np.unique(y)) == 2:  # Binary classification
                metrics["ROC AUC"] = roc_auc_score(y, y_pred_proba[:, 1])
            else:  # Multi-class classification
                try:
                    # One-vs-Rest ROC AUC
                    y_bin = label_binarize(y, classes=np.unique(y))
                    n_classes = y_bin.shape[1]
                    
                    # Compute ROC curve and ROC area for each class
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                        roc_auc[i] = roc_auc_score(y_bin[:, i], y_pred_proba[:, i])
                    
                    # Compute micro-average ROC curve and ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_pred_proba.ravel())
                    roc_auc["micro"] = roc_auc_score(y_bin, y_pred_proba, multi_class="ovr", average="micro")
                    
                    metrics["ROC AUC (micro)"] = roc_auc["micro"]
                    metrics["ROC AUC (macro)"] = np.mean(list(roc_auc.values()))
                    
                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC AUC: {str(e)}")
        
        # Calculate cross-validated metrics if sample size allows
        if len(y) >= 100:  # Only if we have enough samples
            try:
                cv_scores = cross_val_score(
                    model, X, y, 
                    cv=StratifiedKFold(n_splits=min(5, len(np.unique(y))), 
                                    shuffle=True, 
                                    random_state=self.config.random_state),
                    scoring='accuracy',
                    n_jobs=self.config.n_jobs
                )
                metrics["CV Accuracy (mean)"] = np.mean(cv_scores)
                metrics["CV Accuracy (std)"] = np.std(cv_scores)
            except Exception as e:
                self.logger.warning(f"Could not perform cross-validation: {str(e)}")
        
            metrics["ROC AUC"] = roc_auc_score(y, y_pred_proba[:, 1])
            
        # Store predictions and probabilities for plotting
        metrics["y_true"] = y
        metrics["y_pred"] = y_pred
        metrics["y_pred_proba"] = y_pred_proba
        
        return metrics
        
    def hyperparameter_tuning(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Perform hyperparameter tuning for specific models using cross-validation.
        
        Args:
            model: The model to tune
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            
        Returns:
            The best estimator with tuned hyperparameters
        """
        # Define parameter grids for different models
        param_grids = {
            'Logistic Regression': {
                'classifier__C': [0.01, 0.1, 1, 10, 100],
                'classifier__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'classifier__solver': ['saga'],
                'classifier__penalty': ['elasticnet']
            },
            'Random Forest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [5, 10, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__max_features': ['sqrt', 'log2'],
                'classifier__bootstrap': [True],
                'classifier__oob_score': [True]
            },
            'XGBoost': {
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.6, 0.8, 1.0],
                'classifier__colsample_bytree': [0.6, 0.8, 1.0],
                'classifier__reg_alpha': [0, 0.1, 1],
                'classifier__reg_lambda': [0, 0.1, 1]
            }
        }
        
        # Skip tuning if model not in our grid
        param_grid = None
        for name, grid in param_grids.items():
            if name in model_name:
                param_grid = grid
                break
                
        if param_grid is None:
            self.logger.warning(f"No parameter grid defined for {model_name}, skipping tuning")
            return model
            
        # Set up cross-validation
        cv = StratifiedKFold(
            n_splits=min(5, len(np.unique(y_train))),
            shuffle=True,
            random_state=self.config.random_state
        )
        
        # Use RandomizedSearchCV for faster tuning with more parameters
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=self.config.scoring_metric,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbosity,
            error_score='raise',
            refit=True
        )
        
        self.logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_train, y_train)
            
            self.logger.info(f"Best parameters for {model_name}: {search.best_params_}")
            self.logger.info(f"Best {self.config.scoring_metric} score: {search.best_score_:.4f}")
            
            # Log feature importances for tree-based models
            if hasattr(search.best_estimator_.named_steps['classifier'], 'feature_importances_'):
                importances = search.best_estimator_.named_steps['classifier'].feature_importances_
                features = X_train.columns
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                self.logger.info("\nFeature Importances:")
                self.logger.info(importance_df.to_string())
            
            return search.best_estimator_
            
        except Exception as e:
            self.logger.error(f"Error during hyperparameter tuning for {model_name}: {str(e)}")
            self.logger.warning("Returning untuned model due to error")
            return model
            
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, 
                            normalize: bool = False, class_names: List[str] = None) -> plt.Figure:
        """Enhanced confusion matrix plotting with better visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for the plot title
            normalize: Whether to normalize the confusion matrix
            class_names: List of class names for display
            
        Returns:
            Matplotlib figure containing the confusion matrix plot
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = f'Normalized Confusion Matrix\n{model_name}'
            fmt = '.2f'
            vmin, vmax = 0, 1
        else:
            title = f'Confusion Matrix\n{model_name}'
            fmt = 'd'
            vmin, vmax = None, None
            
        # Convert class indices to names if available
        if class_names is not None and len(class_names) == len(cm):
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45, ha='right')
            plt.yticks(tick_marks, class_names)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            square=True,
            cbar=True,
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )
        
        # Customize plot
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Adjust layout to prevent cutoff
        plt.tight_layout()
        
        # Save the figure
        os.makedirs('plots', exist_ok=True)
        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}"
        filename += "_normalized" if normalize else ""
        plt.savefig(f"plots/{filename}.png", dpi=300, bbox_inches='tight')
        
        return fig
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(title)
        plt.tight_layout()
        
        suffix = '_normalized' if normalize else ''
        plt.savefig(f"plots/confusion_matrix_{model_name.replace(' ', '_')}{suffix}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_roc_curves(self, models_metrics: List[Dict], model_names: List[str]):
        """Plot ROC curves for binary and multi-class classification."""
        plt.figure(figsize=(12, 8))
        
        for i, (metrics, name) in enumerate(zip(models_metrics, model_names)):
            if metrics.get("y_pred_proba") is not None:
                y_true = metrics["y_true"]
                y_proba = metrics["y_pred_proba"]
                
                if len(self.class_names) == 2:
                    # Binary classification
                    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
                else:
                    # Multi-class - plot macro average
                    y_bin = label_binarize(y_true, classes=range(len(self.class_names)))
                    try:
                        auc = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
                        # For visualization, use the mean ROC curve
                        fpr_grid = np.linspace(0, 1, 100)
                        mean_tpr = np.zeros_like(fpr_grid)
                        
                        for class_idx in range(len(self.class_names)):
                            fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_proba[:, class_idx])
                            mean_tpr += np.interp(fpr_grid, fpr, tpr)
                            
                        mean_tpr /= len(self.class_names)
                        plt.plot(fpr_grid, mean_tpr, label=f'{name} (Macro AUC = {auc:.3f})')
                    except:
                        continue
                        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_feature_importance(self, model: Any, features: List[str], model_name: str):
        """Enhanced feature importance plotting for classification."""
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, max(6, len(features) * 0.3)))
            sns.barplot(data=feature_importance_df, y='feature', x='importance')
            plt.title(f"Feature Importances - {model_name}")
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f"plots/feature_importance_{model_name.replace(' ', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature importance to CSV
            feature_importance_df.to_csv(
                f"output/feature_importance_{model_name.replace(' ', '_')}.csv", 
                index=False
            )
            
    def plot_results(self, results_df: pd.DataFrame):
        """Enhanced plotting functionality for classification."""
        # Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        sns.barplot(data=results_df, x='Model', y='Accuracy', ax=axes[0,0])
        axes[0,0].set_title('Model Accuracy Comparison')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # F1 Weighted comparison
        sns.barplot(data=results_df, x='Model', y='F1 Weighted', ax=axes[0,1])
        axes[0,1].set_title('Model F1 Weighted Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Balanced Accuracy comparison
        sns.barplot(data=results_df, x='Model', y='Balanced Accuracy', ax=axes[1,0])
        axes[1,0].set_title('Model Balanced Accuracy Comparison')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Matthews Correlation comparison
        sns.barplot(data=results_df, x='Model', y='Matthews Correlation', ax=axes[1,1])
        axes[1,1].set_title('Model Matthews Correlation Comparison')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/classification_model_comparison_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_ensemble_model(self, models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Create an ensemble model from top performers."""
        if not self.config.use_ensemble:
            return None
            
        # Select top 3 models for ensemble (exclude SVM for speed)
        ensemble_models = []
        for name, model in models.items():
            if name not in ['SVM', 'Neural Network']:  # Exclude for speed
                pipeline = Pipeline([
                    ('scaler', self.get_scaler()),
                    ('classifier', model)
                ])
                ensemble_models.append((name.replace(' ', '_'), pipeline))
                
        if len(ensemble_models) >= 3:
            ensemble_models = ensemble_models[:3]  # Top 3
            
            voting_classifier = VotingClassifier(
                estimators=ensemble_models,
                voting='soft',
                n_jobs=self.config.n_jobs
            )
            
            self.logger.info(f"Created ensemble with models: {[name for name, _ in ensemble_models]}")
            return voting_classifier
        
        return None
        
    def save_model_info(self, best_model: Any, selected_features: List[str], target_col: str):
        """Save detailed model information for classification."""
        model_info = {
            'timestamp': datetime.now().isoformat(),
            'task_type': 'classification',
            'target_column': target_col,
            'best_model_type': type(best_model.named_steps['classifier']).__name__ if hasattr(best_model, 'named_steps') else type(best_model).__name__,
            'selected_features': selected_features,
            'class_names': self.class_names.tolist() if self.class_names is not None else None,
            'class_weights': self.class_weights,
            'config': {
                'test_size': self.config.test_size,
                'random_state': self.config.random_state,
                'cv_folds': self.config.cv_folds,
                'max_features': self.config.max_features,
                'outlier_method': self.config.outlier_method,
                'scaling_method': self.config.scaling_method,
                'feature_selection': self.config.feature_selection,
                'handle_class_imbalance': self.config.handle_class_imbalance,
                'use_ensemble': self.config.use_ensemble,
                'scoring_metric': self.config.scoring_metric
            }
        }

        # Save as JSON
        with open('output/classification_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

    def train_models(self, file_path: str = 'data/preprocessed_data.csv') -> pd.DataFrame:
        """Train and evaluate classification models with enhanced pipeline.
        
        Args:
            file_path: Path to the preprocessed data file
            
        Returns:
            DataFrame containing evaluation metrics for all models
        """
        try:
            self.logger.info(f"\n{'='*80}")
            self.logger.info("STARTING MODEL TRAINING PIPELINE")
            self.logger.info(f"{'='*80}\n")
            
            # 1. Load and validate data
            self.logger.info("1. Loading and validating data...")
            df = self.load_and_validate_data(file_path)
            
            # 2. Create AQI classes if needed
            if 'aqi' in df.columns and 'aqi_class' not in df.columns:
                self.logger.info("Creating AQI classes...")
                df['aqi_class'] = df['aqi'].apply(self.classify_aqi)
            
            # 3. Preprocess data
            self.logger.info("Preprocessing data...")
            X, y = self.preprocess_data(df)
            
            # 4. Handle data types and ensure DataFrame
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            
            # 5. Encode target variable if needed
            if y.dtype == 'O':
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                self.class_names = self.label_encoder.classes_.tolist()
                self.logger.info(f"Encoded classes: {dict(zip(self.class_names, range(len(self.class_names))))}")
            else:
                y_encoded = y
                self.class_names = [str(c) for c in np.unique(y)]
            
            # 6. Handle class imbalance
            if self.config.handle_class_imbalance and len(np.unique(y_encoded)) > 1:
                self.logger.info("Handling class imbalance...")
                self.class_weights = self.handle_class_imbalance(y_encoded)
            
            # 7. Split data
            self.logger.info("Splitting data into train/test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
            )
            
            # Store training data and feature names
            self.X_train = X_train
            self.y_train = y_train
            self.feature_names = X_train.columns.tolist()
            
            # Store training data as instance variables
            self.X_train = X_train
            self.y_train = y_train
            self.feature_names = X_train.columns.tolist()
            
            # Ensure we have AQI classes
            X_train_selected, selected_features = self.select_features(X_train, y_train)
            X_test_selected = X_test[selected_features]
            self.logger.info(f"Selected {len(selected_features)} features")
        except Exception as e:
            self.logger.error(f"Feature selection failed: {str(e)}")
            # Fallback to using all features if selection fails
            X_train_selected, X_test_selected = X_train, X_test
            selected_features = X_train.columns
            self.logger.warning("Using all features due to feature selection error")
        
        # 5. Get models
        self.logger.info("\n5. Initializing models...")
        models = self.get_models()
        results = []
        
        # 6. Train and evaluate each model
        self.logger.info("\n6. Training and evaluating models...")
        for name, model in models.items():
            try:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"MODEL: {name}")
                self.logger.info(f"{'='*50}")
                
                # Skip hyperparameter tuning if we have very few samples
                if hasattr(self.config, 'hyperparameter_tuning') and self.config.hyperparameter_tuning:
                    if len(X_train_selected) < 100:  # Too few samples for reliable tuning
                        self.logger.warning("Skipping hyperparameter tuning due to small dataset")
                    else:
                        try:
                            model = self.hyperparameter_tuning(model, name, X_train_selected, y_train)
                        except Exception as e:
                            self.logger.warning(f"Hyperparameter tuning failed: {str(e)}. Using default parameters.")
                
                # Train model with timing
                start_time = time.time()
                try:
                    model.fit(X_train_selected, y_train)
                    train_time = time.time() - start_time
                    
                    # Evaluate model
                    y_pred = model.predict(X_test_selected)
                    metrics = self.evaluate_classification(y_test, y_pred, name)
                    metrics['train_time'] = train_time
                    results.append(metrics)
                    
                    # Update best model if this is the first one or better than current best
                    if not hasattr(self, 'best_model') or metrics[self.config.scoring_metric] > self.best_model['score']:
                        self.best_model = {
                            'model': model,
                            'score': metrics[self.config.scoring_metric],
                            'name': name,
                            'metrics': metrics
                        }
                        self.logger.info(f"New best model: {name} ({self.config.scoring_metric}={metrics[self.config.scoring_metric]:.4f})")
                    
                    # Add to ensemble candidates if performance is good
                    if metrics[self.config.scoring_metric] > 0.7:  # Only include models with decent performance
                        ensemble_candidates.append((name, model))
                    
                    # 10.6 Plot confusion matrix
                    try:
                        y_pred = model.predict(X_test_selected)
                        self.plot_confusion_matrix(
                            y_test, y_pred, 
                            name, 
                            class_names=self.class_names
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not plot confusion matrix: {str(e)}")
                    
                    # 10.7 Plot feature importance for tree-based models
                    try:
                        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                            self.plot_feature_importance(model, X_train_selected.columns, name)
                    except Exception as e:
                        self.logger.warning(f"Could not plot feature importance: {str(e)}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
                    self.logger.exception("Stack trace:")
                    continue
            
            # 11. Create and evaluate ensemble model if enabled
            if self.config.use_ensemble and len(ensemble_candidates) >= 2:
                try:
                    self.logger.info("\nCreating ensemble model...")
                    ensemble = self.create_ensemble_model(
                        dict(ensemble_candidates), 
                        X_train_selected, 
                        y_train
                    )
                    
                    if ensemble:
                        self.logger.info("Evaluating ensemble model...")
                        ensemble_metrics = self.evaluate_classification(
                            ensemble, 
                            X_test_selected, 
                            y_test, 
                            'Ensemble'
                        )
                        
                        # Add ensemble to results
                        results.append(ensemble_metrics)
                        
                        # Update best model if ensemble performs better
                        if ensemble_metrics[self.config.scoring_metric] > self.best_model['score']:
                            self.best_model = {
                                'model': ensemble,
                                'score': ensemble_metrics[self.config.scoring_metric],
                                'name': 'Ensemble',
                                'metrics': ensemble_metrics
                            }
                            self.logger.info(f"New best model: Ensemble ({self.config.scoring_metric}={ensemble_metrics[self.config.scoring_metric]:.4f})")
                        
                        # Plot confusion matrix for ensemble
                        try:
                            y_pred = ensemble.predict(X_test_selected)
                            self.plot_confusion_matrix(
                                y_test, y_pred,
                                'Ensemble',
                                class_names=self.class_names
                            )
                        except Exception as e:
                            self.logger.warning(f"Could not plot confusion matrix for ensemble: {str(e)}")
                            
                except Exception as e:
                    self.logger.error(f"Error creating ensemble model: {str(e)}")
                    self.logger.exception("Stack trace:")
            
            # 12. Process and save results
            self.logger.info("\n4. Processing and saving results...")
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            if not results_df.empty:
                # Sort by the scoring metric
                results_df = results_df.sort_values(
                    by=self.config.scoring_metric, 
                    ascending=False
                )
                
                # Select columns to display in logs
                display_cols = ['Model', self.config.scoring_metric, 'Accuracy', 'training_time']
                if 'F1 Score' in results_df.columns:
                    display_cols.append('F1 Score')
                if 'ROC AUC' in results_df.columns:
                    display_cols.append('ROC AUC')
                
                # Save detailed results
                os.makedirs('output', exist_ok=True)
                results_df.to_csv('output/classification_results.csv', index=False)
                
                # Save best model
                if self.best_model:
                    self.save_model_info(
                        self.best_model['model'], 
                        selected_features, 
                        self.target_column if hasattr(self, 'target_column') else 'target'
                    )
                
                # Plot and save results
                self.plot_results(results_df)
                
                # Log summary
                self.logger.info("\n" + "="*80)
                self.logger.info("TRAINING COMPLETED SUCCESSFULLY!")
                self.logger.info("="*80)
                self.logger.info("\nResults summary (sorted by scoring metric):")
                self.logger.info("-"*60)
                self.logger.info(results_df[display_cols].to_string(index=False, float_format='{:,.4f}'.format))
                
                self.logger.info("\nBest model:")
                self.logger.info(f"- Name: {self.best_model['name']}")
                self.logger.info(f"- {self.config.scoring_metric}: {self.best_model['score']:.4f}")
                
                # Log feature importances for the best model if available
                if hasattr(self.best_model['model'].named_steps['classifier'], 'feature_importances_'):
                    try:
                        importances = self.best_model['model'].named_steps['classifier'].feature_importances_
                        feature_importance = pd.DataFrame({
                            'feature': selected_features,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                        
                        self.logger.info("\nFeature importances for best model:")
                        self.logger.info(feature_importance.to_string(index=False))
                    except Exception as e:
                        self.logger.warning(f"Could not log feature importances: {str(e)}")
                
                return results_df
            else:
                self.logger.error("No models were successfully trained.")
                return pd.DataFrame()
            
                
            return results_df
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise

    def predict(self, X_new: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the best trained model."""
        if self.best_model is None:
            raise ValueError("No trained model available. Run train_models() first.")
            
        # Ensure features match training data
        if self.selected_features:
            X_new = X_new[self.selected_features]
            
        # Get predictions and probabilities
        y_pred = self.best_model.predict(X_new)
        y_pred_proba = self.best_model.predict_proba(X_new) if hasattr(self.best_model, 'predict_proba') else None
        
        # Convert predictions back to original labels
        if self.label_encoder:
            y_pred = self.label_encoder.inverse_transform(y_pred)
            
        return y_pred, y_pred_proba

    def evaluate_new_data(self, X_new: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on new data."""
        if self.best_model is None:
            raise ValueError("No trained model available. Run train_models() first.")
            
        # Ensure features match training data
        if self.selected_features:
            X_new = X_new[self.selected_features]
            
        # Make predictions
        y_pred = self.best_model.predict(X_new)
        y_pred_proba = self.best_model.predict_proba(X_new) if hasattr(self.best_model, 'predict_proba') else None
        
        # Convert predictions back to original labels
        if self.label_encoder:
            y_pred = self.label_encoder.inverse_transform(y_pred)
            y_true = self.label_encoder.inverse_transform(y_true)
            
        # Calculate metrics
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "F1 Weighted": f1_score(y_true, y_pred, average='weighted'),
            "F1 Macro": f1_score(y_true, y_pred, average='macro'),
            "Precision Weighted": precision_score(y_true, y_pred, average='weighted'),
            "Recall Weighted": recall_score(y_true, y_pred, average='weighted'),
            "Matthews Correlation": matthews_corrcoef(y_true, y_pred),
            "Cohen Kappa": cohen_kappa_score(y_true, y_pred)
        }
        
        if y_pred_proba is not None and self.label_encoder:
            try:
                y_bin = label_binarize(y_true, classes=self.label_encoder.classes_)
                metrics["ROC AUC Macro"] = roc_auc_score(y_bin, y_pred_proba, average='macro', multi_class='ovr')
                metrics["ROC AUC Weighted"] = roc_auc_score(y_bin, y_pred_proba, average='weighted', multi_class='ovr')
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC AUC: {str(e)}")
                metrics["ROC AUC Macro"] = np.nan
                metrics["ROC AUC Weighted"] = np.nan
            
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        self.logger.info(f"\nClassification Report:\n{report}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, "New Data", normalize=True)
        
        return metrics
if __name__ == "__main__":
    # Set up paths
    base_dir = Path(__file__).parent.parent  # Move up two levels to project root
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'output'
    plots_dir = base_dir / 'plots'
    models_dir = base_dir / 'models'
    logs_dir = base_dir / 'logs'
    
    # Custom configuration
    config = ClassificationConfig(
        test_size=0.2,
        random_state=42,
        cv_folds=5,
        n_jobs=-1,
        max_features=15,
        outlier_method='iqr',
        scaling_method='robust',
        feature_selection='both',
        handle_class_imbalance=True,
        use_ensemble=True,
        scoring_metric='f1_weighted'
    )
    
    # Create pipeline
    pipeline = ClassificationPipeline(config)
    
    # Train models
    results = pipeline.train_models(str(data_dir / 'preprocessed_data.csv'))
    
    # Save results
    results.to_csv(str(output_dir / 'classification_results.csv'), index=False)
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best model: {results.iloc[0]['Model']}")
    print(f"Best F1 Weighted: {results.iloc[0]['F1 Weighted']:.4f}")
    print(f"Best Accuracy: {results.iloc[0]['Accuracy']:.4f}")
    print("\nCheck the following directories for outputs:")
    print("- plots/: Visualizations and charts")
    print("- output/: Results and feature importance")
    print("- models/: Saved model files")
    print("- logs/: Training logs")