import pandas as pd
import numpy as np
import os
import joblib
import logging
import warnings
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, validation_curve
)
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy import stats
import xgboost as xgb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ClassificationConfig:
    """Configuration class for classification model training."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1
    max_features: int = 15
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation'
    scaling_method: str = 'robust'  # 'robust', 'standard', 'minmax'
    feature_selection: str = 'selectk'  # 'selectk', 'rfe', 'both'
    use_stratified_split: bool = True
    target_column: str = 'aqi_class'
    use_hyperparameter_tuning: bool = True

class ClassificationPipeline:
    """Enhanced ML Pipeline for Air Quality Classification."""
    
    def __init__(self, config: ClassificationConfig = None):
        self.config = config or ClassificationConfig()
        self.results = []
        self.best_model = None
        self.selected_features = None
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        """Set up logging configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.abspath('logs')
        log_file = os.path.join(log_dir, f'classification_training_{timestamp}.log')
        
        # Create logs directory if it doesn't exist
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating logs directory: {e}")
            raise
        
        # Create handlers
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.info('Classification model training started')
        self.logger.info(f'Created logs directory at: {log_dir}')
        
    def setup_directories(self):
        """Create necessary directories."""
        dirs = ["output", "models", "plots", "data"]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
            
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate input data."""
        try:
            self.logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Basic validation
            if df.empty:
                raise ValueError("Dataset is empty")
                
            # If target column doesn't exist, create it based on PM2.5
            if self.config.target_column not in df.columns:
                self.logger.info(f"Target column '{self.config.target_column}' not found. Creating classes based on PM2.5...")
                df = self.create_pm25_classes(df)
            else:
                # Check class distribution
                class_counts = df[self.config.target_column].value_counts()
                self.logger.info(f"Class distribution:\n{class_counts}")
                
                # Check if we have only one class
                if len(class_counts) == 1:
                    self.logger.warning("Only one class found in target column. Creating classes based on PM2.5 percentiles...")
                    df = self.create_pm25_classes(df)
                
            self.logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            self.logger.info(f"Missing values per column:\n{df.isnull().sum()}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def create_pm25_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create air quality classes based on PM2.5 percentiles when only one class exists."""
        self.logger.info("Creating PM2.5-based air quality classes...")
        
        # Check if PM2.5 column exists
        if 'pm2.5' not in df.columns:
            raise ValueError("PM2.5 column not found. Cannot create classes.")
        
        # Get PM2.5 values (handle both original and scaled data)
        pm25_values = df['pm2.5'].values
        
        # Calculate percentiles to create classes
        # For air quality classification, we'll create 3-4 classes based on PM2.5 levels
        p33 = np.percentile(pm25_values, 33)
        p66 = np.percentile(pm25_values, 66)
        
        # Create classes based on PM2.5 levels
        def classify_pm25(pm25_val):
            if pm25_val <= p33:
                return "Low"
            elif pm25_val <= p66:
                return "Medium"
            else:
                return "High"
        
        # Apply classification
        df[self.config.target_column] = df['pm2.5'].apply(classify_pm25)
        
        # Log the new class distribution
        new_class_counts = df[self.config.target_column].value_counts()
        self.logger.info(f"New class distribution based on PM2.5:\n{new_class_counts}")
        self.logger.info(f"PM2.5 thresholds: Low <= {p33:.3f}, Medium <= {p66:.3f}, High > {p66:.3f}")
        
        return df
            
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
        numeric_cols = [col for col in numeric_cols if col not in [self.config.target_column]]
        
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
        
        # Encode categorical features first
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in [self.config.target_column, 'timestamp']]
        
        if len(categorical_cols) > 0:
            self.logger.info(f"Encoding categorical columns: {categorical_cols}")
            df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        
        # Time-based features if timestamp exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
        # Interaction features for air quality
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col not in [self.config.target_column, 'aqi']]
        
        # Create polynomial features for key predictors
        if 'pm2.5' in df.columns and 'temperature' in df.columns:
            df['pm2.5_temp_interaction'] = df['pm2.5'] * df['temperature']
            
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-8)
            
        # Air quality index features
        if 'pm2.5' in df.columns:
            df['pm2.5_squared'] = df['pm2.5'] ** 2
            df['pm2.5_log'] = np.log1p(df['pm2.5'])
            
        self.logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df.dropna()  # Remove rows with NaN from new features
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Enhanced feature selection for classification."""
        self.logger.info(f"Selecting features using {self.config.feature_selection} method...")
        
        if self.config.feature_selection == 'selectk':
            selector = SelectKBest(
                score_func=f_classif, 
                k=min(self.config.max_features, X.shape[1])
            )
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif self.config.feature_selection == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
            selector = RFE(
                estimator=estimator, 
                n_features_to_select=min(self.config.max_features, X.shape[1])
            )
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif self.config.feature_selection == 'both':
            # First use SelectKBest
            selector1 = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
            X_temp = selector1.fit_transform(X, y)
            temp_features = X.columns[selector1.get_support()]
            
            # Then use RFE on selected features
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
            selector2 = RFE(estimator=estimator, n_features_to_select=min(self.config.max_features, len(temp_features)))
            X_selected = selector2.fit_transform(X_temp, y)
            selected_features = temp_features[selector2.get_support()].tolist()
            
        self.selected_features = selected_features
        self.logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
        
    def get_scaler(self):
        """Get the appropriate scaler based on configuration."""
        scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        return scalers[self.config.scaling_method]
        
    def get_models(self) -> Dict[str, Any]:
        """Get classification models with hyperparameter tuning."""
        models = {
            "Logistic Regression": LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000,
                solver='liblinear'
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                max_depth=15,
                min_samples_split=5
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                eval_metric='mlogloss'
            )
        }
        return models
        
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, Any]:
        """Enhanced model evaluation with classification metrics."""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # Handle multi-class vs binary classification
        if len(np.unique(y)) == 2:
            # Binary classification
            metrics = {
                "Model": model_name,
                "Accuracy": accuracy_score(y, y_pred),
                "Precision": precision_score(y, y_pred, average='weighted'),
                "Recall": recall_score(y, y_pred, average='weighted'),
                "F1-Score": f1_score(y, y_pred, average='weighted'),
                "ROC-AUC": roc_auc_score(y, y_pred_proba[:, 1])
            }
        else:
            # Multi-class classification
            metrics = {
                "Model": model_name,
                "Accuracy": accuracy_score(y, y_pred),
                "Precision": precision_score(y, y_pred, average='weighted'),
                "Recall": recall_score(y, y_pred, average='weighted'),
                "F1-Score": f1_score(y, y_pred, average='weighted'),
                "ROC-AUC": roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
            }
        
        return metrics
        
    def hyperparameter_tuning(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Perform hyperparameter tuning for classification models."""
        param_grids = {
            "Logistic Regression": {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            },
            "Random Forest": {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 15, 20, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            },
            "XGBoost": {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if model_name in param_grids and self.config.use_hyperparameter_tuning:
            self.logger.info(f"Tuning hyperparameters for {model_name}...")
            pipeline = Pipeline([
                ('scaler', self.get_scaler()),
                ('classifier', model)
            ])
            
            cv_strategy = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            
            grid_search = GridSearchCV(
                pipeline, 
                param_grids[model_name],
                cv=cv_strategy,
                scoring='f1_weighted',
                n_jobs=self.config.n_jobs,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
        else:
            # Return regular pipeline for models without tuning
            return Pipeline([
                ('scaler', self.get_scaler()),
                ('classifier', model)
            ])
            
    def plot_results(self, results_df: pd.DataFrame):
        """Enhanced plotting functionality for classification."""
        # Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        sns.barplot(data=results_df, x='Model', y='Accuracy', ax=axes[0,0])
        axes[0,0].set_title('Model Accuracy Comparison')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        sns.barplot(data=results_df, x='Model', y='F1-Score', ax=axes[0,1])
        axes[0,1].set_title('Model F1-Score Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        sns.barplot(data=results_df, x='Model', y='Precision', ax=axes[1,0])
        axes[1,0].set_title('Model Precision Comparison')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # ROC-AUC comparison
        sns.barplot(data=results_df, x='Model', y='ROC-AUC', ax=axes[1,1])
        axes[1,1].set_title('Model ROC-AUC Comparison')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/classification_model_comparison_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrix(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
        """Plot confusion matrix for classification model."""
        y_pred = model.predict(X_test)
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap='Blues', ax=plt.gca())
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(f"plots/confusion_matrix_{model_name.replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
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
            
    def plot_roc_curve(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
        """Plot ROC curve for classification model."""
        y_pred_proba = model.predict_proba(X_test)
        
        plt.figure(figsize=(8, 6))
        
        if len(np.unique(y_test)) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        else:
            # Multi-class classification
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            n_classes = y_test_bin.shape[1]
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                auc_score = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plots/roc_curve_{model_name.replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_model_info(self, best_model: Any, selected_features: List[str]):
        """Save detailed model information."""
        model_info = {
            'timestamp': datetime.now().isoformat(),
            'best_model_type': type(best_model.named_steps['classifier']).__name__,
            'selected_features': selected_features,
            'config': {
                'test_size': self.config.test_size,
                'random_state': self.config.random_state,
                'cv_folds': self.config.cv_folds,
                'max_features': self.config.max_features,
                'outlier_method': self.config.outlier_method,
                'scaling_method': self.config.scaling_method,
                'feature_selection': self.config.feature_selection,
                'target_column': self.config.target_column
            }
        }
        
        # Save as JSON
        import json
        with open('output/classification_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
            
    def train_models(self, file_path: str = 'data/preprocessed_data.csv'):
        """Main training pipeline for classification."""
        try:
            # Load and preprocess data
            df = self.load_and_validate_data(file_path)
            df = self.handle_missing_values(df)
            df = self.remove_outliers(df)
            df = self.engineer_features(df)
            
            # Prepare features and target
            exclude_cols = [self.config.target_column, 'aqi', 'timestamp']
            X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
            
            # Ensure we only have numerical features for machine learning
            X = X.select_dtypes(include=[np.number])
            self.logger.info(f"Features after selecting numerical columns: {X.columns.tolist()}")
            
            y = df[self.config.target_column]
            
            # Encode target labels if they are strings
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)
                self.logger.info(f"Encoded target classes: {self.label_encoder.classes_}")
            
            # Final check for single class
            unique_classes = np.unique(y)
            if len(unique_classes) == 1:
                raise ValueError(f"Only one class found after preprocessing: {unique_classes[0]}. "
                               "Classification requires at least 2 classes. Please check your data.")
            
            self.logger.info(f"Final class distribution: {np.bincount(y)}")
            
            # Feature selection
            X_selected, selected_features = self.select_features(X, y)
            
            # Train-test split
            if self.config.use_stratified_split:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y, 
                    test_size=self.config.test_size, 
                    random_state=self.config.random_state,
                    stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y, 
                    test_size=self.config.test_size, 
                    random_state=self.config.random_state
                )
            
            # Train models
            models = self.get_models()
            results = []
            best_f1 = -np.inf
            best_model = None
            
            for name, model in models.items():
                self.logger.info(f"Training {name}...")
                
                # Get model with hyperparameter tuning if applicable
                trained_model = self.hyperparameter_tuning(model, name, X_train, y_train)
                
                # If not tuned, create regular pipeline
                if not isinstance(trained_model, Pipeline):
                    trained_model = Pipeline([
                        ('scaler', self.get_scaler()),
                        ('classifier', model)
                    ])
                
                # Cross-validation
                cv_strategy = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
                cv_scores = cross_val_score(trained_model, X_train, y_train, cv=cv_strategy, scoring='f1_weighted', n_jobs=self.config.n_jobs)
                
                # Fit and evaluate
                trained_model.fit(X_train, y_train)
                metrics = self.evaluate_model(trained_model, X_test, y_test, name)
                metrics["CV F1 Mean"] = cv_scores.mean()
                metrics["CV F1 Std"] = cv_scores.std()
                results.append(metrics)
                
                # Track best model
                if metrics["F1-Score"] > best_f1:
                    best_f1 = metrics["F1-Score"]
                    best_model = trained_model
                    
                # Generate plots
                self.plot_feature_importance(trained_model, selected_features, name)
                self.plot_confusion_matrix(trained_model, X_test, y_test, name)
                self.plot_roc_curve(trained_model, X_test, y_test, name)
                
                self.logger.info(f"{name} - Accuracy: {metrics['Accuracy']:.4f}, F1-Score: {metrics['F1-Score']:.4f}")
            
            # Save results
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('F1-Score', ascending=False)
            results_df.to_csv('output/classification_results.csv', index=False)
            
            # Save best model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_model_path = f"models/best_classification_model_{timestamp}.joblib"
            joblib.dump(best_model, best_model_path)
            
            # Save label encoder
            label_encoder_path = f"models/label_encoder_{timestamp}.joblib"
            joblib.dump(self.label_encoder, label_encoder_path)
            
            # Save model metadata
            self.save_model_info(best_model, selected_features)
            
            # Generate comprehensive plots
            self.plot_results(results_df)
            
            # Log final results
            best_model_name = results_df.iloc[0]['Model']
            self.logger.info(f"Best model: {best_model_name} with F1-Score = {best_f1:.4f}")
            self.logger.info(f"Model saved to: {best_model_path}")
            
            self.best_model = best_model
            self.results = results_df
            
            return results_df
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise
            
    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best trained model."""
        if self.best_model is None:
            raise ValueError("No trained model available. Run train_models() first.")
            
        # Ensure features match training data
        if self.selected_features:
            X_new = X_new[self.selected_features]
            
        predictions = self.best_model.predict(X_new)
        
        # Decode predictions if label encoder was used
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)
            
        return predictions

def main():
    """Main execution function with configuration options."""
    # Custom configuration
    config = ClassificationConfig(
        test_size=0.2,
        random_state=42,
        cv_folds=5,
        max_features=12,
        outlier_method='iqr',
        scaling_method='robust',
        feature_selection='both',
        use_stratified_split=True,
        target_column='aqi_class',
        use_hyperparameter_tuning=True
    )
    
    # Initialize and run pipeline
    pipeline = ClassificationPipeline(config)
    # Use original data to preserve PM2.5 distribution for class creation
    results = pipeline.train_models('data/air_quality_data.csv')
    
    print("\n" + "="*50)
    print("CLASSIFICATION TRAINING COMPLETE")
    print("="*50)
    print(f"Best model: {results.iloc[0]['Model']}")
    print(f"Best Accuracy: {results.iloc[0]['Accuracy']:.4f}")
    print(f"Best F1-Score: {results.iloc[0]['F1-Score']:.4f}")
    print("\nCheck the following directories for outputs:")
    print("- plots/: Visualizations and charts")
    print("- output/: Results and feature importance")
    print("- models/: Saved model files")
    print("- logs/: Training logs")

if __name__ == "__main__":
    main()
