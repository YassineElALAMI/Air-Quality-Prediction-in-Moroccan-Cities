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
    LabelEncoder, label_binarize, OneHotEncoder
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

from scipy import stats
import itertools

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
    handle_class_imbalance: bool = True
    use_ensemble: bool = True
    scoring_metric: str = 'f1_weighted'  # 'accuracy', 'f1_weighted', 'f1_macro'

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
        self.param_grids = {
            "Logistic Regression": {
                'classifier__C': [0.1, 1, 10],
                'classifier__l1_ratio': [0.0, 0.5, 1.0]
            },
            "Random Forest": {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [5, 10, None]
            },
            "XGBoost": {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5, 7]
            }
        }
        
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
            
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate input data for classification."""
        try:
            self.logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Basic validation
            if df.empty:
                raise ValueError("Dataset is empty")
                
            # Check for target column (aqi or aqi_class)
            target_cols = ['aqi', 'aqi_class']
            available_targets = [col for col in target_cols if col in df.columns]
            if not available_targets:
                raise ValueError(f"No target column found. Expected one of: {target_cols}")
                
            self.logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            self.logger.info(f"Available target columns: {available_targets}")
            self.logger.info(f"Missing values per column:\n{df.isnull().sum()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Enhanced data preprocessing for classification."""
        self.logger.info("Preprocessing data for classification...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove outliers
        df = self.remove_outliers(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Select target variable
        target_columns = ['aqi_class']  # Only use categorical AQI class
        available_targets = [col for col in target_columns if col in df.columns]
        if not available_targets:
            raise ValueError("No valid target column found in data")
            
        self.target_column = available_targets[0]
        self.logger.info(f"Target variable: {self.target_column}")
        
        # Get class distribution
        class_dist = df[self.target_column].value_counts()
        self.logger.info(f"Class distribution:\n{class_dist}")
        
        # Check if we have enough classes
        if len(class_dist) < 2:
            # If we only have one class, create synthetic classes based on PM2.5 levels
            self.logger.warning("Only one class found. Creating synthetic classes based on PM2.5 levels...")
            
            def create_synthetic_classes(pm25):
                if pm25 <= 12: return 'Low PM2.5'
                elif pm25 <= 35.5: return 'Medium PM2.5'
                else: return 'High PM2.5'
            
            df['synthetic_class'] = df['pm2.5'].apply(create_synthetic_classes)
            self.target_column = 'synthetic_class'
            class_dist = df[self.target_column].value_counts()
            self.logger.info(f"Created synthetic classes:\n{class_dist}")
        
        # Separate features and target
        X = df.drop([self.target_column, 'timestamp'], axis=1, errors='ignore')
        y = df[self.target_column]
        
        return X, y
        
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
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
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
        scale_pos_weight = 1
        if self.class_weights and max(self.class_weights.keys()) == 1:  # Assume binary with classes 0 and 1
            scale_pos_weight = self.class_weights[1] / self.class_weights[0]
        
        models = {
            # 1. Logistic Regression with elasticnet regularization
            "Logistic Regression": LogisticRegression(
                class_weight=self.class_weights,
                max_iter=1000,
                random_state=self.config.random_state,
                solver='saga',
                penalty='elasticnet',
                l1_ratio=0.5,
                n_jobs=self.config.n_jobs,
                verbose=0
            ),
            
            # 2. Random Forest with balanced class weights
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                verbose=0
            ),
            
            # 3. XGBoost with optimized parameters
            "XGBoost": XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                scale_pos_weight=scale_pos_weight,
                eval_metric='mlogloss',
                verbosity=0
            )
        }
        return models

        
    def evaluate_classification(self, model: Any, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, Any]:
        """Comprehensive classification evaluation."""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Basic metrics
        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y, y_pred),
            "F1 Weighted": f1_score(y, y_pred, average='weighted'),
            "F1 Macro": f1_score(y, y_pred, average='macro'),
            "F1 Micro": f1_score(y, y_pred, average='micro'),
            "Precision Weighted": precision_score(y, y_pred, average='weighted'),
            "Recall Weighted": recall_score(y, y_pred, average='weighted'),
            "Matthews Correlation": matthews_corrcoef(y, y_pred),
            "Cohen Kappa": cohen_kappa_score(y, y_pred)
        }
        
        # Multi-class ROC AUC
        if y_pred_proba is not None and len(self.class_names) > 2:
            try:
                y_bin = label_binarize(y, classes=range(len(self.class_names)))
                metrics["ROC AUC Macro"] = roc_auc_score(y_bin, y_pred_proba, average='macro', multi_class='ovr')
                metrics["ROC AUC Weighted"] = roc_auc_score(y_bin, y_pred_proba, average='weighted', multi_class='ovr')
            except:
                metrics["ROC AUC Macro"] = np.nan
                metrics["ROC AUC Weighted"] = np.nan
        elif y_pred_proba is not None and len(self.class_names) == 2:
            metrics["ROC AUC"] = roc_auc_score(y, y_pred_proba[:, 1])
            
        # Store predictions and probabilities for plotting
        metrics["y_true"] = y
        metrics["y_pred"] = y_pred
        metrics["y_pred_proba"] = y_pred_proba
        
        return metrics
        
    def hyperparameter_tuning(self, estimator: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Perform hyperparameter tuning for specific models."""
        preprocessor = self.get_scaler(X_train)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', estimator)
        ])
        
        if model_name in self.param_grids:
            self.logger.info(f"Tuning hyperparameters for {model_name}...")
            
            cv_strategy = StratifiedKFold(n_splits=min(3, self.config.cv_folds), shuffle=True, random_state=self.config.random_state)
            
            grid_search = GridSearchCV(
                pipeline, 
                self.param_grids[model_name],
                cv=cv_strategy,
                scoring=self.config.scoring_metric,
                n_jobs=self.config.n_jobs,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
        else:
            return pipeline
            
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, normalize: bool = False):
        """Enhanced confusion matrix plotting."""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = f'Normalized Confusion Matrix - {model_name}'
            fmt = '.2f'
        else:
            title = f'Confusion Matrix - {model_name}'
            fmt = 'd'
            
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
                ensemble_models.append((name.replace(' ', '_'), model))
                
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
        """Train and evaluate models."""
        try:
            self.logger.info(f"Training pipeline started with {file_path}")
            
            # Load and preprocess data
            df = self.load_and_validate_data(file_path)
            
            # Create aqi_class if necessary
            if 'aqi' in df.columns and 'aqi_class' not in df.columns:
                def classify_aqi(aqi):
                    if aqi == 1: return 'Good'
                    elif aqi == 2: return 'Fair'
                    elif aqi == 3: return 'Moderate'
                    elif aqi == 4: return 'Poor'
                    elif aqi == 5: return 'Very Poor'
                    else: return 'Unknown'
                
                df['aqi_class'] = df['aqi'].apply(classify_aqi)
                self.logger.info("Created AQI classes from AQI values")
            
            X, y = self.preprocess_data(df)
            
            # Convert X to DataFrame if it's a numpy array
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Store training data as instance variables
            self.X_train = X_train
            self.y_train = y_train
            self.feature_names = X_train.columns.tolist()
            
            # Encode target if categorical
            if y_train.dtype == 'object' or pd.api.types.is_categorical_dtype(y_train):
                self.label_encoder = LabelEncoder()
                y_train = pd.Series(self.label_encoder.fit_transform(y_train))
                y_test = pd.Series(self.label_encoder.transform(y_test))
                self.class_names = self.label_encoder.classes_.tolist()
                self.logger.info(f"Encoded target classes: {self.class_names}")
            
            # Handle class imbalance
            if self.config.handle_class_imbalance:
                self.class_weights = self.handle_class_imbalance(y_train)
            
            # Feature selection
            X_train_selected, selected_features = self.select_features(X_train, y_train)
            X_test_selected = X_test[selected_features]
            
            # Get models
            models = self.get_models()
            results = []
            trained_models = {}
            
            # Train and evaluate each model
            for name, estimator in models.items():
                try:
                    self.logger.info(f"Training {name}...")
                    
                    # Hyperparameter tuning
                    model = self.hyperparameter_tuning(estimator, name, X_train_selected, y_train)
                    
                    # Train model
                    model.fit(X_train_selected, y_train)
                    
                    # Evaluate model
                    metrics = self.evaluate_classification(model, X_test_selected, y_test, name)
                    results.append({**metrics, 'Model': name})
                    
                    # Track trained model for ensemble
                    trained_models[name] = model
                    
                    # Track best model
                    if not self.best_model or metrics[self.config.scoring_metric] > self.best_model['score']:
                        self.best_model = {
                            'model': model,
                            'score': metrics[self.config.scoring_metric],
                            'name': name,
                            'metrics': metrics
                        }
                        
                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
                    continue
            
            # Create ensemble if enabled
            if self.config.use_ensemble:
                ensemble = self.create_ensemble_model(trained_models, X_train_selected, y_train)
                if ensemble:
                    ensemble.fit(X_train_selected, y_train)
                    ensemble_metrics = self.evaluate_classification(ensemble, X_test_selected, y_test, 'Ensemble')
                    results.append({**ensemble_metrics, 'Model': 'Ensemble'})
                    if ensemble_metrics[self.config.scoring_metric] > self.best_model['score']:
                        self.best_model = {
                            'model': ensemble,
                            'score': ensemble_metrics[self.config.scoring_metric],
                            'name': 'Ensemble',
                            'metrics': ensemble_metrics
                        }
            
            # Save results
            results_df = pd.DataFrame(results)
            results_df.to_csv('output/classification_results.csv', index=False)
            
            # Plot results
            self.plot_results(results_df)
            
            # Save model info
            self.save_model_info(self.best_model['model'], selected_features, self.target_column)
            
                
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
        y_pred = self.best_model['model'].predict(X_new)
        y_pred_proba = self.best_model['model'].predict_proba(X_new) if hasattr(self.best_model['model'], 'predict_proba') else None
        
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
        y_pred = self.best_model['model'].predict(X_new)
        y_pred_proba = self.best_model['model'].predict_proba(X_new) if hasattr(self.best_model['model'], 'predict_proba') else None
        
        # Assume y_true is in original label format (strings/categories)
        # Encode y_true to numeric for consistency if needed
        y_true_encoded = y_true
        y_pred_encoded = y_pred
        if self.label_encoder:
            y_true_encoded = self.label_encoder.transform(y_true)
            y_pred = self.label_encoder.inverse_transform(y_pred)  # For string-based metrics
        
        # Calculate metrics using encoded versions where necessary
        metrics = {
            "Accuracy": accuracy_score(y_true_encoded, y_pred_encoded),
            "Balanced Accuracy": balanced_accuracy_score(y_true_encoded, y_pred_encoded),
            "F1 Weighted": f1_score(y_true_encoded, y_pred_encoded, average='weighted'),
            "F1 Macro": f1_score(y_true_encoded, y_pred_encoded, average='macro'),
            "Precision Weighted": precision_score(y_true_encoded, y_pred_encoded, average='weighted'),
            "Recall Weighted": recall_score(y_true_encoded, y_pred_encoded, average='weighted'),
            "Matthews Correlation": matthews_corrcoef(y_true_encoded, y_pred_encoded),
            "Cohen Kappa": cohen_kappa_score(y_true_encoded, y_pred_encoded)
        }
        
        if y_pred_proba is not None and self.label_encoder:
            try:
                y_bin = label_binarize(y_true, classes=self.class_names)
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
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
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
    
    # Sort results to find best
    results_sorted = results.sort_values(by='F1 Weighted', ascending=False)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best model: {results_sorted.iloc[0]['Model']}")
    print(f"Best F1 Weighted: {results_sorted.iloc[0]['F1 Weighted']:.4f}")
    print(f"Best Accuracy: {results_sorted.iloc[0]['Accuracy']:.4f}")
    print("\nCheck the following directories for outputs:")
    print("- plots/: Visualizations and charts")
    print("- output/: Results and feature importance")
    print("- models/: Saved model files")
    print("- logs/: Training logs")