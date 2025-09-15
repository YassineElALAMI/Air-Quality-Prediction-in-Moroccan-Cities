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
    TimeSeriesSplit, validation_curve
)
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Configuration class for model training."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1
    max_features: int = 15
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation'
    scaling_method: str = 'robust'  # 'robust', 'standard', 'minmax'
    feature_selection: str = 'selectk'  # 'selectk', 'rfe', 'both'
    use_time_series_split: bool = False  # Use if data has temporal ordering

class MLPipeline:
    """Enhanced ML Pipeline for PM2.5 prediction."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.results = []
        self.best_model = None
        self.selected_features = None
        self.scaler = None
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        """Set up logging configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.abspath('logs')
        log_file = os.path.join(log_dir, f'model_training_{timestamp}.log')
        
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
        
        self.logger.info('Model training started')
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
                
            if 'pm2.5' not in df.columns:
                raise ValueError("Target column 'pm2.5' not found")
                
            self.logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            self.logger.info(f"Missing values per column:\n{df.isnull().sum()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
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
        numeric_cols = [col for col in numeric_cols if df[col].nunique() > 2]
        
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
        """Feature engineering for air quality data."""
        self.logger.info("Engineering features...")
        
        # Time-based features if timestamp exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
        # Interaction features for air quality
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col not in ['pm2.5', 'aqi']]
        
        # Create polynomial features for key predictors
        if 'no2' in df.columns and 'o3' in df.columns:
            df['no2_o3_interaction'] = df['no2'] * df['o3']
            
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-8)
            
        # Lag features if time series data
        if 'timestamp' in df.columns and len(df) > 24:
            df = df.sort_values('timestamp')
            for lag in [1, 2, 3, 24]:  # 1-3 hours and 1 day lag
                if len(df) > lag:
                    df[f'pm2.5_lag_{lag}'] = df['pm2.5'].shift(lag)
                    
        self.logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df.dropna()  # Remove rows with NaN from lag features
        
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Enhanced feature selection."""
        self.logger.info(f"Selecting features using {self.config.feature_selection} method...")
        
        if self.config.feature_selection == 'selectk':
            selector = SelectKBest(
                score_func=f_regression, 
                k=min(self.config.max_features, X.shape[1])
            )
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif self.config.feature_selection == 'rfe':
            estimator = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
            selector = RFE(
                estimator=estimator, 
                n_features_to_select=min(self.config.max_features, X.shape[1])
            )
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif self.config.feature_selection == 'both':
            # First use SelectKBest
            selector1 = SelectKBest(score_func=f_regression, k=min(20, X.shape[1]))
            X_temp = selector1.fit_transform(X, y)
            temp_features = X.columns[selector1.get_support()]
            
            # Then use RFE on selected features
            estimator = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
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
        """Get enhanced model dictionary with hyperparameter tuning."""
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
            "Decision Tree": DecisionTreeRegressor(
                random_state=self.config.random_state,
                max_depth=10,
                min_samples_split=5
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                max_depth=15,
                min_samples_split=5
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.config.random_state
            ),
            "AdaBoost": AdaBoostRegressor(
                n_estimators=50,
                random_state=self.config.random_state
            ),
            "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "KNN": KNeighborsRegressor(n_neighbors=5)
        }
        return models
        
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, Any]:
        """Enhanced model evaluation with additional metrics."""
        y_pred = model.predict(X)
        
        metrics = {
            "Model": model_name,
            "MAE": mean_absolute_error(y, y_pred),
            "RMSE": mean_squared_error(y, y_pred, squared=False),
            "R²": r2_score(y, y_pred),
            "MAPE": mean_absolute_percentage_error(y, y_pred),
            "Explained Variance": explained_variance_score(y, y_pred)
        }
        
        return metrics
        
    def hyperparameter_tuning(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Perform hyperparameter tuning for specific models."""
        param_grids = {
            "Random Forest": {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [10, 15, 20, None],
                'regressor__min_samples_split': [2, 5, 10]
            },
            "Gradient Boosting": {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.05, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7]
            },
            "SVR": {
                'regressor__C': [0.1, 1, 10],
                'regressor__epsilon': [0.01, 0.1, 0.2],
                'regressor__gamma': ['scale', 'auto']
            }
        }
        
        if model_name in param_grids:
            self.logger.info(f"Tuning hyperparameters for {model_name}...")
            pipeline = Pipeline([
                ('scaler', self.get_scaler()),
                ('regressor', model)
            ])
            
            cv_strategy = TimeSeriesSplit(n_splits=3) if self.config.use_time_series_split else self.config.cv_folds
            
            grid_search = GridSearchCV(
                pipeline, 
                param_grids[model_name],
                cv=cv_strategy,
                scoring='r2',
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
                ('regressor', model)
            ])
            
    def plot_results(self, results_df: pd.DataFrame):
        """Enhanced plotting functionality."""
        # Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # R² comparison
        sns.barplot(data=results_df, x='Model', y='R²', ax=axes[0,0])
        axes[0,0].set_title('Model R² Comparison')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        sns.barplot(data=results_df, x='Model', y='RMSE', ax=axes[0,1])
        axes[0,1].set_title('Model RMSE Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        sns.barplot(data=results_df, x='Model', y='MAE', ax=axes[1,0])
        axes[1,0].set_title('Model MAE Comparison')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        sns.barplot(data=results_df, x='Model', y='MAPE', ax=axes[1,1])
        axes[1,1].set_title('Model MAPE Comparison')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_feature_importance(self, model: Any, features: List[str], model_name: str):
        """Enhanced feature importance plotting."""
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            importances = model.named_steps['regressor'].feature_importances_
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
            
    def plot_predictions(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
        """Plot actual vs predicted values."""
        y_pred = model.predict(X_test)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.6, s=30)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual PM2.5')
        plt.ylabel('Predicted PM2.5')
        plt.title(f'Actual vs Predicted - {model_name}')
        plt.text(0.05, 0.95, f'R² = {r2_score(y_test, y_pred):.3f}', 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig(f"plots/predictions_{model_name.replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_model_info(self, best_model: Any, selected_features: List[str]):
        """Save detailed model information."""
        model_info = {
            'timestamp': datetime.now().isoformat(),
            'best_model_type': type(best_model.named_steps['regressor']).__name__,
            'selected_features': selected_features,
            'config': {
                'test_size': self.config.test_size,
                'random_state': self.config.random_state,
                'cv_folds': self.config.cv_folds,
                'max_features': self.config.max_features,
                'outlier_method': self.config.outlier_method,
                'scaling_method': self.config.scaling_method,
                'feature_selection': self.config.feature_selection
            }
        }
        
        # Save as JSON
        import json
        with open('output/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
            
    def train_models(self, file_path: str = 'data/preprocessed_data.csv'):
        """Main training pipeline with comprehensive improvements."""
        try:
            # Load and preprocess data
            df = self.load_and_validate_data(file_path)
            df = self.handle_missing_values(df)
            df = self.remove_outliers(df)
            df = self.engineer_features(df)
            
            # Prepare features and target
            exclude_cols = ['pm2.5', 'aqi', 'aqi_class', 'timestamp']
            X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
            y = df['pm2.5']
            
            # Feature selection
            X_selected, selected_features = self.select_features(X, y)
            
            # Train-test split
            split_method = train_test_split
            X_train, X_test, y_train, y_test = split_method(
                X_selected, y, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state
            )
            
            # Train models
            models = self.get_models()
            results = []
            best_r2 = -np.inf
            best_model = None
            
            for name, model in models.items():
                self.logger.info(f"Training {name}...")
                
                # Get model with hyperparameter tuning if applicable
                trained_model = self.hyperparameter_tuning(model, name, X_train, y_train)
                
                # If not tuned, create regular pipeline
                if not isinstance(trained_model, Pipeline):
                    trained_model = Pipeline([
                        ('scaler', self.get_scaler()),
                        ('regressor', model)
                    ])
                
                # Cross-validation
                cv_strategy = TimeSeriesSplit(n_splits=self.config.cv_folds) if self.config.use_time_series_split else self.config.cv_folds
                cv_scores = cross_val_score(trained_model, X_train, y_train, cv=cv_strategy, scoring='r2', n_jobs=self.config.n_jobs)
                
                # Fit and evaluate
                trained_model.fit(X_train, y_train)
                metrics = self.evaluate_model(trained_model, X_test, y_test, name)
                metrics["CV R² Mean"] = cv_scores.mean()
                metrics["CV R² Std"] = cv_scores.std()
                results.append(metrics)
                
                # Track best model
                if metrics["R²"] > best_r2:
                    best_r2 = metrics["R²"]
                    best_model = trained_model
                    
                # Generate plots
                self.plot_feature_importance(trained_model, selected_features, name)
                self.plot_predictions(trained_model, X_test, y_test, name)
                
                self.logger.info(f"{name} - R²: {metrics['R²']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            
            # Save results
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('R²', ascending=False)
            results_df.to_csv('output/comprehensive_model_results.csv', index=False)
            
            # Save best model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_model_path = f"models/best_model_{timestamp}.joblib"
            joblib.dump(best_model, best_model_path)
            
            # Save model metadata
            self.save_model_info(best_model, selected_features)
            
            # Generate comprehensive plots
            self.plot_results(results_df)
            
            # Log final results
            best_model_name = results_df.iloc[0]['Model']
            self.logger.info(f"Best model: {best_model_name} with R² = {best_r2:.4f}")
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
            
        return self.best_model.predict(X_new)

def main():
    """Main execution function with configuration options."""
    # Custom configuration
    config = ModelConfig(
        test_size=0.2,
        random_state=42,
        cv_folds=5,
        max_features=12,
        outlier_method='iqr',
        scaling_method='robust',
        feature_selection='both',
        use_time_series_split=False  # Set to True if your data has temporal ordering
    )
    
    # Initialize and run pipeline
    pipeline = MLPipeline(config)
    results = pipeline.train_models('data/preprocessed_data.csv')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best model: {results.iloc[0]['Model']}")
    print(f"Best R²: {results.iloc[0]['R²']:.4f}")
    print(f"Best RMSE: {results.iloc[0]['RMSE']:.4f}")
    print("\nCheck the following directories for outputs:")
    print("- plots/: Visualizations and charts")
    print("- output/: Results and feature importance")
    print("- models/: Saved model files")
    print("- logs/: Training logs")

if __name__ == "__main__":
    main()