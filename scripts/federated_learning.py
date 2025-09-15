import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from datetime import datetime
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FederatedConfig:
    """Configuration class for federated learning parameters"""
    n_rounds: int = 5
    test_size: float = 0.1
    random_state: int = 42
    min_samples_per_city: int = 20
    target_column: str = 'pm2.5'
    city_column: str = 'city'
    columns_to_drop: List[str] = None
    model_type: str = 'rf'  # 'linear', 'rf', 'xgb'
    use_time_features: bool = True
    n_jobs: int = -1
    early_stopping_rounds: int = 3
    feature_selection: bool = True
    k_features: int = 20
    use_robust_scaler: bool = False
    
    def __post_init__(self):
        if self.columns_to_drop is None:
            self.columns_to_drop = ['aqi', 'aqi_class', 'timestamp']
        if self.model_type not in ['linear', 'rf', 'xgb']:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        if self.k_features < 1:
            raise ValueError("k_features must be greater than 0")

class FederatedLearningManager:
    """Main class for managing federated learning process"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.feature_columns = None
        self.global_model = None
        self.feature_importances = None
        self.scaler = RobustScaler() if config.use_robust_scaler else StandardScaler()
        self.training_history = []
        self.best_round = None
        self.model_dir = "models/federated"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def get_model(self):
        """Get appropriate model based on configuration"""
        if self.config.model_type == 'linear':
            return LinearRegression()
        elif self.config.model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=100,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state
            )
        else:  # xgb
            return XGBRegressor(
                n_estimators=100,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state
            )
            
    def create_feature_pipeline(self):
        """Create feature preprocessing pipeline"""
        numeric_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'pm2.5']
        categorical_features = ['city']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', self.scaler)
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', pd.get_dummies)
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
        
    def load_and_preprocess_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess the dataset"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Drop unnecessary columns
        df = df.drop(columns=self.config.columns_to_drop, errors='ignore')
        
        # Remove rows with missing target values
        initial_rows = len(df)
        df = df.dropna(subset=[self.config.target_column])
        logger.info(f"Removed {initial_rows - len(df)} rows with missing target values")
        
        return df
    
    def create_global_test_set(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a global test set and return remaining training data"""
        logger.info(f"Creating global test set ({self.config.test_size:.0%} of data)")
        
        global_test_df = df.sample(
            frac=self.config.test_size, 
            random_state=self.config.random_state
        )
        train_df = df.drop(global_test_df.index)
        
        return train_df, global_test_df
    
    def partition_data_by_city(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Partition data by city and filter cities with sufficient samples"""
        cities = df[self.config.city_column].unique()
        city_data = {}
        
        for city in cities:
            city_df = df[df[self.config.city_column] == city].copy()
            if len(city_df) >= self.config.min_samples_per_city:
                city_data[city] = city_df
            else:
                logger.warning(f"City {city} has only {len(city_df)} samples, skipping")
        
        logger.info(f"Partitioned data for {len(city_data)} cities: {list(city_data.keys())}")
        return city_data
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Prepare features for training/testing"""
        X = df.drop(columns=[self.config.target_column, self.config.city_column])
        X = pd.get_dummies(X, drop_first=True)
        
        # Ensure consistent feature columns
        if self.feature_columns is None:
            self.feature_columns = X.columns.tolist()
        else:
            X = X.reindex(columns=self.feature_columns, fill_value=0)
        
        # Scale features
        if fit_scaler:
            X_scaled = self.global_scaler.fit_transform(X)
        else:
            X_scaled = self.global_scaler.transform(X)
        
        return X_scaled
    
    def train_local_model(self, city_data: pd.DataFrame) -> Tuple[np.ndarray, float, int]:
        """Train a local model for a city"""
        # Prepare local data
        X_local = self.prepare_features(city_data)
        y_local = city_data[self.config.target_column].values
        
        # Train local model
        model = LinearRegression()
        model.fit(X_local, y_local)
        
        return model.coef_, model.intercept_, len(y_local)
    
    def federated_averaging(self, city_weights: List[Tuple], sample_counts: List[int]) -> Tuple[np.ndarray, float]:
        """Perform federated averaging of model weights"""
        total_samples = sum(sample_counts)
        
        # Weighted average of coefficients
        weights_array = np.array([w for w, _, _ in city_weights])
        avg_weights = np.average(weights_array, axis=0, weights=sample_counts)
        
        # Weighted average of intercepts
        intercepts = [b for _, b, _ in city_weights]
        avg_bias = np.average(intercepts, weights=sample_counts)
        
        return avg_weights, avg_bias
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the global model with comprehensive metrics"""
        if self.global_weights is None:
            raise ValueError("Global model not trained yet")
        
        # Make predictions
        y_pred = np.dot(X_test, self.global_weights[0]) + self.global_weights[1]
        
        # Calculate metrics
        metrics = {
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'msle': mean_squared_log_error(y_test, y_pred, multioutput='uniform_average'),
            'explained_variance': explained_variance_score(y_test, y_pred)
        }
        
        # Track training history
        self.training_history.append({
            'round': len(self.training_history) + 1,
            'metrics': metrics,
            'weights': self.global_weights
        })
        
        # Save current model if it's the best
        if not self.best_round or metrics['rmse'] < self.best_round['metrics']['rmse']:
            self.best_round = {
                'round': len(self.training_history),
                'metrics': metrics,
                'weights': self.global_weights
            }
            self.save_model()
        
        return metrics
    
    def save_model(self):
        """Save the current model and its metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"global_model_{timestamp}.joblib"
        metrics_filename = f"metrics_{timestamp}.json"
        
        # Save model weights
        joblib.dump(self.global_weights, os.path.join(self.model_dir, model_filename))
        
        # Save metrics
        metrics = self.best_round['metrics'] if self.best_round else self.training_history[-1]['metrics']
        with open(os.path.join(self.model_dir, metrics_filename), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Saved model and metrics to {self.model_dir}")
        
    def plot_training_progress(self):
        """Plot training progress metrics"""
        if not self.training_history:
            logger.warning("No training history available for plotting")
            return
            
        rounds = [h['round'] for h in self.training_history]
        metrics = self.training_history[0]['metrics'].keys()
        
        plt.figure(figsize=(15, 10))
        for metric in metrics:
            values = [h['metrics'][metric] for h in self.training_history]
            plt.plot(rounds, values, marker='o', label=metric)
        
        plt.title('Training Progress')
        plt.xlabel('Training Round')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_progress.png'))
        plt.close()
        
    def plot_feature_importance(self, feature_names: List[str]):
        """Plot feature importance if available"""
        if not self.feature_importances:
            return
            
        plt.figure(figsize=(12, 8))
        sorted_idx = np.argsort(self.feature_importances)[::-1]
        plt.barh(range(len(sorted_idx)), self.feature_importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Importance')
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'feature_importance.png'))
        plt.close()
    
    def run_federated_learning(self, filepath: str) -> Dict[str, float]:
        """Main method to run the complete federated learning process"""
        try:
            # Load and preprocess data
            df = self.load_and_preprocess_data(filepath)
            
            # Create global test set
            train_df, global_test_df = self.create_global_test_set(df)
            
            # Partition data by city
            city_data = self.partition_data_by_city(train_df)
            
            if len(city_data) == 0:
                raise ValueError("No cities have sufficient data for training")
            
            # Prepare global test features (fit scaler on first use)
            X_global_test = self.prepare_features(global_test_df, fit_scaler=True)
            y_global_test = global_test_df[self.config.target_column].values
            
            logger.info(f"Starting federated learning with {self.config.n_rounds} rounds")
            logger.info(f"Cities participating: {list(city_data.keys())}")
            
            # Federated training rounds
            for round_num in range(self.config.n_rounds):
                logger.info(f"Round {round_num + 1}/{self.config.n_rounds}")
                
                city_weights = []
                sample_counts = []
                
                # Train local models
                for city, data in city_data.items():
                    try:
                        weights, bias, n_samples = self.train_local_model(data)
                        city_weights.append((weights, bias, n_samples))
                        sample_counts.append(n_samples)
                        logger.debug(f"  {city}: {n_samples} samples")
                    except Exception as e:
                        logger.error(f"Error training model for {city}: {e}")
                        continue
                
                if not city_weights:
                    raise ValueError(f"No successful local training in round {round_num + 1}")
                
                # Federated averaging
                self.global_weights = self.federated_averaging(city_weights, sample_counts)
                logger.info(f"  Aggregated weights from {len(city_weights)} cities")
            
            # Final evaluation
            logger.info("Evaluating global model on test set")
            final_metrics = self.evaluate_model(X_global_test, y_global_test)
            
            # Log results
            logger.info("Final Results:")
            for metric, value in final_metrics.items():
                logger.info(f"  {metric.upper()}: {value:.4f}")
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Federated learning failed: {e}")
            raise

def main():
    """Main execution function with comprehensive reporting"""
    # Configuration
    config = FederatedConfig(
        n_rounds=20,  # Increased for better convergence
        test_size=0.1,
        random_state=42,
        min_samples_per_city=20,
        model_type='rf',  # Using RandomForest for better performance
        use_time_features=True,
        n_jobs=-1,
        early_stopping_rounds=3,
        feature_selection=True,
        k_features=20,
        use_robust_scaler=True
    )
    
    # Initialize and run federated learning
    fl_manager = FederatedLearningManager(config)
    
    try:
        # Run federated learning
        metrics = fl_manager.run_federated_learning('data/preprocessed_data.csv')
        
        # Print comprehensive results
        print("\n" + "="*50)
        print("FEDERATED LEARNING RESULTS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        print("="*50)
        
        # Generate plots
        fl_manager.plot_training_progress()
        fl_manager.plot_feature_importance(fl_manager.feature_columns)
        
        # Save final model
        fl_manager.save_model()
        
        print("\nModel training completed successfully!")
        print(f"Best model saved at round {fl_manager.best_round['round']}")
        print(f"Best model RMSE: {fl_manager.best_round['metrics']['rmse']:.4f}")
        print(f"Training history saved in {fl_manager.model_dir}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())