import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import KFold, GridSearchCV
import joblib
from typing import Tuple, Dict, Any, Optional
from dataloader import DataProcessor, TMP_PATH, DATA_PATH
from model import ModelFactory, RANDOM_SEED
class Pipeline:
    """Main pipeline class for training and prediction."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_config = ModelFactory.get_model_config(model_name)
        if self.model_config is None:
            raise ValueError(f"Model {model_name} not found")
            
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE with rounding to 2 decimals."""
        return np.sqrt(mean_squared_error(
            np.round(y_true, 2),
            np.round(y_pred, 2)
        ))
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """Train model and perform grid search CV."""
        print("Selected features:", X.columns.tolist())
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scorer = make_scorer(self.rmse, greater_is_better=False)
        
        grid_search = GridSearchCV(
            estimator=self.model_config.base_model,
            param_grid=self.model_config.param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X, y)
        
        results = {
            'model_name': self.model_name,
            'best_params': grid_search.best_params_,
            'best_rmse': -grid_search.best_score_,
            'std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
            'n_features':  X.shape[1]
        }
        print("Best parameters:", grid_search.best_params_)
        print("Best RMSE:", -grid_search.best_score_, "+/-", results['std'])

        # Save model and results
        joblib.dump(grid_search.best_estimator_, 
                   TMP_PATH / f'best_{self.model_name}.joblib')
        joblib.dump(results, 
                   TMP_PATH / f'results_{self.model_name}.joblib')
        
        return grid_search.best_estimator_, results

    def predict(self, X: pd.DataFrame, y: pd.Series, team_name: str) -> np.ndarray:
        """Make predictions on test data."""
        # Load best model and parameters
        best_results = joblib.load(TMP_PATH / f'results_{self.model_name}.joblib')
        print("Best training results:", best_results)
        
        # Train model with best parameters on full dataset
        self.model_config.base_model.set_params(**best_results['best_params'])
        self.model_config.base_model.fit(X, y)
        print("Model trained on full dataset.")
        print("RMSE on full dataset:", self.rmse(y, self.model_config.base_model.predict(X)))

        # Load and prepare test data
        test_data = pd.read_csv(DATA_PATH / 'test/test_data_clean.csv')
        lotes = test_data['Lote'].values
        test_features = DataProcessor.load_test_data()
        
        # Make predictions
        predictions = np.round(
            self.model_config.base_model.predict(test_features),
            2
        )
        # Save predictions
        pd.DataFrame({
            'LOTE': lotes,
            'PRODUCTO 1': predictions
        }).sort_values('LOTE').to_csv(
            f"{team_name}_UH2024.txt",
            sep='|',
            index=False,
            header=False,
            float_format='%.6f'
        )
        return predictions