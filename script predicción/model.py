
from typing import Dict, Any
from dataclasses import dataclass
from sklearn.ensemble import (
    VotingRegressor, RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.linear_model import (
    ElasticNet, Lasso, Ridge, LinearRegression
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pygam import LinearGAM, s

RANDOM_SEED = 42

@dataclass
class ModelConfig:
    """Configuration class for ML models and their hyperparameters."""
    base_model: Any
    param_grid: Dict[str, Any]
    
class ModelFactory:
    """Factory class for creating ML models with their configurations."""
    
    @staticmethod
    def get_model_config(model_name: str) -> ModelConfig:
        """Returns model and its hyperparameter grid based on model name."""
        models = {
            'LinearRegression': ModelConfig(
                LinearRegression(n_jobs=-1),
                {
                    'fit_intercept': [True, False],
                    'positive': [True, False],
                }
            ),
            'XGBoost': ModelConfig(
                XGBRegressor(n_jobs=-1, random_state=RANDOM_SEED),
                {
                    'n_estimators': [1000, 2000, 1500, 2500],
                    'learning_rate': [0.1, 0.3],
                    'max_depth': [2, 5, 7],
                    'min_child_weight': [1, 2, 5],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1],
                    'gamma': [0, 0.5, 0.75],
                    'reg_alpha': [0.3, 0.1, 0.5],
                    'reg_lambda': [1, 5, 7]
                }
            ),
            'HistGradientBoosting': ModelConfig(
                HistGradientBoostingRegressor(random_state=RANDOM_SEED),
                {
                    'learning_rate': [0.01, 0.1],
                    'max_iter': [100, 300, 500],
                    'max_depth': [None, 5, 10],
                    'min_samples_leaf': [10, 20, 30],
                    'l2_regularization': [0, 0.1, 0.5],
                    'max_bins': [255, 512],
                }
            ),
            'ExtraTrees': ModelConfig(
                ExtraTreesRegressor(n_jobs=-1, random_state=RANDOM_SEED),
                {
                    'n_estimators': [1000,3000,5000],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'bootstrap': [True, False],
                }
            ),
            'RandomForest': ModelConfig(
                RandomForestRegressor(n_jobs=-1, random_state=RANDOM_SEED), 
                {
                    'n_estimators': [5,10,20,22,25,30],
                    'max_depth': [None,3,7, 10,12, 15],
                    'min_samples_split': [2, 5,6],
                    'min_samples_leaf': [1, 2, 3, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                }
            ),
            'LGBM': ModelConfig(LGBMRegressor(random_state=RANDOM_SEED), 
                {
                    'n_estimators': [90,100,110,120],
                    'learning_rate': [0.15,0.2, 0.1,0.3],
                    'max_depth': [-1,3],
                    'num_leaves': [5,7,8],
                    'subsample': [0.5,0.6,0.8],
                    'colsample_bytree': [1.0,1.5,3],
                    'reg_alpha': [0.15, 0.1,0.05],
                    'reg_lambda': [0.3, 0.5],
                    'min_child_weight': [1,2,3],
                    'min_child_samples': [20,30,40,50],
                    'objective': ['regression', 'poisson'],
                }
            ),
            'CatBoost': ModelConfig(CatBoostRegressor(verbose=False, random_state=RANDOM_SEED), 
                {
                    'n_estimators': [200,250,300],           # Reducido para evitar overfitting
                    'learning_rate': [0.3,0.5],         # Tasas moderadas para balance
                    'max_depth': [2,3, 5],                  # Profundidades menores para evitar overfitting
                    'subsample': [0.6,0.8, 1.0],              # Mayor proporción de datos dado el tamaño pequeño
                    'reg_lambda': [3,4,5,6],                # Mayor regularización para controlar la varianza
                    'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
                    'bagging_temperature': [0.5, 0.8, 1.0],
                    'max_bin': [255, 512, 1024],
                    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
                    'min_data_in_leaf': [1, 2, 3],
                    'loss_function': ['RMSE', 'MAE', 'Poisson'],
                }
            ),
            'ElasticNet': ModelConfig(ElasticNet(random_state=RANDOM_SEED), 
                {
                    'alpha': [0.01, 0.1, 0.5, 1.0, 5.0],
                    'l1_ratio': [0.2, 0.5, 0.7, 0.9],
                    'max_iter': [1000, 5000, 10000]
                }
            ),
            'SVR': ModelConfig(SVR(), 
                {
                    'C': [0.1, 1.0, 10, 100],
                    'epsilon': [0.01, 0.1, 0.5, 1],
                    'kernel': ['rbf', 'linear', 'poly']
                }
            ),
            'Lasso': ModelConfig(Lasso(random_state=RANDOM_SEED),
                {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'max_iter': [1000, 3000, 5000],
                    'tol': [1e-4, 1e-3, 1e-2],
                    'selection': ['cyclic', 'random'],
                    'fit_intercept': [True, False]
                }
            ),

            'Ridge': ModelConfig(Ridge(random_state=RANDOM_SEED), 
                    {
                        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                        'max_iter': [1000, 3000, 5000],
                        'tol': [1e-4, 1e-3, 1e-2],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                        'fit_intercept': [True, False]
                    }
                ),

            'AdaBoost': ModelConfig(AdaBoostRegressor(random_state = RANDOM_SEED),
                    {
                        'n_estimators': [28,29,30,31,32,33,40],
                        'learning_rate': [0.001,0.004,0.003,0.01, 0.05],
                        'loss': ['linear', 'square', 'exponential'],
                    }
                ),

            'GradientBoosting': ModelConfig(GradientBoostingRegressor(random_state=RANDOM_SEED), 
                    {
                        'n_estimators': [200,250,100, 300, 500],
                        'learning_rate': [0.5, 0.1, 0.3],
                        'max_depth': [2,3, 5],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2],
                        'subsample': [0.6, 0.8,1],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'min_impurity_decrease': [0.0, 0.01,0.1],
                        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile']
                    }
                ),

            'Best_CatBoost': ModelConfig(CatBoostRegressor(verbose=False,
                                                random_state=RANDOM_SEED,
                                                learning_rate=0.3,
                                                max_depth=3,
                                                n_estimators=309,
                                                reg_lambda=5,
                                                subsample=0.8), {}),

            'Best_RandomForest': ModelConfig(RandomForestRegressor(random_state=RANDOM_SEED,
                                                        criterion='poisson',
                                                        max_depth=10,
                                                        max_features='sqrt',
                                                        min_samples_leaf=2,
                                                        min_samples_split=5,
                                                        n_estimators=25), {}),

            'Best_LGBM': ModelConfig(LGBMRegressor(random_state=RANDOM_SEED,
                                        colsample_bytree=1.0,
                                        learning_rate=0.1,
                                        max_depth=-1,
                                        n_estimators=100,
                                        num_leaves=7,
                                        reg_alpha=0.1,
                                        reg_lambda=0.5,
                                        subsample=0.8), {}),
            'Best_XGBoost': ModelConfig(XGBRegressor(random_state=RANDOM_SEED,
                                        colsample_bytree=1,
                                        gamma=0.5,
                                        learning_rate=0.1,
                                        max_depth=7,
                                        min_child_weight=5,
                                        n_estimators=1000,
                                        reg_alpha=0.1,
                                        reg_lambda=7,
                                        subsample=0.8), {}),
            'Best_GradientBoosting': ModelConfig(GradientBoostingRegressor(random_state=RANDOM_SEED,
                                                                learning_rate=0.3,
                                                                loss='huber',
                                                                max_depth=2,
                                                                max_features='sqrt',
                                                                min_impurity_decrease=0.01,
                                                                min_samples_leaf=1,
                                                                min_samples_split=2,
                                                                n_estimators=300,
                                                                subsample=0.8), {}),
            'Best_AdaBoost': ModelConfig(AdaBoostRegressor(random_state=RANDOM_SEED,
                                                learning_rate=0.003,
                                                loss='exponential',
                                                n_estimators=30), {}),
            'Ensemble': ModelConfig(
                VotingRegressor([
                    ('catboost', CatBoostRegressor(
                        verbose=False, random_state=RANDOM_SEED,
                        learning_rate=0.3, max_depth=3,
                        n_estimators=300, reg_lambda=5,
                        subsample=0.8
                    )),
                    ('randomforest', RandomForestRegressor(
                        random_state=RANDOM_SEED, criterion='poisson',
                        max_depth=10, max_features='sqrt',
                        min_samples_leaf=2, min_samples_split=5,
                        n_estimators=25
                    )),
                    ('lgbm', LGBMRegressor(
                        random_state=RANDOM_SEED, colsample_bytree=1.0,
                        learning_rate=0.1, max_depth=-1,
                        min_child_samples=40, min_child_weight=1,
                        n_estimators=110, num_leaves=5,
                        reg_alpha=0.05, reg_lambda=0.3,
                        subsample=0.5
                    )),
                    ('xgboost', XGBRegressor(
                        random_state=RANDOM_SEED, colsample_bytree=1,
                        gamma=0.5, learning_rate=0.1, max_depth=7,
                        min_child_weight=5, n_estimators=1000,
                        reg_alpha=0.1, reg_lambda=7, subsample=0.8
                    )),
                    ('gradientboosting', GradientBoostingRegressor(
                        random_state=RANDOM_SEED, learning_rate=0.1,
                        loss='huber', max_depth=2, max_features='log2',
                        min_impurity_decrease=0.0, min_samples_leaf=2,
                        min_samples_split=5, n_estimators=500,
                        subsample=1
                    ))
                ]),
                #{'weights': [(6, 3.5, 3, 1, 2)]} 219!!!
                #{'weights': [(6, 3.5, 3, 1, 3)]} 219_mejor!!
                #{'weights': [(6, 3.5, 2.7, 1, 3)]} # 218
                {'weights': [(6, 3.5, 2.7, 1, 3.5)]} # 218.75
                
            )
        }
        return models.get(model_name)