from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import os
import sys
from dataclasses import dataclass
from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("pliting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regressor" : KNeighborsRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoost Regressor" : CatBoostRegressor(verbose=False),
                "Adaboost Regressor" : AdaBoostRegressor(),
            }
            params = {
                "Decision Tree": {
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_features': ['sqrt', 'log2'],  
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                },
                "XGBRegressor": {
                    'learning_rate': [1, 0.1, 0.05, 0.01],
                    'n_estimators': [8, 16, 32, 64, 128, 256]  
                },
                "CatBoost Regressor": {  
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Adaboost Regressor": {
                    'learning_rate': [1, 0.1, 0.5, 0.01],
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
            model_report:dict = evaluate_models(X_train=X_train,
                                               y_train=y_train,
                                               X_test=X_test,
                                               y_test=y_test, 
                                               models=models,
                                               params=params)
            
            logging.info(str(model_report))
            max_score = max(list(model_report.values()))
            model_name = [key for key in model_report.keys() if model_report[key] == max_score]
            best_model = models[model_name[0]]

            if max_score < 0.6:
                raise Custom_Exception("No best model found", sys)
            
            logging.info("found best model")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            y_test_pred = best_model.predict(X_test)

            r2 = r2_score(y_test, y_test_pred)

            return r2

        except Exception as e:
            raise Custom_Exception(e,sys)