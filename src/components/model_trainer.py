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
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighors Classifier" : KNeighborsRegressor(),
                "XGBClassifier" : XGBRegressor(),
                "CatBoosting Classifier" : CatBoostRegressor(verbose=False),
                "AdaboostClassifier" : AdaBoostRegressor(),
            }

            model_report:dict = evaluate_models(X_train=X_train,
                                               y_train=y_train,
                                               X_test=X_test,
                                               y_test=y_test, 
                                               models=models)
            
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