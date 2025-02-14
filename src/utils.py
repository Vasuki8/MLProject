import os
import sys
import dill
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import Custom_Exception
from sklearn.metrics import  r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise Custom_Exception(e,sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for key in models.keys():
            model = models[key]
            param = params[key]

            # gs = GridSearchCV(model, param, cv=5,error_score='raise')
            rs = RandomizedSearchCV(model, param, cv=5, n_iter=10)
            # gs.fit(X_train,y_train)
            rs.fit(X_train,y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train,y_train)

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[key] = test_model_score
            logging.info(f"Completed {key} training")
        return report
    
    except Exception as e:
        raise Custom_Exception(e,sys)