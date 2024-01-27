import os
import sys
import pandas as pd

from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sktime.forecasting.fbprophet import Prophet
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("../../models", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
    ):
        try:
            models = {"XGBRegressor": XGBRegressor()}

            params = {
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001, 0.5],
                    "n_estimators": [4, 8, 16, 32, 64, 128, 256, 512],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # To get the best model score from dict
            best_model_score = max(sorted(list(model_report.values())))

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # if best_model_score < 0.6:
            #     raise CustomException("No best model found.")
            logging.info(f"Found best model on both training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_path, obj=best_model
            )

        except Exception as e:
            raise CustomException(e, sys)
