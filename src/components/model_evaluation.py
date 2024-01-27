import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib


class ModelEvaluation:
    def eval_metrics(self, actual, predicted):
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        return rmse, mae

    def log_into_mlflow(self, X_test, y_test):
        model = joblib.load("../../models/model.pkl")

        mlflow.set_registry_uri(
            "https://dagshub.com/SumeetChougule/battery-forecasting.mlflow"
        )
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted = model.predict(X_test)

            rmse, mae = self.eval_metrics(y_test, predicted)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="XGBRegressor"
                )

            else:
                mlflow.sklearn.log_model(model, "model")
