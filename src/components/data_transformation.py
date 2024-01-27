import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    X_train_path = os.path.join("../../data/processed", "X_train.pkl")
    X_test_path = os.path.join("../../data/processed", "X_test.pkl")
    val_path = os.path.join("../../data/processed", "val.pkl")


class DataTransformation:
    def __init__(self):
        self.data_path = DataTransformationConfig()

    def data_transformation(self, train_path, test_path, val_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            val_data = pd.read_csv(val_path)

            logging.info("Reading of train, test and validation data completed")

            label = "battery_output"

            X_train = train_data.drop(label, axis=1)
            y_train = train_data[label]

            X_test = test_data.drop(label, axis=1)
            y_test = test_data[label]

            val = val_data

            logging.info(
                f"Applying preprocessing object on training and testing dataframe"
            )

            scaler = StandardScaler()
            X_train["windspeed_100mNewcastle upon Tyne_weather"] = scaler.fit_transform(
                X_train[["windspeed_100mNewcastle upon Tyne_weather"]]
            )

            X_test["windspeed_100mNewcastle upon Tyne_weather"] = scaler.transform(
                X_test[["windspeed_100mNewcastle upon Tyne_weather"]]
            )

            val["windspeed_100mNewcastle upon Tyne_weather"] = scaler.transform(
                val[["windspeed_100mNewcastle upon Tyne_weather"]]
            )

            X_train.to_pickle(self.data_path.X_train_path)
            X_test.to_pickle(self.data_path.X_test_path)
            val.to_pickle(self.data_path.val_path)

            return (X_train, X_test, y_train, y_test, val)

        except Exception as e:
            raise CustomException(e, sys)
