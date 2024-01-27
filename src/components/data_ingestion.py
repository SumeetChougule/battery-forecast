import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sktime.split import temporal_train_test_split
from dataclasses import dataclass

from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)

from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.model_evaluation import ModelEvaluation
from src.pipeline.prediction import PredictionPipeline


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("../../data/interim", "train.csv")
    test_data_path: str = os.path.join("../../data/interim", "test.csv")
    val_data_path: str = os.path.join("../../data/interim", "val.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method")
        try:
            df = pd.read_csv("../../data/interim/data.csv")
            logging.info("Read the dataset as df")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            logging.info("Train test split initiated")

            train_df, test_df = temporal_train_test_split(df, train_size=0.8)

            train_df.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            test_df.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Data ingestion completed!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.val_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, val_data = obj.initiate_data_ingestion()

    trans_obj = DataTransformation()
    (X_train, X_test, y_train, y_test, val) = trans_obj.data_transformation(
        train_path=train_data, test_path=test_data, val_path=val_data
    )

    train_obj = ModelTrainer()
    print(train_obj.initiate_model_trainer(X_train, X_test, y_train, y_test))

    eval_obj = ModelEvaluation()
    eval_obj.log_into_mlflow(X_test=X_test, y_test=y_test)

    pred = PredictionPipeline()
    prediction = pred.predict(data=val)
