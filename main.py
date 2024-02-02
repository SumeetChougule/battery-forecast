import pandas as pd

from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.pipeline.prediction import PredictionPipeline


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


predicted_df = pd.DataFrame(prediction, columns=["predicted_values"])

# Save the DataFrame to a CSV file
predicted_df.to_csv("data/prediction.csv", index=False, header=True)

pd.read_csv("data/prediction.csv")
