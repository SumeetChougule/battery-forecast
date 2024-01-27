import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sktime.utils.plotting import plot_series
from sktime.forecasting.fbprophet import Prophet

pd.read_pickle("../data/processed/val.pkl")

train_df = pd.read_csv("../data/interim/train.csv")
test_df = pd.read_csv("../data/interim/test.csv")
pred_df = pd.read_csv("../data/interim/predicted.csv")


test_df = pd.read_csv("../data/raw/val_data.csv")

test_df["battery_output"]

sample_submission = pd.read_csv(
    "../../data/raw/eso-battery-forecasting/sample_submission.csv"
)

df = df.drop(columns="battery_output", axis=1)


# numerical, categorical and datetime features

numerical_col = df.select_dtypes(include=["int", "float"]).columns

categorical_col = df.select_dtypes(include=["object"], exclude=["datetime64"]).columns

datetime_col = df.select_dtypes(include=["datetime64"]).columns


# get a list of numerical columns that don't contain the strings in strings_to_remove in their column names
strings_to_remove = ["id", "longitude", "latitude", "utc_offset", "code"]
select_num_cols = [
    col for col in numerical_col if all(x not in col.lower() for x in strings_to_remove)
]


corr_matrix = df[select_num_cols].corr()

corr_matrix["battery_output"][abs(corr_matrix["battery_output"]) > 0.02]

hidden_categorical_cols = [col for col in numerical_col if "is_" in col.lower()]


df.loc[df["is_dayNewcastle upon Tyne_weather"] == 1, "battery_output"].mean()

df.loc[df["is_dayNewcastle upon Tyne_weather"] == 0, "battery_output"].mean()


t_stat, p_val = stats.ttest_ind(
    df[df["is_dayNewcastle upon Tyne_weather"] == 1]["battery_output"],
    df[df["is_dayNewcastle upon Tyne_weather"] == 0]["battery_output"],
    equal_var=True,
)


autocorrelation_lag1 = df["battery_output"].autocorr(lag=50)


plt.figure(figsize=(12, 4))
plot_acf(df["battery_output"], lags=100, zero=False)
plt.gca().set_ylim(-0.3, 0.3)
plt.title("Autocorrelation Function")
plt.show()

autocorrs = [df["battery_output"].autocorr(lag=i) for i in range(1, 18000)]

autocorr_series = pd.Series(autocorrs)

top_5_lags = autocorr_series.abs().nlargest(5)


# time-lagged variables

df["battery_output_lag1"] = df["battery_output"].shift(1)
df["battery_output_lag2"] = df["battery_output"].shift(2)
df["battery_output_lag48"] = df["battery_output"].shift(48)

mean_value = df["battery_output"].mean()
df["battery_output_lag1"] = df["battery_output_lag1"].fillna(mean_value)
df["battery_output_lag2"] = df["battery_output_lag2"].fillna(mean_value)
df["battery_output_lag48"] = df["battery_output_lag48"].fillna(mean_value)

# train dataset

column_list = [
    "battery_output_lag1",
    "battery_output_lag2",
    "battery_output_lag48",
    "windspeed_100mNewcastle upon Tyne_weather",
    "is_dayNewcastle upon Tyne_weather",
]
y = df["battery_output"]
X = df[column_list]
X = X.fillna(method="ffill").fillna(method="bfill")

scaler = StandardScaler()
X["windspeed_100mNewcastle upon Tyne_weather"] = scaler.fit_transform(
    X[["windspeed_100mNewcastle upon Tyne_weather"]]
)

# Define parameter grid
param_grid = {
    "learning_rate": [0.005, 0.1, 0.01, 0.05, 0.001],
    "n_estimators": [8, 16, 32, 64, 128, 256, 512],
    # "reg_alpha": [0.0, 0.1, 1.0],
    # "objective": ["regression", "regression_l1", "huber"]
}

tscv = TimeSeriesSplit(n_splits=5)

model = XGBRegressor()

grid_search = GridSearchCV(
    model, param_grid=param_grid, cv=tscv, n_jobs=-1, scoring="neg_mean_absolute_error"
)

grid_search.fit(X, y)

best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

best_model = grid_search.best_estimator_

# predictions on test data

test_df["battery_output_lag1"] = 0
test_df["battery_output_lag2"] = 0
test_df["battery_output_lag48"] = 0
test_df = test_df[column_list]

test_df = test_df.ffill().bfill()
test_df["windspeed_100mNewcastle upon Tyne_weather"] = scaler.transform(
    test_df[["windspeed_100mNewcastle upon Tyne_weather"]]
)


last_known_op = df["battery_output"].iloc[-1]
test_df.loc[test_df.index[0], "battery_output_lag1"] = last_known_op
test_df.loc[test_df.index[0:2], "battery_output_lag2"] = df["battery_output"][
    -2:
].values
test_df.loc[test_df.index[0:48], "battery_output_lag48"] = df["battery_output"][
    -48:
].values


actual_preds = []
for i in range(len(test_df)):
    ip_features = test_df.iloc[i]
    predicted_op = best_model.predict([ip_features])[0]
    if i + 1 < len(test_df):
        test_df.loc[test_df.index[i + 1], "battery_output_lag1"] = predicted_op
    if i + 2 < len(test_df):
        test_df.loc[test_df.index[i + 1], "battery_output_lag2"] = predicted_op
    if i + 48 < len(test_df):
        test_df.loc[test_df.index[i + 1], "battery_output_lag48"] = predicted_op

    actual_preds.append(predicted_op)
sample_submission["battery_output"] = actual_preds
sample_submission.head(48)


X_train = pd.read_pickle("../data/processed/X_train.pkl")
X_test = pd.read_pickle("../data/processed/X_test.pkl")


endog = train_df["battery_output"]
exog = train_df[["battery_output_lag1", "battery_output_lag2", "battery_output_lag48"]]

y_true = test_df["battery_output"]
exog_forecast = test_df[
    ["battery_output_lag1", "battery_output_lag2", "battery_output_lag48"]
]

params_arima = {
    "p": [0, 1, 2, 3],
    "d": [0, 1, 2],
    "q": [0, 1, 2, 3],
}

model = ARIMA(endog=endog, exog=exog)
mt = model.fit()

y_pred = mt.forecast(steps=7892, exog=exog_forecast)

mae = mean_absolute_error(y_true, y_pred)

endog.plot()
plot_series(y_true, y_pred)


from itertools import product


def grid_search_arima(train_data, exog, test_data, exog_forecast, params_arima):
    best_score, best_cfg = float("inf"), None
    for p, d, q in product(params_arima["p"], params_arima["d"], params_arima["q"]):
        order = (p, d, q)
        try:
            model = ARIMA(endog=train_data, exog=exog, order=order)
            model_fit = model.fit()
            y_pred = model_fit.forecast(steps=len(test_data), exog=exog_forecast)[0]
            mse = mean_squared_error(test_data, y_pred)
            if mse < best_score:
                best_score, best_cfg = mse, order
            print(f"ARIMA{order} MSE={mse}")
        except:
            continue
    print(f"Best ARIMA{best_cfg} MSE={best_score}")
    return best_cfg


# Example usage:
# Assuming train_data and test_data are your training and testing datasets
params_arima = {
    "p": [1, 2, 3],
    "d": [1, 2],
    "q": [1, 2, 3],
}
best_params = grid_search_arima(
    train_data=endog,
    exog=exog,
    test_data=y_true,
    exog_forecast=exog_forecast,
    params_arima=params_arima,
)


###

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        batch_size = x.size(0)  # Get the batch size from the input tensor
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out


import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Assuming you have your data prepared as X_train, y_train, X_test, y_test

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(train_df["battery_output"], dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(test_df["battery_output"], dtype=torch.float32)

# Create DataLoader for training and test sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64)

# Define model parameters
input_size = X_train.shape[1]  # Number of features
hidden_size = 64  # Number of hidden units in LSTM
num_layers = 2  # Number of LSTM layers
output_size = 1  # Number of output units

# Create LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs.squeeze(), labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Perform predictions
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    y_pred = model(X_test_tensor)

# Compute loss on test set
test_loss = criterion(y_pred.squeeze(), y_test_tensor)
print(f"Test Loss: {test_loss.item():.4f}")


# Assuming train_loader is your DataLoader object
# Iterate over the DataLoader to get one batch of data
for inputs, labels in train_loader:
    # Check the shape of the inputs tensor
    print("Shape of inputs tensor:", inputs.shape)
    # Check the shape of the labels tensor
    print("Shape of labels tensor:", labels.shape)
    # Break after processing one batch
    break


y_test = test_df["battery_output"]

forecaster = Prophet()
forecaster.fit(endog)

from sktime.forecasting.base import ForecastingHorizon

fh = ForecastingHorizon(y_test.index, is_relative=False)

forecaster.add_regressor("exog")

y_pred = forecaster.predict(fh)

ci = forecaster.predict_interval(fh, coverage=0.9)

mae = mean_absolute_error(y_test, y_pred)
