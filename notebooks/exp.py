import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from xgboost import XGBRegressor


df = pd.read_pickle("../../data/interim/train_data_processed.pkl")

test_df = pd.read_pickle("../../data/interim/test_data_processed.pkl")

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
