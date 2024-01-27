from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon

from sktime.split import temporal_train_test_split

from sktime.forecasting.theta import ThetaForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
import pandas as pd
from sktime.utils.plotting import plot_series

from sklearn.metrics import mean_absolute_error, r2_score
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.fbprophet import Prophet
import matplotlib.pyplot as plt
import numpy as np


y = pd.DataFrame(load_airline())

y.plot()

y_train, y_test = temporal_train_test_split(y, train_size=0.8)

fh = ForecastingHorizon(y_test.index, is_relative=False)

forecaster = ThetaForecaster(sp=12)
forecaster.fit(y_train)

y_pred = forecaster.predict(fh)

mean_absolute_percentage_error(y_test, y_pred)

plot_series(y, y_pred)


traffic = pd.read_csv("traffic.csv", parse_dates=[0], index_col=[0])

traffic = traffic.pivot(columns="Junction", values="Vehicles")

forecast_df = traffic.resample(rule="D").sum()

forecaster = Prophet()
horizon = 30
df = forecast_df[1]

y_train = df[:-horizon]
y_test = df.tail(horizon)

forecaster.fit(y_train)

fh = ForecastingHorizon(y_test.index, is_relative=False)

y_pred = forecaster.predict(fh)

ci = forecaster.predict_interval(fh, coverage=0.9)

y_true = df.tail(horizon)

mae = mean_absolute_error(y_true, y_pred)
