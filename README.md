# Battery Forecasting
==============================

## Overview

Battery Forecasting is a project aimed at predicting the charging and discharging patterns of batteries on the electricity network for Great Britain's Electricity System Operator (ESO). By forecasting battery usage, this project contributes to optimizing energy management, enhancing grid stability, and supporting the integration of renewable energy sources into the grid.

## Objective

The primary objective of Battery Forecasting is to develop accurate and reliable models that forecast the behavior of batteries connected to the electricity grid. These forecasts enable energy operators and stakeholders to make informed decisions regarding energy storage, grid management, and renewable energy integration.

## Features

    Time Series Analysis: Utilizes advanced time series analysis techniques to model battery charging and discharging patterns over time.
    
    Machine Learning Model: Employed XGBRegressor, to predict battery behavior based on historical data and relevant features.
    
    Exogenous Variables: Considers exogenous variables such as weather conditions, electricity demand, and grid load to improve the accuracy of battery forecasts.
    
    Grid Optimization: Supports grid optimization efforts by providing timely and accurate predictions of battery usage, aiding in load balancing and grid stability.

## Components

    Data Ingestion: Collects and preprocesses historical data on battery usage, weather conditions, electricity demand, and other relevant variables.
    
    Model Training: Trains machine learning models using historical data to forecast battery behavior.
    
    Evaluation: Assesses the performance of trained models using appropriate evaluation metrics and techniques.
    
    Deployment: Integrates the forecasting models into energy management systems and grid infrastructure for real-time decision-making.


# How to run?
### STEPS:

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/SumeetChougule/battery-forecasting.mlflow \
MLFLOW_TRACKING_USERNAME=SumeetChougule \
MLFLOW_TRACKING_PASSWORD=90cd3850e33bbd72900418c750f9caa930f0deda \
python script.py

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/SumeetChougule/battery-forecasting.mlflow

export MLFLOW_TRACKING_USERNAME=SumeetChougule

export MLFLOW_TRACKING_PASSWORD=90cd3850e33bbd72900418c750f9caa930f0deda

```
