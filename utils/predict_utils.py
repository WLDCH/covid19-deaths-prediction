import datetime

import matplotlib as mpl

mpl.use("Agg")

import pickle

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import (AutoARIMA, BlockRNNModel, ExponentialSmoothing,
                          NBEATSModel, NHiTSModel, RandomForest,
                          RegressionModel, TCNModel, TFTModel,
                          TransformerModel)
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from webdriver_manager.chrome import ChromeDriverManager
from utils.variables import EXPERIMENT_NAME, TRACKING_URI, length_pred, features_indicator, features_test, features, target


@task
def create_preprocess_time_series(covid_df):
    y = TimeSeries.from_series(covid_df[target])
    past_cov = TimeSeries.from_dataframe(covid_df[features])

    with open("models/scalers", "rb") as f_in:
        target_scaler, past_cov_scaler = pickle.load(f_in)

    y_scaled = target_scaler.transform(y)
    past_cov_scaled = past_cov_scaler.transform(past_cov)

    return (y_scaled, past_cov_scaled, target_scaler, past_cov_scaler)



@task
def predict(model, y, past_cov, target_scaler, past_cov_scaler):
    y_pred = model.predict(n=length_pred, series=y, past_covariates=past_cov)

    return np.round(target_scaler.inverse_transform(y_pred).pd_dataframe()).astype(int)