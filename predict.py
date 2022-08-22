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

from train_model import (preprocess_data, read_data, scrap_covid_indicator,
                         scrap_covid_test)

length_pred = 14
features_indicator = [
    "hosp",
    "incid_hosp",
    "rea",
    "incid_rea",
    "rad",
    "incid_rad",
    "pos",
    "pos_7j",
    "tx_pos",
    "tx_incid",
    "TO"
]
features_test = ["pop", "P", "T", "Ti", "Tp", "Td"]
features = features_indicator + features_test
target = ["incid_dchosp"]


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


@flow(name="predict_flow")
def main(covid_indicator_path=None, covid_test_path=None):
    covid_indicator_df, covid_test_df = read_data(covid_indicator_path, covid_test_path)
    covid_df = preprocess_data(covid_indicator_df, covid_test_df)

    y, past_cov, target_scaler, past_cov_scaler = create_preprocess_time_series(
        covid_df
    )

    model_name = "TransformerModel"
    # with open(f"models/{model_name}", "rb") as f_in:
    #     model = pickle.load(f_in)
    model = TransformerModel.load(f"models/{model_name}")
    print(covid_indicator_df)
    y_pred = predict(model, y, past_cov, target_scaler, past_cov_scaler)
    y = target_scaler.inverse_transform(y)
    return y, y_pred

if __name__ == '__main__':
    print(main())
