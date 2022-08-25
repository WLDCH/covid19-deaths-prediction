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

from utils.read_preprocess import (preprocess_data, read_data, scrap_covid_indicator,
                         scrap_covid_test)
from utils.predict_utils import create_preprocess_time_series, predict

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
    y_pred = predict(model, y, past_cov, target_scaler, past_cov_scaler)
    y = target_scaler.inverse_transform(y)
    return y, y_pred

if __name__ == '__main__':
    print(main())
