import datetime

import matplotlib as mpl

mpl.use("Agg")


import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from darts import TimeSeries
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

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("covid-deaths-prediction")


@task
def scrap_covid_indicator():
    # Scraps covid_indicator dataset
    covid_indicator_page = requests.get(
        "https://www.data.gouv.fr/fr/datasets/synthese-des-indicateurs-de-suivi-de-lepidemie-covid-19/"
    )
    soup_indicator = BeautifulSoup(covid_indicator_page.content, "html.parser")
    covid_indicator_path = soup_indicator.find_all(
        "dd",
        {"class": "fr-ml-0 fr-col-8 fr-col-md-9 fr-col-lg-10 text-overflow-ellipsis"},
    )[-1].find("a", href=True)["href"]
    covid_indicator_df = pd.read_csv(covid_indicator_path)

    return covid_indicator_df


@task
def scrap_covid_test():
    # Scraps covid_test dataset
    options = Options()
    options.headless = True
    # driver = webdriver.Chrome("/usr/bin/chromedriver", options=options)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    driver.implicitly_wait(0.5)
    driver.get(
        "https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/"
    )
    bt = driver.find_element(
        by="xpath",
        value="//*[@id='app']/main/section[4]/div/div/section[1]/div/nav/ul/li[4]/a",
    )
    bt.click()
    pa = driver.find_element(
        by="xpath",
        value="//*[@id='resource-4e8d826a-d2a1-4d69-9ed0-b18a1f3d5ce2']/dl/div[2]/dd/a",
    )
    covid_test_path = pa.get_attribute("href")
    covid_test = pd.read_csv(covid_test_path, sep=";")
    driver.quit()

    return covid_test


@flow
def read_data(covid_indicator_path=None, covid_test_path=None):
    if covid_indicator_path is None:
        covid_indicator_df = scrap_covid_indicator()
    else:
        covid_indicator_df = pd.read_csv(covid_indicator_path)

    if covid_test_path is None:
        covid_test_df = scrap_covid_test()
    else:
        covid_test_df = pd.read_csv(covid_test_path, sep=";")
    return (covid_indicator_df, covid_test_df)


@task
def preprocess_data(covid_indicator_df, covid_test_df):

    covid_test_df.loc[:, features_test] = covid_test_df.loc[:, features_test].applymap(
        lambda row: float(row.replace(",", "."))
    )

    covid_df = pd.merge(
        left=covid_indicator_df,
        right=covid_test_df,
        left_on="date",
        right_on="jour",
        how="inner",
    )

    covid_df = covid_df[["date"] + features + target].dropna()
    covid_df.loc[:, "date"] = covid_df.date.apply(
        lambda row: datetime.datetime.strptime(row, "%Y-%m-%d")
    )
    covid_df.set_index("date", inplace=True)

    return covid_df


@task
def train_model(covid_df, model_name, input_chunk_length, output_chunk_length):
    y = TimeSeries.from_series(covid_df[target])
    past_cov = TimeSeries.from_dataframe(covid_df[features])
    y_train = y[:-14]
    y_val = y[-14:]

    if model_name == "RegressionModel":
        model = RegressionModel(
            lags=[-1, -2, -3, -4, -5, -6, -7],
            lags_past_covariates=[-1, -2, -3, -4, -5, -6, -7],
            model=RandomForestRegressor(),
        )
    elif model_name == "RandomForest":
        model = RandomForest(
            lags=[-1, -2, -3, -4, -5, -6, -7],
            lags_past_covariates=[-1, -2, -3, -4, -5, -6, -7],
        )
    elif model_name == "N-BEATS":
        model = NBEATSModel(input_chunk_length=14, output_chunk_length=14,)
    elif model_name == "N-HiTS":
        model = NHiTSModel(input_chunk_length=14, output_chunk_length=14,)
    elif model_name == "TCN":
        model = TCNModel(input_chunk_length=14, output_chunk_length=14,)
    elif model_name == "Transformer":
        model = TransformerModel(input_chunk_length=14, output_chunk_length=14,)

    model.fit(series=y_train, past_covariates=past_cov)
    y_pred = model.predict(n=len(y_val), series=y_train, past_covariates=past_cov)

    mae = mean_absolute_error(y_val.values().reshape(-1), y_pred.values().reshape(-1))
    mse = mean_squared_error(y_val.values().reshape(-1), y_pred.values().reshape(-1))
    mlflow.log_metrics({"mae": mae, "mse": mse})
    mlflow.log_params(
        {
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "model_name": model_name,
        }
    )

    return (y, y_pred)


@task
def plot(y, y_pred, model_name):
    plt.figure()
    y_pred.plot(label="Deaths predictions")
    y[-30:].plot(label="Real deaths")
    plt.legend()
    plt.title(f"Deaths prediction vs groundtruth - {model_name} model")
    plt.savefig("fig/pred_true_plot.png")
    plt.close()
    mlflow.log_artifact(
        local_path="fig/pred_true_plot.png", artifact_path="pred_true_plot.png"
    )


@flow
def main(
    model_name,
    input_chunk_length=14,
    output_chunk_length=14,
    covid_indicator_path=None,
    covid_test_path=None,
):
    with mlflow.start_run():
        global features_indicator
        global features_test
        global features
        global target

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
            "TO",
            "R",
        ]
        features_test = ["pop", "P", "T", "Ti", "Tp", "Td"]
        features = features_indicator + features_test
        target = ["incid_dchosp"]

        covid_indicator_df, covid_test_df = read_data(
            covid_indicator_path, covid_test_path
        )

        covid_df = preprocess_data(covid_indicator_df, covid_test_df)

        y, y_pred = train_model(
            covid_df, model_name, input_chunk_length, output_chunk_length
        )
        plot(y, y_pred, model_name)


if __name__ == "__main__":
    # main(model_name='RegressionModel')
    # main(model_name='RandomForest')
    main(model_name="N-BEATS")
    # main(model_name='N-HiTS')
    # main(model_name='TCN')
    # main(model_name='TransformerModel')
