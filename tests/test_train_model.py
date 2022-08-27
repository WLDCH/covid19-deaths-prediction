import numpy as np
import pandas as pd

from utils.read_preprocess import (
    preprocess_data,
    scrap_covid_indicator,
    scrap_covid_test,
)
from utils.train_utils import create_preprocess_time_series_train_val


def test_scrap_covid_indicator():
    expected_df = pd.read_csv(
        "data/table-indicateurs-open-data-france-2022-08-26-19h01.csv",
        index_col="Unnamed: 0",
    )

    actual_df = scrap_covid_indicator.fn()

    expected_df.dropna(inplace=True)
    actual_df.dropna(inplace=True)

    assert expected_df.equals(actual_df.iloc[: expected_df.shape[0], :])


def test_scrap_covid_test():
    expected_df = pd.read_csv(
        "data/sp-fra-jour-2022-08-26-19h01.csv", sep=",", index_col="Unnamed: 0"
    )
    actual_df = scrap_covid_test.fn()

    expected_df.dropna(inplace=True)
    actual_df.dropna(inplace=True)

    # covid tests value are updated so we can't test dataframe equality
    assert np.all(expected_df.columns == actual_df.columns)


def test_preprocess_data():

    expected_df = pd.read_csv("data/covid_df.csv", index_col="date")
    covid_indicator = pd.read_csv(
        "data/table-indicateurs-open-data-france-2022-08-26-19h01.csv"
    )
    covid_test = pd.read_csv("data/sp-fra-jour-2022-08-26-19h01.csv", sep=",")
    actual_df = preprocess_data.fn(
        covid_indicator_df=covid_indicator, covid_test_df=covid_test
    )

    # same as previous test
    assert np.all(expected_df.columns == actual_df.columns)


def test_create_preprocess_time_series_train_val():

    covid_df = pd.read_csv("data/covid_df.csv")
    (
        y_scaled,
        y_train,
        y_val,
        past_cov_train,
        past_cov_val,
        target_scaler,
        past_cov_scaler,
    ) = create_preprocess_time_series_train_val.fn(covid_df)

    assert y_scaled == y_train.concatenate(y_val)
    assert np.all(y_scaled <= 1) & np.all(y_scaled >= 0)
    assert np.all(y_train <= 1) & np.all(y_train >= 0)
    assert np.all(y_val <= 1) & np.all(y_val >= 0)
