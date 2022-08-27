import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from prefect import flow, task
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from utils.variables import (
    EXPERIMENT_NAME,
    TRACKING_URI,
    features,
    features_indicator,
    features_test,
    length_pred,
    target,
)


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
def preprocess_data(
    covid_indicator_df, covid_test_df, features=features, target=target
):

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
