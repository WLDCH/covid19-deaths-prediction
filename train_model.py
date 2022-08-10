from prefect import task, flow
from prefect.task_runners import SequentialTaskRunner

import mlflow

import pandas as pd
import datetime

import requests
from bs4 import BeautifulSoup


from selenium import webdriver
from selenium.webdriver.chrome.options import Options


# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_experiment("covid-deaths-prediction")


# @task
# def read_data(covid_indicator_path=None, covid_test_path=None):
#     if covid_indicator_path is None:
#         today = datetime.datetime.today()
#         covid_indicator_long_path = f"https://static.data.gouv.fr/resources/synthese-des-indicateurs-de-suivi-de-lepidemie-covid-19/20220809-190002/table-indicateurs-open-data-france-{today.year}-{today.month:02}-{today.day:02}-19h00.csv"
#         covid_indicator = pd.read_csv(covid_indicator_long_path)

#     if covid_test_path is None:
#         today = datetime.datetime.today()
        

def scrap_data():
    covid_indicator_page = requests.get('https://www.data.gouv.fr/fr/datasets/synthese-des-indicateurs-de-suivi-de-lepidemie-covid-19/')
    soup_indicator = BeautifulSoup(covid_indicator_page.content, 'html.parser')
    covid_indicator_path = soup_indicator.find_all('dd', {"class": "fr-ml-0 fr-col-8 fr-col-md-9 fr-col-lg-10 text-overflow-ellipsis"})[-1].find('a', href=True)['href']
    covid_indicator = pd.read_csv(covid_indicator_path)

    options = Options()
    options.headless = True
    driver = webdriver.Chrome("/usr/bin/chromedriver", options=options)
    driver.implicitly_wait(0.5)
    driver.get("https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/")
    bt = driver.find_element(by='xpath', value="//*[@id='app']/main/section[4]/div/div/section[1]/div/nav/ul/li[4]/a")
    bt.click()
    pa = driver.find_element(by='xpath', value="//*[@id='resource-4e8d826a-d2a1-4d69-9ed0-b18a1f3d5ce2']/dl/div[2]/dd/a")
    covid_test_path = pa.get_attribute('href')
    covid_test = pd.read_csv(covid_test_path, sep=';')
    driver.quit()

    print(covid_indicator)
    print(covid_test)


scrap_data()



