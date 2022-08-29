# Covid19 deaths prediction for MLOps zoomcamp's project

This is my final project for both the MLOps Zoomcamp from Data Talks Club. My course work are available in [this other repo](https://github.com/WLDCH/mlops-zoomcamp).

# Project framing

Covid19 has struck every country in the world since late 2019/early 2020. Over 6M person have died since then, and everyday, new people die from covid19.

The goal of this project is to predict the daily number of deaths due to covid19 in the next two weeks in France. The project utilizes MLOps to create a system that predicts the number of deaths due to covid19 and is updated every day. A new model is retrained with new data every Monday at 20:00.

## Datasets

Two datasets are used in this project. They are scrapped directly from www.data.gouv.fr/. They are updated every day (but Saturday & Sunday) at 19:00, so each day after 19:00 we can update streamlit dashboard to have new predictions.

* French covid19 indicators synthesis : https://www.data.gouv.fr/fr/datasets/synthese-des-indicateurs-de-suivi-de-lepidemie-covid-19/
* French covid19 screening : https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/

## MLOps

The project uses tools and concepts from MLOps. Once a week, every Monday at 20:00, three differents models of Deep Learning using Darts library are trained (Transformer, RNN & TCN). We use experiment & model registery with MLFlow with a remote tracking server on GCP
![MLOps_diagram](fig/MLOps_diagram.png)

## Tools

Cloud - Google Cloud Platform
Containerization - Docker and Docker Compose
Workflow Orchestration - Prefect
Data Visualization/Dashboard - Streamlit and Plotly 
Model Development, Experiment Tracking, and Registration - Darts and MLflow
Model Deployment - Darts and MLflow and Streamlit

# Dashboard

You can acces the dashboard [here](http://35.210.155.194:8501/). It contains a graphic that shows current covid19 deaths and predictions by the model.

# Steps to reproduce
