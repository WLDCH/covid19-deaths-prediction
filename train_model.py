import datetime

import matplotlib as mpl

mpl.use("Agg")


import pickle

import mlflow
from darts.models import (BlockRNNModel,
                          NBEATSModel, NHiTSModel, RandomForest,
                          RegressionModel, TCNModel,
                          TransformerModel)
from prefect import flow, task
from sklearn.ensemble import RandomForestRegressor

from utils.read_preprocess import read_data, preprocess_data
from utils.train_utils import create_preprocess_time_series_train_val, compute_metrics, plot
from utils.variables import EXPERIMENT_NAME, TRACKING_URI, length_pred, features_indicator, features_test, features, target

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

@task
def train_model(y, y_train, y_val, past_cov_train, past_cov_val, model_name):

    if model_name == "Regression":
        model = RegressionModel(
            lags=[-1, -2, -3, -4, -5, -6, -7],
            lags_past_covariates=[-1],  # , -2, -3, -4, -5, -6, -7],
            model=RandomForestRegressor(),
        )
    elif model_name == "RandomForest":
        model = RandomForest(
            lags=[-1, -2, -3, -4, -5, -6, -7],
            lags_past_covariates=[-1, -2, -3, -4, -5, -6, -7],
        )
    elif model_name == "N-BEATS":
        model = NBEATSModel(
            input_chunk_length=length_pred,
            output_chunk_length=length_pred,
        )
    elif model_name == "N-HiTS":
        model = NHiTSModel(
            input_chunk_length=length_pred,
            output_chunk_length=length_pred,
        )
    elif model_name == "TCN":
        model = TCNModel(
            input_chunk_length=length_pred + 1,
            output_chunk_length=length_pred,
        )
    elif model_name == "Transformer":
        model = TransformerModel(
            input_chunk_length=length_pred,
            output_chunk_length=length_pred,
        )
    elif model_name == "RNN":
        model = BlockRNNModel(
            input_chunk_length=length_pred, output_chunk_length=length_pred, model="RNN"
        )
    elif model_name == "LSTM":
        model = BlockRNNModel(
            input_chunk_length=length_pred,
            output_chunk_length=length_pred,
            model="LSTM",
        )

    model.fit(series=y_train, past_covariates=past_cov_train)
    y_pred = model.predict(
        n=length_pred, series=y_train, past_covariates=past_cov_train
    )

    return (model, y_pred)




@flow
def main(
    model_name,
    input_chunk_length=length_pred,
    output_chunk_length=length_pred,
    covid_indicator_path=None,
    covid_test_path=None,
):
    with mlflow.start_run(run_name=f"{model_name}_run"):
        # global features_indicator
        # global features_test
        # global features
        # global target

        covid_indicator_df, covid_test_df = read_data(
            covid_indicator_path, covid_test_path
        )

        covid_df = preprocess_data(covid_indicator_df, covid_test_df)

        (
            y,
            y_train,
            y_val,
            past_cov_train,
            past_cov_val,
            target_scaler,
            past_cov_scaler,
        ) = create_preprocess_time_series_train_val(covid_df=covid_df)

        model, y_pred = train_model(
            y, y_train, y_val, past_cov_train, past_cov_val, model_name
        )
        y, y_pred, y_val = (
            target_scaler.inverse_transform(y),
            target_scaler.inverse_transform(y_pred),
            target_scaler.inverse_transform(y_val),
        )

        compute_metrics(
            y_pred=y_pred,
            y_val=y_val,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            model_name=model_name,
        )
        plot(y, y_pred, model_name)
        model.save(f"models/{model_name}Model")
        with open(f"models/scalers", "wb") as f_out:
            pickle.dump((target_scaler, past_cov_scaler), f_out)


if __name__ == "__main__":
    # main(model_name="Regression")
    # main(model_name='RandomForest')
    # main(model_name="N-BEATS")
    # main(model_name="N-HiTS")
    # main(model_name="TCN")
    main(model_name="Transformer")
    # main(model_name='RNN')
    # main(model_name='LSTM')
