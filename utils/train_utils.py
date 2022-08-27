import matplotlib.pyplot as plt
import mlflow
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from prefect import task
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.variables import (
    features,
    features_indicator,
    features_test,
    length_pred,
    target,
)


@task
def create_preprocess_time_series_train_val(covid_df, features=features, target=target):
    y = TimeSeries.from_series(covid_df[target])
    past_cov = TimeSeries.from_dataframe(covid_df[features])

    target_scaler = Scaler()
    past_cov_scaler = Scaler()
    y_scaled = target_scaler.fit_transform(y)
    past_cov_scaled = past_cov_scaler.fit_transform(past_cov)

    y_train = y_scaled[:-length_pred]
    y_val = y_scaled[-length_pred:]
    past_cov_train = past_cov_scaled[:-length_pred]
    past_cov_val = past_cov_scaled

    # target_scaler = Scaler()
    # past_cov_scaler = Scaler()
    # y_train_scaled = target_scaler.fit_transform(y_train)
    # y_val_scaled = target_scaler.transform(y_val)
    # past_cov_train_scaled = past_cov_scaler.fit_transform(past_cov_scaled_train)
    # past_cov_val_scaled = past_cov_scaler.transform(past_cov_val)

    return (
        y_scaled,
        y_train,
        y_val,
        past_cov_train,
        past_cov_val,
        target_scaler,
        past_cov_scaler,
    )


@task
def compute_metrics(y_pred, y_val, model_name, input_chunk_length, output_chunk_length):

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


@task
def plot(y, y_pred, model_name):
    plt.figure()
    y_pred.plot(label="Deaths predictions")
    y[-30:].plot(label="Real deaths")
    plt.legend()
    plt.title(f"Deaths prediction vs groundtruth - {model_name} model")
    plt.savefig(f"fig/{model_name}_pred_true_plot.png")
    plt.close()
    mlflow.log_artifact(
        local_path=f"fig/{model_name}_pred_true_plot.png",
        artifact_path=f"{model_name}_pred_true_plot.png",
    )
