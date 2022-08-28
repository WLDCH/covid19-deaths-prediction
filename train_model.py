import matplotlib as mpl

mpl.use("Agg")
import mlflow
from darts.models import BlockRNNModel, TCNModel, TransformerModel
from google.cloud import storage
from prefect import flow, task

from utils.read_preprocess import preprocess_data, read_data
from utils.train_utils import (
    compute_metrics,
    create_preprocess_time_series_train_val,
    plot,
)
from utils.variables import (
    EXPERIMENT_NAME,
    TRACKING_URI,
    features,
    features_indicator,
    features_test,
    length_pred,
    target,
)

storage_client = storage.Client()
bucket = storage_client.get_bucket("covid19-deaths-prediction-bucket")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


@task
def train_model(y, y_train, y_val, past_cov_train, past_cov_val, model_name):

    if model_name == "TCN":
        model = TCNModel(
            input_chunk_length=length_pred + 1,
            output_chunk_length=length_pred,
            save_checkpoints=True,
            force_reset=True,
            work_dir="checkpoints/",
            model_name=model_name,
        )
    elif model_name == "Transformer":
        model = TransformerModel(
            input_chunk_length=length_pred,
            output_chunk_length=length_pred,
            save_checkpoints=True,
            force_reset=True,
            work_dir="checkpoints/",
            model_name=model_name,
        )
    elif model_name == "RNN":
        model = BlockRNNModel(
            input_chunk_length=length_pred,
            output_chunk_length=length_pred,
            model="RNN",
            save_checkpoints=True,
            force_reset=True,
            work_dir="checkpoints/",
            model_name=model_name,
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
    with mlflow.start_run(run_name=f"{model_name}_run") as run:
        run_id = run.info.run_id
        print(run_id)
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
        # plot(y, y_pred, model_name)
        mlflow.log_artifacts("checkpoints/")
        model.save(f"models/MLmodel")
        mlflow.log_artifact(f"models/MLmodel", "model")
        mlflow.register_model(f"runs:/{run_id}/model", f"{model_name}Model")


if __name__ == "__main__":
    main(model_name="TCN")
    main(model_name="Transformer")
    main(model_name="RNN")
