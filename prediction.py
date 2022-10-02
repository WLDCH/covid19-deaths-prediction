import os

import mlflow
from google.cloud import storage
from prefect import flow, task

from utils.predict_utils import create_preprocess_time_series, predict
from utils.read_preprocess import preprocess_data, read_data
from utils.variables import EXPERIMENT_NAME, TRACKING_URI, BUCKET_NAME, LOCALLY, model_name_to_model_class

print(LOCALLY)
if not LOCALLY:
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.client.MlflowClient()
# else:
#     mlflow.set_tracking_uri("sqlite:///mlflowlocal.db")
#     mlflow.set_experiment('local')
#     client = mlflow.client.MlflowClient()


@task
def load_model(model_name):
    
    # If we run it locally, we don't fetch models from GCP, we use existing ones located in "block_checkpoints"
    if not LOCALLY:
        run_id = dict(client.search_model_versions(f"name='{model_name}Model'")[0])[
            "run_id"
        ]
        blob_model = bucket.blob(f"mlruns/1/{run_id}/artifacts/{model_name}/_model.pth.tar")
        blob_checkpoints = bucket.blob(
            f"mlruns/1/{run_id}/artifacts/{model_name}/checkpoints/last-epoch=99.ckpt"
        )

        if not os.path.exists("blob_checkpoints"):
            os.mkdir("blob_checkpoints")
        if not os.path.exists(f"blob_checkpoints/{model_name}"):
            os.mkdir(f"blob_checkpoints/{model_name}")
        if not os.path.exists(f"blob_checkpoints/{model_name}/checkpoints/"):
            os.mkdir(f"blob_checkpoints/{model_name}/checkpoints/")

        blob_model.download_to_filename(f"blob_checkpoints/{model_name}/_model.pth.tar")
        blob_checkpoints.download_to_filename(
            f"blob_checkpoints/{model_name}/checkpoints/last-epoch=99.ckpt"
        )

    model = model_name_to_model_class[model_name].load_from_checkpoint(
        model_name=model_name, work_dir="blob_checkpoints", best=False
    )

    return model


@task
def load_best_model():

    if not LOCALLY:
        best_model = None
        best_mae = float("inf")
        for model_name in model_name_to_model_class.keys():
            run_id = dict(client.search_model_versions(f"name='{model_name}Model'")[0])[
                "run_id"
            ]
            mae = dict(dict(mlflow.get_run(run_id))["data"])["metrics"]["mae"]
            if mae < best_mae:
                best_mae, best_model = mae, model_name
    else:
        best_model = 'Transformer'

    return load_model.fn(best_model)


@flow(name="predict_flow")
def main(covid_indicator_path=None, covid_test_path=None):
    covid_indicator_df, covid_test_df = read_data(covid_indicator_path, covid_test_path)
    covid_df = preprocess_data(covid_indicator_df, covid_test_df)

    y, past_cov, target_scaler, past_cov_scaler = create_preprocess_time_series(
        covid_df
    )
    model = load_best_model()
    y_pred = predict(model, y, past_cov, target_scaler, past_cov_scaler)
    y = target_scaler.inverse_transform(y)
    return y, y_pred


if __name__ == "__main__":
    main()
