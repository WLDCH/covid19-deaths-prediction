import os

from darts.models import (
    BlockRNNModel,
    NBEATSModel,
    NHiTSModel,
    TCNModel,
    TransformerModel,
)

length_pred = 14

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
]

features_test = ["pop", "P", "T", "Ti", "Tp", "Td"]

features = features_indicator + features_test

target = ["incid_dchosp"]


model_name_to_model_class = {
    # 'N-BEATS': NBEATSModel,
    # 'N-HiTS': NHiTSModel,
    "TCN": TCNModel,
    "Transformer": TransformerModel,
    "RNN": BlockRNNModel,
    # 'LSTM': BlockRNNModel
}

# TRACKING_URI = "sqlite:///mlflow.db"
TRACKING_SERVER_HOST = os.environ.get("TRACKING_SERVER_HOST", "35.210.155.194")
TRACKING_URI = f"http://{TRACKING_SERVER_HOST}:5000"
EXPERIMENT_NAME = "covid-deaths-prediction"
BUCKET_NAME = os.environ.get("BUCKET_NAME", "covid19-deaths-prediction-bucket")
LOCALLY = os.environ.get("LOCALLY", False)
