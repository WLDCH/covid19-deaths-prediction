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

# TRACKING_URI = "sqlite:///mlflow.db"
TRACKING_SERVER_HOST = "34.77.171.28"
TRACKING_URI = f"http://{TRACKING_SERVER_HOST}:5000"
EXPERIMENT_NAME = "covid-deaths-prediction"
