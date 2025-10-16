import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import sys
from json import load
from joblib import dump

from utils_functions import *

from sklearn.model_selection import train_test_split

print("Reading config file")
with open(sys.argv[1], "r") as doc_config:
    config_file = load(doc_config)

print("Get dataset")
df_data = pd.read_csv(f"{config_file["path"]}{config_file["name_dataset"]}")

print("Get response")
response = df_data[config_file["column_label"]].values

print("Remove columns")
df_values = df_data.drop(columns=config_file["columns_to_remove"]).values

print("Divide dataset")
X_train, X_test, y_train, y_test = train_test_split(
    df_values, 
    response,
    random_state=config_file["random_seed"],
    test_size=config_file["test_size"])

X_train, X_val, y_train, y_val = train_test_split(
    X_train, 
    y_train,
    random_state=config_file["random_seed"],
    test_size=config_file["val_size"])

print("Dimensions: ", X_train.shape, X_val.shape, X_test.shape)

print("Standardize dataset")
if config_file["scaler_method"] == "standar_scaler":
    X_train, scaler = apply_standar_scaler(X_train)

elif config_file["scaler_method"] == "min_max":
    X_train, scaler = apply_min_max_scaler(X_train)

elif config_file["scaler_method"] == "max_abs":
    X_train, scaler = apply_max_absolute_scaler(X_train)
else:
    X_train, scaler = apply_robust_scaler(X_train)

print("Scaling X_test and X_val")
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("Exporting data")
columns_to_use = [column for column in df_data.columns if column not in config_file["columns_to_remove"]]

create_and_export_dataset(
    X_train, 
    y_train, 
    columns_to_use, 
    config_file["column_label"],
    f"{config_file["path"]}X_train_dataset.csv")

create_and_export_dataset(
    X_val, 
    y_val, 
    columns_to_use, 
    config_file["column_label"],
    f"{config_file["path"]}X_val_dataset.csv")

create_and_export_dataset(
    X_test, 
    y_test, 
    columns_to_use, 
    config_file["column_label"],
    f"{config_file["path"]}X_test_dataset.csv")

print("Exporting scaler")
name_export_scaler = f"{config_file["path"]}scaler_instance.joblib"
dump(scaler, name_export_scaler)