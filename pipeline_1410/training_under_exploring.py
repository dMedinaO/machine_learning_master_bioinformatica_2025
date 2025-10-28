import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from json import load

from utils_functions import train_model, generate_row

print("Processing input data")
path_input = sys.argv[1]
column_label = sys.argv[2]

print("Reading config hyperparameters")
with open(sys.argv[3], "r") as doc_config:
    config_hyperparams = load(doc_config)

print("Reading datasets")
df_train = pd.read_csv(f"{path_input}X_train_dataset.csv")
df_val = pd.read_csv(f"{path_input}X_val_dataset.csv")
df_test = pd.read_csv(f"{path_input}X_test_dataset.csv")

print("Preparing data for training")
y_train = df_train[column_label].values
X_train = df_train.drop(columns=[column_label]).values

y_val = df_val[column_label].values
X_val = df_val.drop(columns=[column_label]).values

y_test = df_test[column_label].values
X_test = df_test.drop(columns=[column_label]).values

print("Preparing algorithms")

logistic_model = LogisticRegression(**config_hyperparams["logistic_regression"])
knn_model = KNeighborsClassifier(**config_hyperparams["knn"])
svc_model = SVC(random_state=42)
gnb_model = GaussianNB()
rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)

print("Training process")

performance_metrics_summary = []

logistic_model, logistic_model_performances = train_model(
    logistic_model, 
    "LogisticRegression",
    X_train, y_train, X_val, y_val, 
    X_test=X_test, y_test=y_test
)

gnb_model, gnb_model_performances = train_model(
    gnb_model, 
    "GaussianNB",
    X_train, y_train, X_val, y_val, 
    X_test=X_test, y_test=y_test
)

svc_model, svc_model_performances = train_model(
    svc_model, 
    "SVC",
    X_train, y_train, X_val, y_val, 
    X_test=X_test, y_test=y_test
)

rf_model, rf_model_performances = train_model(
    rf_model, 
    "RandomForestClassifier",
    X_train, y_train, X_val, y_val, 
    X_test=X_test, y_test=y_test
)

dt_model, dt_model_performances = train_model(
    dt_model, 
    "DecisionTreeClassifier",
    X_train, y_train, X_val, y_val, 
    X_test=X_test, y_test=y_test
)

knn_model, knn_model_performances = train_model(
    knn_model, 
    "KNeighborsClassifier",
    X_train, y_train, X_val, y_val, 
    X_test=X_test, y_test=y_test
)
# adding performances to matrix
performance_metrics_summary.append(generate_row(logistic_model_performances))
performance_metrics_summary.append(generate_row(knn_model_performances))
performance_metrics_summary.append(generate_row(svc_model_performances))
performance_metrics_summary.append(generate_row(rf_model_performances))
performance_metrics_summary.append(generate_row(dt_model_performances))
performance_metrics_summary.append(generate_row(gnb_model_performances))

print("Exporting metrics from exploration")
df_summary = pd.DataFrame(performance_metrics_summary)
df_summary.to_csv(f"{path_input}summary_metrics_exploration.csv", index=False)