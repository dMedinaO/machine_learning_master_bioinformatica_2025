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

from utils_functions import train_model

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

print("Training process")

logistic_model, logistic_model_performances = train_model(
    logistic_model, 
    "LogisticRegression",
    X_train, y_train, X_val, y_val, 
    X_test=X_test, y_test=y_test
)

print(logistic_model_performances)