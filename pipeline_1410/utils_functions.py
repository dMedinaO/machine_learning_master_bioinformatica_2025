from sklearn.preprocessing import (StandardScaler, MinMaxScaler, 
                                   MaxAbsScaler, RobustScaler)
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef, precision_score

def apply_standar_scaler(data):

    scaler = StandardScaler()
    scaler.fit(data)

    transform_data = scaler.transform(data)

    return transform_data, scaler

def apply_min_max_scaler(data):
    
    scaler = MinMaxScaler()
    scaler.fit(data)

    transform_data = scaler.transform(data)

    return transform_data, scaler

def apply_max_absolute_scaler(data):
    
    scaler = MaxAbsScaler()
    scaler.fit(data)

    transform_data = scaler.transform(data)

    return transform_data, scaler

def apply_robust_scaler(data):
    
    scaler = RobustScaler()
    scaler.fit(data)

    transform_data = scaler.transform(data)

    return transform_data, scaler

def create_and_export_dataset(X_values, response, columns, column_label, name_export):

    df = pd.DataFrame(data=X_values, columns=columns)
    df[column_label] = response

    df.to_csv(name_export, index=False)

def get_metrics(y_true, y_pred):

    return {
        "accuracy_score" : accuracy_score(y_true, y_pred), 
        "precision_score" :precision_score (y_true, y_pred, average="weighted"),
        "recall_score" : recall_score(y_true, y_pred, average="weighted"),
        "f1_score" : f1_score(y_true, y_pred, average="weighted"), 
        "matthews_corrcoef" : matthews_corrcoef(y_true, y_pred),
     }
    
def train_model(model, model_name, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    model.fit(X_train, y_train)

    validation_metrics = get_metrics(y_val, model.predict(X_val))

    dict_process = {
        "model" : model_name,
        "validation_metrics" : validation_metrics
    }

    if X_test is not None:
        testing_metrics = get_metrics(y_test, model.predict(X_test))

        dict_process.update({"testing_metrics" : testing_metrics})
    
    return model, dict_process
