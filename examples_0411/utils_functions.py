from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

class MLFunctions:

    @classmethod
    def plot_models_comparison(cls, model, X_val, y_val, axis1, axis2, axis3):
        ConfusionMatrixDisplay.from_estimator(
            model,
            X=X_val,
            y=y_val,
            ax=axis1,
            cmap="Blues",
            normalize="true"
        )

        PrecisionRecallDisplay.from_estimator(
            model,
            X=X_val,
            y=y_val,
            ax=axis2
        )

        RocCurveDisplay.from_estimator(
            model,
            X=X_val,
            y=y_val,
            ax=axis3
        )
    
    @classmethod
    def get_metrics(cls, X_val, y_val, model, model_name):
        predictions = model.predict(X_val)
        predict_proba = model.predict_proba(X_val)
        list_predict_proba = [value[1] for value in predict_proba]

        performances = {
            "model_name" : model_name,
            "accuracy" : accuracy_score(y_val, predictions), 
            "f1_score" : f1_score(y_val, predictions), 
            "roc_auc" : roc_auc_score(y_val, list_predict_proba), 
            "precision": precision_score(y_val, predictions), 
            "recall" : recall_score(y_val, predictions)
        }

        return performances

    @classmethod
    def train_model(cls, model_instance, X_train, y_train):
        return model_instance.fit(X_train, y_train)