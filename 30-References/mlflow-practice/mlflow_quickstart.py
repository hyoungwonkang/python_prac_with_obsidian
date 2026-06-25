
import mlflow

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

mlflow.sklearn.autolog()

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

with mlflow.start_run():
    mlflow.log_params(params)

    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    model_info = mlflow.sklearn.log_model(sk_model=lr, name="iris_model")

    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.set_tag("Training Info", "Basic LR model for iris data")

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:4])