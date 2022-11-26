
from sklearn.metrics import accuracy_score, mean_squared_error as mse
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


def load_ds(_load):
    data = _load()
    df = pd.DataFrame(np.column_stack((data["data"], data["target"])), columns=[
        *data['feature_names'], "target"])
    df.target = df.target.astype(int).astype("category")
    return df, df.iloc[:, :4], df.target


def load_ds_split(_load=load_iris, **kwargs):
    from sklearn.model_selection import train_test_split
    df, X, y = load_ds(_load)
    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwargs)
    return df,  X_train, X_test, y_train, y_test


def compare(estimator1, estimator2, X, y, metric, name):
    s1, s2 = metric(y, estimator1.predict(X)), metric(y, estimator2.predict(X))
    print(f"Score of {name} 1 is: {s1}")
    print(f"Score of {name} 2 is: {s2}")
    return s1, s2


def compare_classifiers(cls1, cls2, X, y, metric=accuracy_score):
    return compare(cls1, cls2, X, y, metric, "classifier")


def compare_regressors(rgr1, rgr2, X, y, metric=mse):
    return compare(rgr1, rgr2, X, y, metric, "regressor")
