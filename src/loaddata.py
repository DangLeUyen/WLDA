import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_iris


def load_userData():
    originaldata = fetch_ucirepo(id=257)

        # data (as pandas dataframes)
    X = originaldata.data.features
    y = originaldata.data.targets

    y = np.ravel(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X = X.astype(float).to_numpy()
    print("Categorical",  X.shape)
    print("Class", y.shape)
    for i in np.unique(y):
        print(sum(y==i))
    return X, y

def load_thyroidData():
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data',
                  sep = ",", header = None)

    data = data.to_numpy()
    X, y = data[:, [x for x in range(data.shape[1]) if x != 0]].astype(np.float32),data[:,0]
    G = len(np.unique(y))
    le2 = LabelEncoder()
    y = le2.fit_transform(y)
    for g in range(G):
        print(sum(y==g))
    print(X.shape)

    return X, y

def load_irisData():
    # Load the Iris dataset (n_features: 4, n_samples: 150 )
    iris = load_iris()
    X, y = iris.data, iris.target
    column_names = iris.feature_names

    return X, y, column_names
    