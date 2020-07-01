import pandas as pd

import sklearn.model_selection as ms


def load_dataset(path):
    return pd.read_csv(path, sep=',')


def train_test_split(features, labels):
    return ms.train_test_split(features, labels, test_size=0.01, random_state=42, shuffle=True)
