
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd


class MissingIndicatorImputer(TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        # default to all features
        if self.features is None:
            self.features = [c for c in X.columns]
        return self

    def transform(self, X):
        for feature in self.features:
            X[f"{feature}_NA"] = np.where(
                X[feature].isnull(), 1, 0
            )
        return X


