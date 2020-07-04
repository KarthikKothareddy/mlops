
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd


class ModeImputer(TransformerMixin):

    def __init__(self, features=None):
        self.features = features
        # init imputer dict
        self.imputer_dict_ = {}

    def fit(self, X, y=None):
        # default to all numerical features
        if self.features is None:
            self.features = [
                c for c in X.columns
                # select only categorical
                if X[c].dtype == "O"
            ]
        for feature in self.features:
            self.imputer_dict_[feature] = X[feature].mode().iloc[0]
        return self

    def transform(self, X):
        for feature in self.features:
            X[feature] = np.where(
                X[feature].isnull(),
                self.imputer_dict_[feature],
                X[feature]
            )
        return X
