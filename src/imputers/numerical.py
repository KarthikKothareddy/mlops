

from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd


class MeanImputer(TransformerMixin):

    def __init__(self, features=None):
        self.features = features
        # init imputer dict
        self.imputer_dict_ = {}
        # default features to select
        self.numerical = ["int", "float"]

    def fit(self, X, y=None):
        # default to all numerical features
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype in self.numerical
            ]
        for feature in self.features:
            self.imputer_dict_[feature] = X[feature].mean()
        return self

    def transform(self, X):
        for feature in self.features:
            X.loc[:, feature] = np.where(
                X[feature].isnull(),
                self.imputer_dict_[feature],
                X[feature]
            )
        return X


class MedianImputer(MeanImputer):

    def __init__(self, features=None):
        super().__init__(features)

    def fit(self, X, y=None):
        # default to all numerical features
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype in self.numerical
            ]
        for feature in self.features:
            self.imputer_dict_[feature] = X[feature].median()
        return self


class ArbitraryValueImputer(TransformerMixin):

    def __init__(self, features=None, arbitrary_value=9999):
        self.features = features
        self.arbitrary_value = arbitrary_value
        # default features to select
        self.numerical = ["int", "float"]

    def fit(self, X, y=None):
        # default to all numerical features
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype in self.numerical
            ]
        return self

    def transform(self, X):
        for feature in self.features:
            X.loc[:, feature] = np.where(
                X[feature].isnull(),
                self.arbitrary_value,
                X[feature]
            )
        return X


