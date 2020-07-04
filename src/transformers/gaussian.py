

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.base import TransformerMixin


class LogarithmicTransformer(TransformerMixin):

    def __init__(self, features=None):
        self.features = features
        # defaults to numerical
        self.numerical = ["int", "float"]

    def fit(self, X, y=None):
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype in self.numerical
            ]
        return self

    def transform(self, X):
        for feature in self.features:
            X.loc[:, feature] = np.log(X[feature])
        return X


class ReciprocalTransformer(TransformerMixin):

    def __init__(self, features=None):
        self.features = features
        # defaults to numerical
        self.numerical = ["int", "float"]

    def fit(self, X, y=None):
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype in self.numerical
            ]
        return self

    def transform(self, X):
        for feature in self.features:
            X.loc[:, feature] = 1 / (X[feature])
        return X


class ExponentialTransformer(TransformerMixin):

    def __init__(self, features=None, exponent=1):
        self.features = features
        self.exponent = exponent
        # defaults to numerical
        self.numerical = ["int", "float"]

    def fit(self, X, y=None):
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype in self.numerical
            ]
        return self

    def transform(self, X):
        for feature in self.features:
            X.loc[:, feature] = np.power(X[feature], self.exponent)
        return X


class SquareRootTransformer(ExponentialTransformer):

    def __init__(self, features=None):
        super().__init__(features)
        self.exponent = 0.5


class BoxCoxTransformer(TransformerMixin):

    def __init__(self, features=None):
        self.features = features
        # defaults to numerical
        self.numerical = ["int", "float"]
        # derived
        self.maxlog_dict_ = {}

    def fit(self, X, y=None):
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype in self.numerical
            ]
        return self

    def transform(self, X):
        for feature in self.features:
            X.loc[:, feature], self.maxlog_dict_[feature] = stats.boxcox(X[feature])
        return X


class YeoJohnsonTransformer(TransformerMixin):

    def __init__(self, features=None):
        self.features = features
        # defaults to numerical
        self.numerical = ["int", "float"]
        # derived
        self.maxlog_dict_ = {}

    def fit(self, X, y=None):
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype in self.numerical
            ]
        return self

    def transform(self, X):
        for feature in self.features:
            X.loc[:, feature], self.maxlog_dict_[feature] = stats.yeojohnson(X[feature])
        return X
