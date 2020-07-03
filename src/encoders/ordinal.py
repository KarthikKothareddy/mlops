

from sklearn.base import TransformerMixin
import numpy as np


class OrdinalEncoder(TransformerMixin):

    def __init__(self, features=None):
        self.features = features
        # derived properties
        self.encoder_dict_ = {}

    def fit(self, X, y=None):
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype == "O"
            ]
        for feature in self.features:
            # update the mapping dictionary
            self.encoder_dict_[feature] = {
                k: v
                for v, k in enumerate(X[feature].unique(), 0)
            }
        return self

    def transform(self, X):
        for feature in self.features:
            # apply mapping
            X.loc[:, feature] = X[feature].map(self.encoder_dict_[feature])
        return X


class TargetGuidedEncoder(OrdinalEncoder):

    def __init__(self, features=None):
        super().__init__(features)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("target variable is missing")
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype == "O"
            ]
        for feature in self.features:
            __labels = X.groupby([feature])[y.name].mean().sort_values().index
            # update mapping by target guidance
            self.encoder_dict_[feature] = {
                k: v
                for v, k in enumerate(__labels, 0)
            }
        return self


class MeanEncoder(OrdinalEncoder):

    def __init__(self, features=None):
        super().__init__(features)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("target variable is missing")
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype == "O"
            ]
        for feature in self.features:
            __labels = X.groupby([feature])[y.name].mean().to_dict()
            # update mapping by target guidance
            self.encoder_dict_[feature] = __labels
        return self

