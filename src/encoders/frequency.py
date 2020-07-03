

from sklearn.base import TransformerMixin
import numpy as np


class CountEncoder(TransformerMixin):

    def __init__(self, features=None):
        self.features = features
        # derived properties
        self.encoder_dict_ = {}

    def fit(self, X, y=None):
        # by default choose all categorical
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype == "O"
            ]
        for feature in self.features:
            # update the mapping dictionary
            self.encoder_dict_[feature] = X[feature]\
                .value_counts().to_dict()
        return self

    def transform(self, X):
        for feature in self.features:
            # apply mapping
            X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X


class FrequencyEncoder(CountEncoder):

    def __init__(self, features=None):
        super().__init__(features)

    def fit(self, X, y=None):
        # by default choose all categorical
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype == "O"
            ]
        for feature in self.features:
            # update the mapping dictionary with frequency
            self.encoder_dict_[feature] = (
                    X[feature].value_counts() / len(X)
            ).to_dict()
        return self
