

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
            X[feature] = X[feature].map(self.encoder_dict_[feature])
            # drop original
            X = X.drop(columns=[feature])
        return X

