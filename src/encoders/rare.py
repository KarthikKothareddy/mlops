

from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np


class RareLabelEncoder(TransformerMixin):

    def __init__(
            self, features=None, threshold=0.05,
            min_labels=4, label="Rare"
    ):
        self.features = features
        self.threshold = threshold
        self.label = label
        self.min_labels = min_labels
        # derived properties
        self.encoder_dict_ = {}

    def fit(self, X, y=None):
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype == "O"
            ]
        for feature in self.features:
            # extract non rare labels
            s = (X.groupby([feature])[feature].count()/len(X))
            self.encoder_dict_[feature] = [
                x for x in s.loc[s > self.threshold].index.values
            ]
        return self

    def transform(self, X):
        # to avoid errors
        X = X.copy()
        for feature in self.features:
            # apply mapping
            X.loc[:, feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]),
                X[feature],
                self.label
            )
        return X

