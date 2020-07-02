
from sklearn.base import TransformerMixin
import numpy as np


class OneHotTopEncoder(TransformerMixin):

    def __init__(self, top_n=10, features=None):
        self.features = features
        self.top_n = top_n
        # derived properties
        self.encoder_dict_ = {}

    def fit(self, X, y=None):
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype == "O"
            ]
        for feature in self.features:
            # get top n categories for the feature
            # and update encoder_dict_
            self.encoder_dict_[feature] = [
                val for val in X[feature].value_counts().sort_values(
                    ascending=False
                ).head(self.top_n).index
            ]
        return self

    def transform(self, X):
        # loop through the features
        for feature in self.features:
            # apply
            for label in self.encoder_dict_[feature]:
                X[f"{feature}_{label}"] = np.where(
                    X[feature] == label, 1, 0
                )
            # drop original
            X = X.drop(columns=[feature])
        return X

