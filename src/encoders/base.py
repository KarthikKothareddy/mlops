
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np


class BaseEncoder(TransformerMixin):
    # TODO: work in progress
    def __init__(self, features):
        self.features = features
        # derived properties
        self.encoder_dict_ = {}

    def transform(self, X):
        for feature in self.features:
            # apply mapping
            X.loc[:, feature] = X[feature].map(self.encoder_dict_[feature])
        return X

