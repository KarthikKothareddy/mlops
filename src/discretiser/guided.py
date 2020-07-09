

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


class DecisionTreeDiscretiser(BaseEstimator, TransformerMixin):

    # default grid for decision tree
    __param_grid = {
        "max_depth": [2, 3, 4, 5],
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"]
    }

    def __init__(self, features=None, param_grid=None):
        self.features = features
        self.param_grid = self.__param_grid \
            if param_grid is None else param_grid

        # default features to select
        self.numerical = ["int", "float"]

    def fit(self, X, y):
        if y is None:
            raise ValueError("target variable is missing")
        # by default choose all categorical
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype in self.numerical
            ]
        return self

    def transform(self, X):
        # define the gridsearch space

        # Apply grid search over parameter space
        # predict the probability
        # select the best params





