

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


class DecisionTreeDiscretiser(BaseEstimator, TransformerMixin):

    # default grid for decision tree
    __param_grid = {
        "max_depth": [1, 2, 3, 4, 5],
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"]
    }

    def __init__(self, features=None, param_grid=None, cv=5, random_state=None):
        self.features = features
        self.param_grid = self.__param_grid \
            if param_grid is None else param_grid
        self.cv = cv
        self.random_state = random_state
        # derived variables
        self.binner_dict_ = {}
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

        # loop through the features
        for feature in self.features:
            # construct the grid search
            classifier = GridSearchCV(
                DecisionTreeClassifier(
                    random_state=self.random_state
                ),
                param_grid=self.param_grid,
                cv=self.cv, n_jobs=-1
            )
            classifier.fit(X.loc[:, [feature]], y)
            self.binner_dict_[feature] = DecisionTreeClassifier(
                **classifier.best_params_,
                random_state=self.random_state
            ).fit(
                X.loc[:, [feature]], y
            )
        return self

    def transform(self, X):
        # loop through the features and transform
        for feature in self.features:
            # get the classifier
            clf = self.binner_dict_[feature]
            X.loc[:, feature] = clf.predict_proba(X.loc[:, [feature]])[:, 1]
        return X





