

from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np


class ProbabilityRatioEncoder(TransformerMixin):

    def __init__(self, features=None):
        self.features = features
        # derived properties
        self.encoder_dict_ = {}

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("target variable is missing")
        # add target
        X = pd.concat([X, y], axis=1)
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype == "O"
            ]
        for feature in self.features:
            __prob = pd.DataFrame(X.groupby(feature)[y.name].mean())
            __prob["p_1"] = __prob[y.name]
            __prob["p_0"] = 1 - __prob[y.name]
            # division by zero checks
            if not __prob.loc[__prob["p_0"] == 0, :].empty:
                raise ValueError("p(0) for a category is zero")
            # update mapping by guided target
            self.encoder_dict_[feature] = np.divide(
                    __prob["p_1"], __prob["p_0"]
            ).to_dict()
        return self

    def transform(self, X):
        for feature in self.features:
            # apply mapping
            X.loc[:, feature] = X[feature].map(self.encoder_dict_[feature])
        return X


class WeightOfEvidenceEncoder(ProbabilityRatioEncoder):

    def __init__(self, features=None):
        super().__init__(features)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("target variable is missing")
        # add target
        X = pd.concat([X, y], axis=1)
        if self.features is None:
            self.features = [
                c for c in X.columns
                if X[c].dtype == "O"
            ]
        for feature in self.features:
            __prob = pd.DataFrame(X.groupby(feature)[y.name].mean())
            __prob["p_1"] = __prob[y.name]
            __prob["p_0"] = 1 - __prob[y.name]
            # division by zero checks
            if (
                    not __prob.loc[__prob["p_0"] == 0, :].empty
                    or
                    not __prob.loc[__prob["p_1"] == 0, :].empty
            ):
                raise ValueError("p(1) or p(0) for a category is zero")
            # update mapping by guided target, log2
            self.encoder_dict_[feature] = np.log(
                np.divide(__prob["p_1"], __prob["p_0"])
            ).to_dict()
        return self


