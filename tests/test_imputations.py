
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.imputers.missing_indicator import MissingIndicatorImputer
from src.imputers.numerical import MeanImputer, MedianImputer
from src.imputers.categorical import ModeImputer

data_titanic = pd.read_csv(
    "../data/titanic.csv",
    usecols=['age', 'fare', 'survived']
)

"""
x_train, x_test, y_train, y_test = train_test_split(
    data_titanic[['age', 'fare']],  # predictors
    data_titanic['survived'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

print(x_train.shape, x_test.shape)
"""

cols_to_use = [
    'BsmtQual', 'FireplaceQu', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt',
    'SalePrice'
]
data_housing = pd.read_csv('../data/housing.csv', usecols=cols_to_use)
# split
x_train, x_test, y_train, y_test = train_test_split(
    data_housing.drop(columns=["SalePrice"]),
    data_housing['SalePrice'],
    test_size=0.3,
    random_state=0
)
print(x_train.shape, x_test.shape)


def test_missing_imputer(data):
    imputer = MissingIndicatorImputer(
        features=["age"]
    )
    imputer.fit(data)
    data = imputer.transform(X=data)
    print(data.shape)
    print(data.head())
    print(f"Original NA Mean: {data.age.isna().mean()}")
    print(f"Missing Indicator NA Mean: {data.age_NA.mean()}")


def test_mean_imputer(data):
    imputer = MeanImputer()
    imputer.fit(data)
    print(f"Imputer Dict: {imputer.imputer_dict_}")
    data = imputer.transform(X=data)
    print(data.shape)
    print(data.head())
    print(f"Missing Mean: {data.isnull().mean()}")


def test_median_imputer(data):
    imputer = MedianImputer()
    imputer.fit(data)
    print(f"Imputer Dict: {imputer.imputer_dict_}")
    data = imputer.transform(X=data)
    print(data.shape)
    print(data.head())
    print(f"Missing Mean: {data.isnull().mean()}")


def test_mode_imputer(data):
    imputer = ModeImputer()
    imputer.fit(data)
    print(f"Imputer Dict: {imputer.imputer_dict_}")
    data = imputer.transform(X=data)
    print(data.shape)
    print(data.head())
    print(f"Missing Mean: {data.isnull().mean()}")


# test_missing_imputer(x_train)
# test_mean_imputer(x_train)
# test_median_imputer(x_train)
test_mode_imputer(x_train)



