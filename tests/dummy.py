
from src.encoders.onehot import OneHotTopEncoder
from src.encoders.ordinal import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json

data = pd.read_csv("../data/housing.csv")


# split into train and test
x_train, x_test, y_train, y_test = train_test_split(
    data[['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SalePrice']],  # predictors
    data['SalePrice'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

print(type(x_train))
print(x_train.shape, x_test.shape)


def test_one_hot_top(df):
    encoder = OneHotTopEncoder(
        top_n=10,
        # features=['Neighborhood', 'Exterior1st', 'Exterior2nd'],
    )
    encoder.fit(df)
    print(encoder.features)
    print(json.dumps(encoder.encoder_dict_, indent=4))
    df = encoder.transform(data)
    print(df.columns)
    print(len(df.columns))


def test_ordinal_encoding(df):
    encoder = OrdinalEncoder(
        features=['Neighborhood'],
    )
    encoder.fit(df)
    print(encoder.features)
    print(json.dumps(encoder.encoder_dict_, indent=4))
    df = encoder.transform(df)
    print(df.columns)
    print(df.shape)


test_ordinal_encoding(x_train)

