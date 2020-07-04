
from src.encoders.onehot import OneHotTopEncoder
from src.encoders.ordinal import OrdinalEncoder, TargetGuidedEncoder
from src.encoders.mean import MeanEncoder
from src.encoders.rare import RareLabelEncoder
from src.encoders.weighted import ProbabilityRatioEncoder, WeightOfEvidenceEncoder
from src.encoders.frequency import CountEncoder, FrequencyEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json


data = pd.read_csv("../data/housing.csv")
features_to_read = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'SalePrice']
target = "SalePrice"

"""

data = pd.read_csv(
    "../data/titanic.csv",
    usecols=['cabin', 'sex', 'embarked', 'survived']
)
features_to_read = ['cabin', 'sex', 'embarked']
target = "survived"
"""

# split into train and test
x_train, x_test, y_train, y_test = train_test_split(
    data[features_to_read],  # predictors
    data[target],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

print(y_train.name)
print(type(y_train))
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


def test_count_encoding(df):
    encoder = CountEncoder(
        features=['Neighborhood', 'Exterior1st'],
    )
    encoder.fit(df)
    print(encoder.features)
    print(json.dumps(encoder.encoder_dict_, indent=4))
    df = encoder.transform(df)
    print(df.columns)
    print(df.shape)


def test_frequency_encoding(df):
    encoder = FrequencyEncoder(
        features=['Exterior2nd'],
    )
    encoder.fit(df)
    print(encoder.features)
    print(json.dumps(encoder.encoder_dict_, indent=4))
    df = encoder.transform(df)
    print(df.columns)
    print(df.shape)


def test_target_guided_encoding(df, y):
    encoder = TargetGuidedEncoder()
    encoder.fit(df, y)
    print(encoder.features)
    print(json.dumps(encoder.encoder_dict_, indent=4))
    df = encoder.transform(df)
    print(df.head())
    print(df.shape)


def test_mean_encoding(df, y):
    encoder = MeanEncoder(
        features=["cabin"]
    )
    encoder.fit(df, y)
    print(encoder.features)
    print(json.dumps(encoder.encoder_dict_, indent=4))
    df = encoder.transform(df)
    print(df.head())
    print(df.shape)


def test_probability_encoding(df, y):
    # pre-processing steps to sync with tests
    # df = df.dropna(subset=["embarked"])
    # df = df[df["cabin"] != "T"]

    encoder = ProbabilityRatioEncoder(
        features=["cabin", "sex", "embarked"]
    )
    encoder.fit(df, y)
    print(encoder.features)
    print(json.dumps(encoder.encoder_dict_, indent=4))
    df = encoder.transform(df)
    print(df.head())
    print(df.shape)


def test_weighted_encoding(df, y):
    # pre-processing steps to sync with tests
    df = df.dropna(subset=["embarked"])
    df = df[df["cabin"] != "T"]

    encoder = WeightOfEvidenceEncoder(
        features=["cabin", "sex", "embarked"]
    )
    encoder.fit(df, y)
    print(encoder.features)
    print(json.dumps(encoder.encoder_dict_, indent=4))
    df = encoder.transform(df)
    print(df.head())
    print(df.shape)


def test_rare_encoding(df):
    df = df.drop(columns=["SalePrice"])
    encoder = RareLabelEncoder(
        threshold=0.05
    )
    encoder.fit(df)
    print(encoder.features)
    print(json.dumps(encoder.encoder_dict_, indent=4))
    df = encoder.transform(df)
    print(df.head())
    print(df.shape)

# test_weighted_encoding(x_train, y_train)
# test_one_hot_top(x_train)
# test_rare_encoding(x_train)


