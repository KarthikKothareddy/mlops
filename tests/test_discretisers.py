

import pandas as pd
import numpy as np
import json
import pprint
from sklearn.model_selection import train_test_split
from src.discretiser.guided import DecisionTreeDiscretiser


pp = pprint.PrettyPrinter(indent=4)


def impute_na(data, variable):
    # function to fill NA with a random sample
    df = data.copy()
    # random sampling
    df[variable+'_random'] = df[variable]
    # extract the random sample to fill the na
    random_sample = df[variable].dropna().sample(
        df[variable].isnull().sum(), random_state=0)
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable+'_random'] = random_sample

    return df[variable+'_random']


def test_decision_tree_discretiser():
    data = pd.read_csv(
        "../data/titanic.csv",
        usecols=['age', 'fare', 'survived']
    )
    x_train, x_test, y_train, y_test = train_test_split(
        data[['age',]],
        data['survived'],
        test_size=0.3,
        random_state=0
    )
    print(x_train.shape, x_test.shape)
    # replace NA in both train and test sets
    x_train['age'] = impute_na(data, 'age')
    x_train['fare'] = impute_na(data, 'fare')
    x_test['age'] = impute_na(data, 'age')
    x_test['fare'] = impute_na(data, 'fare')

    discretiser = DecisionTreeDiscretiser(
        features=["age"], cv=5
    )

    discretiser.fit(x_train, y_train)
    pp.pprint(discretiser.binner_dict_)
    tmp = discretiser.transform(x_train)
    print(tmp.age.unique())
    # print(tmp.fare.unique())


test_decision_tree_discretiser()

