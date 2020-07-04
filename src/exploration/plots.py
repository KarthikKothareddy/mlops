

import matplotlib.pyplot as plt
import seaborn as sns


def barplot(data, x, y):
    ax = sns.barplot(x=x, y=y, data=data)
    plt.show()


def countplot(data, x, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(x=x, data=data)
    plt.show()


def kde_plot(data, feature, target, figsize=(10, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    # loop through cardinality of feature
    for cat in data[feature].unique():
        data[data[feature] == cat][target].plot(kind="kde", ax=ax)
    # legend
    lines, labels = ax.get_legend_handles_labels()
    labels = data[feature].unique()
    ax.legend(lines, labels, loc='best')
    plt.show()

