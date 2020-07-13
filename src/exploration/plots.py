
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def barplot(data, x, y):
    ax = sns.barplot(x=x, y=y, data=data)
    plt.show()


def countplot(data, x, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(x=x, data=data)
    plt.show()


def hexbin_plot(data, x, y, gridsize=15):
    data.plot.hexbin(
        x=x,
        y=y,
        gridsize=gridsize
    )
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


def diagnostic_plot(data, feature, figsize=(15, 6), bins=30):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    data[feature].hist(bins=bins)
    plt.subplot(1, 2, 2)
    stats.probplot(data[feature], dist="norm", plot=plt)
    plt.show()
