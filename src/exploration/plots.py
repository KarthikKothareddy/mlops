
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


def diagnostic_plot(data, feature, figsize=(16, 5), bins=30):
    # setting figure
    plt.figure(figsize=figsize)
    # histogram
    plt.subplot(1, 3, 1)
    sns.distplot(data[feature], bins=bins)
    plt.title('Histogram')
    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(data[feature], dist="norm", plot=plt)
    plt.ylabel('Variable Quantiles')
    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=data[feature])
    plt.title('Boxplot')
    plt.show()
