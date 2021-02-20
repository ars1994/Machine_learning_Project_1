import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(train_sizes, test_scores_mean, 'o-')
    axes[2].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Training examples")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt



train_filename = "mnist_train.csv"
test_filename = "mnist_test.csv"

# df_train_temp = pd.read_csv("data/"+train_filename)
df_train = pd.read_csv("data/"+train_filename)
# df_test_temp = pd.read_csv("data/"+test_filename)


# df_train = df_train_temp.sample(frac = 0.3)


print(df_train.shape)
# print(df_test.shape)

X = []
y = []
for row in df_train.iterrows() :
    label = row[1][0] # label (the number visible in the image)
    image = list(row[1][1:]) # image information as list, without label
    image = np.array(image) / 255
    X.append(image)
    y.append(label)

X = np.array(X)
y = np.array(y)

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

print("starting code")


# plot decision trees


# fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# title = "Learning Curves (Decision Trees)"

# clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, max_depth=18)
# plot_learning_curve(clf, title, X, y, axes=axes, ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=-1)


# plt.savefig("charts/DT_sample_final")
# plt.clf()


#plot neural nets



# fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# title = "Learning Curves (Nural Net - adam)"

# clf = clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100,), random_state=1, max_iter=100)
# plot_learning_curve(clf, title, X, y, axes=axes, ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=-1)

# plt.savefig("charts/NN")
# plt.clf()


#KNN


# fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# title = "Learning Curves (KNN)"

# clf = KNeighborsClassifier(n_neighbors = 1)
# plot_learning_curve(clf, title, X, y, axes=axes, ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=-1)


# plt.savefig("charts/KNN_final")
# plt.clf()

#SVM

# fig, axes = plt.subplots(3, 3, figsize=(30, 20))

# title = "Learning Curves (SVM - linear)"

# clf = svm.SVC(kernel='linear')
# plot_learning_curve(clf, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=-1)

# title = "Learning Curves (SVM - poly)"

# clf = svm.SVC(kernel='poly')
# plot_learning_curve(clf, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=-1)

# title = "Learning Curves (SVM - rbf)"

# clf = svm.SVC(kernel='rbf')
# plot_learning_curve(clf, title, X, y, axes=axes[:, 2], ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=-1)


# plt.savefig("charts/SVM")
# plt.clf()


#boosting

# fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# title = "Learning Curves (boosting)"

# clf = GradientBoostingClassifier(n_estimators=60, learning_rate=0.3, max_depth=2, random_state=0, min_samples_leaf=5)
# plot_learning_curve(clf, title, X, y, axes=axes, ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=-1)


# plt.savefig("charts/boosting_final)
# plt.clf()