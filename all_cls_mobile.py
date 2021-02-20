import numpy as np
import pandas as pd #
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


train_filename = "train_mobile.csv"

df_train = pd.read_csv("data/"+train_filename)

print(df_train.shape)

X_train, X_test, y_train, y_test = train_test_split(df_train.drop('price_range', axis = 1), df_train['price_range'], test_size = 0.30, random_state = 141)

print("Data loaded, Running Classifiers now")


print("Decision Tree Classifier")

clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=13, min_samples_leaf=8)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


print("Neural Net Classifier")

clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100,), random_state=1, max_iter=400, beta_2=0.870)
clf.fit(X_train, y_train)   
neural_output = clf.predict(X_test)
print(accuracy_score(y_test, neural_output))


print("Gradient Boosting Classifier")

clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.7, max_depth=2, min_samples_leaf=5)
clf.fit(X_train, y_train)
clf_output = clf.predict(X_test)
print(accuracy_score(y_test, clf_output))


print("Support Vector Machine - kernel = linear")

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
clf_output = clf.predict(X_test)
print(accuracy_score(y_test, clf_output))


print("Support Vector Machine - kernel = poly")

clf = svm.SVC(kernel='poly', degree=8)
clf.fit(X_train, y_train)
clf_output = clf.predict(X_test)
print(accuracy_score(y_test, clf_output))


print("kNN")

clf = KNeighborsClassifier(n_neighbors = 36)
clf.fit(X_train, y_train)
clf_output = clf.predict(X_test)
print(accuracy_score(y_test, clf_output))
