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


train_filename = "mnist_train.csv"
test_filename = "mnist_test.csv"

df_train = pd.read_csv("data/"+train_filename)
df_test = pd.read_csv("data/"+test_filename)


print(df_train.shape)
print(df_test.shape)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(X_train), len(y_train))
print(X_train[1].shape)


X_new = []
y_new = []
for row in df_train.iterrows() :
    label = row[1][0] # label (the number visible in the image)
    image = list(row[1][1:]) # image information as list, without label
    image = np.array(image) / 255
    X_new.append(image)
    y_new.append(label)
print(len(X_new))
print(len(df_test))

print("Data loaded, Running Classifiers now")


print("Decision Tree Classifier")

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=18, min_samples_leaf=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_new_pred = clf.predict(X_new)
print(accuracy_score(y_new, y_new_pred))

print("Neural Net Classifier")

clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100,), random_state=1, max_iter=100)
clf.fit(X_train, y_train)   
neural_output = clf.predict(X_test)
print(accuracy_score(y_test, neural_output))


y_new_pred = clf.predict(X_new)
print(accuracy_score(y_new, y_new_pred))

print("Gradient Boosting Classifier")

clf = GradientBoostingClassifier(n_estimators=60, learning_rate=0.3, max_depth=2, random_state=0, min_samples_leaf=5)
clf.fit(X_train, y_train)
clf_output = clf.predict(X_test)
print(accuracy_score(y_test, clf_output))

y_new_pred = clf.predict(X_new)
print(accuracy_score(y_new, y_new_pred))

print("Support Vector Machine - kernel = rbf")

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
clf_output = clf.predict(X_test)
print(accuracy_score(y_test, clf_output))

y_new_pred = clf.predict(X_new)
print(accuracy_score(y_new, y_new_pred))

print("Support Vector Machine - kernel = poly")

clf = svm.SVC(kernel='poly', degree=2)
clf.fit(X_train, y_train)
clf_output = clf.predict(X_test)
print(accuracy_score(y_test, clf_output))

y_new_pred = clf.predict(X_new)
print(accuracy_score(y_new, y_new_pred))


print("kNN")

clf = KNeighborsClassifier(n_neighbors = 1)
clf.fit(X_train, y_train)
clf_output = clf.predict(X_test)
print(accuracy_score(y_test, clf_output))

y_new_pred = clf.predict(X_new)
print(accuracy_score(y_new, y_new_pred))