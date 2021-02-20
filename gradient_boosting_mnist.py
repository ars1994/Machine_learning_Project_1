import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor


# df_train = pd.read_csv("data/mnist_train.csv")
# df_test = pd.read_csv("data/mnist_test.csv")

train_filename = "mnist_train.csv"
test_filename = "mnist_test.csv"

# train_filename = "train_mobile.csv"
# test_filename = "test_mobile.csv"

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
print(len(X))
print(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(X_train), len(y_train))
print(X_train[1].shape)

# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(y_pred[0:20], ".....")
# print(y_test[0:20], ".....")
# print(metrics.accuracy_score(y_test, y_pred))

# clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(10,), random_state=1)
# clf.fit(X_train, y_train)   
# neural_output = clf.predict(X_test)
# print("sgd")
# print(accuracy_score(y_test, neural_output))

regressor = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=3,
    learning_rate=1.0
)
regressor.fit(X_train, y_train)
regressor_output = regressor.predict(X_test)
# print(accuracy_score(y_test, regressor_output))
print(accuracy_score(y_test, regressor_output.round(), normalize=False))


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

y_new_pred = regressor.predict(X_new)
print(y_new_pred)

# print(metrics.accuracy_score(y_new, y_new_pred))
print(accuracy_score(y_new, y_new_pred.round(), normalize=False))