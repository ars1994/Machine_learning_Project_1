import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt


# df_train = pd.read_csv("data/mnist_train.csv")
# df_test = pd.read_csv("data/mnist_test.csv")

# train_filename = "mnist_train.csv"
# test_filename = "mnist_test.csv"

train_filename = "train_mobile.csv"
test_filename = "test_mobile.csv"

df_train = pd.read_csv("data/"+train_filename)
df_test = pd.read_csv("data/"+test_filename)

print(df_train.shape)
print(df_test.shape)

X_train, X_test, y_train, y_test = train_test_split(df_train.drop('price_range', axis = 1), df_train['price_range'], test_size = 0.30, random_state = 141)


# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(metrics.accuracy_score(y_test, y_pred))

# x_axis = []
# score = []
# for i in range(20):
#     clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1)
#     clf.fit(X_train, y_train)   
#     y_pred = clf.predict(X_test)
#     score.append(metrics.accuracy_score(y_test, y_pred))
#     x_axis.append(i+1)

# plt.plot(x_axis, score)
# plt.grid()
# # plt.xticks(np.arange(0, len(x_axis)+1, 1))
# plt.xlabel("Max Depth")
# plt.ylabel("accuracy_score")
# plt.title("Decision Tree - Max Depth(Moile)")
# plt.savefig("charts/dt_max_depth_mobile")
# plt.clf()

# max_val = 0.0
# max_index = -1
# for i in range(len(score)):
#   if score[i] >= max_val:
#       max_val = score[i]
#       max_index = x_axis[i]

# print(max_val)
# print(max_index)


# x_axis = []
# score = []
# for i in range(20):
#     clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=13, min_samples_leaf=i+1)
#     clf.fit(X_train, y_train)   
#     y_pred = clf.predict(X_test)
#     score.append(metrics.accuracy_score(y_test, y_pred))
#     x_axis.append(i+1)

# plt.plot(x_axis, score)
# plt.grid()
# # plt.xticks(np.arange(0, len(x_axis)+1, 1))
# plt.xlabel("Min Samples Leaf")
# plt.ylabel("accuracy_score")
# plt.title("Decision Tree - Min Samples Leaf(Mobile)")
# plt.savefig("charts/dt_min_leaf_mobile")
# plt.clf()

# max_val = 0.0
# max_index = -1
# for i in range(len(score)):
#   if score[i] >= max_val:
#       max_val = score[i]
#       max_index = x_axis[i]

# print(max_val)
# print(max_index)


clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=13, min_samples_leaf=8)
clf.fit(X_train, y_train)   
y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))