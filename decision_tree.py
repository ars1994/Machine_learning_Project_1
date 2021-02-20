import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree


# df_train = pd.read_csv("data/mnist_train.csv")
# df_test = pd.read_csv("data/mnist_test.csv")

train_filename = "mnist_train.csv"
test_filename = "mnist_test.csv"

# train_filename = "train_mobile.csv"
# test_filename = "test_mobile.csv"

df_train = pd.read_csv("data/"+train_filename)
df_test = pd.read_csv("data/"+test_filename)


# df_train_temp = pd.read_csv("data/"+train_filename)
# df_train = df_train_temp.sample(frac = 0.3)

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

X_new = []
y_new = []
for row in df_train.iterrows():
    label = row[1][0] # label (the number visible in the image)
    image = list(row[1][1:]) # image information as list, without label
    image = np.array(image) / 255
    X_new.append(image)
    y_new.append(label)

X_new = np.array(X_new)
y_new = np.array(y_new)
print(len(X_new))
print(len(df_test))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(X_train), len(y_train))
print(X_train[1].shape)

print("data load complete")


# print("training data")

# clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("DT with entropy")
# print(metrics.accuracy_score(y_test, y_pred))

# clf2 = tree.DecisionTreeClassifier(criterion='gini', max_depth=10)
# clf2 = clf2.fit(X_train, y_train)
# y_pred = clf2.predict(X_test)
# print("DT with gini")
# print(metrics.accuracy_score(y_test, y_pred))

# x_axis = []
# score = []
# for i in range(20):
#     clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1)
#     clf.fit(X_train, y_train)   
#     y_pred = clf.predict(X_test)
#     score.append(metrics.accuracy_score(y_test, y_pred))
#     x_axis.append(i+1)

# plt.plot(x_axis, score)
# plt.grid()
# # plt.xticks(np.arange(0, len(x_axis)+1, 1))
# plt.xlabel("Max Depth")
# plt.ylabel("accuracy_score")
# plt.title("Decision Tree - Max Depth(Minst)")
# plt.savefig("charts/dt_max_depth_minst")
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
#     clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=18, min_samples_leaf=i+1)
#     clf.fit(X_train, y_train)   
#     y_pred = clf.predict(X_test)
#     score.append(metrics.accuracy_score(y_test, y_pred))
#     x_axis.append(i+1)

# plt.plot(x_axis, score)
# plt.grid()
# # plt.xticks(np.arange(0, len(x_axis)+1, 1))
# plt.xlabel("Min Samples Leaf")
# plt.ylabel("accuracy_score")
# plt.title("Decision Tree - Min Samples Leaf(Minst)")
# plt.savefig("charts/dt_min_leaf_minst")
# plt.clf()

# max_val = 0.0
# max_index = -1
# for i in range(len(score)):
#   if score[i] >= max_val:
#       max_val = score[i]
#       max_index = x_axis[i]

# print(max_val)
# print(max_index)

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=9, min_samples_leaf=4)
# clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=18, min_samples_leaf=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# print("testing data")

y_new_pred = clf.predict(X_new)
print(metrics.accuracy_score(y_new, y_new_pred))

# y_new_pred = clf2.predict(X_new)
# print(y_new_pred)
# print(metrics.accuracy_score(y_new, y_new_pred))


# clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=9, min_samples_leaf=4)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=18, min_samples_leaf=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# print("testing data")

y_new_pred = clf.predict(X_new)
print(metrics.accuracy_score(y_new, y_new_pred))

# y_new_pred = clf2.predict(X_new)
# print(y_new_pred)
# print(metrics.accuracy_score(y_new, y_new_pred))