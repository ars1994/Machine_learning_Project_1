import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


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


# x_axis = []
# score = []
# for i in range(20):
#     regressor = GradientBoostingClassifier(
#         max_depth=2,
#         n_estimators=((i+1)*10),
#         learning_rate=0.7,
#         min_samples_leaf=5
#     )
#     regressor.fit(X_train, y_train)
#     regressor_output = regressor.predict(X_test)
#     score.append(metrics.accuracy_score(y_test, regressor_output))
#     x_axis.append((i+1)*10)

# plt.plot(x_axis, score)
# plt.grid()
# # plt.xticks(np.arange(0, len(x_axis)+1, 1))
# plt.xlabel("# of Estimators")
# plt.ylabel("accuracy_score")
# plt.title("Boosting - Increasing Estimators")
# plt.savefig("charts/boosting_estimators_mobile")
# plt.clf()

# max_val = 0.0
# max_index = -1
# for i in range(len(score)):
#   if score[i] >= max_val:
#       max_val = score[i]
#       max_index = x_axis[i]

# print(max_val)
# print(max_index)


x_axis = []
score = []
for i in range(10):
    regressor = GradientBoostingClassifier(
        max_depth=2,
        n_estimators=50,
        learning_rate=(i+1)/10.0,
        min_samples_leaf=5
    )
    regressor.fit(X_train, y_train)
    regressor_output = regressor.predict(X_test)
    # print(accuracy_score(y_test, regressor_output))
    score.append(metrics.accuracy_score(y_test, regressor_output))
    x_axis.append((i+1)/10.0)

plt.plot(x_axis, score)
plt.grid()
# plt.xticks(np.arange(0, len(x_axis)+1, 1))
plt.xlabel("Learning Rate")
plt.ylabel("accuracy_score")
plt.title("Boosting - Learning Rate (Mobile)")
plt.savefig("charts/boosting_learningRate_moile")
plt.clf()

max_val = 0.0
max_index = -1
for i in range(len(score)):
  if score[i] >= max_val:
      max_val = score[i]
      max_index = x_axis[i]

print(max_val)
print(max_index)



# clf3 = GradientBoostingClassifier(n_estimators=50, learning_rate=0.7, max_depth=2, min_samples_leaf=5)
# clf3.fit(X_train, y_train)
# clf_output = clf3.predict(X_test)
# print("Gradient Boosting Classifier")
# print(metrics.accuracy_score(y_test, clf_output))