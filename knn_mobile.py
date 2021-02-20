import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

train_filename = "train_mobile.csv"
test_filename = "test_mobile.csv"

df_train = pd.read_csv("data/"+train_filename)
df_test = pd.read_csv("data/"+test_filename)

print(df_train.shape)
print(df_test.shape)


X_train, X_test, y_train, y_test = train_test_split(df_train.drop('price_range', axis = 1), df_train['price_range'], test_size = 0.30, random_state = 141)


# x_axis = []
# score = []
# for i in range(1,40):
#     clf = KNeighborsClassifier(n_neighbors = i)
#     clf.fit(X_train, y_train)  
#     pred_y = clf.predict(X_test)
#     score.append(accuracy_score(y_test, pred_y))
#     x_axis.append(i)


# plt.plot(x_axis, score)
# plt.grid()
# plt.xlabel("Numner of Neighbors")
# plt.ylabel("accuracy_score")
# plt.title("KNN - Number of Neighbours")
# plt.savefig("charts/knn_number_of_n_mobile")
# plt.clf()


# max_val = 0.0
# max_index = -1
# for i in range(len(score)):
#   if score[i] >= max_val:
#       max_val = score[i]
#       max_index = x_axis[i]

# print(max_val)
# print(max_index)


x_axis = ['ball_tree', 'kd_tree', 'brute']
score = []
for i in x_axis:
    print(i)
    clf = KNeighborsClassifier(n_neighbors = 36, algorithm=i)
    clf.fit(X_train, y_train)  
    pred_y = clf.predict(X_test)
    score.append(accuracy_score(y_test, pred_y))

# max_val = 0.0
# max_index = -1
# for i in range(len(score)):
#   if score[i] >= max_val:
#       max_val = score[i]
#       max_index = x_axis[i]

# print(max_val)
# print(max_index)

plt.plot(x_axis, score)
plt.grid()
plt.xlabel("Different Algoritms")
plt.ylabel("accuracy_score")
plt.title("KNN - Different Algoritms")
plt.savefig("charts/knn_algo_mobile")
plt.clf()
