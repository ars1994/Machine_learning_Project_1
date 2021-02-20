import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier



train_filename = "mnist_train.csv"
test_filename = "mnist_test.csv"

# df_train_temp = pd.read_csv("data/"+train_filename)
df_train = pd.read_csv("data/"+train_filename)
df_test = pd.read_csv("data/"+test_filename)


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

print("starting code")

# x_axis = []
# score = []
# for i in range(1,11):
#     print(i)
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
# plt.savefig("charts/knn_number_of_n")
# plt.clf()


# max_val = 0.0
# max_index = -1
# for i in range(len(score)):
#   if score[i] >= max_val:
#       max_val = score[i]
#       max_index = x_axis[i]

# print(max_val)
# print(max_index)


# x_axis = ['ball_tree', 'kd_tree', 'brute']
# score = []
# for i in x_axis:
#     print(i)
#     clf = KNeighborsClassifier(n_neighbors = 1, algorithm=i)
#     clf.fit(X_train, y_train)  
#     pred_y = clf.predict(X_test)
#     score.append(accuracy_score(y_test, pred_y))

# plt.plot(x_axis, score)
# plt.grid()
# plt.xlabel("Different Algoritms")
# plt.ylabel("accuracy_score")
# plt.title("KNN - Different Algoritms")
# plt.savefig("charts/knn_algo")
# plt.clf()

clf = KNeighborsClassifier(n_neighbors = 1)
clf.fit(X_train, y_train)   
pred_y = clf.predict(X_test)
print(accuracy_score(y_test, pred_y))

y_new_pred = clf.predict(X_new)
print(y_new_pred)

print(accuracy_score(y_new, y_new_pred))