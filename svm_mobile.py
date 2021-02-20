import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

train_filename = "train_mobile.csv"
test_filename = "test_mobile.csv"

df_train = pd.read_csv("data/"+train_filename)
df_test = pd.read_csv("data/"+test_filename)

print(df_train.shape)
print(df_test.shape)


X_train, X_test, y_train, y_test = train_test_split(df_train.drop('price_range', axis = 1), df_train['price_range'], test_size = 0.30, random_state = 141)

# x_axis = []
# score = []
# for i in range(1,11):
#     clf = svm.SVC(kernel='poly', degree=i)
#     clf.fit(X_train, y_train)   
#     y_pred = clf.predict(X_test)
#     score.append(metrics.accuracy_score(y_test, y_pred))
#     x_axis.append(i)

# plt.plot(x_axis, score)
# plt.grid()
# plt.xlabel("Degree of Polynomial")
# plt.ylabel("accuracy_score")
# plt.title("SVM (Polynomial) - Degree of Polynomial")
# plt.savefig("charts/svm_poly_degree_mobile")
# plt.clf()

# max_val = 0.0
# max_index = -1
# for i in range(len(score)):
#   if score[i] >= max_val:
#       max_val = score[i]
#       max_index = x_axis[i]

# print(max_val)
# print(max_index)



clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)   
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)   
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))