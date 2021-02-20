import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# import time


# df_train = pd.read_csv("data/mnist_train.csv")
# df_test = pd.read_csv("data/mnist_test.csv")

train_filename = "mnist_train.csv"
test_filename = "mnist_test.csv"

# train_filename = "train_mobile.csv"
# test_filename = "test_mobile.csv"

df_train = pd.read_csv("data/"+train_filename)
df_test = pd.read_csv("data/"+test_filename)

# df_train = df_train_temp.sample(frac = 0.3)

print(df_train.shape)
print(df_test.shape)

# print(df_train.shape)
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
print(len(X))
print(len(y))

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


# clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), random_state=1, max_iter=900)

# x_axis = []
# score = []
# for i in range(10):
#     clf = MLPClassifier(solver='adam', hidden_layer_sizes=((i+1)*10,), random_state=1, max_iter=900)
#     clf.fit(X_train, y_train)   
#     neural_output = clf.predict(X_test)
#     # print(accuracy_score(y_test, regressor_output))
#     score.append(metrics.accuracy_score(y_test, neural_output))
#     x_axis.append((i+1)*10.0)

# plt.plot(x_axis, score)
# plt.grid()
# # plt.xticks(np.arange(0, len(x_axis)+1, 1))
# plt.xlabel("Hidden Layers")
# plt.ylabel("accuracy_score")
# plt.title("Nural Network - Hidden Layers (Minst)")
# plt.savefig("charts/nn_hidden_layers_minst")
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
# fit_time = []
# for i in range(10):
#     clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100,), random_state=1, max_iter=(i+1)*100)
#     # tic = time.perf_counter()
#     clf.fit(X_train, y_train)
#     # toc = time.perf_counter()   
#     neural_output = clf.predict(X_test)
#     score.append(metrics.accuracy_score(y_test, neural_output))
#     x_axis.append((i+1)*100)
#     # fit_time.append(toc-tic)

# plt.plot(x_axis, score)
# plt.grid()
# # plt.xticks(np.arange(0, len(x_axis)+1, 1))
# plt.xlabel("# of iterations")
# plt.ylabel("accuracy_score")
# plt.title("Nural Network - iterations (Minst)")
# plt.savefig("charts/nn_iterations_minst")
# plt.clf()

# plt.plot(x_axis, fit_time)
# plt.grid()
# plt.xlabel("# of iterations")
# plt.ylabel("fit time")
# plt.title("Nural Network - Fit Times(Minst)")
# plt.savefig("charts/nn_fit_times_minst")
# plt.clf()

# max_val = 0.0
# max_index = -1
# for i in range(len(score)):
#   if score[i] >= max_val:
#       max_val = score[i]
#       max_index = x_axis[i]

# print(max_val)
# print(max_index)


# clf = clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(100,), random_state=1, max_iter=900)
# clf.fit(X_train, y_train)   
# neural_output = clf.predict(X_test)
# print("sgd")
# print(accuracy_score(y_test, neural_output))


# y_new_pred = clf.predict(X_new)
# print(y_new_pred)

# print(metrics.accuracy_score(y_new, y_new_pred))


clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100,), random_state=1, max_iter=100)
clf.fit(X_train, y_train)   
neural_output = clf.predict(X_test)
print("adam")
print(accuracy_score(y_test, neural_output))


y_new_pred = clf.predict(X_new)
print(y_new_pred)

print(metrics.accuracy_score(y_new, y_new_pred))