import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

train_filename = "train_mobile.csv"
test_filename = "test_mobile.csv"

df_train = pd.read_csv("data/"+train_filename)
df_test = pd.read_csv("data/"+test_filename)

print(df_train.shape)
print(df_test.shape)


X_train, X_test, y_train, y_test = train_test_split(df_train.drop('price_range', axis = 1), df_train['price_range'], test_size = 0.30, random_state = 141)

x_axis = []
score = []
for i in range(20):
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=((i+1)*10,), random_state=1, max_iter=400)
    clf.fit(X_train, y_train)   
    neural_output = clf.predict(X_test)
    # print(accuracy_score(y_test, regressor_output))
    score.append(metrics.accuracy_score(y_test, neural_output))
    x_axis.append((i+1)*10.0)

plt.plot(x_axis, score)
plt.grid()
# plt.xticks(np.arange(0, len(x_axis)+1, 1))
plt.xlabel("Hidden Layers")
plt.ylabel("accuracy_score")
plt.title("Nural Network - Hidden Layers (Mobile)")
plt.savefig("charts/nn_hidden_layers_mobile")
plt.clf()

max_val = 0.0
max_index = -1
for i in range(len(score)):
  if score[i] >= max_val:
      max_val = score[i]
      max_index = x_axis[i]

print(max_val)
print(max_index)

# x_axis = []
# score = []
# for i in range(10):
#     clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100,), random_state=1, max_iter=(i+1)*100, beta_2=0.870)
#     clf.fit(X_train, y_train)   
#     neural_output = clf.predict(X_test)
#     score.append(metrics.accuracy_score(y_test, neural_output))
#     x_axis.append((i+1)*100)

# plt.plot(x_axis, score)
# plt.grid()
# # plt.xticks(np.arange(0, len(x_axis)+1, 1))
# plt.xlabel("# of iterations")
# plt.ylabel("accuracy_score")
# plt.title("Nural Network - iterations (Mobile)")
# plt.savefig("charts/nn_iterations_mobile")
# plt.clf()

# max_val = 0.0
# max_index = -1
# for i in range(len(score)):
#   if score[i] >= max_val:
#       max_val = score[i]
#       max_index = x_axis[i]

# print(max_val)
# print(max_index)



# clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100,), random_state=1, max_iter=400, beta_2=0.870)
# clf.fit(X_train, y_train)   
# neural_output = clf.predict(X_test)
# print(accuracy_score(y_test, neural_output))