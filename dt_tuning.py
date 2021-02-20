import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree


train_filename = "mnist_train.csv"
test_filename = "mnist_test.csv"

# train_filename = "train_mobile.csv"
# test_filename = "test_mobile.csv"

df_train = pd.read_csv("data/"+train_filename)
df_test = pd.read_csv("data/"+test_filename)

X = []
y = []
if(train_filename == "mnist_train.csv"):
	for row in df_train.iterrows() :
	    label = row[1][0] # label (the number visible in the image)
	    image = list(row[1][1:]) # image information as list, without label
	    image = np.array(image) / 255
	    X.append(image)
	    y.append(label)

	X = np.array(X)
	y = np.array(y)

else:
	X = df_train.drop('price_range', axis = 1)
	y = df_train['price_range']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


print("data load complete")

dt = tree.DecisionTreeClassifier()

param_dict = {
	"criterion":["gini", "entropy"],
	"max_depth" : range(1, 10),
	"min_samples_leaf":range(1,6)
}

grid = GridSearchCV(dt, param_grid=param_dict, cv=10, verbose=1, n_jobs=-1)
grid.fit(X_train,y_train)

print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)