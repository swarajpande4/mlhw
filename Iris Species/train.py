# Importing modules and the dataset
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import LinearSVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 

iris = pd.read_csv('dataset/Iris.csv').drop('Id', axis = 1)

# Data prepping 
X = iris.iloc[:, :-1].values 
y = iris.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Training the models
inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, random_state = 42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize = (7, 7))
plt.plot(range(1, 10), inertia)
plt.title("Elbow method to find optimal clusters")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.plot(2, inertia[1], marker = 'o', color = 'red')
plt.plot(3, inertia[2], marker = 'o', color = 'red')
plt.savefig("elbow.png")

kmeans = KMeans(3)
kmeans.fit(X)
clusters = kmeans.fit_predict(X)

x = iris.iloc[:, [0, 1, 2, 3]].values 
plt.figure(figsize = (7, 7))
plt.scatter(x[clusters == 0, 0], x[clusters == 0, 1],
            c = 'blue', label = 'Iris-setosa')
plt.scatter(x[clusters == 1, 0], x[clusters == 1, 1],
            c = 'orange', label = 'Iris-versicolor')
plt.scatter(x[clusters == 2, 0], x[clusters == 2, 1],
            c = 'green', label = 'Iris-virginica')
plt.legend()
plt.savefig('clusters.png')

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_knn = accuracy_score(y_pred, y_test)
print(f'The accuracy of KNN is: {accuracy_knn}')

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_dt = accuracy_score(y_pred, y_test)
print(f'The accuracy of Decision Tree is: {accuracy_dt}')