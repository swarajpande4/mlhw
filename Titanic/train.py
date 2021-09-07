# Importing modules and the dataset 
import numpy as np
import pandas as pd
import re 

import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix 

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 

from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
target = train.Survived 

# Data cleaning and prepping 

# Taking care of missing data in Training set 
train['Age'].fillna(train['Age'].median(), inplace = True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)
train['Fare'].fillna(train['Fare'].median(), inplace = True)
train['Cabin'].fillna('U', inplace = True)

test['Age'].fillna(test['Age'].median(), inplace = True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace = True)
test['Fare'].fillna(test['Fare'].median(), inplace = True)
test['Cabin'].fillna('U', inplace = True)

# Fixing Sex column with integers as it is a categorical value 
train['Sex'].replace({'male': 0, 'female': 1}, inplace = True)
test['Sex'].replace({'male': 0, 'female': 1}, inplace = True)

# Taking care of the Embarked as it is a categorical value, we will encode
encoder = OneHotEncoder()
temp = pd.DataFrame(encoder.fit_transform(train[['Embarked']]).toarray(), columns = ['S', 'C', 'Q'])
train = train.join(temp)
train.drop(columns='Embarked', inplace = True)

temp = pd.DataFrame(encoder.transform(test[['Embarked']]).toarray(), columns = ['S', 'C', 'Q'])
test = test.join(temp)
test.drop(columns='Embarked', inplace = True)

# Taking care of Cabin list
# Taking only the alphabets using regex 
train['Cabin'] = train['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
test['Cabin'] = test['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())

cabin_category = {'A' : 1, 'B' : 2, 'C' : 3, 'D' : 4, 'E' : 5, 'F' : 6, 'G' : 7, 'T' : 8, 'U' : 9}
train['Cabin'] = train['Cabin'].map(cabin_category)
test['Cabin'] = test['Cabin'].map(cabin_category)

# Extracting titles from names 
train['Name'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
test['Name'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

train.rename(columns={'Name' : 'Title'}, inplace=True)
train['Title'] = train['Title'].replace([   'Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 
                                            'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')

test.rename(columns={'Name' : 'Title'}, inplace=True)
test['Title'] = test['Title'].replace([     'Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 
                                            'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')

encoder = OneHotEncoder()
temp = pd.DataFrame(encoder.fit_transform(train[['Title']]).toarray())
train = train.join(temp)
train.drop(columns='Title', inplace = True)

temp = pd.DataFrame(encoder.transform(test[['Title']]).toarray())
test = test.join(temp)
test.drop(columns='Title', inplace = True)

# Making a new column familySize 
train['familySize'] = train['SibSp'] + train['Parch'] + 1
test['familySize'] = test['SibSp'] + test['Parch'] + 1
train.drop(['SibSp', 'Parch', 'Ticket'], axis = 1, inplace = True)
test.drop(['SibSp', 'Parch', 'Ticket'], axis = 1, inplace = True)

# PCA 
columns = train.columns[2: ]
X_train = StandardScaler().fit_transform(train.drop(columns = ['PassengerId', 'Survived']))
pca_df = pd.DataFrame(X_train, columns = columns)

pca = PCA(n_components = 2)
pca_df = pca.fit_transform(pca_df)

plt.figure(figsize = (8, 6))
plt.scatter(pca_df[:, 0], pca_df[:, 1], c = target)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig('pca.png')

# Model training 
X = train.drop(['PassengerId', 'Survived'], axis = 1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter = 10000)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
accuracy_logreg = accuracy_score(y_pred, y_test)
conf_mat_logreg = confusion_matrix(y_pred, y_test)
plot = plot_confusion_matrix(logreg, X_test_scaled, y_test, cmap = plt.cm.Blues, normalize = 'pred')
plot.ax_.set_title("Nomalized Confusion Matrix for Logistic Regression")
plt.savefig('ConfMatLogReg.png')

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_pred, y_test)
conf_mat_knn = confusion_matrix(y_pred, y_test)
plot = plot_confusion_matrix(knn, X_test_scaled, y_test, cmap = plt.cm.Blues, normalize = 'pred')
plot.ax_.set_title("Nomalized Confusion Matrix for KNN")
plt.savefig('ConfMatKNN.png')

svc = SVC(gamma = 0.1)
svc.fit(X_train_scaled, y_train)
y_pred = svc.predict(X_test_scaled)
accuracy_svc = accuracy_score(y_pred, y_test)
conf_mat_svc = confusion_matrix(y_pred, y_test)
plot = plot_confusion_matrix(svc, X_test_scaled, y_test, cmap = plt.cm.Blues, normalize = 'pred')
plot.ax_.set_title("Nomalized Confusion Matrix for SVC (RBF Kernel)")
plt.savefig('ConfMatSVC.png')

dt = DecisionTreeClassifier(max_depth = 3)
dt.fit(X_train_scaled, y_train)
y_pred = dt.predict(X_test_scaled)
accuracy_dt = accuracy_score(y_pred, y_test)
conf_mat_dt = confusion_matrix(y_pred, y_test)
plot = plot_confusion_matrix(dt, X_test_scaled, y_test, cmap = plt.cm.Blues, normalize = 'pred')
plot.ax_.set_title("Nomalized Confusion Matrix for Decision Tree")
plt.savefig('ConfMatDT.png')

# Plotting accuracies 
model_names = ['Logistic Regression', 'kNN', 'Support Vector Classifier', 'Decision Tree']
accuracy = [accuracy_logreg, accuracy_knn, accuracy_svc, accuracy_dt]

plt.figure(figsize = (9, 5))
plot = sns.barplot(x = model_names, y = accuracy, color = 'navy')
plt.ylim(0.60, 1)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title('Accuracy of Models')

for bar in plot.patches:
    plot.annotate(
        format(bar.get_height(), '.2f'), 
        (bar.get_x() + bar.get_width() / 2, 
        bar.get_height()), 
        ha='center', 
        va='center',
        size=15, 
        xytext=(0, 8),
        textcoords='offset points'
    )
plt.savefig('Accuracies.png')