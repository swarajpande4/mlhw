# Importing modules and dataset
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns

from keras.datasets import mnist 
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf 

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalization 
X_train = X_train / 255.0
X_test = X_test / 255.0 

# Reshaping
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Encoding 
y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)

# Model Training
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout

model = Sequential()
model.add(Conv2D(128, 5, padding="Same", activation="relu", input_shape=(28,28,1)))
model.add(Conv2D(128, 5, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.50))

model.add(Conv2D(64, 3, padding="Same", activation="relu"))
model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.50))

model.add(Conv2D(32, 3, padding="Same", activation="relu"))
model.add(Conv2D(16, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.50))

model.add(Flatten())

model.add(Dense(1000, activation="relu"))
model.add(Dense(500, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer = "SGD", loss = "categorical_crossentropy", metrics = ["accuracy"])

hist = model.fit(X_train, y_train, batch_size = 32, epochs = 50)

# Plotting training accuracy and loss
plt.figure(figsize = (15, 3))

plt.subplot(1, 2, 1)
plt.plot(hist.history["accuracy"])
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")

plt.subplot(1, 2, 2)
plt.plot(hist.history["loss"])
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("accuracy-loss.png")

y_pred = np.argmax(model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_pred, y_test)

print(f'The accuracy of the trained CNN is: {accuracy}')

# Plotting Confusion matrix 

conf_mat_cnn = confusion_matrix(y_pred, y_test)

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat_cnn, annot=True, linewidths=0.01,cmap="cubehelix",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('confmat.png')