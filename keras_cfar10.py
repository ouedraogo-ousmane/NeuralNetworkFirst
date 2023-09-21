# Importation des librairies
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import argparse

# Construction des arguments du script
ap = argparse.ArgumentParser()
ap.add_argument("-o","--output", required=False, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# Chargement des données CIFAR10
print("[INFO] chargement de CIFAR-10 .....")

((trainX,trainY),(testX,testY)) = cifar10.load_data()
# Transformation des données d'inputs
trainX = trainX.astype("float") / 255
testX = testX.astype("float") / 255
trainX = trainX.reshape((trainX.shape[0],3072))
testX = testX.reshape((testX.shape[0],3072))

# Encodage
encoder = LabelBinarizer()
trainY = encoder.fit_transform(trainY)
testY = encoder.fit_transform(testY)
labelNames = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

# Creation du modèle  3072-1024-512-10
print("[INFO] Creation du modèle .........")
model = Sequential()
model.add(Dense(1024, input_shape=(3072,),activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Entrainement du modèle
print("[INFO] entrainement du modèle ......")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,metrics=["accuracy"])
H = model.fit(trainX,trainY, validation_data=(testX,testY),batch_size=32, epochs=100)

# Evaluation du modèle
print("[INFO] evaluation du modèle obtenu ..........")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))


# Visualisation graphique
print("[INFO] visualisation graphique...........")
plt.style.use("ggplot")
plt.figure(figsize=(20,10))
plt.plot(np.arange(0,100), H.history["loss"],label="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,100), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0,100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("# Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.savefig(args["output"])
