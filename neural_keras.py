# Importation des librairies
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import LabelBinarizer

# construction des parametres
ap = argparse.ArgumentParser()

ap.add_argument("-o", "--output",required=False, 
                help="Chemin pour les sorties des plot de loss/accuracy")
args = vars(ap.parse_args())

# Chargements des données
print("[INFO] chargement des données complètes de Mnist.......")
dataset =datasets.fetch_openml("mnist_784", parser='auto')

# Construction du dataset

data = dataset.data.astype("float") / 255

(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, random_state=0, test_size=0.25)

# Encodage des données
encoder = LabelBinarizer()
trainY = encoder.fit_transform(trainY)
testY = encoder.fit_transform(testY)

# Creation du modele avec keras sous la forme 784-256-128-10
print("[INFO] Creation du modele .........")

model = Sequential()
model.add(Dense(256,input_shape=(784,), activation="sigmoid")) # 1ère couche 
model.add(Dense(128,activation="sigmoid"))
model.add(Dense(10, activation="softmax"))  # dernière couche

# Entrainement du modele en utilisant SGD comme optimiseur
print("[INFO] entrainement du modele .....")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])
H = model.fit(trainX,trainY,batch_size=128,epochs=100, validation_data=(testX,testY))

# Evaluation du modele
print("[INFO] evaluation du modele........")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in encoder.classes_]
                            ))

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
