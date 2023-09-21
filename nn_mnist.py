# Importation des librairies à utiliser
from nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Chargement du dataset

print("[INFO] chargement des données du Mnist ....")

digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min()) # Normalisation des données avec MinMax

print("[INFO] samples : {}, dimension:{}".format(data.shape[0],data.shape[1]))

# Division du dataset

(trainX, testX, trainY, testY) = train_test_split(data,digits.target, test_size=0.25)

# Encodage avec LabelBinarizer

encoder = LabelBinarizer()

trainY = encoder.fit_transform(trainY)
testY = encoder.fit_transform(testY)

# Creation et Entrainement du model
print("[INFO] entrainement du model")

nn = NeuralNetwork([trainX.shape[1],32,16,10])

print("[INFO] {}".format(nn))
nn.fit(trainX,trainY, epochs=1000)

# Evaluation du modèle

print("[INFO] evaluation du modele .........")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print("[INFO] Le score obtenu est {} ......".format(classification_report(testY.argmax(axis=1),predictions)))