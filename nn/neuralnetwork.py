# Importation des librairies necessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers,alpha=0.1):
        """Initialisation de la classe

        Args:
            layers (_type_): une liste represente l'architecture à adopter [2,2,1]
            alpha (float, optional): le taux d'apprentissage par Defaults to 0.1.
        """
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # Iniatialisation des valeurs de W
        for i in np.arange(0,len(layers) - 2):

            w = np.random.randn(layers[i] + 1,layers[i+1] + 1)

            self.W.append(w/np.sqrt(layers[i]))
        w = np.random.randn(layers[-2]+1,layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))

    def __repr__(self):

        # Construction de la structure d'un reseau de neurone
        return "Neural Network : {}".format("_".join(str(l) for l in self.layers))

    # definition la fonction d'activation :  La fonction Sigmoid
    def sigmoid(self,x):

        return 1.0/(1 + np.exp(-x))

    # La dérivée partielle pour le backpass
    def derive_sigmoid(self, x):

        return x * (1 - x)

    # Entrainement du reseau de neurone
    def fit(self,X,y, epochs=1000, displayUpdates=100):

        X = np.c_[X,np.ones(X.shape[0])]

        # parcourt des epochs
        for epoch in np.arange(0,epochs):
            
            for (x,target) in zip(X,y):

                self.fit_partial(x,target)

            # epoch
            if (epoch == 0) or ((epoch + 1) % displayUpdates == 0):
                loss = self.calculate_loss(X,y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch+1,loss))

    # Entrainement partielle
    def fit_partial(self,x,y):
        A  = [np.atleast_2d(x)]

        # Parcourt des neurones
        for layer in np.arange(0, len(self.W)):

            net = A[layer].dot(self.W[layer])

            out = self.sigmoid(net)

            A.append(out)
        # Back propagation
        error = A[-1] - y

        D = [error * self.derive_sigmoid(A[-1])]

        for layer in np.arange(len(A)-2,0,-1) :
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.derive_sigmoid(A[layer])
            D.append(delta)

        D = D[::-1]

        for layer in np.arange(0,len(self.W)):
            self.W[layer]  += -self.alpha * A[layer].T.dot(D[layer])
            

    # Fonction de prediction
    def predict(self,X, addBias=True):
        
        p = np.atleast_2d(X) # transformation de X 

        if addBias:
            p = np.c_[p,np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p,self.W[layer]))

        return p


    # La fonction Loss
    def calculate_loss(self,X,targets):

        targets = np.atleast_2d(targets)

        predictions = self.predict(X, addBias=False)

        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss
