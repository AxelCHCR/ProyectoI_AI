# biblioteca para abrir el dataset
import pandas as pd
# biblioteca que permite manejo de vectores
import numpy as np
# biblioteca que permite extraer el porcentaje deseado para cada set
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('ProyectoI_AI\Datasets\diabetes.csv')
# incluir todos los features
X = dataset.iloc[:, 0:8]
# incluir solo los labels
y = dataset.iloc[:, 8]
# separar dataset en training y testing (80-20)
X_Training, X_Testing, Y_Training, Y_Testing = train_test_split( X.values, y.values, test_size=0.2, random_state=42)


def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr, n_iters, meta):
        self.learning_rate = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.meta = meta

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=self.meta else 1 for y in y_pred]
        return class_pred
    
    def accuracy(self, y_pred, y_test):
        return np.sum(y_pred==y_test)/len(y_test)
    
    def setLR(self,lr):
        self.lr = lr


for i in range(1,10):
    print("LR:", i*0.1)
    logistic = LogisticRegression(i*0.1,10000, 0.5)
    logistic.fit(X_Training,Y_Training)

    prediccion = logistic.predict(X_Testing)
    print("Prediccion:", prediccion)
    acc = logistic.accuracy(prediccion, Y_Testing)
    print("Accuracy:", acc)
    print("")