import numpy as np
import pandas as pd
import sys

class Perceptron(object):

    def __init__(self, n_inputs, max_iter = 100000):
        # np.random.seed(1)
        self.max_iter = max_iter
        self.w = 2 * np.random.random((n_inputs, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y).T
        for i in range(self.max_iter):

            out = self.predict(x)
            error = y - out
            adjust = np.dot(x.T, error * self.sigmoid_derivative(out))
            self.w += adjust

    def predict(self, x):
        x = np.array(x)
        x = x.astype(float)  
        return self.sigmoid(np.dot(x, self.w))

if __name__ == "__main__":

    neural = Perceptron(n_inputs = 2)

    print("W random")
    print(neural.w)
    
    x_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y_train = np.array([[0, 1, 1, 0]])

    neural.fit(x_train, y_train)

    print("New w")
    print(neural.w)

    print("Results")
    for i in x_train:
        print(neural.predict(i))