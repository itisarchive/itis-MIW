import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from perceptron import Perceptron
from util.plotka import plot_decision_regions


class MultiClassPerceptron:

    def __init__(self, eta=0.01, n_iter=10):
        self.perceptrons = None
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        self.perceptrons = [Perceptron(eta=self.eta, n_iter=self.n_iter) for _ in range(len(np.unique(y)))]

        for perceptron, target_class in zip(self.perceptrons, np.unique(y)):
            y_binary = np.where(y == target_class, 1, -1)
            perceptron.fit(x, y_binary)

    def net_input(self, x):
        return np.array([perceptron.net_input(x) for perceptron in self.perceptrons])

    def predict(self, x):
        return np.argmax(self.net_input(x), axis=0)


def main():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

    mcp = MultiClassPerceptron(eta=0.2, n_iter=1000)
    mcp.fit(x_train, y_train)

    plot_decision_regions(x, y, classifier=mcp)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
