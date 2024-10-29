import matplotlib.pylab as plt
import numpy as np
from reglog import LogisticRegressionGD
from sklearn import datasets
from sklearn.model_selection import train_test_split

from util.plotka import plot_decision_regions


class MultiClassRegLog:
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.logistic_regression_gds = None
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        self.logistic_regression_gds = [LogisticRegressionGD(eta=self.eta, n_iter=self.n_iter, random_state=self.random_state) for _ in range(len(np.unique(y)))]
        for logistic_regression_gd, target_class in zip(self.logistic_regression_gds, np.unique(y)):
            y_binary = np.where(y == target_class, 1, 0)
            logistic_regression_gd.fit(x, y_binary)


    def net_input(self, x):
        return np.array([logistic_regression_gd.net_input(x) for logistic_regression_gd in self.logistic_regression_gds])

    @staticmethod
    def activation(z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, x):
        return np.argmax(self.net_input(x), axis=0)

    def class_probability(self, x, class_index):
        return self.activation(self.logistic_regression_gds[class_index].net_input(x))


def main():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)


    mcr = MultiClassRegLog(eta=0.2, n_iter=1000, random_state=1)
    mcr.fit(x_train, y_train)

    plot_decision_regions(x, y, classifier=mcr)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()

    test_class = 0
    print(f'Prawdopodobieństwo klasy {test_class} dla pierwszych 3 próbek zbioru testowego:')
    print(mcr.class_probability(x_test[:3], test_class))
    print(x_test[:3])
    print(y_test[:3])


if __name__ == '__main__':
    main()
