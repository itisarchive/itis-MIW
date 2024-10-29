import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt("../dane/dane2.txt")
x = data[:, [0]]
y = data[:, [1]]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

P = (x[~np.isin(x, x_test)]).reshape(1, -1)
T = (y[~np.isin(y, y_test)]).reshape(1, -1)
# P = np.arange(-2, 2.1, 0.1).reshape(1, -1)
# T = P ** 2 + (np.random.rand(P.size) - 0.5).reshape(1, -1)

S1 = 100
W1 = np.random.rand(S1, 1) - 0.5
B1 = np.random.rand(S1, 1) - 0.5
W2 = np.random.rand(1, S1) - 0.5
B2 = np.random.rand(1, 1) - 0.5
lr = 0.001
epochs = 100


def activate(v):
    return np.maximum(0, v)
    # return np.tanh(x)


def activation_derivative(v):
    return np.where(v > 0, 1, 0)
    # return 1 - x ** 2


def fit_batch():
    global W1, B1, W2, B2

    for epoch in range(epochs):
        A1 = activate(W1 @ P + B1 @ np.ones((1, P.size)))
        A2 = W2 @ A1 + B2

        E2 = T - A2
        E1 = W2.T @ E2

        dW2 = lr * E2 @ A1.T
        dB2 = lr * np.sum(E2)
        dW1 = lr * (activation_derivative(A1) * E1) @ P.T
        dB1 = lr * np.sum(activation_derivative(A1) * E1, axis=1, keepdims=True)

        W2 += dW2
        B2 += dB2
        W1 += dW1
        B1 += dB1

        if epoch % 1 == 0:
            plt.clf()
            plt.plot(P.ravel(), T.ravel(), 'r*', label="Target values")
            plt.plot(P.ravel(), A2.ravel(), label="Network output")
            plt.legend(["Target values", "Network output"])
            plt.show()


def fit_online():
    global W1, B1, W2, B2

    for epoch in range(epochs):
        for i in range(P.shape[1]):
            p = P[:, i:i + 1]
            t = T[:, i:i + 1]

            A1 = activate(W1 @ p + B1)
            A2 = W2 @ A1 + B2

            E2 = t - A2
            E1 = W2.T @ E2

            dW2 = lr * E2 @ A1.T
            dB2 = lr * E2
            dW1 = lr * (activation_derivative(A1) * E1) @ p.T
            dB1 = lr * (activation_derivative(A1) * E1)

            W2 += dW2
            B2 += dB2
            W1 += dW1
            B1 += dB1

        if epoch % 1 == 0:
            plt.clf()
            A1_all = activate(W1 @ P + B1 @ np.ones((1, P.shape[1])))
            A2_all = W2 @ A1_all + B2
            plt.plot(P.ravel(), T.ravel(), 'r*', label="Target values")
            plt.plot(P.ravel(), A2_all.ravel(), label="Network output")
            plt.legend(["Target values", "Network output"])
            plt.pause(0.05)


fit_online()

output = W2 @ activate(W1 @ x_test.T + B1 @ np.ones((1, x_test.size))) + B2
plt.scatter(x_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(x_test, y_test, color='green', label='Dane testowe')
plt.scatter(x_test, output, color='red', label='Dane przewidziane')
plt.plot(x, (W2 @ activate(W1 @ x.T + B1 @ np.ones((1, x.size))) + B2).T, color='red', label='Sieć')
plt.legend()
plt.show()
