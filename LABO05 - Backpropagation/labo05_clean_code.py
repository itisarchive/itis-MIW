import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt("../dane/dane2.txt")
x = data[:, [0]]
y = data[:, [1]]
x_train, x_test, y_train, y_test = train_test_split(data[:, [0]], data[:, [1]], random_state=1)

input_values = (x[~np.isin(x, x_test)]).reshape(1, -1)
output_values = (y[~np.isin(y, y_test)]).reshape(1, -1)
# input_values = np.arange(-2, 2.1, 0.1).reshape(1, -1)
# output_values = input_values ** 2 + (np.random.rand(input_values.size) - 0.5).reshape(1, -1)

hidden_layer_size = 100
wieghts_hidden = np.random.rand(hidden_layer_size, 1) - 0.5
bias_hidden = np.random.rand(hidden_layer_size, 1) - 0.5
wieghts_output = np.random.rand(1, hidden_layer_size) - 0.5
bias_output = np.random.rand(1, 1) - 0.5
learning_rate = 0.001


def activate(vector):
    return np.maximum(0, vector)
    # return np.tanh(vector)


def activation_derivative(vector):
    return np.where(vector > 0, 1, 0)
    # return 1 - vector ** 2


epochs = 100
for epoch in range(epochs):
    hidden_layer_input = wieghts_hidden @ input_values + bias_hidden @ np.ones((1, input_values.size))
    hidden_layer_output = activate(hidden_layer_input)
    network_output = wieghts_output @ hidden_layer_output + bias_output

    output_error = output_values - network_output
    hidden_error = wieghts_output.T @ output_error

    weights_hidden_output_gradient = learning_rate * output_error @ hidden_layer_output.T
    bias_output_gradient = learning_rate * np.sum(output_error)

    weights_input_hidden_gradient = learning_rate * (
                activation_derivative(hidden_layer_output) * hidden_error) @ input_values.T
    bias_hidden_gradient = learning_rate * np.sum(activation_derivative(hidden_layer_output) * hidden_error, axis=1,
                                                  keepdims=True)

    wieghts_output += weights_hidden_output_gradient
    bias_output += bias_output_gradient
    wieghts_hidden += weights_input_hidden_gradient
    bias_hidden += bias_hidden_gradient

    if epoch % 1 == 0:
        plt.clf()
        plt.plot(input_values.ravel(), output_values.ravel(), 'r*', label="Target values")
        plt.plot(input_values.ravel(), network_output.ravel(), label="Network output")
        plt.legend(["Target values", "Network output"])
        plt.show()

output = wieghts_output @ activate(wieghts_hidden @ x_test.T + bias_hidden @ np.ones((1, x_test.size))) + bias_output
plt.scatter(x_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(x_test, y_test, color='green', label='Dane testowe')
plt.scatter(x_test, output, color='red', label='Dane przewidziane')
plt.plot(x, (wieghts_output @ activate(wieghts_hidden @ x.T + bias_hidden @ np.ones((1, x.size))) + bias_output).T,
         color='red', label='Sieć')
plt.legend()
plt.show()
