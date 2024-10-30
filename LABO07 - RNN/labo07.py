import os

import numpy as np
import requests
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import Sequential, layers


def load_exchange(company="IBM", series="4. close", look_back=3, limit=40):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={company}&apikey=demo"
    response = requests.get(url)
    data = response.json()["Time Series (Daily)"]

    labels = np.array([key for key in data.keys()][:limit][::-1])
    prices = np.array([float(value[series]) for value in data.values()])[:limit + look_back][::-1]
    volume = np.array([float(value["5. volume"]) for value in data.values()])[:limit][::-1]

    x = np.array([prices[i - look_back:i][::-1] for i in range(look_back, len(prices))])
    y = prices[look_back:]

    return np.array(x), np.array(y), volume, labels


def load_file(file_path="../dane/dane2.txt", look_back=3, limit=40):
    data = np.loadtxt(file_path)

    y = data[:, 1]
    x = np.array([y[i - look_back:i][::-1] for i in range(len(y) - limit, len(y))])
    labels = data[:, 0][len(y) - limit:len(y)]
    y = y[len(y) - limit:len(y)]

    return x, y, labels, labels


def autoregressive_model(x_train, y_train, x_test):
    x_train_aug = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    coef = np.linalg.pinv(x_train_aug) @ y_train

    x_test_aug = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
    predictions = x_test_aug @ coef
    return predictions, coef


def rnn_model(x_train, y_train, x_test, epochs=50, batch_size=16):
    x_train_rnn = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test_rnn = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    model = Sequential([
        layers.Input(shape=(x_train_rnn.shape[1], 1)),
        layers.LSTM(100),
        # layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train_rnn, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    predictions = model.predict(x_test_rnn).flatten()
    return predictions, model


def combined_model(x_train, volume_train, y_train, x_test, volume_test, epochs=50, batch_size=16):
    x_train_combined = np.hstack([x_train, volume_train.reshape(-1, 1)])
    x_test_combined = np.hstack([x_test, volume_test.reshape(-1, 1)])

    model = Sequential([
        layers.Input(shape=(x_train_combined.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train_combined, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    predictions = model.predict(x_test_combined).flatten()
    return predictions


def main():
    x, y, extra_dim, labels = load_exchange()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

    ar_predictions, ar_coef = autoregressive_model(x_train, y_train, x_test)
    ar_mse = mean_squared_error(y_test, ar_predictions)

    rnn_predictions, _ = rnn_model(x_train, y_train, x_test, epochs=100, batch_size=1)
    rnn_mse = mean_squared_error(y_test, rnn_predictions)

    volume_train, volume_test = train_test_split(extra_dim, test_size=0.3, shuffle=False)
    combined_predictions = combined_model(x_train, volume_train, y_train, x_test, volume_test, epochs=100,
                                          batch_size=1)
    combined_mse = mean_squared_error(y_test, combined_predictions)

    plt.figure(figsize=(14, 7))
    plt.plot(labels, y, label='Real Values', color='black')
    plt.plot(labels[len(labels) - len(ar_predictions):], ar_predictions, label='AR Model Predictions', color='red')
    plt.plot(labels[len(labels) - len(rnn_predictions):], rnn_predictions, label='RNN Model Predictions', color='blue')
    plt.plot(labels[len(labels) - len(combined_predictions):], combined_predictions, label='Combined Model Predictions',
             color='purple')
    plt.xticks(labels, rotation=90)
    plt.legend()
    plt.show()

    print(f"AR Model MSE: {ar_mse}")
    print(f"RNN Model MSE: {rnn_mse}")
    print(f"Combined Model (with Volume) MSE: {combined_mse}")


if __name__ == '__main__':
    main()
