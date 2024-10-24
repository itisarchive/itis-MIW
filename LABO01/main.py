import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt("dane1.txt")

x = data[:, [0]]
y = data[:, [1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

F1 = np.hstack([x_train, np.ones(x_train.shape)])
W1 = np.linalg.pinv(F1) @ y_train

F2 = np.hstack([x_train ** 3, x_train ** 2, x_train, np.ones(x_train.shape)])
W2 = np.linalg.pinv(F2) @ y_train

F1_test = np.hstack([x_test, np.ones(x_test.shape)])
y1_pred = F1_test @ W1

F2_test = np.hstack([x_test ** 3, x_test ** 2, x_test, np.ones(x_test.shape)])
y2_pred = F2_test @ W2

mse1 = np.mean((y_test - y1_pred) ** 2)
mse2 = np.mean((y_test - y2_pred) ** 2)

print(f'MSE Model 1: {mse1:.4f}')
print(f'MSE Model 2: {mse2:.4f}')

plt.scatter(x, y, color='blue', label='Dane rzeczywiste')
plt.scatter(x_test, y1_pred, color='red', label='Predykcje Modelu 1')
plt.scatter(x_test, y2_pred, color='green', label='Predykcje Modelu 2')

x_plot = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)

F1_plot = np.hstack([x_plot, np.ones(x_plot.shape)])
y_range_pred1 = F1_plot @ W1
plt.plot(x_plot, y_range_pred1, color='red', label='Model 1 - regresja liniowa')

F2_plot = np.hstack([x_plot ** 3, x_plot ** 2, x_plot, np.ones(x_plot.shape)])
y_range_pred2 = F2_plot @ W2
plt.plot(x_plot, y_range_pred2, color='green', label='Model 2 - regresja wielomianowa')

plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Porównanie modeli')
plt.show()
