import numpy as np
# DATA
X_train = np.loadtxt('x_data.txt', delimiter=",")
y_train = np.loadtxt('y_data.txt', delimiter=",")
train_data = np.stack((X_train, y_train)).T
print(train_data)