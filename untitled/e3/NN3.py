import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import initializers

def nonlin(x):
    return 1 / (1 + (np.e)**(-0.25*x))

x_train = np.loadtxt('X.txt', delimiter=",")
y_train = np.loadtxt('Y.txt', delimiter=",")

W_= np.loadtxt('W_3.txt',delimiter=",")
W=W_.tolist()
for i in range(100):
    model = Sequential()
    model.add(Dense(input_dim=2, units=3, use_bias=False, activation=nonlin, kernel_initializer=initializers.random_normal(mean=0, stddev=1)))
    model.add(Dense(input_dim=3, units=2, use_bias=False, activation=nonlin, kernel_initializer=initializers.random_normal(mean=0, stddev=1)))
    model.add(Dense(input_dim=2, units=1, use_bias=False, activation=None, kernel_initializer=initializers.random_normal(mean=0, stddev=1)))

    model.compile(loss='mse', optimizer=SGD(lr=0.01))
    model.fit(x_train, y_train, batch_size=10, epochs=300)

    W0 = Dense.get_weights(model)
    W1 = W0[0].reshape([1, 6], order='F')
    W2 = W0[1].reshape([1, 6], order='F')
    W3 = W0[2].reshape([1, 2], order='F')
    WW = np.hstack((W1, W2, W3))
    W.append(WW)

WW = np.vstack(W)
np.savetxt('W_3.txt', WW, delimiter=",")



















