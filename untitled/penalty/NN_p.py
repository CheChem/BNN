import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import initializers
from keras import regularizers
from keras.callbacks import Callback

x_train_ = np.loadtxt('Xp1.txt', delimiter=",")
x_train = x_train_[0:50, :]
y_train_ = np.loadtxt('Yp1.txt', delimiter=",")
y_train = y_train_[0:50]

class EarlyStoppingByLoss(Callback):
    def __init__(self, monitor='loss', value=0.0001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def nonlin(x):
    return 1 / (1 + (np.e)**(-0.1*x))

for i in range(25):
    W = np.loadtxt('Wp.txt',delimiter=",")
    Pre = np.loadtxt('Pre.txt',delimiter=",")

    model = Sequential()
    model.add(Dense(input_dim=2, units=2, use_bias=False, activation=nonlin, kernel_initializer=initializers.random_uniform(minval=0, maxval=20, seed=None), kernel_regularizer=regularizers.l2(0)))
    model.add(Dense(input_dim=2, units=2, use_bias=False, activation=nonlin, kernel_initializer=initializers.random_uniform(minval=0, maxval=20, seed=None), kernel_regularizer=regularizers.l2(0)))
    model.add(Dense(input_dim=2, units=1, use_bias=False, activation=None, kernel_initializer=initializers.random_uniform(minval=0, maxval=20, seed=None), kernel_regularizer=regularizers.l2(0)))

    model.compile(loss='mse', optimizer=SGD(lr=0.1))
    callbacks = EarlyStoppingByLoss(monitor='loss', value=0.0001, verbose=0)
    model.fit(x_train, y_train, batch_size=5, epochs=1000,callbacks=[callbacks])

    W0 = Dense.get_weights(model)
    W1 = np.hstack(W0)
    W2 = W1.reshape([1, 10], order='F')
    WW = np.vstack([W, W2])

    X = x_train_[50:60, :]
    pre_y = model.predict(X.reshape(10, 2), verbose=1)
    pre = pre_y.reshape([1, 10], order='F')
    Pre = np.vstack([Pre, pre])

    np.savetxt('Wp.txt',WW,delimiter=",")
    np.savetxt('Pre.txt',Pre,delimiter=",")



