import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import initializers
from keras.callbacks import Callback

x_train=np.loadtxt('X1.txt',delimiter=",")
y_train=np.loadtxt('Y1.txt',delimiter=",")

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

W_=np.loadtxt('W2.txt',delimiter=",")
W=W_.tolist()
for i in range(10):
    model = Sequential()
    model.add(Dense(input_dim=2, units=2, use_bias=False, activation='sigmoid', kernel_initializer=initializers.random_normal(mean=0, stddev=1)))
    model.add(Dense(input_dim=2, units=2, use_bias=False, activation='sigmoid', kernel_initializer=initializers.random_normal(mean=0, stddev=1)))
    model.add(Dense(input_dim=2, units=1, use_bias=False, activation=None, kernel_initializer=initializers.random_normal(mean=0, stddev=1)))

    model.compile(loss='mse',optimizer=SGD(lr=0.2))
    callbacks = EarlyStoppingByLoss(monitor='loss', value=0.0001, verbose=0)
    model.fit(x_train,y_train,batch_size=10,epochs=500,callbacks=[callbacks])
    W0=Dense.get_weights(model)
    W1=np.hstack(W0)
    W2=W1.reshape([1,10],order='F')
    W.append(W2)
WW=np.vstack(W)
print(WW)
np.savetxt('W2.txt',WW,delimiter=",")