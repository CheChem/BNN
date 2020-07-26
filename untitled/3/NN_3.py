import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import initializers
from keras.callbacks import Callback

def nonlin(x):
    return 1 / (1 + (np.e)**(-0.0001*x))
x_train=np.loadtxt('X1.txt',delimiter=",")
y_train=np.loadtxt('Y1.txt',delimiter=",")

W_=np.loadtxt('W.txt',delimiter=",")
W=W_.tolist()
for i in range(100):
    model = Sequential()
    model.add(Dense(input_dim=2, units=2, use_bias=False, activation=nonlin, kernel_initializer=initializers.random_normal(mean=-300,stddev=10)))
    model.add(Dense(input_dim=2, units=2, use_bias=False, activation=nonlin, kernel_initializer=initializers.random_normal(mean=18,stddev=10)))
    model.add(Dense(input_dim=2, units=1, use_bias=False, activation=None, kernel_initializer=initializers.random_normal(mean=0.5,stddev=10)))

    model.compile(loss='mse',optimizer=SGD(lr=0.05))
    model.fit(x_train,y_train,batch_size=10,epochs=50)
    W0=Dense.get_weights(model)
    W1=np.hstack(W0)
    W2=W1.reshape([1,10],order='F')
    W.append(W2)
WW=np.vstack(W)
np.savetxt('W.txt',WW,delimiter=",")