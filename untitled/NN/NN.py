import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import initializers

x_train=np.loadtxt('X1.txt',delimiter=",")
y_train=np.loadtxt('Y1.txt',delimiter=",")



for i in range(3):

    W = np.loadtxt('W1.txt', delimiter=",")
    model = Sequential()
    model.add(Dense(input_dim=2, units=2, use_bias=False, activation='sigmoid', kernel_initializer=initializers.random_normal(mean=0,stddev=1)))
    model.add(Dense(input_dim=2, units=2, use_bias=False, activation='sigmoid', kernel_initializer=initializers.random_normal(mean=0,stddev=1)))
    model.add(Dense(input_dim=2, units=1, use_bias=False, activation=None, kernel_initializer=initializers.random_normal(mean=0,stddev=1)))

    model.compile(loss='mse',optimizer=SGD(lr=0.2))
    model.fit(x_train,y_train,batch_size=10,epochs=200)

    W0 = Dense.get_weights(model)
    W1 = np.hstack(W0)
    W2 = W1.reshape([1,10],order='F')
    WW = np.vstack([W,W2])

    print(WW)
    np.savetxt('W1.txt',WW,delimiter=",")