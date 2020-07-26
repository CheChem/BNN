import numpy as np
# sigmoid function
def nonlin(x, deriv=False):
    return 1 / (1 + np.exp(-0.1*x))

x_train_ = np.loadtxt('Xp1.txt', delimiter=",")
X = x_train_[50:60, :]

for i in range(1000):
    BNN_Y = np.loadtxt('BNN_Y.txt', delimiter=",")

    w1=np.random.normal(loc=5.023201, scale=0.525277, size=None)
    w2=np.random.normal(loc=5.8567324, scale=0.54594266, size=None)
    w3=np.random.normal(loc=5.0230474, scale=0.5201124, size=None)
    w4=np.random.normal(loc=5.856415, scale=0.5327341, size=None)
    w5=np.random.normal(loc=5.4939246, scale=0.18911456, size=None)
    w6=np.random.normal(loc=5.4926085, scale=0.18881862, size=None)
    w7=np.random.normal(loc=5.6506863, scale=0.1933103, size=None)
    w8=np.random.normal(loc=5.6509953, scale=0.1897484, size=None)
    w9=np.random.normal(loc=6.2903233, scale=6.2903233, size=None)
    w10=np.random.normal(loc=6.2856874, scale=0.03509027, size=None)

    W1 = np.array([[w1,w3],
                   [w2,w4]])
    l1 = nonlin(np.dot(X, W1))
    W2 =np.array([[w5,w7],
                  [w6,w8]])
    l2 = nonlin(np.dot(l1, W2))
    W3 =np.array([w9,w10])
    Y = np.dot(l2, W3)

    Y = Y.reshape([1, 10], order='F')
    BNN_Y = np.vstack([BNN_Y, Y])

    np.savetxt('BNN_Y.txt',BNN_Y,delimiter=",")
