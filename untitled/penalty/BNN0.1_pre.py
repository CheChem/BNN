import numpy as np
# sigmoid function
def nonlin(x, deriv=False):
    return 1 / (1 + np.exp(-0.1*x))

x_train_ = np.loadtxt('Xp1.txt', delimiter=",")
X = x_train_[50:60, :]

for i in range(999):
    BNN_Y = np.loadtxt('BNN_Y2.txt', delimiter=",")

    w1=np.random.normal(loc=5.3839583, scale=0.17471199, size=None)
    w2=np.random.normal(loc=6.3465853, scale=0.18440042, size=None)
    w3=np.random.normal(loc=5.3777027, scale=0.17394714, size=None)
    w4=np.random.normal(loc=6.3453283, scale=0.18393835, size=None)
    w5=np.random.normal(loc=6.677633, scale=0.10425177, size=None)
    w6=np.random.normal(loc=6.6733446, scale=0.11020287, size=None)
    w7=np.random.normal(loc=7.1410317, scale=0.11410772, size=None)
    w8=np.random.normal(loc=7.140176, scale=0.11393753, size=None)
    w9=np.random.normal(loc=6.012008, scale=0.03917574, size=None)
    w10=np.random.normal(loc=5.996803, scale=0.03058068, size=None)

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

    np.savetxt('BNN_Y2.txt',BNN_Y,delimiter=",")
