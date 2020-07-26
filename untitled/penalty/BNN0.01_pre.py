import numpy as np
# sigmoid function
def nonlin(x, deriv=False):
    return 1 / (1 + np.exp(-0.1*x))

x_train_ = np.loadtxt('Xp1.txt', delimiter=",")
X = x_train_[50:60, :]

for i in range(999):
    BNN_Y = np.loadtxt('BNN_Y3.txt', delimiter=",")

    w1=np.random.normal(loc=5.4141917, scale=0.05654705, size=None)
    w2=np.random.normal(loc=6.365416, scale=0.05843281, size=None)
    w3=np.random.normal(loc=5.3633294, scale=0.05911443, size=None)
    w4=np.random.normal(loc=6.3628283, scale=0.05819137, size=None)
    w5=np.random.normal(loc=6.8540287, scale=0.09385913, size=None)
    w6=np.random.normal(loc=6.8489866, scale=0.10083112, size=None)
    w7=np.random.normal(loc=7.3748293, scale=0.1046375, size=None)
    w8=np.random.normal(loc=7.3736115, scale=0.10533288, size=None)
    w9=np.random.normal(loc=5.972986, scale=0.03878856, size=None)
    w10=np.random.normal(loc=5.955712, scale=0.03013417, size=None)

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

    np.savetxt('BNN_Y3.txt',BNN_Y,delimiter=",")
