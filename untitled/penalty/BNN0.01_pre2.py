import numpy as np
# sigmoid function
def nonlin(x, deriv=False):
    return 1 / (1 + np.exp(-0.1*x))

x_train_ = np.loadtxt('Xp1.txt', delimiter=",")
X = x_train_[50:60, :]

for i in range(999):
    BNN_Y = np.loadtxt('BNN_Y4.txt', delimiter=",")

    w1=np.random.normal(loc=4.684586, scale=0.11556519, size=None)
    w2=np.random.normal(loc=5.5049167, scale=0.1895491, size=None)
    w3=np.random.normal(loc=4.585351, scale=0.1154027, size=None)
    w4=np.random.normal(loc=5.5162735, scale=0.21438992, size=None)
    w5=np.random.normal(loc=10.362337, scale=0.35692176, size=None)
    w6=np.random.normal(loc=10.356963, scale=0.35902855, size=None)
    w7=np.random.normal(loc=10.154272, scale=0.28616515, size=None)
    w8=np.random.normal(loc=10.147759, scale=0.21487275, size=None)
    w9=np.random.normal(loc=5.4187126, scale=0.05807068, size=None)
    w10=np.random.normal(loc=5.460766, scale=0.05446154, size=None)

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

    np.savetxt('BNN_Y4.txt',BNN_Y,delimiter=",")
