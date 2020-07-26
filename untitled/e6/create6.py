import numpy as np
# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
X1 = 2*np.random.random((50,2))-1
W1 = np.array([[0.01,-0.03],
               [0.04,0.02]])
l1 = nonlin(np.dot(X1, W1))
W2 =np.array([[0.05,0.04],
              [-0.05,0.02]])
l2 = nonlin(np.dot(l1, W2))
W3 =np.array([0.03,0.04]).T
Y1= np.dot(l2, W3)
print(X1,Y1)
np.savetxt('X1.txt',X1,delimiter=",")
np.savetxt('Y1.txt',Y1,delimiter=",")