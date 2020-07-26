import numpy as np
# sigmoid function
def nonlin(x):
    return 1 / (1 + (np.e)**(-0.1*x))


X1 = 2*np.random.random((60,2))-1
W1 = np.array([[5,6],
               [6,7]])
l1 = nonlin(np.dot(X1, W1))
W2 =np.array([[6,7],
              [5,9]])
l2 = nonlin(np.dot(l1, W2))
W3 =np.array([5,7]).T
Y1= np.dot(l2, W3)
print(X1,Y1)
np.savetxt('Xp1.txt',X1,delimiter=",")
np.savetxt('Yp1.txt',Y1,delimiter=",")