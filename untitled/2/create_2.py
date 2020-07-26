import numpy as np
# sigmoid function
def nonlin(x):
    return 2 / (1 + np.exp(-0.0001*x))-1


X1 = 2*np.random.random((50,2))-1
W1 = np.array([[10,-100],
               [237,3]])
l1 = nonlin(np.dot(X1, W1))
W2 =np.array([[-78,0.5],
              [321,1000]])
l2 = nonlin(np.dot(l1, W2))
W3 =np.array([12,-378]).T
Y1= np.dot(l2, W3)
print(X1,Y1)
np.savetxt('X1.txt',X1,delimiter=",")
np.savetxt('Y1.txt',Y1,delimiter=",")