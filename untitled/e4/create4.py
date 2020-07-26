import numpy as np
# tanh function
def nonlin(x):
    return 2 / (1 + np.exp(-0.25*x))-1

W1 = np.random.normal(loc=0,scale=1,size=(50,1))
W2 = np.random.uniform(low=-0.5, high=0.5, size=(50,1))
W3 = np.random.exponential(1, (50,1))
W4 = np.random.normal(loc=0.5,scale=0.5,size=(50,1))
W5 = np.random.uniform(low=0, high=1, size=(50,1))
W6 = -np.random.exponential(0.25, (50,1))
W7 = np.random.normal(loc=-0.3,scale=1,size=(50,1))
W8 = np.random.uniform(low=-1, high=0.5, size=(50,1))
W9 = np.random.normal(loc=-0.7,scale=0.5,size=(50,1))
W10 = np.random.exponential(1/3, (50,1))
W11 = np.random.uniform(low=-0.3, high=0.6, size=(50,1))
W12 = -np.random.exponential(0.25, (50,1))
W13 = np.random.normal(loc=0.4,scale=0.5,size=(50,1))
W14 = np.random.uniform(low=-0.9, high=-0.1, size=(50,1))


X = 20*np.random.random((50,2))-10
Y=[]
for i in range(50):
    X_i = X[i,:]
    W_1 = np.array([[W1[i, 0], W3[i, 0],W5[i, 0]],
                    [W2[i, 0], W4[i, 0],W6[i, 0]]])
    l1 = nonlin(np.dot(X_i, W_1))
    W_2 = np.array([[W7[i, 0], W10[i, 0]],
                    [W8[i, 0], W11[i, 0]],
                    [W9[i, 0], W12[i, 0]]
                    ])
    l2 = nonlin(np.dot(l1, W_2))
    W_3 = np.array([W13[i, 0], W14[i, 0]]).T
    Y_i = np.dot(l2, W_3)
    Y.append(Y_i)
print(X,Y)
np.savetxt('X.txt', X, delimiter=",")
np.savetxt('Y.txt', Y, delimiter=",")
np.savetxt('W1.txt', W1, delimiter=",")
np.savetxt('W2.txt', W2, delimiter=",")
np.savetxt('W3.txt', W3, delimiter=",")
np.savetxt('W4.txt', W4, delimiter=",")
np.savetxt('W5.txt', W5, delimiter=",")
np.savetxt('W6.txt', W6, delimiter=",")
np.savetxt('W7.txt', W7, delimiter=",")
np.savetxt('W8.txt', W8, delimiter=",")
np.savetxt('W9.txt', W9, delimiter=",")
np.savetxt('W10.txt', W10, delimiter=",")
np.savetxt('W11.txt', W11, delimiter=",")
np.savetxt('W12.txt', W12, delimiter=",")
np.savetxt('W13.txt', W13, delimiter=",")
np.savetxt('W14.txt', W14, delimiter=",")







