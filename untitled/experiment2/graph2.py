import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-4, 4, 0.01)
def f(x):
    return (np.e) ** (-(x+0.48773092) ** 2 / (2 * 0.14178446 ** 2)) / (2 * np.pi * 0.14178446 ** 2) ** 0.5
x_ = np.arange (-4, 0, 0.01)
def g(x):
    return 4*(np.e) ** (4*x)

data1 = np.loadtxt('W.txt',delimiter=",")
data = data1[:, 11]
mean = np.mean(data)
std = np.std(data)
print(mean, std)

plt.plot(x, f(x), label='BNN')
plt.plot(x_, g(x_), label='True')
plt.hist(data, bins=100, density=True, histtype='stepfilled', label='NN')
plt.legend(loc='upper left')

plt.title('w12')
plt.show()







