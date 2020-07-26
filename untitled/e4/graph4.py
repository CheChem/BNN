import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-4, 4, 0.01)
def f(x):
    return (np.e) ** (-(x+0.00788786) ** 2 / (2 * 0.9933789 ** 2)) / (2 * np.pi * 0.9933789 ** 2) ** 0.5

def g(x):
    return (np.e) ** (-x ** 2 / (2 * 1 ** 2)) / (2 * np.pi * 1 ** 2) ** 0.5

data1 = np.loadtxt('W_4.txt',delimiter=",")
data = data1[:, 0]
mean = np.mean(data)
std = np.std(data)
print(mean, std)

plt.plot(x, f(x), label='BNN')
plt.plot(x, g(x), label='True')
plt.hist(data, bins=100, density=True, histtype='stepfilled', label='NN')
plt.legend(loc='upper left')

plt.title('w1')
plt.show()







