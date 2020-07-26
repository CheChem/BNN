import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-4, 4, 0.01)
def f(x):
    return (np.e) ** (-(x+0.04030497) ** 2 / (2 * 0.02825426 ** 2)) / (2 * np.pi * 0.02825426 ** 2) ** 0.5

def g(x):
    return (np.e) ** (-(x-0.4) ** 2 / (2 * 0.5 ** 2)) / (2 * np.pi * 0.5 ** 2) ** 0.5

data1 = np.loadtxt('W_3.txt',delimiter=",")
data = data1[:, 10]
mean = np.mean(data)
std = np.std(data)
print(mean, std)

plt.plot(x, f(x), label='BNN')
plt.plot(x, g(x), label='True')
plt.hist(data, bins=100, density=True, histtype='stepfilled', label='NN')
plt.legend(loc='upper left')

plt.title('w13')
plt.show()







