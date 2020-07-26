import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

x = np.arange(0, 10, 0.01)

def f(x):
    return (np.e) ** (-(x-6.2903233) ** 2 / (2* 6.2903233 ** 2)) / (2 * np.pi * 6.2903233 ** 2) ** 0.5
def g(x):
    return (np.e) ** (-(x-6.012008) ** 2 / (2* 0.03917574 ** 2)) / (2 * np.pi * 0.03917574 ** 2) ** 0.5
def h(x):
    return (np.e) ** (-(x-5.972986) ** 2 / (2*0.03878856**2)) / (2 * np.pi * 0.03878856 ** 2) ** 0.5


data1 = np.loadtxt('Wp.txt',delimiter=",")
data=data1[:, 8]

plt.plot(x, f(x), label='BNN')
plt.plot(x, g(x), label='BNN-0.1')
plt.plot(x, h(x), label='BNN-0.01')
plt.plot([5,5],[0,4],label='True')

plt.hist(data, bins=100, density=True, histtype='stepfilled', label='NN')
plt.legend(loc='upper left')

plt.title('w5')
plt.show()







