import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

x = np.arange(-2, 4, 0.01)
def f(x):
    return (np.e) ** (-(x+4.4911794e-05) ** 2 / (2 * 1.0000005 **2)) / (2 * np.pi * 1.0000005 ** 2) ** 0.5

data1 = np.loadtxt('W.txt',delimiter=",")
data=data1[:, 1]
mean = np.mean(data)
std = np.std(data)
print(mean,std)

plt.plot(x, f(x), label='BNN')
plt.hist(data, bins=100, density=True, histtype='stepfilled', label='NN')
plt.legend(loc='upper left')

plt.title('w10')
plt.show()







