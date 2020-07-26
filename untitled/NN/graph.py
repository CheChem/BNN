import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

x = np.arange(-4, 4, 0.01)
def f(x):
    return (np.e) ** (-(x-0.03751191) ** 2 / (2*0.03111862**2)) / (2 * np.pi *0.03111862 ** 2) ** 0.5

data1 = np.loadtxt('W2.txt',delimiter=",")
data=data1[:, 9]
mean = np.mean(data)
std = np.std(data)
print(mean,std)

plt.plot(x, f(x), label='BNN')
plt.hist(data, bins=100, density=True, histtype='stepfilled', label='NN')
plt.legend(loc='upper left')

plt.title('w10')
plt.show()







