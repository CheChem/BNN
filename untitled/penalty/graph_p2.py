import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter



data1 = np.loadtxt('Pre.txt',delimiter=",")
data1=data1[:, 3]
data2 = np.loadtxt('BNN_Y4.txt',delimiter=",")
data2=data2[:, 3]

plt.plot([8.531501839307304991,8.531501839307304991],[0,15],label='True')

plt.hist(data1, bins=50, density=True, alpha=0.8, histtype='stepfilled', label='NN')
plt.hist(data2, bins=50, density=True, alpha=0.8, histtype='stepfilled', label='BNN')
plt.legend(loc='upper left')

plt.title('Y2-Y1')
plt.show()







