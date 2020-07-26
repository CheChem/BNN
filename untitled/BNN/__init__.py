import torch
import torch.nn as nn
import torch.nn.functional as F  # 激励函数都在这
from torch.autograd import Variable
import numpy as np
dtype = torch.float
device = torch.device("cpu")

x=np.loadtxt('X1.txt',delimiter=",")
y=np.loadtxt('Y1.txt',delimiter=",")
x = torch.from_numpy(x)        # x data (tensor), shape=(50, 2)
y = torch.from_numpy(y)        # y data (tensor), shape=(50, 1)
x,y = Variable(x).float(),Variable(y).float()


class Net(torch.nn.Module):  # 继承 torch 的 Module（固定）

    # an affine operation: y = Wx + b
    def __init__(self,input_num,neural_num,output_num):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_num, neural_num,bias=None), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(neural_num, neural_num,bias=None), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(neural_num, output_num,bias=None))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

net = Net(2,2,1)
print(net)


optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
loss_function = torch.nn.MSELoss()

for i in range(10):
    prediction = net(x)
    loss = loss_function(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
