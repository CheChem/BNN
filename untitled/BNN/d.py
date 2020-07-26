import pyvarinf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

x_train_data=np.loadtxt('X1.txt',delimiter=",")
y_train_data=np.loadtxt('Y1.txt',delimiter=",")
x_train_data = torch.from_numpy(x_train_data)
y_train_data = torch.from_numpy(y_train_data)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
    def forward(self, x):
        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn2(F.relu(F.max_pool2d(self.conv2(x), 2)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
model = Net()
var_model = pyvarinf.Variationalize(model)


ptimizer = optim.Adam(var_model.parameters(), lr=0.01)
def train(epoch):
    var_model.train()
    for batch_idx, (data, target) in enumerate(x_train_data):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = var_model(data)
        loss_error = F.nll_loss(output, target)
  # The model is only sent once, thus the division by
  # the number of datapoints used to train
        loss_prior = var_model.prior_loss() / 60000
        loss = loss_error + loss_prior
        loss.backward()
        optimizer.step()
for epoch in range(1, 500):
    train(epoch)