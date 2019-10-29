# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:10:35 2019

@author: mhrahman
"""

import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd 

TRAIN_TRANSFORMS = torchvision.transforms.ToTensor()
TEST_TRANSFORMS = torchvision.transforms.ToTensor()
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
BATCH_SIZE = 4

cifar_train = CIFAR10('./data',download=True, train= True, transform = transform)
cifar_test = CIFAR10('./data',download=True, train = False, transform= transform) 

train_dataloader = torch.utils.data.DataLoader(cifar_train, batch_size= BATCH_SIZE, shuffle= True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(cifar_test,batch_size= BATCH_SIZE, shuffle= True, num_workers= 0)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show
    
dataiter = iter(train_dataloader)
images, label = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(''.join('%5s' % classes[label[j]] for j in range (BATCH_SIZE)))

class Net(nn.Module):
    def __init__ (self):
        super(Net, self).__init__ ()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 6, kernel_size= 5)
        self.pool1 = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels= 16, kernel_size= 5)
        self.fc1 = nn.Linear(in_features=5*5*16,out_features= 120)
        self.fc2 = nn.Linear(in_features= 120, out_features= 84)
        self.fc3 = nn.Linear(in_features= 84, out_features= 10)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.view(-1, 16*5*5)        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params= net.parameters(), lr=0.001, momentum= 0.9 ) 

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Finished Training')
    
    
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
         100*correct / total))
