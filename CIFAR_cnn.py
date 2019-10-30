# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 03:28:08 2019

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRAIN_TRANSFORMS = torchvision.transforms.ToTensor()
TEST_TRANSFORMS = torchvision.transforms.ToTensor()
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
BATCH_SIZE = 16
total_epoch = 2

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
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size= 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size= 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels= 256, kernel_size= 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels= 512, kernel_size= 3, padding = 1)

        self.pool1 = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.fc1 = nn.Linear(in_features=512*2*2,out_features= 256)
        self.fc2 = nn.Linear(in_features= 256, out_features= 128)
        self.fc3 = nn.Linear(in_features= 128, out_features= 10)
        
    def forward(self,x):
        x = F.relu(self.conv1(x)) # 32
        x = self.pool1(x) # 16
        x = F.relu(self.conv2(x)) # 16
        x = self.pool1(x) # 8
        x = F.relu(self.conv3(x)) # 8
        x = self.pool1(x) # 4
        x = F.relu(self.conv4(x)) # 4
        x = self.pool1(x) # 2
        x = x.view(-1, 512*2*2)        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params= net.parameters(), lr=0.001, momentum= 0.9 ) 

for epoch in range(total_epoch):
    
    def calc_accu(series):
        running_loss, correct, total = 0,0,0
        with torch.no_grad():
            for i,data in enumerate(series,0):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss = criterion(outputs,labels)
                running_loss += loss.item()
                _,predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            return 100*correct/total
    
    #inital check
    if epoch == 0:
        inital = calc_accu(test_dataloader)
        print('Inital accuracy on the test image ======== {}'.format(inital))
    
    
    running_loss, correct,total = 0,0,0
    for i,data in enumerate(train_dataloader,0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum().item()
#        if i % 2000 == 1999:    # print every 2000 mini-batches
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 2000))
#            running_loss = 0.0

    test_acc = calc_accu(test_dataloader)
    print('epoch {}/{} ============================= loss {}, Train accuracy {}, Test accuracy {}'.format(epoch+1,total_epoch,running_loss/12000, 100*correct/total,test_acc))

# Ground truth vs predicted
Groud_truth = []
for j in range (BATCH_SIZE):
    gt = classes[label[j]]
    Groud_truth.append(gt)
    
Predicted = []
out = net(images.cuda())
_,predicted = torch.max(out,1)
for i in range(BATCH_SIZE):
    pr = classes[predicted[i]]
    Predicted.append(pr)

df = pd.DataFrame({'Ground Truth':Groud_truth,'Predicted':Predicted})

    
