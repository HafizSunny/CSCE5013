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
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import pandas as pd
from Models import Net_new
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Data augmetation + Data normalization
train_transform = transforms.Compose([transforms.RandomCrop(32,padding = 4),transforms.RandomHorizontalFlip(),transforms.ToTensor()
                                        ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

BATCH_SIZE = 16
total_epoch = 150

cifar_training = CIFAR10('./data',download=True, train= True, transform = train_transform)
cifar_test = CIFAR10('./data',download=True, train = False, transform= test_transform) 

train_size = int(0.8*len(cifar_training))
val_size = len(cifar_training) - train_size
cifar_train, cifar_val = torch.utils.data.random_split(cifar_training,[train_size,val_size])

cifar_val.transform = test_transform

train_dataloader = torch.utils.data.DataLoader(cifar_train, batch_size= BATCH_SIZE, shuffle= True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(cifar_test,batch_size= BATCH_SIZE, shuffle= True, num_workers= 2)
val_dataloader = torch.utils.data.DataLoader(cifar_val,batch_size= BATCH_SIZE, shuffle= True, num_workers= 2)

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

#weight initial
def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)


#net = Net_BN()
#net = Net_BN_Dr()        
net = Net_new() #Best one
#net = CNN()        
#net.apply(weight_init)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params= net.parameters(), lr=0.001, momentum= 0.9, weight_decay = 0.0005) 
#optimizer = optim.Adam(params=net.parameters(), lr= 0.001, weight_decay= 0.0005)
scheduler = lr_scheduler.StepLR(optimizer,step_size= 25, gamma= 0.5)

Training_acc = []
Validation_acc = []

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
    running_loss /= len(train_dataloader)     
    val_acc = calc_accu(val_dataloader)
    Training_acc.append(100*correct/total)
    Validation_acc.append(val_acc)
    print('epoch {}/{} ============================= loss {}, Train accuracy {},Validation accuracy {}'.format(epoch+1,total_epoch,running_loss, 100*correct/total, val_acc))

#Testing accuracy
test_acc = calc_accu(test_dataloader)
print('Testing accuracy on 10000 images =========================== {}'.format(test_acc))    


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
df.to_csv('Report.csv')

#Accuracy plot
plt.figure()
plt.plot(Training_acc,label = 'Training accuracy')
plt.plot(Validation_acc, label = 'Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracies')
plt.legend(loc = 4)
plt.savefig('acc.png',dpi = 600)
plt.close()
    
