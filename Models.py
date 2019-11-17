# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:09:49 2019

@author: mhrahman
"""

import torch.nn as nn
import torch.nn.functional as F

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

class Net_BN(nn.Module):
    def __init__ (self):
        super(Net_BN, self).__init__ ()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size= 3, padding = 1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 3, padding = 1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size= 3, padding = 1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 3, padding = 1)
        self.bn4   = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels= 256, kernel_size= 3, padding = 1)
        self.bn5   = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels= 256, kernel_size= 3, padding = 1)
        self.bn6   = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels= 512, kernel_size= 3, padding = 1)
        self.bn7   = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels= 512, kernel_size= 3, padding = 1)
        self.bn8   = nn.BatchNorm2d(512)
        
        self.dropout = nn.Dropout2d(p = 0.3)
        self.pool1 = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.fc1 = nn.Linear(in_features=512*2*2,out_features= 512)
        self.fc2 = nn.Linear(in_features= 512, out_features= 10)
        
    def forward(self,x):
        x = F.relu(self.bn2(self.conv2(self.bn1(self.conv1(x))))) #outsize = 32
        x = self.pool1(x) # 16
        x = F.relu(self.bn4(self.conv4(self.bn3(self.conv3(x))))) # 16
        x = self.pool1(x) # 8
        x = F.relu(self.bn6(self.conv6(self.bn5(self.conv5(x))))) # 8
        x = self.pool1(x) # 4
        x = F.relu(self.bn8(self.conv8(self.bn7(self.conv7(x))))) # 4
        x = self.pool1(x) # 2
        x = x.view(-1, 512*2*2)        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
class Net_BN_Dr(nn.Module):
    def __init__ (self):
        super(Net_BN_Dr, self).__init__ ()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size= 3, padding = 1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 3, padding = 1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size= 3, padding = 1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 3, padding = 1)
        self.bn4   = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels= 256, kernel_size= 3, padding = 1)
        self.bn5   = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels= 256, kernel_size= 3, padding = 1)
        self.bn6   = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels= 512, kernel_size= 3, padding = 1)
        self.bn7   = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels= 512, kernel_size= 3, padding = 1)
        self.bn8   = nn.BatchNorm2d(512)
        
        self.dropout = nn.Dropout2d(p = 0.3)
        self.dropout2 = nn.Dropout2d(p = 0.5)
        self.pool1 = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.fc1 = nn.Linear(in_features=512,out_features= 512)
        self.fc2 = nn.Linear(in_features= 512, out_features= 10)
        
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x))) #outsize = 32
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))        
        x = self.pool1(x) # 16
        
        x = F.relu(self.bn3(self.conv3(x))) # 16 
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))       
        x = self.pool1(x) # 8
        
        x = F.relu(self.bn5(self.conv5(x))) # 8
        x = self.dropout(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool1(x) # 4
               
        x = F.relu(self.bn7(self.conv7(x))) # 4
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.dropout(x)
        x = self.pool1(x) # 2
                
        x = F.relu(self.bn8(self.conv8(x))) #2
        x = self.dropout(x)        
        x = F.relu(self.bn8(self.conv8(x)))    
        x = self.pool1(x) # 1
                
        x = x.view(-1, 512)
        x= self.dropout2(x)        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class Net_new(nn.Module):
    def __init__ (self):
        super(Net_new, self).__init__ ()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size= 3, padding = 1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size= 3, padding = 1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size= 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 3, padding = 1)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels= 256, kernel_size= 3, padding = 1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels= 256, kernel_size= 3, padding = 1)
        self.bn4   = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels= 512, kernel_size= 3, padding = 1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels= 512, kernel_size= 3, padding = 1)
        self.bn5   = nn.BatchNorm2d(512)
        

        self.pool1 = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.dropout1 = nn.Dropout2d(p = 0.1)
        self.dropout2 = nn.Dropout2d(p = 0.2)


        
        self.fc1 = nn.Linear(in_features=512*2*2,out_features= 512)
        self.fc2 = nn.Linear(in_features= 512, out_features= 10)

        
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)),inplace = True) #outsize = 32
        x = F.relu(self.conv2(x),inplace = True)        
        x = self.pool1(x) # 16
        
        x = F.relu(self.bn3(self.conv3(x)),inplace = True) # 16 
        x = F.relu(self.conv4(x),inplace = True)       
        x = self.pool1(x) # 8
        x = self.dropout1(x)
        
        x = F.relu(self.bn4(self.conv5(x)),inplace = True) # 8
        x = F.relu(self.conv6(x),inplace = True)
        x = self.pool1(x) # 4
        x = self.dropout1(x)
        
        x = F.relu(self.bn5(self.conv7(x)),inplace = True) # 4
        x = F.relu(self.conv8(x),inplace = True)
        x = self.pool1(x) # 2
        x = self.dropout1(x)
        
        x = x.view(-1, 512*2*2)       
        x = self.fc1(x)
        x = F.relu(x,inplace = True)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

          
#Try with sequential 
        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x
