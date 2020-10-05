#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3 , 64 , 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64 , 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128,  3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128,  3, padding = 1)
        self.fc1   = nn.Linear(128 * 8 * 8, 256)
        self.fc2   = nn.Linear(256       ,256) 
        self.fc3   = nn.Linear(256       , 10)
        
    def forward(self, x):
        out_conv1 = F.relu(self.conv1(x))
        out_conv2 = F.relu(self.conv2(out_conv1))
        out_pool1 = F.max_pool2d(out_conv2, kernel_size = (2, 2))
        out_conv3 = F.relu(self.conv3(out_pool1))
        out_conv4 = F.relu(self.conv4(out_conv3))
        out_pool2 = F.max_pool2d(out_conv4, kernel_size = (2, 2))
        out_view  = out_pool2.reshape(-1, 128 * 8 * 8)
        out_fc1   = F.dropout(F.relu(self.fc1(out_view)), 0.5, training = self.training)
        out_fc2   = F.relu(self.fc2(out_fc1))
        out       = F.relu(self.fc2(out_fc2))
        
        return out_conv1, out_conv2, out_conv3, out_conv4, out

