#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1 , 64 , 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64 , 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128,  3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128,  3, padding = 1)
        self.fc1   = nn.Linear(128 * 7 * 7, 256)
        self.fc2   = nn.Linear(256       , 256)
        self.fc3   = nn.Linear(256       , 10)
    
    def truncated_forward(self,x,truncate=None):
        assert truncate is not None,"truncate must be specified"
        if truncate == 0:
            return self.partial_forward_1(x)
        elif truncate == 1:
            return self.partial_forward_2(x)
        elif truncate == 2:
            return self.partial_forward_3(x)
        else:
            return self.partial_forward_4(x)
        
    def partial_forward_1(self,x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training = self.training)
        return out_conv1
    
    def partial_forward_2(self,x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training = self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training = self.training)
        return out_conv2
    
    def partial_forward_3(self,x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training = self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training = self.training)
        out_pool1 = F.max_pool2d(out_conv2, kernel_size = (2, 2))
        out_conv3 = F.dropout(F.relu(self.conv3(out_pool1)), 0.1, training = self.training)
        return out_conv3
    
    def partial_forward_4(self,x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training = self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training = self.training)
        out_pool1 = F.max_pool2d(out_conv2, kernel_size = (2, 2))
        out_conv3 = F.dropout(F.relu(self.conv3(out_pool1)), 0.1, training = self.training)
        out_conv4 = F.dropout(F.relu(self.conv4(out_conv3)), 0.1, training = self.training)
        return out_conv4
    
    def forward(self, x):
        out_conv1 = F.dropout(F.relu(self.conv1(x)), 0.1, training = self.training)
        out_conv2 = F.dropout(F.relu(self.conv2(out_conv1)), 0.1, training = self.training)
        out_pool1 = F.max_pool2d(out_conv2, kernel_size = (2, 2))
        out_conv3 = F.dropout(F.relu(self.conv3(out_pool1)), 0.1, training = self.training)
        out_conv4 = F.dropout(F.relu(self.conv4(out_conv3)), 0.1, training = self.training)
        out_pool2 = F.max_pool2d(out_conv4, kernel_size = (2, 2))
        out_view  = out_pool2.view(-1, 128 * 7 * 7)
        out_fc1    = F.dropout(F.relu(self.fc1(out_view)), 0.1, training = self.training)
        out_fc2    = F.dropout(F.relu(self.fc2(out_fc1)), 0.1, training = self.training)
        out       = self.fc3(out_fc2)
        
        return out_conv1, out_conv2, out_conv3, out_conv4, out

