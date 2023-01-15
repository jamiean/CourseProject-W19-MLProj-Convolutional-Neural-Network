'''
EECS 445 - Introduction to Machine Learning
Winter 2019 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Challenge(nn.Module):
    def __init__(self):
        super().__init__()
        
        # TODO:
        # self.maxp = nn.maxpool2d(3)
        # self.conv1 = nn.Conv2d(3, 6, (5 ,5), stride = (2, 2), padding = 2)
        # self.conv2 = nn.Conv2d(6, 16, (5 ,5), stride = (2, 2), padding = 2)
        # self.conv3 = nn.Conv2d(16, 240, (5 ,5), stride = (2, 2), padding = 2)
        # # self.drop = nn.dropout2d(p = 0.1)
        # self.fc1 = nn.Linear(240, 80, True)
        # self.fc2 = nn.Linear(80, 32, True)
        # self.fc3 = nn.Linear(32, 5, True)
        #


        self.conv1 = nn.Conv2d(3, 24, (4 ,4), padding = 2)
        self.conv2 = nn.Conv2d(24, 84, (4 ,4), padding = 2)
        self.conv3 = nn.Conv2d(84, 120, (3 ,3), padding = 2)
        self.conv4 = nn.Conv2d(120, 160, (5 ,5), stride = (2, 2), padding = 2)
        self.conv5 = nn.Conv2d(160, 200, (4, 4))
        self.fc1 = nn.Linear(200 * 2 * 2, 120, True)
        self.fc2 = nn.Linear(120, 84, True)
        self.fc3 = nn.Linear(84, 10, True)
        self.maxp = nn.MaxPool2d(2, stride = 2)
        self.drop1 = nn.Dropout2d(p = 0.5)
        self.drop2 = nn.Dropout2d(p = 0.3)

        self.init_weights()

    def init_weights(self):
        # TODO:
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)
        
        # TODO: initialize the parameters for [self.fc1, self.fc2, self.fc3]
        for fc in [self.fc1, self.fc2, self.fc3]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1/ sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)
        #
        #
        
    def forward(self, x):
        N, C, H, W = x.shape

        # TODO:
        # TODO: forward pass
        # # x = self.maxp(x)s
        # x = F.relu(self.conv1(x))
        # # x = self.drop(x)
        # # appears not so effective on the accuracy
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = x.view(-1, 240)
        # x = self.fc1(x)
        # x = F.relu(x)
        # # important this relu, would a affect a lot during the training
        # x = self.fc2(x)
        # # x = f.relu(x)
        # x = self.fc3(x)
        # #

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxp(x)
        x = F.relu(self.conv3(x))
        x = self.maxp(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.drop1(x)
        x = x.view(-1, 200 * 2 * 2)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.fc3(x)

        return x
