import importlib
import pydicom as dicom

import pickle
import data
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pylab as plt
from PIL import Image
import numpy as np
import cv2


def get_model_class(part):
    return models[part]




class LungBinaryClassifierCNN(nn.Module):
    def __init__(self):
        super(LungBinaryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(num_features=16)
        self.dropout1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(num_features=32)
        self.dropout2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(num_features=64)
        self.dropout3 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.relu4 = nn.ReLU()
        self.dropout4=nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)

        x = x.view(-1, 64 * 32 * 32)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class BreastBinaryClassifierCNN(nn.Module):
    def __init__(self):
        super(BreastBinaryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(num_features=16)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(num_features=32)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(num_features=64)
        self.dropout3 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.relu4 = nn.ReLU()
        self.dropout4=nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)

        x = x.view(-1, 64 * 32 * 32)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    


class BrainBinaryClassifierCNN(nn.Module):
    def __init__(self):
        super(BrainBinaryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)


        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(-1, 64 * 32 * 32)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


models = {}
models['lung'] = LungBinaryClassifierCNN()
models['breast'] = BreastBinaryClassifierCNN()
models['brain'] = BrainBinaryClassifierCNN()