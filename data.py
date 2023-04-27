import numpy as np
import pydicom as dicom
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os

class BodyPartDataset(Dataset):
    def __init__(self, part, load_function, data_path):
        self.part = part
        self.load_function = load_function
        self.data_path = data_path

        self.images, self.labels = load_function(data_path)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    


def load_lung_data(data_dir):
    files_list = os.listdir(data_dir)
    images = [torch.tensor(dicom.dcmread(data_dir+image_path).pixel_array.astype('int16')) for image_path in files_list]
    X = torch.stack(images)
    
    labels = 50*[1] + 50*[0]
    y = torch.tensor(labels)
    
    return X, y


lung_data = BodyPartDataset('lung', load_lung_data, 'data/lung/')
