import numpy as np
import pydicom as dicom
import torch
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
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

def load_brain_data(data_dir):
    files_list = os.listdir(data_dir)
    images = [torch.from_numpy(cv2.imread(data_dir+image_path)) for image_path in files_list]
    X = images
    
    labels = [get_brain_label(data_dir+image_path) for image_path in files_list]
    y = torch.tensor(labels)
    
    return X, y

def get_brain_label(image_path):
  value = np.max(cv2.imread(image_path))
  if value > 0:
    return 1
  else:
    return 0

# lung_data = BodyPartDataset('lung', load_lung_data, 'data/lung/')
