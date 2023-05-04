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
import collections

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

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MultiPartDataset(Dataset):
    def __init__(self, part_datasets, size=256):
        self.part_datasets = part_datasets
        parts = [part_dataset.part for part_dataset in part_datasets]
        self.parts_idx = {part: idx for idx, part in enumerate(parts)}

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.unsqueeze(0)),  # Add channel dimension
            transforms.Resize((size, size)),
        ])

        images = []
        labels = []
        for dataset in part_datasets:
            for i in range(len(dataset)):
                image, part = dataset[i]
                image = self.transform(image)
                label = self.parts_idx[dataset.part]
                images.append(image)
                labels.append(label)

        self.images = torch.stack(images)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def rgb_to_grayscale(tensor):
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1).to(tensor.device)
    grayscale_tensor = (tensor * weights).sum(dim=1, keepdim=True)
    return grayscale_tensor


def load_lung_data(data_dir):
    files_list = os.listdir(data_dir)
    images = [torch.tensor(dicom.dcmread(data_dir+image_path).pixel_array.astype('int16')) for image_path in files_list]
    X = torch.stack(images)
    
    labels = 50*[1] + 50*[0]
    y = torch.tensor(labels)
    
    return X, y

def load_brain_data(data_dir):
    files_list = os.listdir(data_dir)
    images = []
    labels = []

    for image_path in files_list:
       
       if image_path != '.DS_Store':
        
        trimmed_path = image_path[:-4]
        
        read_image = cv2.imread(data_dir+image_path)
        if 'mask' not in image_path:
            images.append(torch.from_numpy(read_image))
            
            masked = trimmed_path + '_mask.tif'
            masked_read = cv2.imread(data_dir+masked)
            if 255 in masked_read:
                labels.append(1)
            else:
                labels.append(0)
    
    X = torch.stack(images)


    X = X.permute(0, 3, 1, 2)

    X = rgb_to_grayscale(X)

    X = X.squeeze(1)
    y = torch.tensor(labels)
    
    return X, y

def load_breast_data(data_dir):
    files_list = os.listdir(data_dir)
    images = [torch.from_numpy(cv2.imread(data_dir+image_path)) for image_path in files_list if image_path != '.DS_Store']
    X = torch.stack(images)

    labels = [int(image_path[-5]) for image_path in files_list if image_path != '.DS_Store']
    y = torch.tensor(labels)
    
    X = X.permute(0, 3, 1, 2)

    X = rgb_to_grayscale(X)

    X = X.squeeze(1)



    return X, y

# lung_data = BodyPartDataset('lung', load_lung_data, 'data/lung/')