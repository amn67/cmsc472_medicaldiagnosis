import numpy as np
import pydicom as dicom
import torch
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import cv2
import os
import collections

IMG_SIZE = 256

class BodyPartDataset(Dataset):
    def __init__(self, part, load_function, data_path):
        self.part = part
        self.load_function = load_function
        self.data_path = data_path

        self.images, self.labels = load_function(data_path)

        # Resize to 256x256
        self.images = self.images.unsqueeze(1) 
        self.images = F.interpolate(self.images, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        self.images = self.images.squeeze(1)
        self.images = self.images.to(torch.float32)  # Force float32



    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MultiPartDataset(Dataset):
    def __init__(self, part_datasets, parts_list=None):
        self.part_datasets = part_datasets

        # If we pass in dataset Subset, we need to pass in a parts_list to know
        # which body part each dataset object corresponds to
        if parts_list is None:
            self.parts = [part_dataset.part for part_dataset in part_datasets]
        else:
            self.parts = parts_list
        # One hot encode our body part classes
        self.parts_idx = {part: idx for idx, part in enumerate(self.parts)}


        images = []
        labels = []
        binary_labels = []
        for idx, dataset in enumerate(part_datasets):
            for i in range(len(dataset)):
                image, bin_label = dataset[i]
                label = self.parts_idx[self.parts[idx]]
                images.append(image)
                labels.append(label)
                binary_labels.append(bin_label)

        self.images = torch.stack(images)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.labels = F.one_hot(self.labels, num_classes=3)
        self.binary_labels = torch.tensor(binary_labels, dtype=torch.long)


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
    
    window_level = -600
    window_width = 2250

    lower_limit = window_level - window_width // 2
    upper_limit = window_level + window_width // 2

    images_scaled = []

    for image in images:
        pixel_data = image

        windowed_data = torch.clamp(pixel_data, lower_limit, upper_limit)

        normalized_data = (windowed_data - lower_limit) * (255.0 / window_width)
        grayscale_data = normalized_data.to(torch.uint8)
        images_scaled.append(grayscale_data)
    
    X = torch.stack(images_scaled)
    
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
    images = [cv2.imread(data_dir+image_path) for image_path in files_list if image_path != '.DS_Store']
    resized = [torch.from_numpy(cv2.resize(image, (50,50), interpolation= cv2.INTER_LINEAR))for image in images]
    X = torch.stack(resized)

    labels = [int(image_path[-5]) for image_path in files_list if image_path != '.DS_Store']
    y = torch.tensor(labels)
    
    X = X.permute(0, 3, 1, 2)

    X = rgb_to_grayscale(X)

    X = X.squeeze(1)



    return X, y

def load_datasets():
    # Returns a dictionary of datasets, where each dataset is a dictionary of train, val, test


    lung_dataset = BodyPartDataset('lung', load_lung_data, 'data/lung/')
    breast_dataset = BodyPartDataset('breast', load_breast_data, 'data/breast/')
    brain_dataset = BodyPartDataset('brain', load_brain_data, 'data/brain/')

    datasets = [lung_dataset, breast_dataset, brain_dataset]
    generator = torch.Generator().manual_seed(42)
    all_data = {}
    parts_list = []
    for dataset in datasets:
        part = dataset.part
        parts_list.append(part)
        all_data[part] = {}
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1], generator=generator)
        all_data[part]['train'] = train_set
        all_data[part]['val'] = val_set
        all_data[part]['test'] = test_set

    all_data['combined'] = {}
    all_data['combined']['train'] = MultiPartDataset([all_data[part]['train'] for part in parts_list], parts_list)
    all_data['combined']['val'] = MultiPartDataset([all_data[part]['val'] for part in parts_list], parts_list)
    all_data['combined']['test'] = MultiPartDataset([all_data[part]['test'] for part in parts_list], parts_list)

    return all_data