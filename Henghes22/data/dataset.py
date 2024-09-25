import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys 
from data.load_utils import load_data,load_catalog,read_config,compute_mean_std
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from data.z_compute import compute_z
from data.mag_compute import check_dataset_magnitude_dist

def image_Augmentation():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90, fill=(0,))  # Random rotation with fill mode set to 0
    ])
    
    
# def image_Transform(mean=None,std=None):
#     return transforms.Compose([
#             transforms.ToTensor(), # 0-1 
#             transforms.Normalize(mean=mean,std=std)
#         ])
    

def data_split(indices,data,config):
    # split indices:
    # split train and test
    indices_train,indices_test = train_test_split(indices,shuffle=True,test_size=config['Data']['TEST_SIZE'], random_state=42) 
    # split train and valid
    indices_train,indices_valid = train_test_split(indices_train,shuffle=True,test_size=config['Data']['VALIDATION_SIZE'],random_state=42)

    # split data
    train_data,valid_data,test_data = [],[],[]
    for d in data:
        train_data.append(d[indices_train])
        valid_data.append(d[indices_valid])
        test_data.append(d[indices_test])
    
    print(f"training set size:{len(indices_train)} \t validation set size:{len(indices_valid)} \t test set size:{len(indices_test)}")

    return train_data,valid_data,test_data



class AstroDataset(Dataset):
    def __init__(self, images,labels,catalog_data,indices,augmentation=None,transform=None):
        self.images = images
        self.labels = labels
        self.catalog_data = catalog_data
        self.indices = indices
        self.augmentation  =  augmentation
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        # augmentation:
        if self.augmentation:
            image = self.augmentation(image)
        # normalize and totensor:0-1
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        catalog_data = self.catalog_data[idx]
        indice = self.indices[idx]
        return image.to(torch.float32), label,torch.Tensor(catalog_data),indice
        # return image, label,catalog_data,indice



def make_dataset(config):
    # load image and label
    images,labels = load_data(config['Data']['IMG_PATH'],
        config['Data']['LABEL_PATH'],
        config['Data']['DATA_TYPE']
    )

    # reshape images：
    images = torch.tensor(images).permute(0,3,1,2) # n,channel,width,height
    
    # load catalog:
    catalog_data,catalog = load_catalog(config['Data']['CATALOG_PATH'],config['Data']['CATALOG_COLUMN'])

    # indices:
    indices = np.arange(len(images))


    # data split:
    train_data,valid_data,test_data = data_split(indices,[images,labels,catalog_data,indices],config)

    # # compute mean and std: only compute training set and apply it to validation and test set
    # image_mean,image_std =  compute_mean_std(train_data[0])
    # print(image_mean,image_std)

    # check z and encode(optional)
    labels_data,zbins_midpoint,Nbins = compute_z([labels,train_data[1],valid_data[1],test_data[1]],config)

    # check magnitude:
    check_dataset_magnitude_dist(catalog,train_data[3],valid_data[3],test_data[3],config['Experiment']['Run_name'])

    # make dataset:
    # image_transform = image_Transform(image_mean,image_std)
    train_dataset = AstroDataset(images=train_data[0],labels=labels_data[0],catalog_data=train_data[2],indices=train_data[3],augmentation=image_Augmentation(),transform=None)
    valid_dataset = AstroDataset(images=valid_data[0],labels=labels_data[1],catalog_data=valid_data[2],indices=valid_data[3],augmentation=None,transform=None)
    test_dataset = AstroDataset(images=test_data[0],labels=labels_data[2],catalog_data=test_data[2],indices=test_data[3],augmentation=None,transform=None)


    # dataloader:
    train_loader = DataLoader(train_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True)
    valid_loader = DataLoader(valid_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=config['Train']['BATCH_SIZE'],shuffle=True)

    return train_loader,valid_loader,test_loader,catalog,zbins_midpoint,Nbins


if __name__ =='__main__':
    config = read_config()
    train_dataset, val_dataset, test_dataset = make_dataset(config)
    


