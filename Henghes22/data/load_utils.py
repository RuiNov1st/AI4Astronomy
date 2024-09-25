import torch
import numpy as np
import yaml
import pandas as pd

def load_data(img_path,label_path,data_type):
    """
    载入数据
    """
    images = np.load(img_path)
    labels = np.load(label_path)

    if data_type == 'DESI':
        images = images[:,:,:,:4] # g,r,i,z
    elif data_type == 'WISE_COLOR-WISE':
        images = images[:,:,:,[0,1,2,3,6,7,8]] # g,r,i,z,g-r,r-i,i-z 
    elif data_type == 'DESI_COLOR':
        images = images[:,:,:,:7] # g,r,i,z,g-r,r-i,i-z 
    elif data_type == 'WISE':
        images = images[:,:,:,:6] # g,r,i,z,w1,w2
    elif data_type == 'WISE_COLOR':
        images = images[:,:,:,:11] # g,r,i,z,w1,w2,g-r,r-i,i-z,z-w1,w1-w2

    return images,labels

def load_catalog(catalog_path,colsname=['MAG_R']):
    df = pd.read_csv(catalog_path)
    # 读取其中所需要的列：
    arr = df[colsname[0]].values
    arr = arr.reshape(-1,1)
    if len(colsname)>1:
        for i in range(1,len(colsname)):
            arr = np.concatenate([arr,df[colsname[i]].values.reshape(-1,1)],axis=1)
    return arr,df

def read_config():
    """
    读取配置
    """
    with open("../config.yaml") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


# compute mean & std for normalize:
def compute_mean_std(images):
    # required images shape: (n,channel,width,height)
    mean = images.mean(dim=[0, 2, 3])
    std = images.std(dim=[0,2,3])

    return mean,std


