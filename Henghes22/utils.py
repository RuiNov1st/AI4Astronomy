import numpy as np
import matplotlib.pyplot as plt
# from keras.utils.vis_utils import plot_model
import gc
''' modules for scatter density'''
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import yaml
import torch
import wandb
import os
os.environ["WANDB_API_KEY"] = "e37d4bd09540220f7c81d663c47480e474dbbe2f"

def read_config():
    """
    读取配置
    """
    with open("./config.yaml") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def write_config(config):
    """
    将每次实验的config文件写入文件中，留存以便之后检查
    """
    with open(f"./output/{config['Experiment']['Run_name']}/config.yaml", 'w',encoding='utf-8') as yaml_file:
        yaml.dump(config, yaml_file,allow_unicode=True)
    
    print("write config.yaml successfully!")


def set_gpu():
    if torch.cuda.is_available():
        device_ids = list(range(torch.cuda.device_count()))
        print(f"CUDA NUMBER:{torch.cuda.device_count()}")
        return True,device_ids
    else:
        device = torch.device("cpu")
        print("using cpu")
    return False,device


    
def set_wandb(config,resume_run=False):
    """
    配置wandb
    """
    wandb.login()
    # Resume run
    if resume_run:
        run = wandb.init(
                project=config['Experiment']['Project_name'], 
                id=config['Experiment']['Run_id'], 
                resume="must")
    # new run
    else:
        run = wandb.init(
            project = config['Experiment']['Project_name'],
            config = config,
            notes = config['Experiment']['Description'],
            name = config['Experiment']['Run_name'],
            group = config['Experiment']['Group']
        )
        # 将此次run id写入config中
        config['Experiment']['Run_id'] = wandb.run.id
    return run,config

def upload_wandb(config):
    # log
    arti_log = wandb.Artifact('log',type='log')
    arti_log.add_file('./logs/log.txt',name=config['Experiment']['Run_name']+"_log")
    wandb.log_artifact(arti_log)


def write_result(test_labels,pred_red,indices_test,probability,model_name):
    file_name = f'./output/{model_name}/{model_name}.npz'
    np.savez(file_name,z_true=test_labels,z_pred=pred_red,indices = indices_test,pdf = probability)
    print(f"save {file_name} success!")

def log_result(deltaz,bias,nmad,outlier,model_name):
    log_file = open(f'./output/{model_name}/{model_name}_log_result.txt','w')
    log_file.write(f'Bias:{bias}\n')
    log_file.write(f'NMAD:{nmad}\n')
    log_file.write(f'Outlier fraction:{outlier}\n')
    log_file.close()
    print(f'log ./output/{model_name}/{model_name}_log_result.txt success!')