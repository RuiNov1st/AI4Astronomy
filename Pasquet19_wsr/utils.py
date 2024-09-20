import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from keras.utils.vis_utils import plot_model
import gc
from keras.models import load_model
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

def update_config(config):
    """
    更新配置
    """
    with open("./config_new.yaml", 'w') as yaml_file:
        yaml.dump(config, yaml_file)
    
    print("Update config.yaml successfully!")


def set_gpu():
    """
    配置GPU
    """
    # 检查GPU是否可用
    gpus = tf.config.list_physical_devices('GPU')
    print(f'{len(gpus)} GPUs are available: {gpus}')
    if gpus: # 有GPU
        # 显式指定使用哪个GPU
        use_gpu = gpus[:]
        tf.config.experimental.set_visible_devices(devices=use_gpu, device_type='GPU')
        print(f"Use {use_gpu}")

        # 设置按需申请显存空间
        for gpu in use_gpu:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print('No GPU available.')

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


def set_zbins(z_min,z_max,Nbins):
    """
    设置红移bin：留两个特殊的红移bin，第0个为小于等于0的红移值；最后一个bin作为超出最大红移之外的红移值
    """
    # zbins = np.array([i/Nbins for i in range(Nbins)])*(z_max+zbins_gap) 
    zbins = np.linspace(z_min,z_max,Nbins-1) # 一共Nbins-1个值，但编码出来的Bin有Nbins
    zbins_gap = np.mean(np.diff(zbins))

    return zbins,zbins_gap


def make_classes(zbins,labels):
    label_encode = np.digitize(labels,zbins,right = True) # digitize函数：用bins给labels编码，right=True表示在bins是升序的情况下是右等号的，即bin[i-1]<x<=bin[i]。https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
    # 实际上，传进来的只有Nbins-1个数值，但是np.digitize编码会编码出Nbins个数值：从0到Nbins-1，即Nbins-1+1=Nbins个值
    return label_encode

def compute_zbins_midpoint(zbins):
    """
    计算每个红移bin的中值
    """
    zbins_midpoint = np.array([(zbins[i]+zbins[i-1])/2 for i in range(1,len(zbins))]) # 计算每个bin对应的中值，那就只有Nbins-2个值
    return zbins_midpoint


def load_data(img_path,label_path,ebv_path,data_type):
    """
    载入数据
    """
    images = np.load(img_path)
    labels = np.load(label_path)
    ebv = np.load(ebv_path)

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

    return images,labels,ebv

def img_preprocess(images,config):
    # 裁剪
    if config['Data']['IMAGE_SIZE'] != images.shape[1]:
        crop_size = config['Data']['IMAGE_SIZE']
        offset = (images.shape[1]-crop_size)//2
        images = tf.image.crop_to_bounding_box(images,offset,offset,crop_size,crop_size)

        images = images.numpy() if isinstance(images, tf.Tensor) else images

    return images


def check_input(labels,model_name,dataset_name):
    """
    检查数据集情况：绘制数据集红移的分布情况
    """
    # 查看数据集红移分布
    fig = plt.figure()
    hist = plt.hist(labels,bins=50)
    plt.title(f'{model_name}_z_distribution.png')
    plt.savefig(f'./output/{model_name}/{model_name}_{dataset_name}_z_distribution.png')
    z_min = np.min(labels)
    z_max = np.max(labels)
    print(f"z_min:{z_min} z_max:{z_max}")
    return z_min,z_max
    


def write_result(test_labels,pred_red,indices_test,pred_red_max,probability,model_name):
    file_name = f'./output/{model_name}/{model_name}.npz'
    np.savez(file_name,z_true=test_labels,z_pred=pred_red,indices = indices_test,z_pred_max = pred_red_max,pdf = probability)
    print(f"save {file_name} success!")

def log_result(deltaz,bias,nmad,outlier,model_name):
    log_file = open(f'./output/{model_name}/{model_name}_log_result.txt','w')
    log_file.write(f'Bias:{bias}\n')
    log_file.write(f'NMAD:{nmad}\n')
    log_file.write(f'Outlier fraction:{outlier}\n')
    log_file.close()
    print(f'log ./output/{model_name}/{model_name}_log_result.txt success!')


def check_dataset_redshift_dist(labels,y_train,y_train_label,y_test,y_test_label,y_valid,y_valid_label,z_max,model_name):
    # # 检查训练集和测试集的分布情况
    print("dataset size(full and <=z_max): 1) all; 2) training; 3) validation; 4) test")
    print(labels.shape,labels[labels<=z_max].shape)
    print(y_train.shape,y_train[y_train<=z_max].shape)
    print(y_valid.shape,y_valid[y_valid<=z_max].shape)
    print(y_test.shape,y_test[y_test<=z_max].shape)

    check_input(y_train[y_train<=z_max],model_name,"training_set")
    check_input(y_test[y_test<=z_max],model_name,"test_set")
    check_input(y_valid[y_valid<=z_max],model_name,"validation set")
    print("origin result: 1) train vs test; 2) train vs valid")
    print(ks_2samp(y_train[y_train<=z_max],y_test[y_test<=z_max]))
    print(ks_2samp(y_train[y_train<=z_max],y_valid[y_valid<=z_max]))

    
    # 检查编号情况
    check_input(y_train_label,model_name,"traing_encode")
    check_input(y_test_label,model_name,"test_encode")
    check_input(y_valid_label,model_name,"valid encode")
    
    # ks检验训练集和测试集是否服从相同分布
    print("encode result: 1)train vs test 2)train vs valid")
    print(ks_2samp(y_train_label,y_test_label))
    print(ks_2samp(y_train_label,y_valid_label))
    

    # # 检验真实值和label之前是否服从相同分布
    # print("origin vs encode")
    # print(ks_2samp(y_train[y_train<=z_max],y_train_label))
    # print(ks_2samp(y_test[y_test<=z_max],y_test_label))


# 看一下星等分布情况
def check_dataset_magnitude_dist(df_all,train_idx,valid_idx,test_idx,model_name):
    bands = ['G','R','I','Z','W1','W2']
    ## 星等分布绘图：
    def show_magnitude_dist(df,model_name,name):
        fig,ax = plt.subplots(6,1,figsize=(7,14))
        plt.tight_layout()
        for b in range(len(bands)):
            tmp_df = df[f'MAG_{bands[b]}']
            idx = np.where(df[f'MAG_{bands[b]}_flag']==1)[0]
            ax[b].hist(df[f'MAG_{bands[b]}'].iloc[idx])
            ax[b].set_title(f'MAG_{bands[b]}')
        plt.savefig(f'./output/{model_name}/MAG_dist_{name}.png')
    # 总体：
    show_magnitude_dist(df_all,model_name,'all')
    # 训练集：
    df_train = df_all.iloc[train_idx]
    show_magnitude_dist(df_train,model_name,'train')
    # 验证集
    df_valid = df_all.iloc[valid_idx]
    show_magnitude_dist(df_valid,model_name,'valid')
    # 测试集
    df_test= df_all.iloc[test_idx]
    show_magnitude_dist(df_test,model_name,'test')

    
def get_model_size(model):
    param_num = sum([np.prod(w.shape) for w in model.get_weights()])
    param_size = param_num *4/1024/1024 # MB
    return param_size

def write_config(config):
    """
    将每次实验的config文件写入文件中，留存以便之后检查
    """

    with open(f"./output/{config['Experiment']['Run_name']}/config.yaml", 'w',encoding='utf-8') as yaml_file:
        yaml.dump(config, yaml_file,allow_unicode=True)
    
    print("write config.yaml successfully!")

    # try:
    #     f = open('./config.yaml','r')
    #     config_contents = f.readlines()
    #     f_write = open(f'./output/{model_name}/config.txt','w')
    #     for line in config_contents:
    #         f_write.write(line)
    # finally:
    #     f_write.close()
    #     f.close()

