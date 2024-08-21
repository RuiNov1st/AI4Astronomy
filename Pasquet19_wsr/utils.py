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


def load_data(img_path,label_path,ebv_path):
    """
    载入数据
    """
    images = np.load(img_path)
    labels = np.load(label_path)
    ebv = np.load(ebv_path)

    return images,labels,ebv

def check_input(labels,model_name):
    """
    检查数据集情况：绘制数据集红移的分布情况
    """
    # 查看数据集红移分布
    fig = plt.figure()
    hist = plt.hist(labels,bins=50)
    plt.title(f'{model_name}_z_distribution.png')
    plt.savefig(f'./output/{model_name}_z_distribution.png')
    z_min = np.min(labels)
    z_max = np.max(labels)
    print(f"z_min:{z_min} z_max:{z_max}")
    return z_min,z_max
    


def write_result(pred_red,pred_red_max,probability,model_name):
    file_name = f'./output/{model_name}.npz'
    np.savez(file_name,z_pred=pred_red,z_pred_max = pred_red_max,pdf = probability)
    print(f"save {file_name} success!")

def log_result(deltaz,bias,nmad,outlier,model_name):
    log_file = open(f'./output/{model_name}_log_result.txt','w')
    log_file.write(f'Bias:{bias}\n')
    log_file.write(f'NMAD:{nmad}\n')
    log_file.write(f'Outlier fraction:{outlier}\n')
    log_file.close()
    print(f'log ./output/{model_name}_log_result.txt success!')


def check_dataset_redshift_dist(labels,y_train,y_train_label,y_test,y_test_label,y_valid,y_valid_label,z_max):
    # # 检查训练集和测试集的分布情况
    print("dataset size(full and <=z_max): 1) all; 2) training; 3) validation; 4) test")
    print(labels.shape,labels[labels<=z_max].shape)
    print(y_train.shape,y_train[y_train<=z_max].shape)
    print(y_valid.shape,y_valid[y_valid<=z_max].shape)
    print(y_test.shape,y_test[y_test<=z_max].shape)

    check_input(y_train[y_train<=z_max],"training_set")
    check_input(y_test[y_test<=z_max],"test_set")
    check_input(y_valid[y_valid<=z_max],"validation set")
    print("origin result: 1) train vs test; 2) train vs valid")
    print(ks_2samp(y_train[y_train<=z_max],y_test[y_test<=z_max]))
    print(ks_2samp(y_train[y_train<=z_max],y_valid[y_valid<=z_max]))

    
    # 检查编号情况
    check_input(y_train_label,"traing_encode")
    check_input(y_test_label,"test_encode")
    check_input(y_valid_label,"valid encode")
    
    # ks检验训练集和测试集是否服从相同分布
    print("encode result: 1)train vs test 2)train vs valid")
    print(ks_2samp(y_train_label,y_test_label))
    print(ks_2samp(y_train_label,y_valid_label))
    

    # # 检验真实值和label之前是否服从相同分布
    # print("origin vs encode")
    # print(ks_2samp(y_train[y_train<=z_max],y_train_label))
    # print(ks_2samp(y_test[y_test<=z_max],y_test_label))

    
def get_model_size(model):
    param_num = sum([np.prod(w.shape) for w in model.get_weights()])
    param_size = param_num *4/1024/1024 # MB
    return param_size
    