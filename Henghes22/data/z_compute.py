import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import torch

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

def check_z_dist(labels,model_name):
    """
    检查数据集情况：绘制数据集红移的分布情况
    labels:[all_labels,train_labels,valid_labels,test_labels]
    """
    z_min = np.min(labels[0])
    z_max = np.max(labels[0])
    print(f"z_min:{z_min} z_max:{z_max}")

    # 查看数据集红移分布
    name = ['all','train','valid','test']
    fig,ax = plt.subplots(4,1)
    plt.tight_layout()
    for i in range(len(name)):
        ax[i].hist(labels[i],bins=50)
        ax[i].set_title(f'{name[i]}_z_distribution')
    plt.savefig(f'./output/{model_name}/{model_name}_z_distribution.png')

    # kstest:
    print("ks test for train and valid",sep=',')
    print(ks_2samp(labels[1],labels[2])) # train vs valid
    print("ks test for train and test",sep=',')
    print(ks_2samp(labels[1],labels[3])) # train vs test
    print("ks test for valid and test",sep=',')
    print(ks_2samp(labels[2],labels[3])) # valid vs test
    
    return z_min,z_max



def compute_z(labels,config):
    """
    labels:[all_labels,train_labels,valid_labels,test_labels]
    """
    # check z distribution:
    z_min,z_max = check_z_dist(labels,config['Experiment']['Run_name'])
    # z value to z encode:
    if config['Data']['LABEL_ENCODE']:
        if config['Data']['Z_USE_DEFAULT']: # 使用事先设定的默认值
            z_min = config['Data']['Z_MIN']
            z_max = config['Data']['Z_MAX']
        # set zbins
        zbins_gap = config['Data']['ZBINS_GAP_DEFAULT'] # 默认的zbins_gap
        Nbins = np.min([int((z_max-z_min)/zbins_gap)+2,config['Data']['NBINS_MAX']]) # Nbins与默认值对比取最小的。留两个：第0个为小于等于0的红移；最后一个为大于最大值的红移。
        zbins,zbins_gap = set_zbins(z_min,z_max,Nbins) # 设置zbins
        print(zbins)
        print(zbins.shape,zbins_gap)
        # compute zbin midpoint:
        zbins_midpoint = compute_zbins_midpoint(zbins)

        # save midpoint:
        np.save(f"./output/{config['Experiment']['Run_name']}/{config['Experiment']['Run_name']}_zbins_midpoint.npy",zbins_midpoint)

        # encode z labels:
        encode_labels = []
        for l in range(1,len(labels)):
            encode_labels.append(make_classes(zbins,labels[l]))
    
        return encode_labels,zbins_midpoint,Nbins
    else:
        type_labels = [torch.Tensor(i).to(torch.float32) for i in labels] # to float32
        return type_labels[1:],None,1
