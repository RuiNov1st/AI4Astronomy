"""
Author: weishirui
Date: 2024-07-20
Description: 专门用来预测数据集红移的脚本
"""
import tensorflow as tf
from keras.models import load_model
from metrics import compute_metrics,check_probability
from utils import set_zbins,log_result,write_result
import numpy as np

def predict(data_path,checkpoint,zbins_midpoint_path,model_name):
    """
    data_path: 要求数据集已经整理成(width,height,channel)

    """
    # load dataset
    images = np.load(data_path) # width*height*channel
    # make zbins:
    zbins_midpoint = np.load(zbins_midpoint_path) # load zbins中值
    # load model
    model = load_model(checkpoint)
    # predict:
    predictions = model.predict(images)
    # redshift:只报道Integer zp(z)作为红移结果
    pred_red = np.sum(predictions[:,1:-1]*zbins_midpoint,axis=1)

    # pdf
    check_probability(images,pred_red,None,predictions[:,1:-1],zbins_midpoint,model_name,predict=True)

    # output
    write_result(pred_red,predictions,model_name)
    print("================ Done!================")


if __name__ == '__main__':
    data_path = '/data/home/wsr/Workspace/dl/dr10_dataset/21.5_r_23.4/images_0.npy'
    checkpoint = '/data/home/wsr/Workspace/dl/Algorithm/Pasquet19_wsr/weights/21.5_r_23.4_Pasquet19_validation_best.keras'
    zbins_midpoint_path = '/data/home/wsr/Workspace/dl/Algorithm/Pasquet19_wsr/data/21.5_r_23.4_Pasquet19_zbins_midpoint.npy'
    
    predict(data_path,checkpoint,zbins_midpoint_path,'img1_predict')