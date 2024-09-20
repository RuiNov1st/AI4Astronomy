from model import Pasquet_model
from models.ResNet import ResNet18
from utils import load_data,set_zbins,log_result,make_classes,write_result,check_input,compute_zbins_midpoint,check_dataset_redshift_dist,get_model_size,write_config,set_gpu,read_config,set_wandb,img_preprocess,check_dataset_magnitude_dist
from sklearn.model_selection import train_test_split
from metrics import compute_metrics,make_plot,check_probability,print_history,compute_maxpdf_z,metrics_z_plot
import tensorflow as tf
from keras.models import load_model
import numpy as np
import os
from config import *
import math
import sys
import wandb # weights & bias experiment track
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from PIL import Image
from analysis import outlier_analysis
import pandas as pd


# use generator as input to prevent cuda out of memory:
def batch_iterator(data, labels, batch_size,predict=False,shuffle=False):
    num_batches_per_epoch = int((len(labels) - 1) / batch_size) + 1
    
    def data_generator():
        data_size = len(labels)  # Assuming labels length matches the data's first dimension
        while True:
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size)) # 随机排序
                if isinstance(data,list):
                    shuffle_data = [d[shuffle_indices] for d in data]
                else:
                    shuffle_data = data[shuffle_indices]
                shuffle_labels = labels[shuffle_indices]
            else:
                shuffle_data = data
                shuffle_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                
                # If data is a list (e.g., [X_train, ebv_train]), process each part separately
                if isinstance(shuffle_data, list):
                    batch_X = [d[start_index:end_index] for d in shuffle_data]
                    
                else:
                    batch_X = shuffle_data[start_index:end_index]
                
                batch_y = shuffle_labels[start_index:end_index]
                if predict:
                    yield batch_X,None
                    
                else:
                    yield batch_X, batch_y
    
    return num_batches_per_epoch, data_generator()


def evaluation(test_images,test_ebv,test_labels_encode,test_labels,indices_test,zbins_midpoint,z_max,model_checkpoint,config):
    # muiltiple GPUs:
    strategy = tf.distribute.MirroredStrategy()
    print(print('Number of devices: {}'.format(strategy.num_replicas_in_sync)))

    model_name = config['Experiment']['Run_name']

    # evaluate
    if config['Data']['DATA_EBV']:
        test_input = [test_images,test_ebv]
    else:
        test_input = test_images

    # use generator
    test_steps,test_generator = batch_iterator(test_input,test_labels_encode,BATCH_SIZE,predict=False,shuffle=False)

    with strategy.scope():
        # 载入模型
        model = load_model(model_checkpoint)
        # model = ResNet18(180)
        # model.build(input_shape=(None,64,64,4))
        
        # model.load_weights(model_checkpoint)
        # model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        #             optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),#learning_rate=1e-3),
        #             metrics = ['accuracy']
        #     )
        

    # score = model.evaluate(test_input,test_labels_encode,verbose=2)

    score = model.evaluate(x = test_generator,
                            verbose=2,
                            steps = test_steps)
    
    print(f"test loss, test acc:{score}")

    # predict:
    # predictions = model.predict(test_input)

    # evaluate
    test_steps,test_generator = batch_iterator(test_input,test_labels_encode,BATCH_SIZE,predict=True,shuffle=False)
    
    predictions = model.predict(x = test_generator,
                                steps = test_steps
                                ) # predictions.shape = (size,Nbins)
    print(predictions.shape)

    # redshift: 目前只计算Integer zp(z)dz作为红移值，to-do：max pdf对应的红移值
    # 最后一个bin作为额外的bin，只打印概率值，不计入真实值的计算
    # 第一个bin作为小于等于0的红移值，也只打印概率值，不计入真实值的计算
    pred_red = np.sum(predictions[:,1:-1]*zbins_midpoint,axis=1)
    # 加上最大pdf所对应的红移值
    pred_red_max = compute_maxpdf_z(predictions,zbins_midpoint,z_max)

    # 报道Integer zp(z)dz作为红移值
    # metrics
    deltaz,bias,nmad,outlier = compute_metrics(pred_red,test_labels)
    # 计算各个指标随红移的变化：
    metrics_z_plot(pred_red,test_labels,model_name)
    # plot
    make_plot(pred_red,test_labels,deltaz,bias,nmad,outlier,model_name)
    # pdf
    check_probability(test_images,pred_red,test_labels,predictions[:,1:-1],zbins_midpoint,model_name)
    # output
    log_result(deltaz,bias,nmad,outlier,model_name)

    

    # # max pdf对应的红移值
    # deltaz,bias,nmad,outlier = compute_metrics(pred_red_max,test_labels)
    # make_plot(pred_red_max,test_labels,deltaz,nmad,model_name+'_max_pdf')
    # check_probability(test_images,pred_red_max,test_labels,predictions[:,1:-1],zbins_midpoint,model_name+'_max_pdf')
    # log_result(deltaz,bias,nmad,outlier,model_name+'_max_pdf')

    write_result(test_labels,pred_red,indices_test,pred_red_max,predictions,model_name)

    # 分析离群点特征：
    outlier_analysis(f'./output/{model_name}/{model_name}.npz',config['Data']['CATALOG_PATH'],model_name)

    # 上传到Wandb：
    # metrics:
    wandb.define_metric("Bias",summary = "best")
    wandb.define_metric("NMAD",summary = "best")
    wandb.define_metric("Outlier fraction",summary="best")

    log_dict = {
        "Bias":bias,
        "NMAD":nmad,
        "Outlier fraction":outlier
    }
    wandb.log(log_dict,commit=False)

    # images:
    if os.path.exists(f'./output/{model_name}/{model_name}_loss_acc.png'):
        wandb.log({"train_metrics_monitor":wandb.Image(Image.open(f'./output/{model_name}/{model_name}_loss_acc.png'))},commit=False)
    wandb.log({"pdf_check":wandb.Image(Image.open(f'./output/{model_name}/check_pdf_{model_name}.png'))},commit=False)
    wandb.log({"z plot":wandb.Image(Image.open(f'./output/{model_name}/Plot_{model_name}.png')),
                "Residuals":wandb.Image(Image.open(f'./output/{model_name}/Residuals_{model_name}.png'))
                           })
    # 上传指标随红移变化以及离群点分析图：
    if os.path.exists(f'./output/{model_name}/outlier_analysis_{model_name}.png'):
        wandb.log({"outlier_analysis":wandb.Image(Image.open(f'./output/{model_name}/outlier_analysis_{model_name}.png'))})
    if os.path.exists(f'./output/{model_name}/{model_name}_metrics-z.png'):
        wandb.log({"metrics-z":wandb.Image(Image.open(f'./output/{model_name}/{model_name}_metrics-z.png'))})

    

def train(X_train,ebv_train,y_train_label,X_valid,ebv_valid,y_valid_label,Nbins,config):
    # muiltiple GPUs:
    strategy = tf.distribute.MirroredStrategy()
    print(print('Number of devices: {}'.format(strategy.num_replicas_in_sync)))
    
    model_name = config['Experiment']['Run_name']
    with strategy.scope():
        # init model:
        print("===========Model Initialization===========")
        img_shape = X_train.shape[1:]
        model = Pasquet_model(img_shape=img_shape,Nbins=Nbins,reddening_correction=config['Data']['DATA_EBV'],Augm=config['Data']['DATA_AUGMENTATION'])
        # model = ResNet18(Nbins)
        # model.build(input_shape=(None,64,64,4))
        print(model.summary())

        # compile:
        model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),#learning_rate=1e-3),
                    metrics = ['accuracy']
            )
        
        # 断点续训
        if config['Train']['CONTINUE_TRAIN']:
            model.load_weights(f"./weights/{model_name}_validation_best.keras")
    
    # callback
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6,verbose = 1, restore_best_weights = True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f"./weights/{model_name}_validation_best.keras",
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f"./weights/{model_name}_validation_best.weights.h5",
    verbose=1,
    save_best_only=True,
    save_weights_only=False, # False
    monitor='val_loss',
    mode='auto',
    )
    # n_batches = math.ceil(len(X_train)/BATCH_SIZE)
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./weights/validation_{epoch:04d}.keras",
    # verbose=1,
    # save_weights_only=False,
    # monitor='val_loss',
    # save_freq = 5*n_batches,
    # mode='auto',
    # )

    if config['Data']['DATA_EBV']:
        train_input = [X_train,ebv_train]
        valid_input = [X_valid,ebv_valid]
    else:
        train_input = X_train
        valid_input = X_valid
    
    # use generator
    train_steps,train_generator = batch_iterator(train_input,y_train_label,BATCH_SIZE)
    valid_steps,valid_generator = batch_iterator(valid_input,y_valid_label,BATCH_SIZE)
  
    # fit:
    print("===========Fit begining===========")
    if not CONTINUE_TRAIN:
        # model_fit = model.fit(train_input,y_train_label,                   
        #                 batch_size = BATCH_SIZE,
        #                 epochs = EPOCH,                                        
        #                 # validation_split = VALIDATION_SIZE,
        #                 validation_data = (valid_input,y_valid_label),
        #                 # validation_freq=5,
        #                 verbose = 2,
        #                 callbacks = [early_stop_callback,checkpoint_callback] 
        #                 )
        model_fit = model.fit(train_generator,
                            steps_per_epoch=train_steps,
                            epochs=EPOCH,
                            validation_data = valid_generator,
                            validation_steps = valid_steps,
                            verbose = 2,
                            callbacks =  [
                                early_stop_callback,
                                checkpoint_callback,
                                WandbMetricsLogger()
                            ]
                            )

    else:
        model_fit = model.fit(train_generator,      
                        steps_per_epoch=train_steps,             
                        epochs = EPOCH,            
                        initial_epoch=CONTINUE_EPOCH,                            
                        validation_data = valid_generator,
                        validation_steps = valid_steps,
                        # validation_freq=5,
                        verbose = 2,
                        callbacks = [early_stop_callback,checkpoint_callback,WandbMetricsLogger()] 
                        )
    # 绘制loss和accuracy曲线
    best_epoch = print_history(model_fit,model_name)
    
    # save model weight:
    path_save =  "./weights/" + model_name+'.keras'
    model.save(path_save)
    # weight_path = './weights/'+model_name+'.weights.h5'
    # model.save_weights(weight_path)
    

    print(f"Model {model_name} Save Successfully!")
    return best_epoch


def main(only_test=False,best_epoch=30,resume_run=False):
    # 设置GPU
    set_gpu()

    # 读取配置文件
    config = read_config()

    # 创建本次实验所使用的文件夹
    if not os.path.exists(f"./output/{config['Experiment']['Run_name']}"):
        os.mkdir(f"./output/{config['Experiment']['Run_name']}")
    
   
    # init wandb:
    run,config = set_wandb(config,resume_run)


    # load data
    images,labels,ebv = load_data(
        config['Data']['IMG_PATH'],
        config['Data']['LABEL_PATH'],
        config['Data']['EBV_PATH'],
        config['Data']['DATA_TYPE']
        )
    
    
    
    # 图像预处理：裁剪为32*32的图片
    images = img_preprocess(images,config)

    # 为数据集维护一个indices索引
    indices = np.arange(len(images))

    print(images.shape,labels.shape,ebv.shape)


    # 检查数据分布情况和红移范围
    z_min,z_max = check_input(labels,config['Experiment']['Run_name'],'total')
    if config['Data']['Z_USE_DEFAULT']: # 使用事先设定的默认值
        z_min = config['Data']['Z_MIN']
        z_max = config['Data']['Z_MAX']
    
    # set zbins
    zbins_gap = config['Data']['ZBINS_GAP_DEFAULT'] # 默认的zbins_gap
    Nbins = np.min([int((z_max-z_min)/zbins_gap)+2,config['Data']['NBINS_MAX']]) # Nbins与默认值对比取最小的。留两个：第0个为小于等于0的红移；最后一个为大于最大值的红移。

    zbins,zbins_gap = set_zbins(z_min,z_max,Nbins) # 设置zbins
    print(zbins)
    print(zbins.shape,zbins_gap)
    
    # 计算每个红移bin的中值，用于之后使用相同的模型进行预测
    zbins_midpoint = compute_zbins_midpoint(zbins)

    # 保存对应的zbins_midpoint，用于后续测试
    if not only_test:
        np.save(f"./output/{config['Experiment']['Run_name']}/{config['Experiment']['Run_name']}_zbins_midpoint.npy",zbins_midpoint)
        print(f"Save {config['Experiment']['Run_name']}_zbins_midpoint.npy Successfully!")
    
    
    # 数据分割
    X_train, X_test,ebv_train,ebv_test,y_train, y_test,indices_train,indices_test = train_test_split(images,ebv,labels,indices,shuffle=True,test_size=config['Data']['TEST_SIZE'], random_state=42) # 先分训练和测试
    X_train,X_valid,ebv_train,ebv_valid,y_train, y_valid,indices_train,indices_valid = train_test_split(X_train,ebv_train,y_train,indices_train,shuffle=True,test_size=config['Data']['VALIDATION_SIZE'],random_state=42) # 再分训练和验证
    
    # 将红移值编码
    y_train_label = make_classes(zbins,y_train)
    y_test_label = make_classes(zbins,y_test)
    y_valid_label = make_classes(zbins,y_valid)

    
    if not only_test:
        # 检查数据集红移分布情况
        check_dataset_redshift_dist(labels,y_train,y_train_label,y_test,y_test_label,y_valid,y_valid_label,z_max,config['Experiment']['Run_name'])
        # 检查数据集星等分布情况
        df_all = pd.read_csv(config['Data']['CATALOG_PATH'])
        check_dataset_magnitude_dist(df_all,indices_train,indices_valid,indices_test,config['Experiment']['Run_name'])    
    
        # 训练
        best_epoch = train(X_train,ebv_train,y_train_label,X_valid,ebv_valid,y_valid_label,Nbins,config)
    
    # 测试
    # model_checkpoint = "./weights/validation_{:04d}.keras".format(best_epoch)
    model_checkpoint = f"./weights/{config['Experiment']['Run_name']}_validation_best.keras"
    
    evaluation(X_test,ebv_test,y_test_label,y_test,indices_test,zbins_midpoint,z_max,model_checkpoint,config)
    
    # 记录log文件：
    if not only_test:
        arti_log = wandb.Artifact('log',type='log')
        arti_log.add_file('./logs/log.txt',name=config['Experiment']['Run_name']+"_log")
        wandb.log_artifact(arti_log)

    # 记录本次实验的config
    write_config(config)

    wandb.finish()

if __name__ == '__main__':
    main(only_test = False,best_epoch=None,resume_run=False)


