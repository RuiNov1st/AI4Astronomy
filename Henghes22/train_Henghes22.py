from data.dataset import make_dataset
import numpy as np
import os
import math
import sys
import wandb # weights & bias experiment track
from PIL import Image
import pandas as pd
from models.Henghes22 import Henghes22_model
import torch
from utils import set_gpu,read_config,log_result,write_result
from torch import nn
import torch.nn.functional as F
from visual import training_monitor
from metrics import compute_metrics,metrics_z_plot,make_plot
from analysis import outlier_analysis
from torch.nn.parallel import DataParallel 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
# from torchinfo import summary

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def evaluation(test_dataloader,model_checkpoint,config,Nbins,gpu_flag,device):
    run_name = config['Experiment']['Run_name']

    model = torch.load(model_checkpoint)
    if gpu_flag:
        model = model.to(device[0])
        if len(device)>1:
            model = torch.nn.DataParallel(model,device_ids=device)
    

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    test_loss = 0.

    # use for metrics compute:
    output_arr = []
    label_arr = []
    indice_arr = []

    model.eval()
    with torch.no_grad():
        for idx,data in enumerate(test_dataloader):
            image, label,catalog_data,indice = data
            # to cuda:
            image,label,catalog_data = image.to(device[0]),label.to(device[0]),catalog_data.to(device[0])

            output = model(image,catalog_data)
            probability = output.view(-1) # pdf
            loss = torch.sqrt(loss_fn(output.view(-1),label))

            test_loss+=loss.item()

            # log data:
            output_arr.append(probability)
            label_arr.append(label)
            indice_arr.append(indice)
            
        print(f"loss/test:{test_loss/(idx+1)}")
    
    pred_red = np.array(torch.cat(output_arr, dim=0).cpu())  # Concatenate along the first dimension
    true_red = np.array(torch.cat(label_arr, dim=0).cpu())    # Concatenate labels
    indice_arr = np.array(torch.cat(indice_arr, dim=0).cpu())  # Concatenate indices

    # metrics compute:
    deltaz,bias,nmad,outlier = compute_metrics(pred_red,true_red)
    # 计算各个指标随红移的变化：
    metrics_z_plot(pred_red,true_red,run_name)
    # plot
    make_plot(pred_red,true_red,deltaz,bias,nmad,outlier,run_name)
    # output
    log_result(deltaz,bias,nmad,outlier,run_name)
    write_result(true_red,pred_red,indice_arr,None,run_name)
    # 分析离群点特征：
    outlier_analysis(f'./output/{run_name}/{run_name}.npz',config['Data']['CATALOG_PATH'],run_name)
    





"""
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
"""

class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.min_delta = float('inf')
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.min_delta = 0.5*self.min_validation_loss
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def train(train_dataloader,valid_dataloader,config,Nbins,gpu_flag,device):
    
    # define model:
    model = Henghes22_model(Nbins)
    # print(summary(model,input_size=(config['Train']['BATCH_SIZE'],4,64,64)))
    if gpu_flag:
        model = model.to(device[0])
        if len(device)>1:
            model = torch.nn.DataParallel(model,device_ids=device)
    
    
    model_save_path = f"./weights/{config['Experiment']['Run_name']}.pth"

    # define loss and optimizer
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config['Train']['LEARNING_RATE'])

    running_loss_list = []
    val_loss_list = []
    best_epoch = 0
    best_vloss = np.inf
    
    validation_freq = 1
    # early stopper:
    early_stopper = EarlyStopper(patience=10)

    # training:
    for e in range(config['Train']['EPOCH']):
        print(f"Epoch {e}:",end=' ')
        # make sure gradient tracking is on 
        model.train(True)
        running_loss = 0. # loss around one epoch in training

        for idx,data in enumerate(train_dataloader):
            image, label,catalog_data,indice = data
            # to cuda:
            label = label.to(torch.float32)
            image,label,catalog_data = image.to(device[0]),label.to(device[0]),catalog_data.to(device[0])

            optimizer.zero_grad()

            output = model(image,catalog_data)
            loss = torch.sqrt(loss_fn(output.view(-1),label))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"loss/train:{running_loss/(idx+1)}",end=' ')

        running_loss_list.append(running_loss/(idx+1))

        
        # validation
        if e % validation_freq == 0:
            model.eval() # set the model to evaluation mode
            # disable gradient computation and reduce memory consumption
            with torch.no_grad():
                val_loss = 0.
                for idx,data in enumerate(valid_dataloader):
                    image, label,catalog_data,indice = data
                    image,label,catalog_data = image.to(device[0]),label.to(device[0]),catalog_data.to(device[0])
                    voutput = model(image,catalog_data)
                    vloss = torch.sqrt(loss_fn(voutput,label))

                    val_loss+=vloss.item()
                
                val_loss = val_loss/(idx+1)
                val_loss_list.append(val_loss)
                print(f"loss/valid:{val_loss}")

                # gain improvement in validation set:
                if val_loss<best_vloss:
                    best_vloss = val_loss
                    best_epoch = e
                    torch.save(model,model_save_path)
                    print(f"Best Epoch currently is {e}. Best Val loss:{best_vloss}")
                
                # judge whether to stop:
                if early_stopper.early_stop(val_loss):
                    break
    
    # visual loss:
    training_monitor(running_loss_list,val_loss_list,config['Experiment']['Run_name'])




def main(only_test=False,best_epoch=30,resume_run=False):
    # 设置GPU
    gpu_flag,device = set_gpu()
    setup(rank, world_size)

    # 读取配置文件
    config = read_config()

    # 创建本次实验所使用的文件夹
    if not os.path.exists(f"./output/{config['Experiment']['Run_name']}"):
        os.mkdir(f"./output/{config['Experiment']['Run_name']}")
    
    # init wandb:
    # run,config = set_wandb(config,resume_run)

    # generate dataloder:
    train_loader,val_loader,test_loader,catalog,zbins_midpoint,Nbins = make_dataset(config) # image,label,catalog,indice

    if not only_test:
        train(train_loader,val_loader,config,Nbins,gpu_flag,device)
    

    model_checkpoint = f"./weights/{config['Experiment']['Run_name']}.pth"
    evaluation(test_loader,model_checkpoint,config,Nbins,gpu_flag,device)
    """
    # 记录log文件：
    # upload_wandb()
    
    # 记录本次实验的config
    # write_config(config)

    # wandb.finish()
    """

if __name__ == '__main__':
    main(only_test = False,best_epoch=None,resume_run=False)


