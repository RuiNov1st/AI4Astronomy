import numpy as np
import random
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb

def compute_metrics(pred_red,labels):
    def outlier_compute(deltaz,nmad):
        return len(np.where(np.abs(deltaz)>5*nmad)[0])/len(deltaz)
    # delta z
    deltaz = (pred_red - labels)/(1+labels)
    # bias
    bias = np.mean(deltaz)
    # NMAD
    nmad = 1.48*(np.median(abs(deltaz-np.median(deltaz))))
    # outlier:
    outlier = outlier_compute(deltaz,nmad)
    
    return deltaz,bias,nmad,outlier

def compute_maxpdf_z(predictions,zbins_midpoint,z_max):
    """
    计算最大pdf所对应的红移值
    """
    # 选取最大的概率值对应的bin的中值作为红移预测值
    pred_red_max_idx = np.argmax(predictions,axis=1)
    
    pred_red_max = []
    for i in pred_red_max_idx:
        if i == 0: # 第0个bin用最小值替代
            pred_red_max.append(z_min)
        elif i==len(predictions)-1: # 最后一个bin用最大值替代
            pred_red_max.append(z_max) # 超过最大阈值的红移值
        else:
            pred_red_max.append(zbins_midpoint[i-1]) # count from 1
    pred_red_max = np.array(pred_red_max)

    return pred_red_max


''' Function that makes plots '''
def make_plot(pred_red,labels,deltaz,nmad,model_name = 'model'):
    """
        Makes a histogram of the prediction bias and a plot of the estimated Photometric redshift 
        and the Prediction bias versus the Spectroscopic redshift given the labels, red, dz, smad,
        name and lim and it saves both of them in .png files. (Mostly used for debugging purposes)


        Arguments:
            labels (ndarray): The labels for the images used.

            red (ndarray): Contains the predicted redshift values for the test images.

            dz (ndarray): Residuals for every test image.

            smad (float):The MAD deviation.

            name (str): The name of the model.

            lim (float): The limit of the axes of the plot

    """
    # Constructing the filenames
    
    file_name_01 = f'./output/Residuals_{model_name}.png'
    file_name_02 = f'./output/Plot_{model_name}.png'
    # Plotting the residuals 
    plt.figure()
    plt.hist(deltaz, bins=100)
    plt.xticks(np.arange(-0.1, 0.1, step=0.02))
    plt.xlim(-0.1,0.1)
    plt.xlabel('Δz')
    plt.ylabel('Relative Frequency')
    plt.title('Δz Distribution')
    plt.savefig(file_name_01, bbox_inches='tight')

    # Plotting the predictions vs the data 
    fig = plt.figure(figsize = (3,3))
    axis = fig.add_axes([0,0.4,1,1])
    axis2 = fig.add_axes([0,0,1,0.3])
    axis.set_ylabel('Photometric Redshift')
    axis2.set_ylabel('bias (Δz)')
    axis2.set_xlabel('Spectroscopic Redshift')
    lim = np.max(labels)
    axis.plot([0, lim], [0, lim], 'k-', lw=1)
    axis.set_xlim([0, lim])
    axis.set_ylim([0,lim])
    axis2.plot([0, lim], [nmad, nmad], 'k-', lw=1)
    axis2.set_xlim([0, lim])
    # axis.plot(labels,red,'ko', markersize=0.3, alpha=0.3)
    axis.scatter(labels,pred_red,marker='o',color='k',s=0.3,alpha=0.3)
    #axis.hist2d(labels,red,bins =150)
    # axis2.plot(labels,  np.asarray(red) - np.asarray(labels),'ko', markersize=0.3, alpha=0.3)
    axis2.scatter(labels,  np.asarray(pred_red) - np.asarray(labels),color='k',marker='o',s=0.3,alpha=0.3)
    
    
    plt.savefig(file_name_02,dpi = 300,transparent = False,bbox_inches = 'tight')


def plot_probability(pred_red,label,predictions,zbins,predict=False):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    pred_red_label = np.digitize(pred_red,zbins,right=True)
    if not predict:
        true_red_label = np.digitize(label,zbins,right=True)
    """
    thisplot = plt.bar(range(len(zbins)),predictions,color = "#777777")
    # plt.ylim([0,1])
    
    # encode:
    pred_red_label = np.digitize(pred_red,zbins,right=True)
    # 防止超出范围
    if pred_red_label<len(zbins):
        thisplot[pred_red_label].set_color('red')

    if not predict:
        true_red_label = np.digitize(label,zbins,right=True)
        if true_red_label<len(zbins):
            thisplot[true_red_label].set_color('green')
    """
    plt.plot(range(len(zbins)),predictions)
    plt.axvline(pred_red_label,linestyle='--',color='red')
    if not predict:
        plt.axvline(true_red_label,linestyle='-.',color='green')
    

    
def plot_img(image,pred_red,label,predict=False):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    # 图像处理以显示出图形：
    # image: g,r,i,z
    g = image[:,:,0]
    r = image[:,:,1]
    i = image[:,:,2]
    rgb = make_lupton_rgb(i,r,g,Q=10,stretch=0.05)
    plt.imshow(rgb,origin='lower')
    # plt.imshow(image[:,:,:3])# select the first 3 bands
    if predict:
        plt.title(f"pred(r):{round(pred_red,3)}")
    else:
        plt.title(f"pred(r):{round(pred_red,3)} true(g):{round(label,3)}")


def check_probability(images,pred_red,labels,predictions,zbins,model_name,predict=False):
    """
    随机检查9个预测的PDF情况
    """
    
    num_rows,num_cols=3,3
    num_img = num_rows * num_cols
    idx_list = [random.randint(0,len(pred_red)-1) for i in range(num_img)]
    plt.figure(figsize=(2*2*num_cols,2*num_rows))
    for i in range(num_img):
        idx = idx_list[i]
        plt.subplot(num_rows,2*num_cols,2*i+1)
        if predict:
            plot_img(images[idx],pred_red[idx],None,predict)
            plt.subplot(num_rows,2*num_cols,2*i+2)
            plot_probability(pred_red[idx],None,predictions[i],zbins,predict)
        else:
            plot_img(images[idx],pred_red[idx],labels[idx],predict)
            plt.subplot(num_rows,2*num_cols,2*i+2)
            plot_probability(pred_red[idx],labels[idx],predictions[i],zbins,predict)

    
    # plt.show()
    plt.savefig(f'./output/check_pdf_{model_name}.png',dpi=200)


def print_history(history,model_name):
    fig = plt.figure()
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model accuracy&loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Val_acc', 'Train_loss', 'Val_loss'])
    plt.savefig(f'./output/{model_name}_loss_acc.png')
    np.save(f'./output/{model_name}_val_loss.npy', np.array(history.history['val_loss']))
    np.save(f'./output/{model_name}_val_accuracy.npy', np.array(history.history['val_accuracy']))


    print("Save loss&accuracy Curve successfully!")

    # 找到最小loss和最大acc处
    print("min val_loss:",sep='\t')
    print(int(np.argmin(history.history['val_loss']))+1) # count from 1
    print("max val_acc:",sep='\t')
    print(int(np.argmax(history.history['val_accuracy']))+1)
    return int(np.argmin(history.history['val_loss']))+1

    
