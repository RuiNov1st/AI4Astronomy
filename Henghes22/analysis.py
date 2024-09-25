import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from metrics import compute_metrics



def galaxy_type_fun(row):
    if not isinstance(row['main_type'],str):
        return row['SUBCLASS']
    elif not isinstance(row['SUBCLASS'],str):
        return row['main_type']
    else:
        return f"{row['main_type']}_{row['SUBCLASS']}"



def feature_visual(normal_df,outlier_df,df,model_name):
   # 检查特征上的共性：
    patterns = ['z_spec', 'MAG_G', 'MAG_R', 'MAG_I', 'MAG_Z', 'MAG_W1',
        'MAG_W2', 'ra_2', 'dec_2', 'ebv','galaxy_type']
    means_diff = []

    # 创建一个图形对象，指定大小
    fig = plt.figure(figsize=(10, 15))
    plt.subplots_adjust(wspace=0.3, hspace=0.5) 
    # 创建5行2列的GridSpec对象
    gs = gridspec.GridSpec(int(np.ceil(len(patterns)/2)), 2)

    # 创建11个子图
    for i in range(11):
        p = patterns[i]
        # 如果 i >= 10，需要单独调整最后一个子图的位置
        if i == 10:
            ax = fig.add_subplot(gs[-1, :])  # 最后一个子图占据整行
            normal_galaxy_type_counts = normal_df[p].value_counts(normalize=True).sort_index()
            outlier_galaxy_type_counts = outlier_df[p].value_counts(normalize=True).sort_index()

            normal_galaxy_type_name = list(normal_galaxy_type_counts.index)
            normal_galaxy_type_count = list(normal_galaxy_type_counts.values)
            outlier_galaxy_type_name = list(outlier_galaxy_type_counts.index)
            outlier_galaxy_type_count = list(outlier_galaxy_type_counts.values)

            ax.plot(normal_galaxy_type_name,normal_galaxy_type_count,label='normal')
            ax.scatter(normal_galaxy_type_name,normal_galaxy_type_count)
            ax.plot(outlier_galaxy_type_name,outlier_galaxy_type_count,label='outlier',color='orange')
            ax.scatter(outlier_galaxy_type_name,outlier_galaxy_type_count,color='orange')

            ax.set_title(p)
            ax.legend()
            
        else:
            ax = fig.add_subplot(gs[i // 2, i % 2])  # 其余子图按行列放置
            
            counts_normal, bin_edges_normal = np.histogram(normal_df[p], bins=10,range=(df[df[p].apply(np.isfinite)][p].min(),df[df[p].apply(np.isfinite)][p].max()))
            probabilities_normal = counts_normal / len(normal_df[p])
            counts_outlier, bin_edges_outlier = np.histogram(outlier_df[p], bins=10,range=(df[df[p].apply(np.isfinite)][p].min(),df[df[p].apply(np.isfinite)][p].max()))
            probabilities_outlier = counts_outlier / len(outlier_df[p])
            ax.plot((bin_edges_normal[:-1]+bin_edges_normal[1:])/2, probabilities_normal,label='normal')
            ax.scatter((bin_edges_normal[:-1]+bin_edges_normal[1:])/2, probabilities_normal)
            ax.axvline(x = np.mean(normal_df[p]),linestyle='--')
            ax.plot((bin_edges_outlier[:-1]+bin_edges_outlier[1:])/2, probabilities_outlier,label='outlier',color='orange')
            ax.scatter((bin_edges_outlier[:-1]+bin_edges_outlier[1:])/2, probabilities_outlier,color='orange')
            ax.axvline(x = np.mean(outlier_df[p]),color='orange',linestyle='--')
            ax.legend()
            ax.set_title(p)
            # means_diff.append(np.abs(np.mean(normal_df[p])-np.mean(outlier_df[p])))
    
    # means_dict = dict(zip(patterns,means_diff))
    plt.suptitle("Outliers Analysis",y=0.93)
    # 保存图片：
    plt.savefig(f'./output/{model_name}/outlier_analysis_{model_name}.png')

        

def outlier_analysis(data_path,catalog_path,model_name):
    # 读入待分析的数据：
    data = np.load(data_path)
    z_pred = data['z_pred']
    z_true = data['z_true']
    test_indices = data['indices']
    df = pd.read_csv(catalog_path)

    # 计算df的gaxly type:结合main_type和subclass
    df['galaxy_type'] = df.apply(lambda x:galaxy_type_fun(x),axis=1)


    # 离群点计算：
    deltaz,bias,nmad,outlier = compute_metrics(z_pred,z_true)
    outlier_indices_temp = np.where(np.abs(deltaz)>0.15)[0]
    outlier_indices = test_indices[outlier_indices_temp]
    # 正常点：
    normal_indices_temp = np.where(np.abs(deltaz)<=0.15)[0]
    normal_indices = test_indices[normal_indices_temp]
    # 离群点和正常点的df：
    normal_df = df.iloc[normal_indices]
    outlier_df = df.iloc[outlier_indices]

    # 分析离群点特征：
    feature_visual(normal_df,outlier_df,df,model_name)


if __name__  == '__main__':
    data_path = '/data/home/wsr/Workspace/dl/Algorithm/Pasquet19_wsr/output/EmG_AGN_Pasquet19_0916_wise2/EmG_AGN_Pasquet19_0916_wise2.npz'
    catalog_path = '/data/home/wsr/Workspace/dl/dr10_dataset/EmG_AGN/dataset/fullband/EmG_AGN_all_fullband.csv'
    model_name = 'EmG_AGN_Pasquet19_0916_wise2'
    
    outlier_analysis(data_path,catalog_path,model_name)

    
