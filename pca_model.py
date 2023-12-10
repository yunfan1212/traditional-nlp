#coding=utf8

'''pca算法用于降维'''
#sklearn.decomposition 降维模块
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

def PCA_Model():
    pca=PCA(n_components=2)    #降低的维度为       min(n_components,size(0),size(1))

    data=np.random.random(size=(40,64))
    print(data.shape)
    new_pca=pca.fit_transform(data)          #fit_transform 是fit 与transform的结合，fit()为训练，transform进行数据转换
    print(new_pca.shape)
    new_pca=pd.DataFrame(new_pca)
    print("降维后的数据：")
    print(new_pca)

if __name__ == '__main__':
    PCA_Model()





