import numpy as np
import pandas as pd
test_data=pd.read_csv('test_data.csv',encoding='gbk')
#导入数据
test_data.drop('route',axis=1,inplace=True)
#去除无用列
attr=test_data.columns
type1=[1,2,2,1,2,2]
test_data.rename(columns={'distance':0,
                          'sta_old_pop':1,
                          'des_old_pop':2,
                          'toll_num':3,
                          'sta_old_volume':4,
                          'des_old_volume':5},inplace=True)
test_data=test_data.iloc[:,[0,1,2,3,4,5]]
#更换标签
def scaler(data,type,xmin,xmax):
    m = data.iloc[:, 0].size
    n = data.columns.size
    for i in range(m):
        for j in range(n):
            if type1[j] == 1:  # 正向指标
                data.loc[i, j] = (xmax - xmin) * (data.loc[i, j] - min(data[j])) / (max(data[j]) - min(data[j])) + xmin
            else:
                data.loc[i, j] = (xmax - xmin) * (max(data[j]) - data.loc[i, j]) / (max(data[j]) - min(data[j])) + xmin
    return data
test_data=scaler(test_data,type1,0.002,0.996)
photo_data=test_data.copy()
#标准化方法二：需要调试，修改xmin、xmax（归一化的区间端点）
m=test_data.iloc[: , 0].size#数据集行数
n=test_data.columns.size#数据集列数
P=test_data.copy()
SUM=[]
for j in range(n):
    SUM.append(test_data[j].sum())
for i in range(m):
    for j in range(n):
        P.loc[i,j]=P.loc[i,j]/SUM[j]
#第j个指标下，第i个样本占该指标的比重
import math
k=1/math.log(m)
e=[]#存储熵值
for j in range(n):
    sum1=0
    for i in range(m):
        t=P.loc[i,j]*math.log(P.loc[i,j])
        sum1+=t
    e.append(-k * sum1)
e = [1 - i for i in e]
#信息熵冗余度
w=[]
for j in range(n):
    w.append(e[j]/sum(e))
#各项指标的权重
print(w)
S=[]#各指标压力系数
for i in range(m):
    sum2=0
    for j in range(n):
        t=w[j]*test_data.loc[i,j]
        sum2+=t
    S.append(sum2)
print(S)
#可视化
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
fig=plt.figure(figsize=(3,4))
name=[u'线路全长',u'起始城市人口数',u'终点城市人口数',u'收费站数量',u'起始站车流量',u'终点站车流量']
for i in range(n):
    plt.subplot(2,3,i+1)
    sns.boxplot(photo_data[i],
                showmeans=True,
                orient='v',
                width=0.5,
                meanprops = {"marker":"D","markerfacecolor":"black"}
                )
    plt.ylabel(u'数值范围')
    plt.xlabel(name[i])
plt.suptitle(u'归一化后各指标分布',size=24)
plt.show()
plt.plot(name,w)
plt.xticks(rotation=45)
plt.xlabel(u'压力指标')
plt.ylabel(u'权重值')
plt.suptitle(u'各指标所占权重',size=24)
plt.show()