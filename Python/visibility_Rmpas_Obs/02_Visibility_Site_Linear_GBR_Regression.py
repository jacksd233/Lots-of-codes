# -*- coding: utf-8 -*-
"""
#**********************************************
Created on 2021, Aug 25, Ji Weiwen

尝试机器学习对六个站点模式能见度订正

#**********************************************
Verified on 2022.1.19, Ji Weiwen
对六站点数据进行线性回归拟合、梯度提升回归（GBR）

"""

# In[0]
### 采用算法

# 梯度提升算法， https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py
# gradientboostingregressor https://www.huaweicloud.com/articles/69b02033464d3a98c1db9944e125a42f.html    

# 神经网络：https://blog.csdn.net/kun_csdn/article/details/88853907
#         https://zhuanlan.zhihu.com/p/58964140

### ETS评估
# https://confluence.ecmwf.int/display/FUG/Equitable+Threat+Score
# http://www.atmos.albany.edu/daes/atmclasses/atm401/spring_2014/Scores1.pdf
# https://www.jma.go.jp/jma/jma-eng/jma-center/nwp/outline2013-nwp/pdf/outline2013_Appendix_A.pdf

# In[0.5]
### 定义函数

# 计算Equitable Threat Score
def ets_cal(y_obs,y_pred):
    low_vis = 10 # 设定10km以下的观测能见度为低值
    obs_low_vis = np.logical_and(y_obs <= low_vis,y_pred > low_vis) # 观测低能见度次数
    pred_low_vis = np.logical_and(y_pred <= low_vis, y_obs > low_vis) # 模拟低能见度次数
    hit_low_vis = np.logical_and((y_obs < low_vis), (y_pred < low_vis)) 
    # 模拟到的低能见度次数
    ETS_ref = (len(y_obs[hit_low_vis])+len(y_obs[obs_low_vis]))*(len(y_obs[hit_low_vis])+len(y_pred[pred_low_vis]))/(len(y_obs))
    ETS = (len(y_obs[hit_low_vis])-ETS_ref)/(len(y_obs[hit_low_vis])+ len(y_obs[obs_low_vis])+len(y_pred[pred_low_vis])-ETS_ref)
    return ETS
  
def ts_cal(y_obs,y_pred):
    low_vis = 10 # 设定10km以下的观测能见度为低值
    obs_low_vis = np.logical_and(y_obs <= low_vis,y_pred > low_vis) # 观测低能见度次数
    pred_low_vis = np.logical_and(y_pred <= low_vis, y_obs > low_vis) # 模拟低能见度次数
    hit_low_vis = np.logical_and((y_obs < low_vis), (y_pred < low_vis)) 
    # 模拟到的低能见度次数
    # ETS_ref = (len(y_obs[hit_low_vis])+len(y_obs[obs_low_vis]))*(len(y_obs[hit_low_vis])+len(y_pred[pred_low_vis]))/(len(y_obs))
    ETS = (len(y_obs[hit_low_vis]))/(len(y_obs[hit_low_vis])+ len(y_obs[obs_low_vis])+len(y_pred[pred_low_vis]))
    return ETS
      



# In[1]
### 多元线性回归拟合

##################################     定义函数       ####################################
#(1) 计算Equitable Threat Score
# def ets_cal(y_obs,y_pred):
#     low_vis = 10 # 设定10km以下的观测能见度为低值
#     obs_low_vis = np.logical_and(y_obs <= low_vis,y_pred > low_vis) # 观测低能见度次数
#     pred_low_vis = np.logical_and(y_pred <= low_vis, y_obs > low_vis) # 模拟低能见度次数
#     hit_low_vis = np.logical_and((y_obs < low_vis), (y_pred < low_vis)) 
#     # 模拟到的低能见度次数
#     ETS_ref = (len(y_obs[hit_low_vis])+len(y_obs[obs_low_vis]))*(len(y_obs[hit_low_vis])+len(y_pred[pred_low_vis]))/(len(y_obs))
#     ETS = (len(y_obs[hit_low_vis])-ETS_ref)/(len(y_obs[hit_low_vis])+ len(y_obs[obs_low_vis])+len(y_pred[pred_low_vis])-ETS_ref)
#     return ETS

# Threat Score!!!
# 计算TS评分
def ts_cal(y_obs,y_pred):
    low_vis = 1 # 设定10km以下的观测能见度为低值
    obs_low_vis = np.logical_and(y_obs <= low_vis,y_pred > low_vis) # 观测低能见度次数
    pred_low_vis = np.logical_and(y_pred <= low_vis, y_obs > low_vis) # 模拟低能见度次数
    hit_low_vis = np.logical_and((y_obs < low_vis), (y_pred < low_vis)) 
    # 模拟到的低能见度次数
    # ETS_ref = (len(y_obs[hit_low_vis])+len(y_obs[obs_low_vis]))*(len(y_obs[hit_low_vis])+len(y_pred[pred_low_vis]))/(len(y_obs))
    TS = (len(y_obs[hit_low_vis]))/(len(y_obs[hit_low_vis])+ len(y_obs[obs_low_vis])+len(y_pred[pred_low_vis]))
    return TS

###################################     导入库        ####################################
import glob,os
import numpy as np
import pandas as pd

# 引入多元线性回归算法模块
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


###################################     读取数据       ####################################
# Mac
Data_Path = r'/Users/jiweiwen/Dropbox/Vis/Python_Work/Visibility' # 六个站点的观测数据，观测数据时间为世界时间

# Windows
# Data_Path = r'C:\Users\季伟文\Dropbox\Vis\Python_Work\Visibility' # 模式数据与观测数据

# 读取模式输出的六个站点位置的能见度结果，模式时间为北京时间，从2020.10.01 20:00，世界时间12:00开始
npy_name = "*_clean.npy"
npy_file = glob.glob(os.path.join(Data_Path,npy_name))
npy_file.sort()

# 读取模式数据格点对应的经纬度
Data = [[] for i in range(5)]
for i in range(5):
    # print(npy_file[i])
    Data[i] = np.load(npy_file[i],allow_pickle=True)

# 数据Data[i][j][k]
# i:0-4，依次为pm10(ug/m3)，pm2.5(ug/m3)，rh2，vis_obs,vis_rmaps
# j:0-5,六个站点
# k, 筛选后每个站点不同时刻的数据


####################################     回归拟合      #####################################

x_train_rmaps = [[] for i in range(6)]
x_test_rmaps  = [[] for i in range(6)]
y_train   = [[] for i in range(6)]
y_test    = [[] for i in range(6)]


y_predict = [[] for i in range(6)]

MSE_ini   = [[] for i in range(6)]
MSE       = [[] for i in range(6)]
coef_lr   = [[] for i in range(6)]
inter_lr  = [[] for i in range(6)]
ini_r2    = [[] for i in range(6)]
train_r2  = [[] for i in range(6)]
ini_test  = [[] for i in range(6)]
test_r2   = [[] for i in range(6)]
lr_r2     = [[] for i in range(6)]
ets_ini   = [[] for i in range(6)]
ets_train = [[] for i in range(6)]
ets_ini_test = [[] for i in range(6)]
ets_test  = [[] for i in range(6)]
ets_total = [[] for i in range(6)]
for ii in range(6):
    y = Data[3][ii][:]
    x = [[] for i in range(4)]
    x_i = [0,1,2,4]
    for i in range(4):
        x[i] = Data[x_i[i]][ii][:]
    
    # sc_X = StandardScaler()
    # x[0:3] = sc_X.fit_transform(x[0:3])
    # x[3] = Data[4][ii][:]
    lr = LinearRegression()
    x = pd.DataFrame(x)
    para_names = ["PM10","PM2.5","RH2","VIS_Rmaps"]
    x.index = para_names
    x = x.T
    # Cross validate
    # result = cross_validate(lr, x, y)
    
    
    
    x_train1,x_test1,y_train[ii],y_test[ii] = train_test_split(x,y,train_size=0.8)#,shuffle=False)#random_state=23)
    scaler = StandardScaler()
    scaler.fit(x_test1)
    x_test = scaler.transform(x_test1)
    scaler.fit(x_train1)
    x_train = scaler.transform(x_train1)
    x_train = pd.DataFrame(x_train)
    x_test  = pd.DataFrame(x_test)
    x_train.columns = para_names
    x_test.columns  = para_names
    
    # x_train,x_test,y_train[ii],y_test[ii] = train_test_split(x,y,train_size=0.80,shuffle=False)
    x_train_rmaps[ii] = x_train1['VIS_Rmaps']
    x_test_rmaps[ii]  = x_test1['VIS_Rmaps']
    lr.fit(x_train,y_train[ii])
    y_predict[ii] = lr.predict(x_test)
    
    coef_lr[ii]  = lr.coef_
    inter_lr[ii] = lr.intercept_
    ini_r2[ii]   = r2_score(y,x['VIS_Rmaps'])
    train_r2[ii] = r2_score(y_train[ii],lr.predict(x_train))
    cor1 = np.corrcoef(y_test[ii],np.array(x_test['VIS_Rmaps']))
    ini_test[ii] = cor1[0][1]# r2_score(y_test[ii],np.array(x_test['VIS_Rmaps']))
    cor2 = np.corrcoef(y_test[ii],y_predict[ii])
    test_r2[ii]  = cor2[0][1] # r2_score(y_test[ii],y_predict[ii])
    # ini_test[ii] = r2_score(y_test[ii],np.array(x_test1['VIS_Rmaps']))
    # test_r2[ii]  = r2_score(y_test[ii],y_predict[ii])
    lr_r2[ii]    = r2_score(y,lr.predict(x))
    MSE_ini[ii] = mean_squared_error(y_test[ii], np.array(x_test['VIS_Rmaps'])) 
    MSE[ii]     = mean_squared_error(y_test[ii], y_predict[ii])
    
    ets_ini[ii]   = ts_cal(y, x['VIS_Rmaps'])
    ets_train[ii] = ts_cal(y_train[ii], lr.predict(x_train)) 
    ets_ini_test[ii] = ts_cal(y_test[ii],np.array(x_test1['VIS_Rmaps']))
    ets_test[ii]  = ts_cal(y_test[ii], y_predict[ii])
    ets_total[ii] = ts_cal(y,lr.predict(x))
    del(x,y)
 


#####################################     输出参数      #######################################
# print("cross_validate score:",result['test_score'])

# print("linear regression:")
# print("coef（各项系数）",lr.coef_)
# print("intercept（常数项）",lr.intercept_)

# print("r2_score initial:",r2_score(y,x['VIS_Rmaps']))
# print("r2_score train:",r2_score(y_train,lr.predict(x_train)))
# print("R2_score initial test:",r2_score(y_test,np.array(x_test['VIS_Rmaps'])))
# print("r2_score test:",r2_score(y_test,y_predict))
# print("r2_score lr:",r2_score(y,lr.predict(x)))

# print("ETS initial is:",ets_cal(y, x['VIS_Rmaps']))
# print("ETS train is:",ets_cal(y_train, lr.predict(x_train)))
# print("ETS boost test is:",ets_cal(y_test, y_predict))
# print("ETS boost total is:",ets_cal(y,lr.predict(x)))



# print("\n")
# print("mean squared error:",mean_squared_error(y_test,y_predict))
# print("mean absolute error:",mean_absolute_error(y_test,y_predict))
# print("r2_score:",r2_score(y_test,y_predict))
# print("Regression score:",lr.score(x_test,y_test))
# print(lr.coef_)
# print(np.argsort(lr.coef_))

train_text = [[]for i in range(6)]
test_text  = [[]for i in range(6)]
for i in range(6):
    train_text[i].append(ini_test[i])
    train_text[i].append(ets_ini_test[i])
    test_text[i].append(test_r2[i])
    test_text[i].append(ets_test[i])
    
train_text = np.around(train_text,3)
test_text  = np.around(test_text,3)

#####################################     绘图       ############################################
# %matplotlib inline 
loc_name = ['zhuo zhou','gu an','nan yuan','lang fang','yong qing','da xing']

plt.figure(dpi=600,figsize=(6,16))
# plt.rcParams['font.family'] = ['Heiti']
for j in range(6):
    ax = plt.subplot(6,2,1+j*2)
    
    ax.scatter(y_test[j], x_test_rmaps[j],s=2,marker='.')
    ax.set_xlim([0, 31])
    ax.set_ylim([0, 31])
    ax.set_ylabel("Vis_Rmaps (km)")
    at = AnchoredText(loc_name[j]+'\n'+'Corr, TS\n'+str(train_text[j]),
              prop=dict(size=6), frameon=True,
              loc='upper right',
              )    
    ax.add_artist(at)
    x_y_ticks = np.arange(0,32,5)
    if j==0:
        ax.set_title("Before Regression")
    else:
        ax.set_title("")
    if j==5:
        ax.set_xlabel("Vis_obs (km)")
        ax.set_xticks(np.arange(0,32,5))
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    
    ax=plt.subplot(6,2,2+j*2)
    ax.scatter(y_test[j], y_predict[j],s=2,marker='.')
    ax.set_xlim([0, 31])
    ax.set_ylim([0, 31])
    ax.tick_params(labelleft=False)
    at = AnchoredText(loc_name[j]+'\n'+'Corr, TS\n'+str(test_text[j]),
              prop=dict(size=6), frameon=True,
              loc='upper right',
              )
    ax.add_artist(at)
    x_y_ticks = np.arange(0,32,5)   
    if j==0:
        ax.set_title("After Regression")
    else:
        ax.set_title("")
    if j==5:
        ax.set_xlabel("Vis_obs (km)")
        ax.set_xticks(np.arange(0,32,5))
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

#%% Linear Regression
# 画折线图

loc_name = ['zhuo zhou','gu an','nan yuan','lang fang','yong qing','da xing']

plt.figure(dpi=600,figsize=(6,16))

for j in range(6):
    ax = plt.subplot(6,2,1+j*2)
    
    ax.plot(y_test[j])
    ax.plot(np.array(x_test_rmaps[j]),'--',linewidth=1)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 31])
    ax.set_ylabel("Vis_Rmaps (km)")
    # at = AnchoredText(loc_name[j]+'\n'+'R2, ETS\n'+str(train_text[j]),
    #           prop=dict(size=6), frameon=True,
    #           loc='upper right',
    #           )    
    # ax.add_artist(at)
    # x_y_ticks = np.arange(0,32,5)
    if j==0:
        ax.set_title("Before Regression")
    else:
        ax.set_title("")
    if j==5:
        ax.set_xlabel("Vis_obs (km)")
        # ax.set_xticks(np.arange(0,32,5))
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    
    ax=plt.subplot(6,2,2+j*2)
    ax.plot(y_test[j])
    ax.plot(y_predict[j],'--',linewidth=1)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 31])
    ax.tick_params(labelleft=False)
    # at = AnchoredText(loc_name[j]+'\n'+'R2, ETS\n'+str(test_text[j]),
    #           prop=dict(size=6), frameon=True,
    #           loc='upper right',
    #           )
    # ax.add_artist(at)
    # x_y_ticks = np.arange(0,32,5)   
    if j==0:
        ax.set_title("After Regression")
    else:
        ax.set_title("")
    if j==5:
        ax.set_xlabel("Vis_obs (km)")
        # ax.set_xticks(np.arange(0,32,5))
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)


# In[2]
### Gradient Boosting Regression


###########################################    定义函数    ###########################################
# 计算Equitable Threat Score
# def ets_cal(y_obs,y_pred):
#     low_vis = 10 # 设定10km以下的观测能见度为低值
#     obs_low_vis = np.logical_and((y_obs <= low_vis),(y_pred > low_vis)) # 观测低能见度次数
#     pred_low_vis = np.logical_and((y_pred <= low_vis), (y_obs > low_vis)) # 模拟低能见度次数
#     hit_low_vis = np.logical_and((y_obs < low_vis), (y_pred < low_vis)) 
#     # 模拟到的低能见度次数
#     ETS_ref = (len(y_obs[hit_low_vis])+len(y_obs[obs_low_vis]))*(len(y_obs[hit_low_vis])+len(y_pred[pred_low_vis]))/(len(y_obs))
#     ETS = (len(y_obs[hit_low_vis])-ETS_ref)/(len(y_obs[hit_low_vis])+ len(y_obs[obs_low_vis])+len(y_pred[pred_low_vis])-ETS_ref)
#     return ETS

# Threat Score!!!
def ts_cal(y_obs,y_pred):
    low_vis = 1 # 设定10km以下的观测能见度为低值
    obs_low_vis = np.logical_and(y_obs <= low_vis,y_pred > low_vis) # 观测低能见度次数
    pred_low_vis = np.logical_and(y_pred <= low_vis, y_obs > low_vis) # 模拟低能见度次数
    hit_low_vis = np.logical_and((y_obs < low_vis), (y_pred < low_vis)) 
    # 模拟到的低能见度次数
    # ETS_ref = (len(y_obs[hit_low_vis])+len(y_obs[obs_low_vis]))*(len(y_obs[hit_low_vis])+len(y_pred[pred_low_vis]))/(len(y_obs))
    TS = (len(y_obs[hit_low_vis]))/(len(y_obs[hit_low_vis])+ len(y_obs[obs_low_vis])+len(y_pred[pred_low_vis]))
    return TS

###########################################     导入库     ###########################################
# import package
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import glob,os
import numpy as np
import pandas as pd
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import AnchoredText


###########################################    读取数据    ############################################
# read in dataset
# Mac
Data_Path = r'/Users/jiweiwen/Dropbox/Vis/Python_Work/Visibility' # 六个站点的观测数据，观测数据时间为世界时间

# Windows
# Data_Path = r'C:\Users\季伟文\Dropbox\Vis\Python_Work\Visibility' # 模式数据与观测数据

# 读取模式输出的六个站点位置的能见度结果，模式时间为北京时间，从2020.10.01 20:00，世界时间12:00开始
npy_name = "*_clean.npy"
npy_file = glob.glob(os.path.join(Data_Path,npy_name))
npy_file.sort()

# 读取模式数据格点对应的经纬度
Data = [[] for i in range(5)]
for i in range(5):
    # print(npy_file[i])
    Data[i] = np.load(npy_file[i],allow_pickle=True)
 
# 数据Data[i][j][k]
# i:0-4，依次为pm10，pm25，rh2，vis_obs,vis_rmaps
# j:0-5,六个站点
# k, 筛选后每个站点不同时刻的数据



############################################    回归拟合    ############################################

x_train_rmaps = [[] for i in range(6)]
x_test_rmaps  = [[] for i in range(6)]
y_train   = [[] for i in range(6)]
y_test    = [[] for i in range(6)]


y_predict = [[] for i in range(6)]

MSE_ini   = [[] for i in range(6)]
MSE       = [[] for i in range(6)]
# coef_lr   = [[] for i in range(6)]
# inter_lr  = [[] for i in range(6)]
ini_r2    = [[] for i in range(6)]
train_r2  = [[] for i in range(6)]
ini_test  = [[] for i in range(6)]
test_r2   = [[] for i in range(6)]
lr_r2     = [[] for i in range(6)]
ets_ini   = [[] for i in range(6)]
ets_train = [[] for i in range(6)]
ets_ini_test = [[] for i in range(6)]
ets_test  = [[] for i in range(6)]
ets_total = [[] for i in range(6)]
for ii in range(6):
    y = Data[3][ii][:]    
    x = [[] for i in range(4)]
    x_i = [0,1,2,4]
    for i in range(4):
        x[i] = Data[x_i[i]][ii][:]
    # lr = LinearRegression()
    # x = np.array(x)
    x = pd.DataFrame(x)
    para_names = ["PM10","PM2.5","RH2","VIS_Rmaps"]
    x.index = para_names
    x = x.T

    x_train1,x_test1,y_train[ii],y_test[ii] = train_test_split(x,y,train_size=0.8)#,shuffle=False)#random_state=23)
    scaler = StandardScaler()
    scaler.fit(x_test1)
    x_test = scaler.transform(x_test1)
    scaler.fit(x_train1)
    x_train = scaler.transform(x_train1)
    x_train = pd.DataFrame(x_train)
    x_test  = pd.DataFrame(x_test)
    x_train.columns = para_names
    x_test.columns  = para_names
    
#    x_train,x_test,y_train[ii],y_test[ii] = train_test_split(x,y,train_size=0.8,shuffle=False)
    x_train_rmaps[ii] = x_train1['VIS_Rmaps']
    x_test_rmaps[ii]  = x_test1['VIS_Rmaps']    
    params = {'n_estimators': 300, # 1000,初始化的弱学习器
              'max_depth': 10, # default=3，常用10-100之间。树最大深度，越深容易过度拟合
              'min_samples_split': 2, # default = 2，定义树中节点用来分裂的最少样本数
              'learning_rate': 0.01,
              'loss': 'ls'}
    
    lr = ensemble.GradientBoostingRegressor(**params)
    lr.fit(x_train, y_train[ii])
    
    y_predict[ii] = lr.predict(x_test)
    
#    coef_lr[ii]  = lr.coef_
#    inter_lr[ii] = lr.intercept_
    ini_r2[ii]   = r2_score(y,x['VIS_Rmaps'])
    train_r2[ii] = r2_score(y_train[ii],lr.predict(x_train))
    
    cor1 = np.corrcoef(y_test[ii],np.array(x_test['VIS_Rmaps']))
    ini_test[ii] = cor1[0][1]# r2_score(y_test[ii],np.array(x_test['VIS_Rmaps']))
    cor2 = np.corrcoef(y_test[ii],y_predict[ii])
    test_r2[ii]  = cor2[0][1] # r2_score(y_test[ii],y_predict[ii])
    
    # ini_test[ii] = r2_score(y_test[ii],np.array(x_test['VIS_Rmaps']))
    # test_r2[ii]  = r2_score(y_test[ii],y_predict[ii])
    lr_r2[ii]    = r2_score(y,lr.predict(x))
    MSE_ini[ii] = mean_squared_error(y_test[ii], np.array(x_test['VIS_Rmaps'])) 
    MSE[ii]     = mean_squared_error(y_test[ii], y_predict[ii])
    
    ets_ini[ii]   = ts_cal(y, x['VIS_Rmaps'])
    ets_train[ii] = ts_cal(y_train[ii], lr.predict(x_train)) 
    ets_ini_test[ii] = ts_cal(y_test[ii],np.array(x_test1['VIS_Rmaps']))
    ets_test[ii]  = ts_cal(y_test[ii], y_predict[ii])
    ets_total[ii] = ts_cal(y,lr.predict(x))
    del(x,y)
  
    
  
train_text = [[]for i in range(6)]
test_text  = [[]for i in range(6)]
for i in range(6):
    train_text[i].append(ini_test[i])
    train_text[i].append(ets_ini_test[i])
    test_text[i].append(test_r2[i])
    test_text[i].append(ets_test[i])
    
train_text = np.around(train_text,3)
test_text  = np.around(test_text,3)

#####################################     绘图       ############################################
# %matplotlib inline 
loc_name = ['zhuo zhou','gu an','nan yuan','lang fang','yong qing','da xing']

plt.figure(dpi=300,figsize=(6,16))

for j in range(6):
    ax = plt.subplot(6,2,1+j*2)
    
    ax.scatter(y_test[j], x_test_rmaps[j],s=2,marker='.')
    ax.set_xlim([0, 31])
    ax.set_ylim([0, 31])
    ax.set_ylabel("Vis_Rmaps (km)")
    at = AnchoredText(loc_name[j]+'\n'+'Corr, TS\n'+str(train_text[j]),
              prop=dict(size=6), frameon=True,
              loc='upper right',
              )    
    ax.add_artist(at)
    x_y_ticks = np.arange(0,32,5)
    if j==0:
        ax.set_title("Before GBR")
    else:
        ax.set_title("")
    if j==5:
        ax.set_xlabel("Vis_obs (km)")
        ax.set_xticks(np.arange(0,32,5))
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    
    ax=plt.subplot(6,2,2+j*2)
    ax.scatter(y_test[j], y_predict[j],s=2,marker='.')
    ax.set_xlim([0, 31])
    ax.set_ylim([0, 31])
    ax.tick_params(labelleft=False)
    at = AnchoredText(loc_name[j]+'\n'+'Corr, TS\n'+str(test_text[j]),
              prop=dict(size=6), frameon=True,
              loc='upper right',
              )
    ax.add_artist(at)
    x_y_ticks = np.arange(0,32,5)   
    if j==0:
        ax.set_title("After GBR")
    else:
        ax.set_title("")
    if j==5:
        ax.set_xlabel("Vis_obs (km)")
        ax.set_xticks(np.arange(0,32,5))
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
        
#%% GBR
# 画折线图

loc_name = ['zhuo zhou','gu an','nan yuan','lang fang','yong qing','da xing']

plt.figure(dpi=300,figsize=(6,16))

for j in range(6):
    ax = plt.subplot(6,2,1+j*2)
    
    ax.plot(y_test[j])
    ax.plot(np.array(x_test_rmaps[j]),'--',linewidth=1)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 31])
    ax.set_ylabel("Vis_Rmaps (km)")
    # at = AnchoredText(loc_name[j]+'\n'+'R2, ETS\n'+str(train_text[j]),
    #           prop=dict(size=6), frameon=True,
    #           loc='upper right',
    #           )    
    # ax.add_artist(at)
    # x_y_ticks = np.arange(0,32,5)
    if j==0:
        ax.set_title("Before GBR")
    else:
        ax.set_title("")
    if j==5:
        ax.set_xlabel("Vis_obs (km)")
        # ax.set_xticks(np.arange(0,32,5))
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    
    ax=plt.subplot(6,2,2+j*2)
    ax.plot(y_test[j])
    ax.plot(y_predict[j],'--',linewidth=1)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 31])
    ax.tick_params(labelleft=False)
    # at = AnchoredText(loc_name[j]+'\n'+'R2, ETS\n'+str(test_text[j]),
    #           prop=dict(size=6), frameon=True,
    #           loc='upper right',
    #           )
    # ax.add_artist(at)
    # x_y_ticks = np.arange(0,32,5)   
    if j==0:
        ax.set_title("After GBR")
    else:
        ax.set_title("")
    if j==5:
        ax.set_xlabel("Vis_obs (km)")
        # ax.set_xticks(np.arange(0,32,5))
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
        





# In[5]
# 观测数据共6个站点，20201001-20210331
# 54502 涿州   54512 固安
# 54514 南苑   54515 廊坊
# 54519 永清   54594 大兴


# 六个站点的观测数据，观测数据时间为世界时间

File_name = ["54502_202[0-1]*.csv","54512_202[0-1]*.csv","54514_202[0-1]*.csv",\
             "54515_202[0-1]*.csv","54519_202[0-1]*.csv","54594_202[0-1]*.csv"]

    
vis_loc = [[39.48 ,116.03], # 54502 涿洲   *******  Y=136, X=106  ******* Grid: (0-252,0-231)
           [39.42 ,116.28], # 54512 固安   *******  Y=133, X=113  ******* Grid: (0-252,0-231)
           [39.87 ,116.25], # 54514 丰台   *******  Y=150, X=112  ******* Grid: (0-252,0-231)
           [39.5  ,116.7 ], # 54515 廊坊   *******  Y=137, X=124  ******* Grid: (0-252,0-231)
           [39.3  ,116.48], # 54519 永清   *******  Y=129, X=118  ******* Grid: (0-252,0-231)
           [39.72 ,116.35]] # 54594 大兴   *******  Y=144, X=114  ******* Grid: (0-252,0-231)
# ！！！！！！！！！！！！！！ 廊坊 54515 2020年10月3日 22:00(6645-6646位置)缺失数据，采用前后时间均值填充 
# 最新的文件中已矫正数据 2021.08.17



   
