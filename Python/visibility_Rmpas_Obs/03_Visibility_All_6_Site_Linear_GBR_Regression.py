# -*- coding: utf-8 -*-
"""
#***********************************************
Created on 2021, Aug 25, Ji Weiwen

尝试机器学习对六个站点模式能见度订正

#***********************************************
Verified on 2022.1.19, Ji Weiwen
综合六站点所有数据进行拟合，
绘制图像为1X3，原始数据+线性回归+GBR
In[0]: 参考来源
In[0.5]: 定义函数
In[1]: 线性回归拟合+绘图
In[2]: GBR，首先运行得到数据

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
    TS = (len(y_obs[hit_low_vis]))/(len(y_obs[hit_low_vis])+ len(y_obs[obs_low_vis])+len(y_pred[pred_low_vis]))
    return TS
      



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
    TS = (len(y_obs[hit_low_vis]))/(len(y_obs[hit_low_vis])+ len(y_obs[obs_low_vis])+len(y_pred[pred_low_vis]))
    return TS

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

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
    if i == 3:
        tmp = np.load(npy_file[i],allow_pickle=True)
        tmp2 = tmp[0]
        for j in range(len(tmp)-1):
            tmp2 = np.append(tmp2,tmp[j+1])
        Data[i] = tmp2
    else:
        tmp = np.load(npy_file[i],allow_pickle=True)
        tmp2 = tmp[0].compressed()
        for j in range(len(tmp)-1):
            tmp2 = np.append(tmp2,tmp[j+1].compressed())
        Data[i] = tmp2





# 数据Data[i][j][k]
# i:0-4，依次为pm10(ug/m3)，pm2.5(ug/m3)，rh2，vis_obs,vis_rmaps
# j:0-5,六个站点
# k, 筛选后每个站点不同时刻的数据


####################################     回归拟合      #####################################


y = Data[3][:]
x = [[] for i in range(4)]
x_i = [0,1,2,4]
for i in range(4):
    x[i] = Data[x_i[i]][:]

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



x_train1,x_test1,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=42)
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
x_train_rmaps = x_train1['VIS_Rmaps']
x_test_rmaps  = x_test1['VIS_Rmaps']
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test)

coef_lr  = lr.coef_
inter_lr = lr.intercept_
ini_r2   = r2_score(y,x['VIS_Rmaps'])
train_r2 = r2_score(y_train,lr.predict(x_train))
cor1 = np.corrcoef(y_test,np.array(x_test['VIS_Rmaps']))
ini_test = cor1[0][1]# r2_score(y_test[ii],np.array(x_test['VIS_Rmaps']))
cor2 = np.corrcoef(y_test,y_predict)
test_r2  = cor2[0][1] # r2_score(y_test[ii],y_predict[ii])
# ini_test[ii] = r2_score(y_test[ii],np.array(x_test1['VIS_Rmaps']))
# test_r2[ii]  = r2_score(y_test[ii],y_predict[ii])
lr_r2    = r2_score(y,lr.predict(x))
RMSE_ini = rmse(y_test, np.array(x_test['VIS_Rmaps'])) 
RMSE     = rmse(y_test, y_predict)

ets_ini   = ts_cal(y, x['VIS_Rmaps'])
ets_train = ts_cal(y_train, lr.predict(x_train)) 
ets_ini_test = ts_cal(y_test,np.array(x_test1['VIS_Rmaps']))
ets_test  = ts_cal(y_test, y_predict)
ets_total = ts_cal(y,lr.predict(x))
del(x,y)
 


#####################################     输出参数      #######################################

train_text=[]
test_train=[]
train_text=ini_test
# train_text.append(ets_ini_test)
train_text=np.append(train_text,ets_ini_test)
test_text=test_r2
# test_text.append(ets_test)
test_text=np.append(test_text,ets_test)
    
train_text = np.around(train_text,3)
test_text  = np.around(test_text,3)

#####################################     绘图       ############################################

# ===============  运行In[2]得到GBR拟合结果  =================
GBR_Path = r'/Users/jiweiwen/Dropbox/vis/Python_Work/Visibility/GBR_results_data_six_sites' # 六个站点的观测数据，观测数据时间为世界时间

# 读取模式输出的六个站点位置的能见度结果，模式时间为北京时间，从2020.10.01 20:00，世界时间12:00开始
GBR_name = "*.npy"
GBR_file = glob.glob(os.path.join(GBR_Path,GBR_name))
GBR_file.sort()
x_test_rmaps_GBR = np.load(GBR_file[0],allow_pickle=True)
y_predict_GBR = np.load(GBR_file[1],allow_pickle=True)
y_test_GBR = np.load(GBR_file[2],allow_pickle=True)
# ===============  从In[2]获得  =================

loc_name = 'six sites'#['zhuo zhou','gu an','nan yuan','lang fang','yong qing','da xing']

plt.figure(dpi=900,figsize=(12,4))
# plt.rcParams['font.family'] = ['Heiti']
# for j in range(6):
ax = plt.subplot(1,3,1)

ax.scatter(y_test, x_test_rmaps,s=2,marker='.')
ax.set_xlim([0, 31])
ax.set_ylim([0, 31])
ax.set_ylabel("Vis_Rmaps (km)")
# at = AnchoredText(loc_name+'\n'+'Corr, ETS(1km)\n'+str(train_text[0])+str(train_text[1]),
#           prop=dict(size=6), frameon=True,
#           loc='upper right',
#           )    
at = AnchoredText("(a)",
          prop=dict(size=8), frameon=True,
          loc='upper right',
          )   
ax.add_artist(at)
x_y_ticks = np.arange(0,32,5)
ax.set_title("Initial Condition")

ax.set_xlabel("Vis_obs (km)")
ax.set_xticks(np.arange(0,32,5))


ax=plt.subplot(1,3,2)
ax.scatter(y_test, y_predict,s=2,marker='.')
ax.set_xlim([0, 31])
ax.set_ylim([0, 31])
ax.tick_params(labelleft=False)
# at = AnchoredText(loc_name+'\n'+'Corr, ETS(1km)\n'+str(list(test_text)),
#           prop=dict(size=6), frameon=True,
#           loc='upper right',
#           )
at = AnchoredText("(b)",
          prop=dict(size=8), frameon=True,
          loc='upper right',
          )   
ax.add_artist(at)
x_y_ticks = np.arange(0,32,5)   

ax.set_title("After Linear Regression")

ax.set_xlabel("Vis_obs (km)")
ax.set_xticks(np.arange(0,32,5))

ax=plt.subplot(1,3,3)
ax.scatter(y_test_GBR, y_predict_GBR,s=2,marker='.')
ax.set_xlim([0, 31])
ax.set_ylim([0, 31])
ax.tick_params(labelleft=False)
# at = AnchoredText(loc_name+'\n'+'Corr, ETS(1km)\n'+str(list(test_text)),
#           prop=dict(size=6), frameon=True,
#           loc='upper right',
#           )
at = AnchoredText("(c)",
          prop=dict(size=8), frameon=True,
          loc='upper right',
          )   
ax.add_artist(at)
x_y_ticks = np.arange(0,32,5)   

ax.set_title("After GBR")

ax.set_xlabel("Vis_obs (km)")
ax.set_xticks(np.arange(0,32,5))

print("RMSE_ini_LR:"+" "+str(RMSE_ini))

print("RMSE_LR:"+" "+ str(RMSE))







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

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

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
    if i == 3:
        tmp = np.load(npy_file[i],allow_pickle=True)
        tmp2 = tmp[0]
        for j in range(len(tmp)-1):
            tmp2 = np.append(tmp2,tmp[j+1])
        Data[i] = tmp2
    else:
        tmp = np.load(npy_file[i],allow_pickle=True)
        tmp2 = tmp[0].compressed()
        for j in range(len(tmp)-1):
            tmp2 = np.append(tmp2,tmp[j+1].compressed())
        Data[i] = tmp2
 
# 数据Data[i][j][k]
# i:0-4，依次为pm10，pm25，rh2，vis_obs,vis_rmaps
# j:0-5,六个站点
# k, 筛选后每个站点不同时刻的数据



############################################    回归拟合    ############################################


y = Data[3][:]    
x = [[] for i in range(4)]
x_i = [0,1,2,4]
for i in range(4):
    x[i] = Data[x_i[i]][:]
# lr = LinearRegression()
# x = np.array(x)
x = pd.DataFrame(x)
para_names = ["PM10","PM2.5","RH2","VIS_Rmaps"]
x.index = para_names
x = x.T

x_train1,x_test1,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=42)
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
x_train_rmaps = x_train1['VIS_Rmaps']
x_test_rmaps  = x_test1['VIS_Rmaps']    
params = {'n_estimators': 300, # 1000,初始化的弱学习器
          'max_depth': 10, # default=3，常用10-100之间。树最大深度，越深容易过度拟合
          'min_samples_split': 2, # default = 2，定义树中节点用来分裂的最少样本数
          'learning_rate': 0.01,
          'loss': 'ls'}

lr = ensemble.GradientBoostingRegressor(**params)
lr.fit(x_train, y_train)

y_predict = lr.predict(x_test)

#    coef_lr[ii]  = lr.coef_
#    inter_lr[ii] = lr.intercept_
ini_r2   = r2_score(y,x['VIS_Rmaps'])
train_r2 = r2_score(y_train,lr.predict(x_train))

cor1 = np.corrcoef(y_test,np.array(x_test['VIS_Rmaps']))
ini_test = cor1[0][1]# r2_score(y_test[ii],np.array(x_test['VIS_Rmaps']))
cor2 = np.corrcoef(y_test,y_predict)
test_r2  = cor2[0][1] # r2_score(y_test[ii],y_predict[ii])

# ini_test[ii] = r2_score(y_test[ii],np.array(x_test['VIS_Rmaps']))
# test_r2[ii]  = r2_score(y_test[ii],y_predict[ii])
lr_r2    = r2_score(y,lr.predict(x))
# MSE_ini = mean_squared_error(y_test, np.array(x_test['VIS_Rmaps'])) 
# MSE     = mean_squared_error(y_test, y_predict)
RMSE_ini = rmse(y_test, np.array(x_test['VIS_Rmaps'])) 
RMSE     = rmse(y_test, y_predict)


ets_ini   = ts_cal(y, x['VIS_Rmaps'])
ets_train = ts_cal(y_train, lr.predict(x_train)) 
ets_ini_test = ts_cal(y_test,np.array(x_test1['VIS_Rmaps']))
ets_test  = ts_cal(y_test, y_predict)
ets_total = ts_cal(y,lr.predict(x))
del(x,y)
  
    
train_text=[]
test_train=[]
train_text=ini_test
train_text=np.append(train_text,ets_ini_test)
test_text=test_r2
test_text=np.append(test_text,ets_test)
    
train_text = np.around(train_text,3)
test_text  = np.around(test_text,3)  


#####################################     绘图       ############################################
# %matplotlib inline 
loc_name = 'six sites'#['zhuo zhou','gu an','nan yuan','lang fang','yong qing','da xing']

plt.figure(dpi=300,figsize=(8,4))

# for j in range(6):
ax = plt.subplot(1,2,1)

ax.scatter(y_test, x_test_rmaps,s=2,marker='.')
ax.set_xlim([0, 31])
ax.set_ylim([0, 31])
ax.set_ylabel("Vis_Rmaps (km)")
at = AnchoredText(loc_name+'\n'+'Corr, ETS(1km)\n'+str(train_text),
          prop=dict(size=6), frameon=True,
          loc='upper right',
          )    
ax.add_artist(at)
x_y_ticks = np.arange(0,32,5)

ax.set_title("Before GBR")

ax.set_xlabel("Vis_obs (km)")
ax.set_xticks(np.arange(0,32,5))

ax=plt.subplot(1,2,2)
ax.scatter(y_test, y_predict,s=2,marker='.')
ax.set_xlim([0, 31])
ax.set_ylim([0, 31])
ax.tick_params(labelleft=False)
at = AnchoredText(loc_name+'\n'+'Corr, ETS(1km)\n'+str(test_text),
          prop=dict(size=6), frameon=True,
          loc='upper right',
          )
ax.add_artist(at)
x_y_ticks = np.arange(0,32,5)   

ax.set_title("After GBR")



ax.set_xlabel("Vis_obs (km)")
ax.set_xticks(np.arange(0,32,5))

print("RMSE_ini_GBR:"+" "+str(RMSE_ini))

print("RMSE_GBR:"+" "+ str(RMSE))


#保存GBR的拟合结果
np.save("/Users/jiweiwen/Dropbox/vis/Python_Work/Visibility/GBR_results_data_six_sites/y_test.npy",
            y_test)
np.save("/Users/jiweiwen/Dropbox/vis/Python_Work/Visibility/GBR_results_data_six_sites/x_test_rmaps.npy",
            x_test_rmaps)
np.save("/Users/jiweiwen/Dropbox/vis/Python_Work/Visibility/GBR_results_data_six_sites/y_predict.npy",
            y_predict)

        



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



   
