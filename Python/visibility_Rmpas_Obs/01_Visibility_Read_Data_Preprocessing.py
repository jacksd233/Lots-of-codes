# -*- coding: utf-8 -*-
"""
**************************************************
Created on 2021, Aug 5, Ji Weiwen

Edited on 2021, Aug 12, Ji Weiwen

程序功能：
    读取 观测能见度（CSV） 与 模式模拟能见度（nc）
    绘制散点图对比模式准确度
    
**************************************************
Edited on 2021, Aug 16, Ji Weiwen
对数据进行清洗，去掉能见度极值30km的观测和模式数据
接下来在另外的文档中对模拟结果进行订正

**************************************************
verifed on 2022.1.19, Ji Weiwen
整理精简代码
In[0]:参考来源
In[0.5]:定义函数（查找最近站点经纬度）
In[1]:导入库、读取数据（模式+观测）、
In[2]:利用函数找出站点位置对应的模式格点
In[3]:预处理观测数据
In[3.5]:筛选掉30公里的能见度数据
In[4]:绘图
In[5]:保存数据到本地
In[999]:备用代码



"""

# In[0]

## Boosting Tree
# https://zhuanlan.zhihu.com/p/108622550


# In[0.5]
### 定义函数

# 1. 查找模式输出中最接近站点经纬度的位置格点
def locate_obs_site(lat,lon,obs_site,L=0.01):
    # lat,lon分别为维度和经度数组，obs_site为站点维度和经度坐标，L为threshold，默认为0.03
#    L = 0.03
    loc = []
    mask_latlon = (abs(lat-obs_site[0][0])+abs(lon-obs_site[0][1]) < L)
    loc_temp = np.where(mask_latlon == True)
    # temp=loc_temp
    for site in obs_site:
        while (len(loc_temp[0]) == 0):
            #del(loc_temp)
            mask_latlon = (abs(lat-site[0])+abs(lon-site[1]) < L)
            loc_temp = np.where(mask_latlon == True)
            #temp=loc_temp
            L=L+0.0002
        loc.append(loc_temp)
        loc_temp=([],[],[])
    # while not len(temp[0]):
    #      mask_latlon = (abs(lat-obs_site[0])+abs(lon-obs_site[1]) < L)
    #      loc = np.where(mask_latlon == True)
    #      temp=loc
    #      L=L+0.001
    # mask_latlon = (abs(lat-obs_site[0])+abs(lon-obs_site[1]) < L)
    # loc = np.where(mask_latlon == True)
    return(loc)

# In[1]
#**************************************************
### 导入库
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as GBR
import netCDF4 as nc
import glob,os
import numpy as np
import codecs
import matplotlib.pyplot as plt

#**************************************************
### 读取数据，windows/mac
# Mac
Rmaps_Path = r'/Users/jiweiwen/Documents/WORK_mnt/DATA/rmaps_CHEM'
Cord_Path = r'/Users/jiweiwen/Documents/WORK_mnt/DATA/rmaps_CHEM/cheminfo/v1.0/wrfout_d02_2021-05-01_12_00_00.grib2'
Obs_Path = r'/Users/jiweiwen/Dropbox/Vis/vis_hourly_new' # 六个站点的观测数据，观测数据时间为世界时间

# Windows
# Rmaps_Path = r'D:\DATA\rmaps_chem_v1.0'
# Cord_Path = r'D:\DATA\rmaps_chem_v1.0\cheminfo\v1.0\wrfout_d02_2021-05-01_12_00_00.grib2'
# Obs_Path = r'C:\Users\季伟文\Dropbox\Vis\vis_hourly_new' # 六个站点的观测数据，观测数据时间为世界时间

### 读取模式结果，模式数据来自于RMAPS_chem
# 读取模式输出的六个站点位置的能见度结果，模式时间为北京时间，从2020.10.01 20:00，世界时间12:00开始
nc_name = "*.nc"
nc_file = glob.glob(os.path.join(Rmaps_Path,nc_name))
nc_file.sort()

# 读取模式数据格点对应的经纬度
nc_cord = nc.Dataset(Cord_Path)
# X=231, Y =252
rmaps_lat = nc_cord.variables['XLAT'][:,:]
rmaps_lon = nc_cord.variables['XLONG'][:,:]


# vis_rmaps = nc_obj.variables['vis'][0:96:4,:,:]


# In[1.5]
### 读取观测数据

# 观测数据共6个站点，20201001-20210331
# 54502 涿州   54512 固安
# 54514 南苑   54515 廊坊
# 54519 永清   54594 大兴

# 六个站点的观测数据，观测数据时间为世界时间

File_name = ["54502_202[0-1]*.csv","54512_202[0-1]*.csv","54514_202[0-1]*.csv",\
             "54515_202[0-1]*.csv","54519_202[0-1]*.csv","54594_202[0-1]*.csv"]

file_obs = [] 
for files in File_name:
    temp = glob.glob(os.path.join(Obs_Path, files))
    file_obs.append(temp)
    
vis_loc = [[39.48 ,116.03], # 54502 涿洲   *******  Y=136, X=106  ******* Grid: (0-252,0-231)
           [39.42 ,116.28], # 54512 固安   *******  Y=133, X=113  ******* Grid: (0-252,0-231)
           [39.87 ,116.25], # 54514 丰台   *******  Y=150, X=112  ******* Grid: (0-252,0-231)
           [39.5  ,116.7 ], # 54515 廊坊   *******  Y=137, X=124  ******* Grid: (0-252,0-231)
           [39.3  ,116.48], # 54519 永清   *******  Y=129, X=118  ******* Grid: (0-252,0-231)
           [39.72 ,116.35]] # 54594 大兴   *******  Y=144, X=114  ******* Grid: (0-252,0-231)
# ！！！！！！！！！！！！！！ 廊坊 54515 2020年10月3日 22:00(6645-6646位置)缺失数据，采用前后时间均值填充


# In[2]
### 筛选出模式中站点位置处数据以对比

# 利用In[0.5]中函数 locate_obs_site 和站点经纬度（vis_loc）找出站点在模式中的位置格点
site_tuple = locate_obs_site(rmaps_lat, rmaps_lon, vis_loc) # 此时为元组数据
site_loc = vis_loc
for i in range(6):
    site_loc[i][0] = int(site_tuple[i][1])
    site_loc[i][1] = int(site_tuple[i][2])

# 从每天输出的nc文件（共182，起始时间从2020.10.1 20:00到2021.3.31 20:00）中读取能见度参数（VIS）
# 每天的nc文件中输出间隔为1小时，共97个数据（预测后四天），包含起始的20:00到4天后的20:00
# 间隔4位取值将结果转化为按小时输出的数据后，每天24个，从当日20:00到下一天的19：00

# 读取模式输出参量用来订正，包括相对湿度（rh2），pm2.5和pm10
vis_rmaps  = [[] for i in range(6)]
rh2_rmaps  = [[] for i in range(6)]
pm25_rmaps = [[] for i in range(6)] 
pm10_rmaps = [[] for i in range(6)]
# vis_rmaps = []

for j in range(6):
    for i in range(182):
        nc_obj = nc.Dataset(nc_file[i])
        lat_s = site_loc[j][0]
        lon_s = site_loc[j][1]
        temp_v    = nc_obj.variables['vis'][0:24,lat_s,lon_s] # 只取模式预测数据的前24小时
        temp_rh   = nc_obj.variables['rh2'][0:24,lat_s,lon_s]
        temp_pm25 = nc_obj.variables['pm25'][0:24,lat_s,lon_s]
        temp_pm10 = nc_obj.variables['pm10'][0:24,lat_s,lon_s]
        vis_rmaps[j]  = np.append(vis_rmaps[j], temp_v)# = np.append(vis_rmaps , temp_v)
        rh2_rmaps[j]  = np.append(rh2_rmaps[j], temp_rh)
        pm25_rmaps[j] = np.append(pm25_rmaps[j], temp_pm25)
        pm10_rmaps[j] = np.append(pm10_rmaps[j], temp_pm10)
        del(nc_obj,temp_v,temp_pm25,temp_pm10)
    # vis_rmaps.append(temp_r)
   
    
# vis_rmaps为6个站点的模式预测结果


# vis_rmaps_54502_zz = vis_rmaps# [:,136,106]

# In[3]   
### 处理观测数据

# 将观测数据按名称排序
for i in range(6):
    file_obs[i-1].sort()
file_obs.sort(reverse=False)
# print(file)

df = []
### 读取观测CSV文件
# 首先尝试一个站点54502
for i in range(6):
    temp1 = pd.read_csv(file_obs[i][0])
    df.append(temp1)
    temp2 = pd.read_csv(file_obs[i][1])
    df.append(temp2)
    del(temp1,temp2)
# df = pd.read_csv(file[0][1])
# df = pd.read_csv('/Users/jiweiwen/Dropbox/Vis/vis_hourly/54502_20200101_20201231_obs.csv')
# print(df[0].VIS)

# 读取6个站点的观测数据
vis_obs_site = [[] for i in range(6)]
for i in range(6):
    temp_obs = df[2*i].VIS
    temp_obs_2021= df[2*i+1].VIS
    
    ### 处理数据，异常值，缺失值
    temp_obs[temp_obs == 999999] = np.median(temp_obs)
    temp_obs_2021[temp_obs_2021 == 999999] = np.median(temp_obs_2021)
    # if len(df[2*i])<8784: # 添加廊坊的缺失值
    #     miss_value = np.mean([temp_obs[6645],temp_obs[6646]])
    #     insert_pos = temp_obs[6645:6646]
    #     insert_pos[6645] = miss_value
    #     temp_obs = pd.concat([temp_obs[0:6646],insert_pos,temp_obs[6646:]])
    #     temp_obs.index = range(8784)
    vis_obs_site[i] = pd.concat([temp_obs[6588:], temp_obs_2021])
    vis_obs_site[i] = vis_obs_site[i]/1000
    vis_obs_site[i].index = np.arange(4356)
    

# In[3.5]
### 筛掉观测和模式中 30 km 的能见度结果，以提高拟合的准确性
# 得到能见度，相对湿度，pm10，pm2.5的

vis_rmaps_clean  = [[] for i in range(6)]
rh2_rmaps_clean  = [[] for i in range(6)]
pm25_rmaps_clean = [[] for i in range(6)] 
pm10_rmaps_clean = [[] for i in range(6)]
vis_obs_clean    = [[] for i in range(6)]

for i in range(6):
    rmaps_temp = vis_rmaps[i][0:4356]
    obs_temp   = np.array(vis_obs_site[i])
    rh2_temp   = rh2_rmaps[i][0:4356]
    pm25_temp  = pm25_rmaps[i][0:4356]
    pm10_temp  = pm10_rmaps[i][0:4356]
    vis_filter = np.logical_and(rmaps_temp<40,obs_temp<40)
    vis_rmaps_clean[i]  = rmaps_temp[vis_filter]
    rh2_rmaps_clean[i]  = rh2_temp[vis_filter]
    pm25_rmaps_clean[i] = pm25_temp[vis_filter]
    pm10_rmaps_clean[i] = pm10_temp[vis_filter]
    vis_obs_clean[i]    = obs_temp[vis_filter]

# plt.figure()
# plt.plot(vis_rmaps[0][0:100])
# print(vis_obs_site[1].describe())
# a = vis_obs_site[0]

# print(a[a['VIS']>29.999])

# In[4]
### 绘图

site_name = ['zhuo zhou','gu an','nan yuan','lang fang','yong qing','da xing']
# site_name = ['涿州','固安','南苑','廊坊','永清','大兴']
plt.figure(dpi=600,figsize=(6,10))
for i in range(6):
    plt.subplot(321+i)
    # plt.scatter(vis_obs_site[i],vis_rmaps[i][0:4356],s=1,marker='.')
    plt.scatter(vis_obs_clean[i],vis_rmaps_clean[i],s=1,marker='.')
    plt.xlim([0, 31])
    plt.ylim([0, 31])
    x_y_ticks = np.arange(0,32,5)
    plt.xticks(x_y_ticks)
    plt.yticks(x_y_ticks)
    plt.title(site_name[i])
    if (i == 0) or (i == 2) or (i ==4):
        plt.ylabel('Vis Rmaps(km)')
    else:
        plt.ylabel('')
    if (i == 4) or (i == 5):
        plt.xlabel('Vis Obs(km)')
    else:
        plt.xlabel('')


plt.show()




# In[5]
### 保存数据到本地

for i in range(6):
    print(np.corrcoef(vis_obs_clean[i],vis_rmaps_clean[i]))

# obs_data = pd.concat(dat for dat in vis_obs_site)


# obs_data.to_csv("./test.csv",index = False)

np.save("./vis_data_with_30km/vis_obs_clean.npy",
            vis_obs_clean)


np.save("./vis_data_with_30km/vis_rmaps_clean.npy",
            vis_rmaps_clean)

np.save("./vis_data_with_30km/rh2_rmaps_clean.npy",
            rh2_rmaps_clean)


np.save("./vis_data_with_30km/pm25_rmaps_clean.npy",
            pm25_rmaps_clean)

np.save("./vis_data_with_30km/pm10_rmaps_clean.npy",
            pm10_rmaps_clean)

# In[999]
### 备份用代码

# 将csv文件转化为utf-8格式保存
# =============================================================================
# ## convert csv to csv utf-8
# src = "/Users/jiweiwen/Dropbox/Vis/vis_hourly/*_202[0-1]*.csv"
# def ReadFile(filePath):  
#     with codecs.open(filePath,"r") as f:
#         return f.read()
# 
# def WriteFile(filePath,u,encoding="utf-8"):
#     with codecs.open(filePath,"wb") as f:
#         f.write(u.encode(encoding,errors="ignore"))
#         
# def CSV_2_UTF8(src,dst):
#     content = ReadFile(src)
#     WriteFile(src,content, encoding="utf-8")
# CSV_2_UTF8(src, src)
# =============================================================================
    

# =============================================================================
# file_1 = glob.glob(os.path.join(Obs_Path,"54502_202[0-1]*.csv"))
# file_2 = glob.glob(os.path.join(Obs_Path,"54512_202[0-1]*.csv"))
# file_3 = glob.glob(os.path.join(Obs_Path,"54514_202[0-1]*.csv"))
# file_4 = glob.glob(os.path.join(Obs_Path,"54515_202[0-1]*.csv"))
# file_5 = glob.glob(os.path.join(Obs_Path,"54519_202[0-1]*.csv"))
# file_6 = glob.glob(os.path.join(Obs_Path,"54594_202[0-1]*.csv"))
# =============================================================================

# =============================================================================
# vis_obs = df[0].VIS # 从6588时刻开始2020.10.01 12:00
# vis_obs_2021 = df[1].VIS
# 
# 
# ### 处理数据，异常值，缺失值
# # print(vis_obs.describe())
# vis_obs[vis_obs == 99999] = np.median(vis_obs)
# vis_obs_2021[vis_obs_2021 == 99999] = np.median(vis_obs_2021)
# 
# # print(vis_obs.describe())
# 
# ### 读取Rmaps模式输出结果
# 
# # vis_obs_temp = np.array(vis_obs[6588:-1])
# # vis_obs_temp1 = np.array(vis_obs_2021)
# # vis_obs_54502 = np.append(vis_obs_temp, vis_obs_temp1)
# 
# vis_obs_temp = vis_obs[6588:-1] # 2020年10月1日 北京时间20:00
# vis_obs_temp1 = vis_obs_2021
# vis_obs_54502 = pd.concat([vis_obs_temp, vis_obs_temp1])
# vis_obs_54502.index = np.arange(4355)
# vis_obs_54502 = vis_obs_54502/1000
# =============================================================================

# =============================================================================
# plt.subplot(322)
# plt.scatter(vis_obs_54502,vis_rmaps_54502_zz[0:4355])
# plt.xlim([0, 30])
# plt.ylim([0, 30])
# 
# plt.subplot(323)
# plt.scatter(vis_obs_54502,vis_rmaps_54502_zz[0:4355])
# plt.xlim([0, 30])
# plt.ylim([0, 30])
# 
# plt.subplot(324)
# plt.scatter(vis_obs_54502,vis_rmaps_54502_zz[0:4355])
# plt.xlim([0, 30])
# plt.ylim([0, 30])
# 
# plt.subplot(325)
# plt.scatter(vis_obs_54502,vis_rmaps_54502_zz[0:4355])
# plt.xlim([0, 30])
# plt.ylim([0, 30])
# 
# plt.subplot(326)
# plt.scatter(vis_obs_54502,vis_rmaps_54502_zz[0:4355])
# 
# plt.subplot(224)
# plt.scatter(vis_obs_54502,vis_rmaps_54502_zz[0:4355])
# plt.xlim([0, 30])
# plt.ylim([0, 30])
# =============================================================================
