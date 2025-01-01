#%%

import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
import datetime
from netCDF4 import num2date
from tqdm import tqdm
tqdm.pandas()
from shapely.geometry import Point
import matplotlib.pyplot as plt
import platform
from cartopy import crs as ccrs

if platform.system() == 'Darwin':
    plt.rcParams["font.family"] = 'Arial Unicode MS'
elif platform.system() == 'Windows':
    plt.rcParams["font.family"] = 'SimHei'
else:
    pass
# %%

#读取nc数据.将时间和温度数据转换为一维列表
# eg:raw_time_data[ 380.410. 439. ... 44118. 44149. 44179.]
nc_data = nc.Dataset(r"D:\data\dataset\climate data\cru_ts4.05.1901.2020.tmp.dat.nc")
raw_time_data=np.array(nc_data.variables['time'])
raw_tmp_data=np.array(nc_data.variables['tmp'])
#[datetime.datetime.strptime(num2date(times=i, units='days since 1900-1-1').strftime('%Y-%m-%d'),'%Y-%m-%d') for i in raw_time_data]
#使用了列表推导式，利用num2date将raw_time_data中的每个时间数据i转换为日期对象,再通过.strftime('%Y-%m-%d')方法将日期对象转为字符串.
# 最后通过.strptime(...,'%Y-%m-%d')方法将字符串解析为datetime对象,最后以列表形式返回.
# %%

#将上述方法一函数形式写出并使用,最后将上述方法得到的datetime对象转换为pandas中的Series对象,以便于后续处理
def cftime2datetime(cftime,units,format='%Y-%m-%d'):

    return datetime.datetime.strptime(num2date(times=cftime, units=units).strftime(format), 
                                      format)
clean_time_data=pd.Series([cftime2datetime(i,'days since 1900-1-1') for i in raw_time_data])
# %%
#对一定时间范围内数据进行切片
raw_tmp_data[(clean_time_data>='2000-08-01') & (clean_time_data<='2003-08-01'),:,:].shape

# %%
#对应年份切片
raw_tmp_data[clean_time_data.dt.year==2012,:,:].shape
#%%
#对对应月份切片
raw_tmp_data[clean_time_data.dt.month==1,:,:].shape
# %%
#对对应天切片
raw_tmp_data[clean_time_data.dt.day==16,:,:].shape