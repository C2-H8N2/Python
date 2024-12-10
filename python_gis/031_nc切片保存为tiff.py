#%%

import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt #将pyplot简写为plt 改库可用于绘制图表
import rasterio

#%%

#读取nc数据
nc_data = nc.Dataset("D:\data\dataset\climate data\cru_ts4.05.1901.2020.tmp.dat.nc")
#%%
print(nc_data.variables['tmp'])
# %%

#提取行数据
raw_temp_data = np.array(nc_data.variables['tmp'])
raw_temp_data.shape 

# %%

#提取缺失值
tmp_missing_value = nc_data.variables['tmp'].missing_value
# %%

#将raw_temp_data 中的缺失值替换为nan
raw_temp_data[raw_temp_data == tmp_missing_value] = np.nan
# %%

#提取第一层数据
small_data = raw_temp_data[0,:,:][::-1,:]#将图件上下颠倒(time, lat, lon)纬度颠倒,经度不变
small_data.shape
# %%

with rasterio.open(r"D:\data\dataset\\result\\test1901.tiff", 'w', driver='GTiff',
                       height=small_data.shape[0],
                       width=small_data.shape[1],
                       count=1,
                       dtype=str(small_data.dtype)) as f:
    f.write(small_data,1)
# %%

#将nc文件转换为tiff文件的函数
def array2tiff(matrix,filename):
     with rasterio.open(filename, 'w', driver='GTiff',#Geotiff格式文件
                       height=matrix.shape[0],#获得高,取决于矩阵的行数
                       width=matrix.shape[1],#获得长,取决于矩阵的列数
                       count=1,#写入的波段数
                       dtype=str(matrix.dtype)) as f:#指定数据类型为与矩阵相同类型
        f.write(matrix,1)#将矩阵传递给文件,并写入第一个波段中
array2tiff(raw_temp_data[2,:,:][::-1,:],r"D:\data\dataset\\result\\test1903.tiff")