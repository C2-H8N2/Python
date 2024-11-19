#%%
#导入库
import numpy as np
import pandas as pd
import netCDF4 as nc
import geopandas as gpd
from netCDF4 import num2date
import datetime
from tqdm import tqdm
tqdm.pandas()
from shapely.geometry import Point
import matplotlib.pyplot as plt
import platform
from cartopy import crs as ccrs
plt.rcParams["font.family"] = 'SimHei'

# %%
#读取nc数据集
nc_data=nc.Dataset(r"D:\data\dataset\climate data\cru_ts4.05.1901.2020.tmp.dat.nc")
print(nc_data)
nc_data.variables['tmp']
# %%
#提取经纬度 时间和温度，并将数据转为一维列表
raw_lat_data=np.array(nc_data.variables['lat'])
raw_lon_data=np.array(nc_data.variables['lon'])
raw_time_data=np.array(nc_data.variables['time'])
raw_tmp_data=np.array(nc_data.variables['tmp'])

# %%
#提取温度变量的缺失值属性,并转为nan
tmp_missing_value=nc_data.variables['tmp'].missing_value

raw_tmp_data[raw_tmp_data==tmp_missing_value]=np.nan

# %%
#导入中国边界数据
china_boundary = gpd.read_file(r"D:\data\dataset\climate data\中国地图边界202111版.json")
#geopandas 库中的 read_file 函数从指定的路径读取地理数据文件（GeoJSON 文件）
china_boundary_valid=china_boundary.copy()
#避免对原数据进行修改，保留原始数据的完整性建立一个副本
china_boundary_valid['geometry']=china_boundary.buffer(0)
boundary=china_boundary_valid.total_bounds
min_x,min_y,max_x,max_y=boundary

# %%
#判断点是否在中国边界内
def pic(lon,lat):
#def pic(lon,lat) -> bool:另一种写法
    return china_boundary_valid.contains(Point(lon,lat))[0]

# %%
#复制原数据进行操作，避免更改原数据
sample_data = raw_tmp_data[0,:,:].copy()
sample_data

# %%
#建立numpy数组
mask_matrix = np.full(shape=(360,720),fill_value=False)

# %%
#tqdm()用在循环里可以展示进度条，指导代码在不在跑
for index_lat in tqdm(range(raw_lat_data.shape[0])):
    for index_lon in range(raw_lon_data.shape[0]):
        point = (raw_lon_data[index_lon],raw_lat_data[index_lat])
        value = pic(point[0],point[1])
        mask_matrix[index_lat,index_lon]=value

# %%
fig,ax=plt.subplots()
ax.imshow(sample_data[::-1,:])


# %%
#将原数据赋给china_data，并根据mask_matrix中值为False的位置在china——data中赋值为nan
china_data= sample_data.copy()
china_data[~mask_matrix] = np.nan
#~mask_matrix是对mask_matrix进行布尔取反操作
fig,ax=plt.subplots()
ax.imshow(china_data[::-1, :])
#在坐标轴 ax 上显示 china_data 数组的图像，imshow() 函数将数组作为图像显示
#[::-1, :] 的作用是将数组的行顺序反转，而列保持不变

# %%
import rasterio
from rasterio.transform import from_origin
def array2tiff(matrix,filename):
     transform = from_origin(west=min_x, north=max_y, xsize=0.5, ysize=0.5)#
     with rasterio.open(filename, 'w', driver='GTiff',#Geotiff格式文件
                       height=matrix.shape[0],#获得高,取决于矩阵的行数
                       width=matrix.shape[1],#获得长,取决于矩阵的列数
                       count=1,#写入的波段数
                       dtype=str(matrix.dtype)
                       crs='EPSG:4326',  # WGS-84坐标系
                       transform=transform) as f:#指定数据类型为与矩阵相同类型
        f.write(matrix,1)#将矩阵传递给文件,并写入第一个波段中
array2tiff(china_data[::-1, :],filename=r"D:\data\dataset\result\1901.tiff")
