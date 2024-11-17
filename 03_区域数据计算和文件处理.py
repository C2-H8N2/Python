#%%
#导入包
# 基础的数据处理工具
import numpy as np
import pandas as pd

#可视化
import matplotlib.pyplot as plt

#处理python时间函数
import datetime

#处理nc数据
import netCDF4 as nc
from netCDF4 import num2date

#处理网格数据eg：shp
import geopandas as gpd

#处理tiff 文件
import rasterio

#gis逻辑判断
from shapely.geometry import Point

#设置坐标投影
from cartopy import crs as ccrs

#打印进度条
from tqdm import tqdm 
tqdm.pandas()

#检测系统
import platform

# matplotlib 显示中文的问题
if platform.system() == 'Darwin':
    plt.rcParams["font.family"] = 'Arial Unicode MS'
elif platform.system() == 'Windows':
    plt.rcParams["font.family"] = 'SimHei'
else:
    pass
##加载数据并提取
# %%
#加载数据
nc_data = nc.Dataset(r"D:\data\dataset\climate data\cru_ts4.05.1901.2020.tmp.dat.nc")
nc_data
# %%
#提取变量
raw_lat_data = np.array(nc_data.variables['lat'])#纬度
raw_lon_data = np.array(nc_data.variables['lon'])#经度
raw_time_data = np.array(nc_data.variables['time'])#时间
raw_tmp_data = np.array(nc_data.variables['tmp'])#温度
raw_tmp_data
#%%
#提取温度缺失值，并换为nan
tmp_missing_value = nc_data.variables['tmp'].missing_value
raw_tmp_data[raw_tmp_data==tmp_missing_value] = np.nan 

##处理时间
# %%

def cftime2datetime(cftime,units,format='%Y-%m-%d %H:%M:%S'):
    #将nc文件时间格式从cftime转换为date
    return datetime.datetime.strptime(num2date(times=cftime, units=units).strftime(format), format)

clean_time_data = pd.Series([cftime2datetime(i, units='days since 1900-1-1') for i in raw_time_data])
clean_time_data[:4]

##处理边界数据并改变坐标系
# %%
#导入中国边界数据
china_boundary = gpd.read_file(filename="D:\data\dataset\中国地图边界202111版.json")
china_boundary_valid = china_boundary.copy()#建立副本防治改变元数据
china_boundary_valid['geometry'] = china_boundary.buffer(0)
china_boundary_valid
boundary=china_boundary_valid.total_bounds
min_x,min_y,max_x,max_y=boundary
#%%
#计算边界中心点
center_china_point = china_boundary_valid.centroid[0]
center_china_point.x,center_china_point.y
# %%
#提取原投影
raw_crs = china_boundary_valid.crs

# %%
#设置新投影
new_crs = ccrs.LambertConformal(central_longitude=center_china_point.x,
                                central_latitude=center_china_point.y
                                )

##裁剪
# %%
#计算中国地图掩膜
def pic(lon,lat)->bool:
    #监测点是否在中国边界线内
    return china_boundary_valid.contains(Point(lon,lat))[0]
    #eg:返回值为False
#raw_tmp_data.shape返回为（1440,360,720）
mask_for_china = np.full(shape=raw_tmp_data.shape[1:],fill_value=False)
#raw_lat_data.shape返回为(720,)
for index_lat in tqdm(range(raw_lat_data.shape[0])):
    for index_lon in range(raw_lon_data.shape[0]):
        point = (raw_lon_data[index_lon],raw_lat_data[index_lat])
        mask_for_china[index_lat,index_lon]=pic(point[0],point[1])
mask_for_china
# %%
#将要写入tiff文件的数据，不在中国范围内的转为nan

china_data= raw_tmp_data[0,:,:].copy()
china_data[~mask_for_china] = np.nan

# %%
#将矩阵保存在tiff文件中

def array2gtiff(array,filename):
    with rasterio.open(filename,'w',driver='GTiff',
                   height=array.shape[0],
                   width=array.shape[1],
                   count=1,
                   dtype=str(array.dtype)) as f:
        f.write(array,1)
array2gtiff(array=china_data.astype('float64')[::-1,],
            filename=r"D:\data\dataset\result\test.tiff")

# %% 
need1 = pd.DataFrame()
need1['date'] = clean_time_data
#计算世界和中国月均值
need1['world_value'] = [np.nanmean(raw_tmp_data[i,:,:]) for i in tqdm(range(raw_tmp_data.shape[0]))]
need1['china_value'] = [np.nanmean(raw_tmp_data[i,:,:][mask_for_china]) for i in tqdm(range(raw_tmp_data.shape[0]))]
need1.head()
# %%
#画图
fig,ax = plt.subplots(figsize=(10,4),dpi=150)
ax.plot(need1['date'],need1['world_value'],label ='世界平均温度' )
ax.plot(need1['date'],need1['china_value'],label ='中国平均温度' )
ax.legend()

##需求3、4
# %%
#设计求年均函数
def cal_need34(year,type):
    out = np.nanmean(raw_tmp_data[clean_time_data.dt.year==year,:,:],axis=0)

    if type == 'world':
        value = np.nanmean(out)
    elif type == 'china':
        value = np.nanmean(out[mask_for_china])
    else:
        value = None
    return value
#测试
print(cal_need34(1901,'china'))
print(cal_need34(1901,'world'))
# %%
#设置年np列表
year = np.arange(start=np.min(clean_time_data.dt.year),stop=np.max(clean_time_data.dt.year))
year
need34 = pd.DataFrame({'year':year})
need34['china_value'] = need34['year'].apply(lambda x:cal_need34(year = x,type='china'))
need34['world_value'] = need34['year'].apply(lambda x:cal_need34(year = x,type='world'))
need34
# %%
#%matplotlib

#%%
#趋势线计算函数
def smooth_data(y_value, deg=4):
    x_new = np.arange(y_value.shape[0])
    new_param = np.polyfit(x_new, y_value, deg=deg)
    value = np.zeros_like(x_new)
    for index, param in enumerate(new_param[::-1]):
        value = value + param * x_new ** index
    return value


need34['smooth_china_value'] = smooth_data(y_value=need34['china_value'])
need34['smooth_world_value'] = smooth_data(y_value=need34['world_value'])
# %%
fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
ax.plot(need34['year'], need34['world_value'], label='世界平均温度', linestyle='-', marker='o')
ax.plot(need34['year'], need34['china_value'], label='中国平均温度', linestyle='-', marker='o')

# 增加拟合曲线
ax.plot(need34['year'], need34['smooth_china_value'], linestyle='--', color='gray')
ax.plot(need34['year'], need34['smooth_world_value'], linestyle='--', color='gray')

ax.set_title("每年平均气温")
ax.legend()
ax.set_xlabel("年份")
ax.set_ylabel("温度平均数$ ^\circ C $")
plt.tight_layout()
fig.savefig(r"D:\data\dataset\result\结果.png")

# %%
##需求5 6
#5全世界每一年的平均气温
#6中国每年平均气温
def cal_need56(year,type):
    out=np.nanmean(raw_tmp_data[clean_time_data.dt.year==year,:,:],axis=0)
    #在第一个维度（时间维度）上计算平均值，即对同一纬度和经度位置上的不同时间点的数值求平均。
    if type=='world':
        value=out
    elif type=='china':
        out[~mask_for_china]=np.nan
        value=out
    else:
        value=np.nan
    return value
#测试
out=cal_need56(year=2000,type='china')
array2gtiff(array=out[::-1,:],filename=r'D:\data\dataset\result\2000ch.tiff')

out=cal_need56(year=2000,type='world')
array2gtiff(array=out[::-1,:],filename=r'D:\data\dataset\result\2000.tiff')
# %%
#需求5和6
for temp_year in tqdm(range(np.min(clean_time_data.dt.year),np.max(clean_time_data.dt.year)+1)):
    out=cal_need56(temp_year,type='world')
    array2gtiff(array=out[::-1,:],filename=f'D:\data\dataset\\result\{temp_year}.tiff')

    out=cal_need56(temp_year,type='china')
    array2gtiff(array=out[::-1,:],filename=f'D:\data\dataset\\result\{temp_year}ch.tiff')
# %%
#需求7 特定时间范围保存tiff
def cal_need7(start_year,end_year,type):
    out=np.nanmean(raw_tmp_data[(start_year<=clean_time_data) & (clean_time_data<=end_year),:,:],axis=0)
    #在第一个维度（时间维度）上计算平均值，即对同一纬度和经度位置上的不同时间点的数值求平均。
    if type=='world':
        value=out
    elif type=='china':
        out[~mask_for_china]=np.nan
        value=out
    else:
        value=np.nan
    return value
#测试
out=cal_need7(start_year='1902-01',end_year='2020-01',type='china')
array2gtiff(array=out[::-1,:],filename=r'D:\data\dataset\result\A190201-202001ch.tiff')

out=cal_need7(start_year='1902-01',end_year='2020-01',type='world')
array2gtiff(array=out[::-1,:],filename=r'D:\data\dataset\result\A190201-202001.tiff')

#需求8特定时间中国数据且用兰伯特投影
# %%
out8=cal_need7(start_year='1902-01',end_year='2020-01',type='china')
#%%
Lon_data, Lat_data = np.meshgrid(raw_lon_data, raw_lat_data)

need_8_df = pd.DataFrame({'value':out8.flatten(),
                          'mask_china':mask_for_china.flatten(),
                          'lon':Lon_data.flatten(),
                          'lat':Lat_data.flatten()})
need_8_df = need_8_df.loc[need_8_df['mask_china']]
#loc 是 Pandas 用于标签索引的方法，专门用于按行的标签或布尔索引进行选择。
# 使用布尔索引，只保留 mask_china 列为 True 的行。
need_8_df

# %%
need_8_df_gd = gpd.GeoDataFrame(
    need_8_df,
    geometry=gpd.points_from_xy(x=need_8_df['lon'], y=need_8_df['lat']),
    crs= raw_crs#china_boundary.crs
)
need_8_df_gd = need_8_df_gd.to_crs(new_crs.proj4_init)
need_8_df_gd
# %%
china_boundary_valid_new_crs = china_boundary_valid.to_crs(new_crs.proj4_init)
china_boundary_valid_new_crs
# %%
fig, ax = plt.subplots(figsize=(8, 7), dpi=150, subplot_kw={'projection': new_crs})
ax.gridlines(draw_labels=True,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
china_boundary_valid_new_crs.boundary.plot(ax=ax, color='black', marker='s')
need_8_df_gd.plot(ax=ax, column='value', cmap=plt.cm.get_cmap('RdYlBu'), legend=True)
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_title(f"时间范围为: 1902-01 ~ 2020-01")
plt.tight_layout()

fig.savefig(r"D:\data\dataset\result\中国可视化.png")
# %%
