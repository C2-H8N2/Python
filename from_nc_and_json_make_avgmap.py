#%%
#导入包
#本代码为从nc文件中提取某一天数据，再利用json的边界数据，分别统计nc文件中各省的均值，将每个省的均值作为其值出一副图
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc

from tqdm import tqdm
from netCDF4 import num2date

import matplotlib.pyplot as plt

from shapely.geometry import Point
# from shapely.validation import make_valid

import platform
# %%
#加载nc文件
nc_data=nc.Dataset(r"D:\data\lftx.sfc.2021.nc")

# %%
#处理nc数据
time_units = [temp_variable.units for temp_variable in nc_data.variables.values() if temp_variable.name == 'time'][0]
# [pd.to_datetime(str(num2date(temp_time,units=time_units))) for temp_time in np.array(nc_data.variables['time'])]
time_list = [num2date(temp_time, units=time_units).strftime('%Y-%m-%d') for temp_time in
             np.array(nc_data.variables['time'])]
# %%
#
lftx_data = np.array(nc_data.variables['lftx'])
lftx_data = lftx_data[np.where([i == '2021-02-13' for i in time_list])[0][0], :, :]

lon_data = np.array(nc_data.variables['lon'])
lat_data = np.array(nc_data.variables['lat'])
# %%
#读取json数据
chinamap_data = gpd.read_file(filename=r"D:\BaiduNetdiskDownload\climate data\100000_中华人民共和国_full.json")
chinamap_data.head()
# %%

#%matplotlib inline
fig, ax = plt.subplots(figsize=(6, 6))
chinamap_data.iloc[:35, :].plot(ax=ax, color='black')
#plt.show()
# %%
#处理地图数据，将行列文件展开为一行
Lon_data, Lat_data = np.meshgrid(lon_data, lat_data)
lftx_data_with_location = pd.DataFrame({'lon': Lon_data.flatten(),
                                        'lat': Lat_data.flatten(),
                                        'lftx': lftx_data.flatten()})
lftx_data_with_location.head()
# %%
map_value = chinamap_data[['adcode', 'name', 'geometry']]
map_value['lftx_value'] = 0
map_value['num'] = 0


# map_value.head()

def trans(lon):
    """

    :param lon:
    :return:
    """
    if lon <= 180:
        return lon
    else:
        return lon - 360


# 这个代码非常重要
# https://stackoverflow.com/questions/63955752/topologicalerror-the-operation-geosintersection-r-could-not-be-performed

# map_value['geometry'] = map_value.buffer(0)



map_value['geometry'] = map_value.buffer(0) #['geometry'].apply(lambda x: make_valid(x))
map_value.loc[~map_value.is_valid]

# %%
for index in tqdm(range(lftx_data_with_location.shape[0])):
    temp_df = lftx_data_with_location.iloc[index, :]

    temp_mask = map_value['geometry'].contains(Point(trans(temp_df.lon),
                                                     temp_df.lat))
    map_value['lftx_value'] = map_value['lftx_value'] + temp_mask * temp_df.lftx

    map_value['num'] = map_value['num'] + temp_mask * 1
# %%
map_value['mean_lftx'] = map_value['lftx_value'] / map_value['num']  #(map_value['num'] + 0.00001)
#map_value.loc[pd.isna(map_value['mean_lftx']) , :]['mean_lftx'] = 0这种方法没有修改原始值
map_value.loc[pd.isna(map_value['mean_lftx']), 'mean_lftx'] = 0
map_value.head()
# %%
#画图
# 检测系统
import platform

if platform.system() == 'Darwin':
    plt.rcParams["font.family"] = 'Arial Unicode MS'
elif platform.system() == 'Windows':
    plt.rcParams["font.family"] = 'SimHei'
else:
    pass
# Linux
# Windows


fig, ax = plt.subplots()

map_value.plot(ax=ax, column='mean_lftx', legend=True)

map_value['center_lon'] = map_value['geometry'].centroid.x
map_value['center_lat'] = map_value['geometry'].centroid.y

for index in range(map_value.shape[0]):
    # ax.scatter(map_value['center_lon'], map_value['center_lat'])
    temp_df = map_value.iloc[index, :]
    ax.text(x=temp_df.center_lon, y=temp_df.center_lat, s=str(temp_df['name']))

ax.set_title("demo 中国地图, 公众号：pypi", fontdict={"size": 20})
ax.autoscale()
plt.tight_layout()
plt.show()
# %%
