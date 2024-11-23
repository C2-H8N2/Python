#1 nc文件保存
#2 插值nc文件

#%% 导入包
#基础数据处理根据
import numpy as np
import pandas as pd

#可视化工具
import matplotlib.pyplot as plt

#处理nc数据
import netCDF4 as nc
from netCDF4 import num2date

#处理网格、shp文件
import geopandas as gpd

#处理tiff文件
import rasterio

#gis逻辑判断
from shapely.geometry import Point

#打印进度条
from tqdm import tqdm
tqdm.pandas()

#并行
from joblib import Parallel,delayed

#检测系统
import platform

#处理时间
import datetime

# matplotlib 显示中文的问题
if platform.system() == 'Darwin':
    plt.rcParams["font.family"] = 'Arial Unicode MS'
elif platform.system() == 'Windows':
    plt.rcParams["font.family"] = 'SimHei'
else:
    pass

#%% 设置类
class GetMask(object):
    def __init__(self,
                 geopandas: gpd.GeoDataFrame,
                 nc_data: nc.Dataset,
                 nc_variable: str,
                 lat_variable: str,
                 lon_variable: str,
                 time_variable: str):
        self.geopandas = geopandas
        self.nc_data = nc_data
        self.nc_variable = nc_variable
        self.lat_variable = lat_variable
        self.lon_variable = lon_variable
        self.time_variable = time_variable
        self.nc_target_data = None
        self.target_data_missing_value = None
        self.time_dim = None
        self.lat_dim = None
        self.lon_dim = None
        self.mask_matrix = None
        self.longitude_data = None
        self.latitude_data = None
        self.time_data = None
        self.time_units = None
        self.clean_time_data = None

    def num2datetime(self, cftime, units, format='%Y-%m-%d %H:%M:%S'):
        """
        将nc文件里面的时间格式 从cftime 转换到 datetime格式
        :param cftime:
        :param units:
        :param format:
        :return:
        """
        return datetime.datetime.strptime(num2date(times=cftime, units=units).strftime(format), format)

    @staticmethod
    def array2gtiff(array, filename):
        """
        将一个矩阵保存为tiff文件,
        这里还可以设置tiff的crs和transofrm。更多，可以查看rasterio的官网或者下面的这个链接
        https://gis.stackexchange.com/questions/279953/numpy-array-to-gtiff-using-rasterio-without-source-raster
        :param array: shape:(row, col)
        :param filename:
        :return:
        """
        with rasterio.open(filename, 'w', driver='GTiff',
                           height=array.shape[0], width=array.shape[1],
                           count=1, dtype=str(array.dtype)) as f:
            f.write(array, 1)

    def pic(self, lon, lat) -> bool:

        """
        检测一个点是否在中国边界线内
        lon:东经
        lat:北纬
        :param lon:
        :param lat:
        :return:
        """
        return self.geopandas.contains(Point(lon, lat))[0]

    def parallel_mask(self, index_lon, index_lat):
        point = (self.longitude_data[index_lon], self.latitude_data[index_lat])
        value = self.pic(lon=point[0], lat=point[1])
        # return value
        self.mask_matrix[index_lat, index_lon] = value

    def run(self):
        # 处理geopandas数据
        # self.geopandas = self.geopandas.iloc[0, :]
        self.geopandas['geometry'] = self.geopandas.buffer(0)

        # 处理nc数据
        self.nc_target_data = np.array(self.nc_data.variables[self.nc_variable])

        if 'missing_value' in dir(self.nc_data.variables[self.nc_variable]):
            self.target_data_missing_value = self.nc_data.variables[self.nc_variable].missing_value
        else:
            self.target_data_missing_value = np.nan

        self.nc_target_data[self.nc_target_data == self.target_data_missing_value] = np.nan

        # 提取lat,lon,lat 变量
        self.longitude_data = np.array(self.nc_data.variables[self.lon_variable])
        self.latitude_data = np.array(self.nc_data.variables[self.lat_variable])
        self.time_units = self.nc_data.variables[self.time_variable].units
        self.time_data = np.array(self.nc_data.variables[self.time_variable])
        self.clean_time_data = [self.num2datetime(cftime=i, units=self.time_units) for i in self.time_data]

        # 创建一个mask
        nc_target_data_shape = self.nc_target_data.shape

        if len(nc_target_data_shape) == 3:
            (self.time_dim, self.lat_dim, self.lon_dim) = nc_target_data_shape
        else:
            (self.lat_dim, self.lon_dim) = nc_target_data_shape

        self.mask_matrix = np.full(shape=(self.lat_dim, self.lon_dim), fill_value=False)

        _ = Parallel(n_jobs=-1, backend='threading', verbose=0)(
            delayed(self.parallel_mask)(index_lon, index_lat)
            for index_lon in tqdm(range(self.lon_dim))
            for index_lat in range(self.lat_dim))

    def getclipdata(self):
        """
        返回一个mask处理好的矩阵
        :return:
        """
        value = self.nc_target_data.copy()
        for i in tqdm(range(self.time_data.shape[0])):
            temp = value[i, :, :]
            temp[~self.mask_matrix] = np.nan
            value[i, :, :] = temp

        return value

#%% 加载数据
shp_data = gpd.read_file(r"D:\data\dataset\climate data\Pearl\shp\ca_Union.shp")
nc_1988tp = nc.Dataset(r"D:\data\dataset\climate data\Pearl\1988tp.nc")

#%% 查看nc数据变量
for item in nc_1988tp.variables.values():
    print('*' * 70)
    print(item)

#%% 获得各变量
raw_longitude = np.array(nc_1988tp.variables['longitude'])
raw_latitude = np.array(nc_1988tp.variables['latitude'])
raw_time = np.array(nc_1988tp.variables['time'])
raw_tp = np.array(nc_1988tp.variables['tp'])

#*********************************************************************
# %%插值目标
# [ 89.75 -89.75]
# [-179.75 179.75]
target_lon = -179.75 + 0.5 * np.arange(0, 720)
target_lat = -89.75 + 0.5 * np.arange(0, 360)
target_points = np.array([[lat, lon] for lat in target_lat for lon in target_lon])
#%%测试
from scipy.interpolate import RegularGridInterpolator

f = RegularGridInterpolator(
    (raw_latitude,raw_longitude),raw_tp[0,:,:],bounds_error=False,
    fill_value=None)
new_value = f(target_points)
new_value_grid = new_value.reshape(len(target_lat), len(target_lon))
new_value_grid.shape

#%% 更改分辨率
target_value = []

for i in tqdm(range(raw_time.shape[0])):
    f = RegularGridInterpolator(
    (raw_latitude,raw_longitude),raw_tp[0,:,:],bounds_error=False,
    fill_value=None)

    temp_value = f(target_points)
    temp_value = new_value.reshape(len(target_lat), len(target_lon))
    target_value.append(temp_value)

target_value = np.array(target_value)
target_value.shape


#%%保存为nc文件
with nc.Dataset(r"D:\data\dataset\result\new_test1115.nc", mode='w', format='NETCDF4_CLASSIC') as ncfile:
    # 创建维度
    lat_dim = ncfile.createDimension('lat', 360)  # latitude axis
    lon_dim = ncfile.createDimension('lon', 720)  # longitude axis
    time_dim = ncfile.createDimension('time', None)

    # 创建变量
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'

    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'

    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'days since 1988-01-01 00:00:00'#根据nc_1988tp.variables['time']
    time.long_name = 'time'

    temp = ncfile.createVariable('temp', np.float64, ('time', 'lat', 'lon'))  # note: unlimited dimension is leftmost
    # temp.units = 'K' # degrees Kelvin
    temp.standard_name = 'air_temperature'

    # 写入变量
    lat[:] = target_lat
    lon[:] = target_lon
    time[:] = raw_time
    temp[:, :, :] = target_value

#*********************************************************************
# %% 测试
import xarray as xr

test_xr = xr.open_dataset(r"D:\data\dataset\result\new_test1115.nc")
test_xr
# %%
test_nc = nc.Dataset(r"D:\data\dataset\result\new_test1115.nc")
for item in nc_1988tp.variables.values():
    print('*' * 70)
    print(item)

#*********************************************************************
# %%掩膜
nc_mask = GetMask(geopandas=shp_data, nc_data=test_nc, nc_variable='temp', lat_variable='lat',
                  lon_variable='lon', time_variable='time')

nc_mask.run()

# %%
clip_test = nc_mask.getclipdata()

#*********************************************************************
# %%测试函数
Lon_data, Lat_data = np.meshgrid(target_lon, target_lat)

plot_data = pd.DataFrame({'lon': Lon_data.flatten(),
                          'lat': Lat_data.flatten(),
                          'mask': nc_mask.mask_matrix.flatten()})

#%%
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
shp_data.boundary.plot(ax=ax, color='black')
ax.grid()

plot_data_in = plot_data.loc[plot_data['mask']]
ax.scatter(plot_data_in['lon'], plot_data_in['lat'], s=0.1)

plot_data_out = plot_data.loc[~plot_data['mask']]
ax.scatter(plot_data_out['lon'], plot_data_out['lat'], s=0.1, c='red')
plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

im = ax.imshow(clip_test[1, :, :][::-1, :], cmap=plt.cm.get_cmap('RdYlBu'))

fig.colorbar(im, orientation='vertical')
#*********************************************************************
# %% 保存数据

with nc.Dataset(r"D:\data\dataset\result\1new_test1115.nc", mode='w', format='NETCDF4_CLASSIC') as ncfile:
    # 创建维度
    lat_dim = ncfile.createDimension('lat', 360)  # latitude axis
    lon_dim = ncfile.createDimension('lon', 720)  # longitude axis
    time_dim = ncfile.createDimension('time', None)

    # 创建变量
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'

    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'

    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'days since 1988-01-01 00:00:00'
    time.long_name = 'time'

    temp = ncfile.createVariable('temp', np.float64, ('time', 'lat', 'lon'))  # note: unlimited dimension is leftmost
    # temp.units = 'K' # degrees Kelvin
    temp.standard_name = 'air_temperature'

    # 写入变量
    lat[:] = target_lat
    lon[:] = target_lon
    time[:] = raw_time
    temp[:, :, :] = clip_test
# %%
