import netCDF4 as nc
import numpy as np
#导入数据集
nc_data = nc.Dataset("D:\data\\remote_sensing_data\pre_2023.nc")
'''
#查看数据集维度数据
for temp_dim in nc_data.dimensions.values():
    print(f'diamenson_name:{temp_dim.name},dimension_size:{temp_dim.size}')

#查看变量数据
for temp in nc_data.variables.values():
    print('-'*50)
    print(temp)
'''
#提取数据
processing_data = np.array(nc_data.variables['pre'])
(tiem,lat,lon) = processing_data.shape #time=12,lat=5146,lon=7849

#处理数据
result_data = np.ones(shape=(lat,lon))

#计算
import time
import random
from tqdm import tqdm
'''
for temp_lat in tqdm(range(lat)):
    time.sleep(random.randint(0,3))
    for temp_lon in range(lon):
        result_data[temp_lat,temp_lon] = processing_data[:,temp_lat,temp_lon].sum()
'''
lon_data = np.array(nc_data.variables['lon'])
lat_data = np.array(nc_data.variables['lat'])
Lon_data,Lat_data = np.meshgrid(lon_data,lat_data)#将纬度和经度数据结合

#画图
import matplotlib.pyplot as plt
import numpy as np
%matplotlib
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(Lon_data, Lat_data, result_data, cmap=plt.cm.YlGnBu_r)

ax.set_xlabel(r'Longitude (degrees_east)')
ax.set_ylabel(r'Latitude (degrees_north)')
ax.set_zlabel(r'sum (degK)')

plt.show()