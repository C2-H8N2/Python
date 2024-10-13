#%%
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import platform
plt.rcParams["font.family"] = 'SimHei'

# %%
#导入数据
china_map = gpd.read_file(filename=r"D:\data\dataset\100000_中华人民共和国_full.json")
china_map['geometry'] = china_map.buffer(0)

# %%
print(china_map.crs)

# %%
china_map.head()

# %%
china_map.centroid
# %%
china_boundary = gpd.read_file("D:\data\dataset\中国地图边界202111版.json")
china_boundary['geometry'] = china_boundary.buffer(0)
china_boundary

# %%
#获得中心点x,y数据
china_boundary.centroid[0].x,china_boundary.centroid[0].y

# %%
#设置兰伯特投影
new_crs = ccrs.LambertConformal(central_longitude=china_boundary.centroid[0].x,
                                central_latitude=china_boundary.centroid[0].y)
print(new_crs)

# %%
#进行兰伯特投影
china_boundary_new_crs = china_boundary.to_crs(new_crs.proj4_init)
china_boundary_new_crs

# %%
ax = plt.subplot(121)
china_boundary.boundary.plot(ax=ax)
ax = plt.subplot(122,projection=new_crs)
china_boundary_new_crs.boundary.plot(ax=ax)
ax.gridlines(draw_labels=True)#绘制经纬度线

# %%
#以下为对样本数据点进行投影
sample_data = pd.DataFrame({
    'lon':np.linspace(start=100,stop=105,num=100),
    'lat':np.linspace(start=30,stop=45,num=100),
    'show_value':np.random.rand(100)
})
#设置样本点地理数据框
sample_data_gpd = gpd.GeoDataFrame(
    data=sample_data,
    geometry=gpd.points_from_xy(x=sample_data['lon'],y=sample_data['lat']),
    crs='epsg:4326'
)
#更改为兰伯特投影
sample_data_gpd=sample_data_gpd.to_crs(new_crs.proj4_init)
sample_data_gpd

# %%
fig, ax = plt.subplots(subplot_kw={'projection':new_crs},figsize = (10,6),dpi=100)
china_boundary_new_crs.boundary.plot(ax=ax,color='black')
sample_data_gpd.plot(ax=ax,column='show_value')
ax.gridlines(draw_labels=True)
