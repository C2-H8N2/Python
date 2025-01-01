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
china_map['geometry'] = china_map.buffer(0)#buffer(0)通常用于修复几何对象中的拓扑问题。当地理数据中存在自交、多边形不闭合或拓扑错误等问题时，可以有效地修复这些问题,对几何对象进行了重新构造，并且会消除一些小的几何错误。

# %%
print(china_map.crs)#Coordinate Reference System（坐标参考系）的缩写crs,提取坐标参考系属性,WGS84（EPSG:4326）

# %%
china_map.head()

# %%
china_map.centroid#获得中心点坐标
# %%
china_boundary = gpd.read_file("D:\data\dataset\中国地图边界202111版.json")
china_boundary['geometry'] = china_boundary.buffer(0)
china_boundary

# %%
#获得中心点x,y数据
china_boundary.centroid[0]
china_boundary.centroid[0].x,china_boundary.centroid[0].y#利用{}.centroid[0]方法提取{}的第一个元素x方向和y方向数据

# %%
#设置兰伯特投影及中心点,使其更适合特定区域的地图显示。
new_crs = ccrs.LambertConformal(central_longitude=china_boundary.centroid[0].x,
                                central_latitude=china_boundary.centroid[0].y)
print(new_crs)

# %%
#进行兰伯特投影
china_boundary_new_crs = china_boundary.to_crs(new_crs.proj4_init)
#proj4_init是Cartopy中投影的Proj4字符串，它是投影系统的一种文本表示方式，描述了投影的参数和具体的定义。
#通过上述方法可将目标 CRS 设置为你之前定义的 Lambert 正形投影的 Proj4 表示。
china_boundary_new_crs

# %%
'''
121 是三个数字的组合，表示子图的布局方式：
第一个数字1表示整个图形窗口的行数,即有 1 行子图。
第二个数字2表示整个图形窗口的列数,即有 2 列子图。
第三个数字1表示当前激活的子图是第 1 个子图（从左到右依次排列）。
plt.subplot(121) 表示将图形窗口分为 1 行 2 列，并创建或选择第 1 个子图。
'''
ax = plt.subplot(121) #创建一个子图（subplot），并将其分配给变量 ax
china_boundary.boundary.plot(ax=ax) #.plot(ax=ax)是一个绘图函数，它将 china_boundary.boundary的边界线绘制在指定的 ax（即第一个子图）上。
ax = plt.subplot(122,projection=new_crs)
china_boundary_new_crs.boundary.plot(ax=ax)
ax.gridlines(draw_labels=True)#绘制经纬度线

# %%
#以下为对样本数据点进行投影
sample_data = pd.DataFrame({
    'lon':np.linspace(start=100,stop=105,num=100),#numpy库中的linspace()函数生成了100个从100到105之间等间距的经度值。
    'lat':np.linspace(start=30,stop=45,num=100),
    'show_value':np.random.rand(100)#生成100个[0,1]随机数
})
#设置样本点地理数据框,将点坐标转换为相应的点对象，作为几何信息存储在 geometry 列中。
sample_data_gpd = gpd.GeoDataFrame(
    data=sample_data,
    geometry=gpd.points_from_xy(x=sample_data['lon'],y=sample_data['lat']),
    crs='epsg:4326'
)
#更改为兰伯特投影
sample_data_gpd=sample_data_gpd.to_crs(new_crs.proj4_init)
sample_data_gpd

# %%
#fig是图形对象,ax是坐标轴对象
fig, ax = plt.subplots(subplot_kw={'projection':new_crs},figsize = (10,6),dpi=100)
sample_data_gpd.plot(ax=ax,column='show_value')
#这行代码创建了一个新的图形和坐标轴,返回对象为(Figure, Axes)
#subplot_kw 是一个字典参数,{'projection': new_crs}指定坐标轴的投影类型为之前定义的new_crs
#figsize = (10,6)设置图形的宽度为 10 英寸，高度为 6 英寸。
#dpi=100每英寸 100 个像素的分辨率进行渲染
china_boundary_new_crs.boundary.plot(ax=ax,color='black')
sample_data_gpd.plot(ax=ax,column='show_value')
ax.gridlines(draw_labels=True)#draw_labels 是一个布尔参数，指定是否在网格线旁边绘制标签。当设置为 True 时，地图的经度和纬度值会显示在对应的网格线旁边。
