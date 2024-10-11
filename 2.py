#%%

#数据处理包
import numpy as np
import pandas as pd

#可视化包
import matplotlib.pyplot as plt

#处理python时间函数
import datetime

#处理nc数据
import netCDF4 as nc
from netCDF4 import num2date #时间转换

#处理网格数据eg:shp
import geopandas as gpd

#处理栅格数据tiff
import rasterio

#gis的逻辑判断
from shapely.geometry import Point

#设置投影坐标系
from cartopy import crs as ccrs

#打印进度条
from tqdm import tqdm
tqdm.pandas()

# 监测系统
import platform

# matplotlib 显示中文的问题
if platform.system() == 'Darwin':
    plt.rcParams["font.family"] = 'Arial Unicode MS'
elif platform.system() == 'Windows':
    plt.rcParams["font.family"] = 'SimHei'
else:
    pass

# %%
#nc文件保存为tiff文件