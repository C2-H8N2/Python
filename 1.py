#%%
import rasterio
from rasterio.transform import from_origin

# 假设你的二维列表是 data
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# 将二维列表转换为 NumPy 数组
import numpy as np
data_array = np.array(data)
data_array
print(str(data_array.dtype))
#%%
# 定义输出 TIFF 文件的路径
output_tiff_path = r"D:\data\dataset\result\output.tiff"

# 创建一个变换对象，指定左上角的坐标（经度和纬度）以及像素的大小
# 这里的例子假设每个像素的大小为 1度
transform = from_origin(west=0, north=10, xsize=1, ysize=1)

# 写入 TIFF 文件
with rasterio.open(
    output_tiff_path,
    'w',
    driver='GTiff',
    height=data_array.shape[0],
    width=data_array.shape[1],
    count=1,  # 单波段
    dtype=str(data_array.dtype),
    crs='EPSG:4326',  # WGS-84坐标系
    transform=transform
) as dst:
    dst.write(data_array, 1)
