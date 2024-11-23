import numpy as np
from scipy.interpolate import RegularGridInterpolator

# 假设原始数据
raw_longitude = np.linspace(100, 110, 611)  # 长度为 611
raw_latitude = np.linspace(20, 30, 221)    # 长度为 221
raw_tp = np.random.random((1, 221, 611))   # 数据形状为 (1, 221, 611)

# 检查数据形状是否匹配
assert raw_tp.shape[1:] == (len(raw_latitude), len(raw_longitude)), "维度不匹配"

# 创建插值函数
f = RegularGridInterpolator((raw_latitude, raw_longitude), raw_tp[0, :, :])

# 目标点
target_lon = 105
target_lat = 25
target_points = np.array([[target_lat, target_lon]])

# 插值
new_value = f(target_points)
print(f"在经度 {target_lon}, 纬度 {target_lat} 的插值值为: {new_value}")
