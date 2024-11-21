#%%
import numpy as np
import matplotlib.pyplot as plt

# 假设 y_value 是已经给定的 y 值数据
y_value = np.array([1, 4, 9, 16, 25, 36])

# 创建对应的 x_new 数据，假设这里是 0 到 5 的索引
x_new = np.arange(y_value.shape[0])

# 拟合的多项式阶数
deg = 1

# 使用 np.polyfit 进行多项式拟合，得到系数
new_param = np.polyfit(x_new, y_value, deg=deg)
print(new_param)
#%%
# 初始化一个全零的数组来存储拟合值
value = np.zeros_like(x_new)
value
#%%
# 通过系数和指数计算多项式值
for index, param in enumerate(new_param[::-1]):
    print('{} {}'.format(index,param))
    value = value + param * x_new ** index
value
#%%
# 打印计算的多项式值
print("拟合值：", value)

# 绘制原始数据和拟合曲线
plt.plot(x_new, y_value, 'bo', label='Original Data')
plt.plot(x_new, value, 'r-', label='Fitted Polynomial')
plt.legend()
plt.show()

# %%
