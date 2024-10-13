#%%
#本节内容np.mean(data,axis=x),计算第x轴数据均值(包括nan数据)
#np.nanmean(data,axis=x),同理,但是会忽略nan值
import numpy as np

#%%
data = np.random.randint(low=10,high=30,size=(12,360,720)).astype('float')
data.shape

# %%
np.mean(data,axis=0).shape
#表示沿着第 0 维度（即“矩阵集合”这个维度）对每个位置的元素进行平均。及对三维数组中纵向轴的数据求均值

# %%
data[0,0,0]=np.nan
np.mean(data,axis=0)

# %%
np.nanmean(data,axis=0)#nanmean不会将缺失值nan计入平均数计算中
# %%
