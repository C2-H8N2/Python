import pandas as pd
from tqdm import tqdm

# 激活 tqdm 的 Pandas 扩展
tqdm.pandas()

# 示例数据
df = pd.DataFrame({'a': range(100000)})

# 使用 tqdm 的进度条跟踪 apply 操作
df['b'] = df['a'].progress_apply(lambda x: x ** 2)