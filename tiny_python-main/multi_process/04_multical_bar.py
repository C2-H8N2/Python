from multiprocessing import Pool
from tqdm import tqdm
import time


def myf(x):
    time.sleep(1)
    return x * x


if __name__ == '__main__':
    value_x= range(200)
    P = Pool(processes=20)

    # 这里计算很快
    res = [P.apply_async(func=myf, args=(i, )) for i in value_x]

    # 主要是看这里
    result = [i.get(timeout=2) for i in tqdm(res)]

    print(result)