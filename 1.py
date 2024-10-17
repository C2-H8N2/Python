import rasterio
from rasterio.transform import from_origin
import numpy as np
arr = np.random.randint(5, size=(100,100)).astype(np.float)

transform = from_origin(472137, 5015782, 0.5, 0.5)

new_dataset = rasterio.open(r'D:\data\dataset\result\test1.tif', 'w', driver='GTiff',
                            height = arr.shape[0], width = arr.shape[1],
                            count=1, dtype=str(arr.dtype),
                            crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
                            transform=transform)

new_dataset.write(arr, 1)
new_dataset.close()