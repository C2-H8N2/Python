#%%
import arcpy


# 初始化许可
if not arcpy.CheckProduct("ArcGISPro") == "Available":
    raise RuntimeError("ArcGIS Pro许可不可用。")

# %%
