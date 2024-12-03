# GIS工具

[get_grid_data]: ../Func&API/grid.md#get_grid_data

[StraightLineToArc]: ../Func&API/StraightLineToArc.md#init
[arc_curve_line]: ../Func&API/StraightLineToArc.md#arc_curve_line
[arc_curve_cor]: ../Func&API/StraightLineToArc.md#arc_curve_cor
[bezier_curve_line]: ../Func&API/StraightLineToArc.md#bezier_curve_line
[bezier_curve_cor]: ../Func&API/StraightLineToArc.md#bezier_curve_cor

[LngLatTransfer]: ../Func&API/LngLatTransfer.md#init
[loc_convert]: ../Func&API/LngLatTransfer.md#loc_convert
[obj_convert]: ../Func&API/LngLatTransfer.md#obj_convert
[gdf_convert]: ../Func&API/LngLatTransfer.md#gdf_convert
[file_convert]: ../Func&API/LngLatTransfer.md#file_convert

## 生成渔网图层(切分栅格)

传入一个geometry列是Polygon类型的面域GeoDataFrame，[get_grid_data]函数可以帮助你切分栅格

```python
# 1. 从gotrackit导入栅格切分函数
from gotrackit.tools.grid import get_grid_data
import geopandas as gpd

if __name__ == '__main__':
    region_gdf = gpd.read_file(r'region.shp')
    grid_gdf = get_grid_data(polygon_gdf=region_gdf, 
                             meter_step=100.0, is_geo_coord=True, generate_index=True)
    grid_gdf.to_file(r'grid.shp', encoding='gbk')
```

=== "栅格化前"

    <figure markdown="span">
      ![Image title](../_static/images/before_grid.png)
    </figure>

=== "栅格化后"

    <figure markdown="span">
      ![Image title](../_static/images/after_grid.png)
    </figure>


## 坐标转换

提供了 GCJ-02、wgs84、百度 坐标系之间的相互转换接口

### 单点转换
使用[loc_convert]函数：

```shell
>>> from gotrackit.tools.coord_trans import LngLatTransfer
>>> trans = LngLatTransfer()
>>> trans_x, trans_y = trans.loc_convert(lng=114.361, lat=22.362, con_type='bd-84')
```

### 多点转换
使用[loc_convert]函数：

```shell
>>> from gotrackit.tools.coord_trans import LngLatTransfer
>>> trans = LngLatTransfer()
>>> trans_x, trans_y = trans.loc_convert(lng=np.array([114.361, 114.362]), lat=np.array([22.362, 33.361]), con_type='gc-84')
```

### 几何对象转换
使用[obj_convert]函数：

```shell
>>> from gotrackit.tools.coord_trans import LngLatTransfer
>>> from shapely.geometry import LineString
>>> trans = LngLatTransfer()
>>> l = LineString([(114.325, 22.36), (114.66, 22.365), (114.321, 22.452)])
>>> trans_line = trans.obj_convert(geo_obj=l, con_type='gc-bd', ignore_z=True)
```

### GeoDataFrame转换
使用[gdf_convert]函数：

```shell
>>> from gotrackit.tools.coord_trans import LngLatTransfer
>>> import geopandas as gpd
>>> trans = LngLatTransfer()
>>> gdf = gpd.read_file(r'./data/XXX.geojson')
>>> gdf = gdf.to_crs('EPSG:4326')
>>> new_gdf = trans.geo_convert(gdf=gdf, ignore_z=True, con_type='84-gc')
```
### 文件转换
使用[file_convert]函数：

```shell
>>> from gotrackit.tools.coord_trans import LngLatTransfer
>>> trans = LngLatTransfer()
>>> trans.file_convert(file_path=r'./xxx.geojson', ignore_z=True, con_type='bd-gc', out_fldr=r'./', out_file_name=r'xfer', file_type='geojson')
```

## 直线转弧线
[StraightLineToArc]类支持直线转弧线的功能，提供圆弧、贝塞尔弧线两种类型。

### 基于起终点坐标得到圆弧坐标
使用类静态方法[arc_curve_cor]:

```shell
>>> from gotrackit.tools.geo_process import StraightLineToArc
>>> sla = StraightLineToArc()
>>> coords_list = sla.arc_curve_cor(o_loc=[114.212, 22.31], d_loc=[114.312, 22.131], r=1.2, sample_num=30)
>>> print(coords_list)
```


### 基于LineString得到圆弧线对象
使用类静态方法[arc_curve_line]，接收LineString对象，返回圆弧线LineString对象

```shell
>>> from gotrackit.tools.geo_process import StraightLineToArc
>>> sla = StraightLineToArc()
>>> l = LineString([(114.212, 22.31), (114.312, 22.131)])
>>> arc_line = sla.arc_curve_line(l, r=1.5, sample_num=30)
>>> print(arc_line)
```


### 基于起终点坐标得到贝塞尔弧线坐标
使用使用类静态方法[bezier_curve_cor]:
```shell
>>> from gotrackit.tools.geo_process import StraightLineToArc
>>> sla = StraightLineToArc()
>>> coords_list = sla.bezier_curve_cor(o_loc=[114.212, 22.31], d_loc=[114.312, 22.131], r=1.2, sample_num=30, right_side=True)
>>> print(coords_list)
```


### 基于LineString得到贝塞尔弧线对象

使用类静态方法[bezier_curve_line]，接收LineString对象，返回贝塞尔弧线LineString对象
```shell
>>> from gotrackit.tools.geo_process import StraightLineToArc
>>> sla = StraightLineToArc()
>>> bezier_line = sla.bezier_curve_line(LineString([(114.212, 22.31), (114.312, 22.131)]), r=1.5, sample_num=30, right_side=False)
>>> print(bezier_line)
```

<figure markdown="span">
  ![Image title](../_static/images/straight_arc.png)
</figure>


## 地理配准
----------------------------


