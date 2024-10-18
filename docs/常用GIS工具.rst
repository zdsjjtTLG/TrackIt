🔧 常用GIS工具
===================================

生成渔网图层(切分栅格)
----------------------------

传入一个geometry列是Polygon类型的面域GeoDataFrame，该函数可以帮助你切分栅格

.. code-block:: python
    :linenos:

    # 1. 从gotrackit导入栅格切分函数
    from gotrackit.tools.grid import get_grid_data
    import geopandas as gpd

    if __name__ == '__main__':
        region_gdf = gpd.read_file(r'region.shp')
        grid_gdf = get_grid_data(polygon_gdf=region_gdf, meter_step=100.0, is_geo_coord=True, generate_index=True)
        grid_gdf.to_file(r'grid.shp', encoding='gbk')

get_grid_data函数参数解释：

* polygon_gdf
    gdf.GeoDataFrame, 面域数据

* meter_step
    float, 栅格边长区域大小, m

* is_geo_coord
    传入的面域文件是否是经纬度坐标，默认True

* generate_index
    是否输出栅格矩阵索引，默认True

栅格化前如下图：

.. image:: _static/images/before_grid.png
    :align: center
-----------------------------------------------

栅格化后如下图：

.. image:: _static/images/after_grid.png
    :align: center
-----------------------------------------------

坐标转换
----------------------------

提供了 GCJ-02、wgs84、百度 坐标系之间的相互转换接口

单点转换
````````````
使用loc_convert方法代码示例::

    >>> from gotrackit.tools.coord_trans import LngLatTransfer
    >>> trans = LngLatTransfer()
    >>> trans_x, trans_y = trans.loc_convert(lng=114.361, lat=22.362, con_type='bd-84')



多点转换
````````````
.. note::
    v0.3.11推出

使用loc_convert方法代码示例::

    >>> from gotrackit.tools.coord_trans import LngLatTransfer
    >>> trans = LngLatTransfer()
    >>> trans_x, trans_y = trans.loc_convert(lng=np.array([114.361, 114.362]), lat=np.array([22.362, 33.361]), con_type='gc-84')


几何对象转换
````````````````````
使用obj_convert方法代码示例::

    >>> from gotrackit.tools.coord_trans import LngLatTransfer
    >>> from shapely.geometry import LineString
    >>> trans = LngLatTransfer()
    >>> l = LineString([(114.325, 22.36), (114.66, 22.365), (114.321, 22.452)])
    >>> trans_line = trans.obj_convert(geo_obj=l, con_type='gc-bd', ignore_z=True)


GeoDataFrame转换
````````````````````

.. note::
    v0.3.11推出

使用geo_convert方法代码示例::

    >>> from gotrackit.tools.coord_trans import LngLatTransfer
    >>> import geopandas as gpd
    >>> trans = LngLatTransfer()
    >>> gdf = gpd.read_file(r'./data/XXX.geojson')
    >>> gdf = gdf.to_crs('EPSG:4326')
    >>> new_gdf = trans.geo_convert(gdf=gdf, ignore_z=True, con_type='84-gc')


文件转换
````````````````````

.. note::
    v0.3.11推出

使用file_convert方法代码示例::

    >>> from gotrackit.tools.coord_trans import LngLatTransfer
    >>> trans = LngLatTransfer()
    >>> trans.file_convert(file_path=r'./xxx.geojson', ignore_z=True, con_type='bd-gc', out_fldr=r'./', out_file_name=r'xfer', file_type='geojson')


参数含义：

* con_type
    gc-84: GCJ-02向WGS84转换

    gc-bd: GCJ-02向百度转换

    84-gc: WGS84向GCJ-02转换

    84-bd: WGS84向百度转换

    bd-84: 百度向WGS84转换

    bd-gc: 百度向GCJ-02转换


* ignore_z
    是否忽略Z坐标，默认True，当且仅当几何对象含有Z坐标时才能指定ignore_z=Fasle


直线转弧线
----------------------------

将直线转化为弧线，提供圆弧、贝塞尔弧线两种类型。

基于起终点坐标得到圆弧坐标
`````````````````````````
使用arc_curve_cor方法代码示例::

    >>> from gotrackit.tools.geo_process import StraightLineToArc
    >>> sla = StraightLineToArc()
    >>> coords_list = sla.arc_curve_cor(o_loc=[114.212, 22.31], d_loc=[114.312, 22.131], r=1.2, sample_num=30)
    >>> print(coords_list)

arc_curve_cor方法接收起终点坐标，返回起终点之间的圆弧坐标
其中：

* r
    float, 代表圆弧的曲度，值越大，曲度越大
* sample_num
    int, 代表采样点数目

基于LineString得到圆弧线对象
````````````````````````````
使用arc_curve_line方法代码示例::

    >>> from gotrackit.tools.geo_process import StraightLineToArc
    >>> sla = StraightLineToArc()
    >>> l = LineString([(114.212, 22.31), (114.312, 22.131)])
    >>> arc_line = sla.arc_curve_line(l, r=1.5, sample_num=30)
    >>> print(arc_line)

arc_curve_line方法接收LineString对象，返回圆弧线LineString对象



基于起终点坐标得到贝塞尔弧线坐标
``````````````````````````````
使用bezier_curve_cor方法代码示例::

    >>> from gotrackit.tools.geo_process import StraightLineToArc
    >>> sla = StraightLineToArc()
    >>> coords_list = sla.bezier_curve_cor(o_loc=[114.212, 22.31], d_loc=[114.312, 22.131], r=1.2, sample_num=30, right_side=True)
    >>> print(coords_list)

bezier_curve_cor方法接收起终点坐标，返回起终点之间的贝塞尔弧线坐标
其中：

* r
    float, 代表弧的曲度，值越大，曲度越大

* sample_num
    int, 代表采样点数目

* right_side
    bool, 是否在拓扑方向右侧生成弧线，默认True


基于LineString得到贝塞尔弧线对象
``````````````````````````````````
使用bezier_curve_line方法代码示例::

    >>> from gotrackit.tools.geo_process import StraightLineToArc
    >>> sla = StraightLineToArc()
    >>> bezier_line = sla.bezier_curve_line(LineString([(114.212, 22.31), (114.312, 22.131)]), r=1.5, sample_num=30, right_side=False)
    >>> print(bezier_line)

bezier_curve_line方法接收LineString对象，返回贝塞尔弧线LineString对象


.. image:: _static/images/straight_arc.png
    :align: center
-----------------------------------------------


地理配准
----------------------------

.. note::
    v0.3.12推出

