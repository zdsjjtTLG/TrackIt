# -- coding: utf-8 --
# @Time    : 2023/11/27 11:00
# @Author  : TangKai
# @Team    : ZheChengData


"""划分栅格"""

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import distance
from shapely.ops import unary_union
from shapely.geometry import Polygon

geometry_field = 'geometry'
lon_field = 'lon'
lat_field = 'lat'


# 获取栅格预测点
def get_grid_data(polygon_gdf: gpd.GeoDataFrame, meter_step: float = None, is_geo_coord: bool = True,
                  generate_index: bool = True) -> gpd.GeoDataFrame:
    """渔网(栅格)划分

    对面域图层进行栅格划分

    Args:
        polygon_gdf: 面域GeoDataFrame
        meter_step: 栅格边长(米)
        is_geo_coord: 是否是地理坐标
        generate_index: 是否生成行列索引

    Returns:
        栅格GeoDataFrame
    """
    crs = polygon_gdf.crs
    geo_list = polygon_gdf[geometry_field].to_list()
    polygon_obj = unary_union(geo_list)

    # 根据栅格区域大小对面域进行栅格划分
    grid_gdf = generate_mesh(polygon_obj=polygon_obj, meter_step=meter_step, is_geo_coord=is_geo_coord, crs=crs,
                             generate_index=generate_index)

    grid_gdf = gpd.GeoDataFrame(grid_gdf, geometry=geometry_field, crs=crs)
    grid_gdf = gpd.sjoin(grid_gdf, polygon_gdf[[geometry_field]])
    del grid_gdf['index_right']
    grid_gdf.reset_index(inplace=True, drop=True)
    if generate_index:
        return grid_gdf[['grid_id', 'dx', 'dy', geometry_field]]
    else:
        return grid_gdf[['grid_id', geometry_field]]

def generate_range(polygon_obj: Polygon = None, meter_step: float = 100.0, is_geo_coord: bool = True) -> \
        tuple[float, float, int, int, float, float]:
    (min_x, min_y, max_x, max_y) = polygon_obj.bounds

    cen_x = polygon_obj.centroid.x
    cen_y = polygon_obj.centroid.y

    # 计算区域的长宽
    _width = max_y - min_y
    _length = max_x - min_x

    # 根据区域的中心点确定经纬度步长
    lon_step = get_geo_step(lon=cen_x, lat=cen_y, direction=1, step=meter_step, is_geo_coord=is_geo_coord)
    lat_step = get_geo_step(lon=cen_x, lat=cen_y, direction=0, step=meter_step, is_geo_coord=is_geo_coord)

    # 计算长宽多少个格子, 多生成一个, 做边界保护
    width_n = math.ceil(_width / lat_step) + 1
    length_n = math.ceil(_length / lon_step) + 1

    return lon_step, lat_step, width_n, length_n, min_x, max_y

# 逻辑子模块：生成栅格用于获取预测点
def generate_mesh(polygon_obj: Polygon = None, meter_step: float = 100.0, is_geo_coord: bool = True,
                  crs: str = 'EPSG:4326', generate_index: bool = True) -> pd.DataFrame:
    """
    生成栅格用于获取预测点
    :param polygon_obj: gdf.GeoDataFrame, 面域数据
    :param meter_step: int, 栅格大小
    :param is_geo_coord:
    :param crs:
    :param generate_index:
    :return: gdf.GeoDataFrame
    """
    lon_step, lat_step, width_n, length_n, min_x, max_y = \
        generate_range(polygon_obj=polygon_obj, meter_step=meter_step, is_geo_coord=is_geo_coord)

    def generate(xy):
        return Polygon([(xy[0], xy[1]), (xy[0] + lon_step, xy[1]),
                        (xy[0] + lon_step, xy[1] - lat_step), (xy[0], xy[1] - lat_step)])
    all_grid_list = []
    for n in range(width_n):
        point_list = [(min_x + k * lon_step, max_y - n * lat_step) for k in range(length_n)]
        grid_list = list(map(generate, point_list))
        all_grid_list.extend(grid_list)
    grid_gdf = gpd.GeoDataFrame({'grid_id': [i + 1 for i in range(len(all_grid_list))]}, geometry=all_grid_list,
                                crs=crs)
    if generate_index:
        grid_gdf['mat_index'] = [[i, j] for i in range(width_n) for j in range(length_n)]
        # dx代表行索引, dy代表列索引
        grid_gdf['dx'] = grid_gdf['mat_index'].apply(lambda x: x[0])
        grid_gdf['dy'] = grid_gdf['mat_index'].apply(lambda x: x[1])
        del grid_gdf['mat_index']
    return grid_gdf

def generate_mesh_alpha(polygon_obj: Polygon = None, meter_step: float = 100.0, is_geo_coord: bool = True,
                        crs: str = 'EPSG:4326', generate_index: bool = True) -> pd.DataFrame:
    """
    生成栅格用于获取预测点
    :param polygon_obj: gdf.GeoDataFrame, 面域数据
    :param meter_step: int, 栅格大小
    :param is_geo_coord:
    :param crs:
    :param generate_index:
    :return: gdf.GeoDataFrame
    """
    lon_step, lat_step, width_n, length_n, min_x, max_y = \
        generate_range(polygon_obj=polygon_obj, meter_step=meter_step, is_geo_coord=is_geo_coord)

    def generate(xy):
        return Polygon([(xy[0], xy[1]), (xy[2], xy[3]), (xy[4], xy[5]), (xy[6], xy[7])])

    x_d = np.array([min_x + k * lon_step for k in range(length_n)])
    y_d = np.array([max_y - n * lat_step for n in range(width_n)])
    x_array, y_array = np.meshgrid(x_d, y_d)
    x_array, y_array = x_array.flatten(), y_array.flatten()
    d1, d2, d3, d4, d5, d6, d7, d8 = \
        x_array, y_array, x_array + lon_step, y_array, \
        x_array + lon_step, y_array - lat_step, x_array, y_array - lat_step
    all_grid_list = list(map(generate, zip(d1, d2, d3, d4, d5, d6, d7, d8)))
    grid_gdf = gpd.GeoDataFrame({'grid_id': [i + 1 for i in range(len(all_grid_list))]}, geometry=all_grid_list,
                                crs=crs)
    if generate_index:
        grid_gdf['mat_index'] = [[i, j] for i in range(width_n) for j in range(length_n)]
        # dx代表行索引, dy代表列索引
        grid_gdf['dx'] = grid_gdf['mat_index'].apply(lambda x: x[0])
        grid_gdf['dy'] = grid_gdf['mat_index'].apply(lambda x: x[1])
        del grid_gdf['mat_index']
    return grid_gdf


# 逻辑子模块：确定经纬度步长
def get_geo_step(lon: float = None, lat: float = None, direction: int = 1, step: float = 100,
                 is_geo_coord: bool = True) -> float:
    """
    根据区域中心点确定经纬度步长
    :param lon: float, 经度
    :param lat: float, 纬度
    :param direction: int, 方向
    :param step: int, 步长
    :param is_geo_coord:
    :return:
    """

    if direction == 1:
        new_lon = lon + 0.1
        if is_geo_coord:
            dis = distance((lat, lon), (lat, new_lon)).m
        else:
            dis = 0.1
        return 0.1 / (dis / step)
    else:
        new_lat = lat + 0.1
        if is_geo_coord:
            dis = distance((lat, lon), (new_lat, lon)).m
        else:
            dis = 0.1
        return 0.1 / (dis / step)