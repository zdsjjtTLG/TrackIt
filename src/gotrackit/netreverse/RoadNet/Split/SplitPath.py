# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData

"""按照相邻坐标点切分路径"""

import pandas as pd
import geopandas as gpd
from itertools import chain
from ...GlobalVal import NetField
from shapely.geometry import LineString

net_field = NetField()


def split_path_main(path_gdf: gpd.GeoDataFrame = None, restrict_region_gdf: gpd.GeoDataFrame = None, slice_num: int = 1,
                    attr_name_list: list = None, cut_slice: bool = False, drop_ft_loc: bool = True) -> gpd.GeoDataFrame:
    """
    这里考虑path_gdf特别多, 一次切分可能会内存不够, 可以分块处理, 输入的gdf是什么坐标系, 输出的gdf就是什么坐标系
    :param path_gdf: 路径gdf
    :param restrict_region_gdf: 用于区域限制
    :param slice_num:
    :param attr_name_list:
    :param cut_slice:
    :param drop_ft_loc:
    :return:
    """
    print(rf'##########   Roughly Remove Duplicates')
    path_gdf = path_gdf[attr_name_list + [net_field.GEOMETRY_FIELD]].copy()
    if cut_slice:
        print(rf'###   Cut Into {slice_num} Pieces')
        origin_crs = path_gdf.crs
        path_gdf['id'] = [i for i in range(1, len(path_gdf) + 1)]
        sum_path_gdf = gpd.GeoDataFrame()
        path_gdf['label'] = list(pd.cut(path_gdf['id'], bins=slice_num, labels=[i for i in range(1, slice_num + 1)]))
        for label in list(set(path_gdf['label'])):
            slice_path_gdf = path_gdf[path_gdf['label'] == label].copy()
            slice_path_gdf.reset_index(inplace=True, drop=True)

            # 切分去重
            _slice_path_gdf = split_path(path_gdf=slice_path_gdf, restrict_region_gdf=restrict_region_gdf)
            sum_path_gdf = pd.concat([sum_path_gdf, _slice_path_gdf])
        sum_path_gdf.reset_index(inplace=True, drop=True)
        sum_path_gdf.drop_duplicates(subset=['ft_loc'], inplace=True, keep='first')
        sum_path_gdf.reset_index(inplace=True, drop=True)
        sum_path_gdf.drop(columns=['label', 'id'], inplace=True, axis=1)
        sum_path_gdf = gpd.GeoDataFrame(sum_path_gdf, geometry=net_field.GEOMETRY_FIELD, crs=origin_crs)
    else:
        sum_path_gdf = split_path(path_gdf=path_gdf, restrict_region_gdf=restrict_region_gdf)

    if drop_ft_loc:
        sum_path_gdf.drop(columns=['ft_loc'], axis=1, inplace=True)
    else:
        pass
    
    del path_gdf

    return sum_path_gdf


def split_path(path_gdf: gpd.GeoDataFrame = None, restrict_region_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """
    拆解轨迹坐标, 并且粗去重(按照路段的起终点坐标)
    :param path_gdf: gpd.GeoDataFrame(), 必需参数, 必须字段: [geometry], crs要求EPSG:4326
    :param restrict_region_gdf: gpd.GeoDataFrame(), 非必需参数, 必须字段: [geometry], crs要求EPSG:4326
    :return: gpd.GeoDataFrame(),
    """
    origin_crs = path_gdf.crs
    path_gdf['point_list'] = path_gdf[net_field.GEOMETRY_FIELD].apply(lambda x: list(x.coords))
    path_gdf['line_list'] = path_gdf['point_list'].apply(
        lambda x: [LineString((x[i], x[i + 1])) for i in range(0, len(x) - 1)])
    path_gdf.drop(columns=[net_field.GEOMETRY_FIELD, 'point_list'], axis=1, inplace=True)
    path_gdf = path_gdf.explode('line_list', ignore_index=True)
    path_gdf.rename(columns={'line_list': net_field.GEOMETRY_FIELD}, inplace=True)
    path_gdf['ft_loc'] = path_gdf[net_field.GEOMETRY_FIELD].apply(lambda x: tuple(list(chain(*list(x.coords)))))
    path_gdf.drop_duplicates(subset=['ft_loc'], inplace=True, keep='first')
    path_gdf = gpd.GeoDataFrame(path_gdf, geometry=net_field.GEOMETRY_FIELD, crs=origin_crs)
    if restrict_region_gdf is None or restrict_region_gdf.empty:
        print(rf"##########   Don't Enable Region Restrictions")
    else:
        print(rf'##########   Enable Region Restrictions')
        path_gdf['index'] = [i for i in range(len(path_gdf))]
        path_gdf = gpd.sjoin(path_gdf, restrict_region_gdf[[net_field.GEOMETRY_FIELD]])
        path_gdf.drop_duplicates(subset=['index'], inplace=True, keep='first')
        del path_gdf['index_right'], path_gdf['index']
        path_gdf = gpd.GeoDataFrame(path_gdf, geometry=net_field.GEOMETRY_FIELD, crs=origin_crs)
    path_gdf.reset_index(inplace=True, drop=True)
    return path_gdf
