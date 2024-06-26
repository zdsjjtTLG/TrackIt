# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
import geopandas as gpd
from itertools import chain
from ...GlobalVal import NetField
from shapely.geometry import LineString
from ....WrapsFunc import function_time_cost

net_field = NetField()

# 线层数据、点层数据必需字段
length_field = net_field.LENGTH_FIELD
direction_field = net_field.DIRECTION_FIELD
link_id_field = net_field.LINK_ID_FIELD
from_node_id_field = net_field.FROM_NODE_FIELD
to_node_id_field = net_field.TO_NODE_FIELD
node_id_field = net_field.NODE_ID_FIELD
geometry_field = net_field.GEOMETRY_FIELD

required_field_list = [link_id_field, length_field, direction_field,
                       from_node_id_field, to_node_id_field, geometry_field]


@function_time_cost
def merge_double_link(link_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """
    same crs as input
    输入的link的dir只能为0 or 1
    将那些连接在同一from_node和to_node上的合并为dir为0的
    :param link_gdf:
    :return:
    """
    origin_crs = link_gdf.crs

    # 消除from_node = to_node的记录
    link_gdf.drop(index=link_gdf[link_gdf[from_node_id_field] == link_gdf[to_node_id_field]].index, inplace=True, axis=0)

    # 按照from_node, to_node去重
    link_gdf.drop_duplicates(subset=[from_node_id_field, to_node_id_field], keep='first', inplace=True)
    link_gdf.reset_index(inplace=True, drop=True)

    link_gdf['ft'] = link_gdf.apply(lambda x: tuple(sorted([x[from_node_id_field], x[to_node_id_field]])), axis=1)
    link_gdf['true_ft'] = link_gdf.apply(lambda x: (x[from_node_id_field], x[to_node_id_field]), axis=1)
    dup_ft_list = link_gdf[link_gdf['ft'].duplicated()]['ft'].to_list()

    dup_link_index = link_gdf[link_gdf['ft'].isin(dup_ft_list)].index
    dup_link_gdf = link_gdf.loc[dup_link_index, :].copy()
    link_gdf.drop(index=dup_link_index, axis=0, inplace=True)
    link_gdf.reset_index(inplace=True, drop=True)

    # merge_link_gdf = gpd.GeoDataFrame([], columns=list(dup_link_gdf.columns))
    # for ft, df in dup_link_gdf.groupby('ft'):
    #     # df必然有两条记录
    #     assert len(df) == 2, 'df记录数有误......'
    #     df.reset_index(inplace=True, drop=True)
    #
    #     attr_dict = {col: df.at[0, col] for col in list(df.columns)}
    #     attr_dict[direction_field] = 0
    #     merge_link_gdf.loc[len(merge_link_gdf), :] = attr_dict

    merge_link_gdf = dup_link_gdf.drop_duplicates(subset=['ft'], keep='first').copy()
    del dup_link_gdf
    merge_link_gdf[direction_field] = 0

    merge_link_gdf.set_geometry(geometry_field, crs=origin_crs, inplace=True)
    link_gdf = pd.concat([link_gdf, merge_link_gdf])
    link_gdf = gpd.GeoDataFrame(link_gdf, geometry=geometry_field, crs=origin_crs)
    link_gdf.drop(columns=['ft', 'true_ft'], axis=1, inplace=True)
    link_gdf.reset_index(inplace=True, drop=True)
    for col in [link_id_field, from_node_id_field, to_node_id_field, direction_field]:
        link_gdf[col] = link_gdf[col].astype(int)
    return link_gdf


def convert_neg_to_pos(link_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """
    same crs as input
    将路网文件中的dir为-1的link转化为dir为1, 0的保持不变
    :param link_gdf: gpd.GeoDataFrame
    :return: gpd.GeoDataFrame
    """
    # 找出dir为-1的记录
    link_neg = link_gdf[link_gdf[direction_field] == -1].copy()

    if link_neg.empty:
        return link_gdf
    else:
        origin_crs = link_gdf.crs

        # 从原路网中删除dir为-1的记录
        link_gdf.drop(index=link_gdf[link_gdf[direction_field] == -1].index, axis=0, inplace=True)

        # 改变几何列的拓扑方向, 同时反转from_node和to_node字段
        if geometry_field in list(link_gdf.columns):
            link_neg[geometry_field] = link_neg[geometry_field].apply(lambda x: LineString(list(x.coords)[::-1]))
        link_neg[[from_node_id_field, to_node_id_field]] = link_neg[[to_node_id_field, from_node_id_field]]
        link_neg[direction_field] = 1

        # 双向字段取反
        col_list = list(link_gdf.columns)
        if geometry_field in col_list:
            col_list.remove(geometry_field)
        double_cols_list = list(set([col[:-3] for col in col_list if (col.endswith('_ab') or col.endswith('_ba'))]))

        ab_list = list(chain(*[['_'.join([col, 'ab']), '_'.join([col, 'ba'])] for col in double_cols_list]))
        ba_list = list(chain(*[['_'.join([col, 'ba']), '_'.join([col, 'ab'])] for col in double_cols_list]))

        link_neg[ab_list] = link_neg[ba_list]

        # 合并到原路网
        link_gdf = pd.concat([link_gdf, link_neg])
        link_gdf.reset_index(inplace=True, drop=True)

        link_gdf = gpd.GeoDataFrame(link_gdf, geometry=geometry_field, crs=origin_crs)

        return link_gdf


def create_single_link(link_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    基于原来路网创建单向路网
    :return:
    """
    link_gdf[net_field.DIRECTION_FIELD] = link_gdf[net_field.DIRECTION_FIELD].astype(int)
    neg_link = link_gdf[link_gdf[net_field.DIRECTION_FIELD] == 0].copy()
    if neg_link.empty:
        return link_gdf
    else:
        neg_link[net_field.GEOMETRY_FIELD] = neg_link[net_field.GEOMETRY_FIELD].apply(
            lambda line_geo: LineString(list(line_geo.coords)[::-1]))

        single_link_gdf = pd.concat([link_gdf, neg_link])
        single_link_gdf.reset_index(inplace=True, drop=True)
        return single_link_gdf


if __name__ == '__main__':
    pass





