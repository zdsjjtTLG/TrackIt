# -- coding: utf-8 --
# @Time    : 2024/9/27 11:02
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
import geopandas as gpd
from .map.Net import Net
from .netreverse.RoadNet.save_file import save_file
from .netreverse.book_mark import generate_book_mark
from .GlobalVal import NetField, GpsField, MarkovField


gps_field = GpsField()
net_field = NetField()
markov_field = MarkovField()


def generate_check_file(warn_info_dict: dict = None, net: Net = None, out_fldr: str = r'./', file_name: str = 'check'):
    single_link_gdf = net.get_link_data()
    single_link_gdf.reset_index(inplace=True, drop=True)
    may_error_list, book_mark_dict = list(), dict()
    for agent_id in warn_info_dict.keys():
        _may_error_gdf = format_warn_info_to_geo(warn_info=warn_info_dict[agent_id], single_link_gdf=single_link_gdf)
        _may_error_gdf['ft_gps'] = str(agent_id) + '-' + _may_error_gdf['ft_gps']
        may_error_list.append(_may_error_gdf)

    if may_error_list:
        may_error_gdf = pd.concat(may_error_list).to_crs('EPSG:4326')
        book_mark_df = may_error_gdf.groupby('ft_gps').apply(
            lambda df: df[['geometry']].values[0, 0].centroid).reset_index(drop=False)
        book_mark_dict.update({k: (v.x, v.y) for k, v in zip(book_mark_df['ft_gps'], book_mark_df[0])})
        generate_book_mark(name_loc_dict=book_mark_dict, input_fldr=out_fldr, prj_name=file_name)
        save_file(data_item=may_error_gdf, file_name=file_name, out_fldr=out_fldr, file_type='shp')


def format_warn_info_to_geo(warn_info: pd.DataFrame = None,
                            single_link_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """

    :param warn_info:
    :param single_link_gdf:
    :return:
    """
    warn_info[['from_single', 'to_single', 'ft_gps']] = warn_info.apply(
        lambda row: (row['from_ft'][0], row['to_ft'][0], row['from_ft'][3]),
        axis=1, result_type='expand')
    format_warn_df = pd.concat([warn_info[['from_single', 'ft_gps']].rename(
        columns={'from_single': net_field.SINGLE_LINK_ID_FIELD}),
        warn_info[['to_single', 'ft_gps']].rename(
            columns={'to_single': net_field.SINGLE_LINK_ID_FIELD})])
    format_warn_df.reset_index(inplace=True, drop=True)
    may_error_gdf = pd.merge(format_warn_df,
                             single_link_gdf[
                                 [net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD,
                                  net_field.DIRECTION_FIELD,
                                  net_field.SINGLE_LINK_ID_FIELD, net_field.GEOMETRY_FIELD]],
                             on=net_field.SINGLE_LINK_ID_FIELD, how='left')
    del may_error_gdf[net_field.SINGLE_LINK_ID_FIELD]
    may_error_gdf = gpd.GeoDataFrame(may_error_gdf, geometry=net_field.GEOMETRY_FIELD,
                                     crs=single_link_gdf.crs)
    return may_error_gdf
