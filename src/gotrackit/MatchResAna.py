# -- coding: utf-8 --
# @Time    : 2024/9/27 11:02
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
import geopandas as gpd
from .map.Net import Net
from .gps.LocGps import GpsPointsGdf
from .netreverse.RoadNet.save_file import save_file
from .netreverse.book_mark import generate_book_mark
from .GlobalVal import NetField, GpsField, MarkovField
from .tools.common import avoid_duplicate_cols
from .tools.time_build import build_time_col

gps_field = GpsField()
net_field = NetField()
markov_field = MarkovField()

agent_id_field = gps_field.AGENT_ID_FIELD
node_id_field = net_field.NODE_ID_FIELD
time_field = gps_field.TIME_FIELD
from_node_field, to_node_field = net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD

def generate_check_file(net: Net, warn_info_dict: dict = None, out_fldr: str = r'./', file_name: str = 'check'):
    """generate_check_file函数：

    - 依据匹配警告信息生成空间书签文件

    Args:
        net: Net路网对象
        warn_info_dict: MapMatch匹配警告信息
        out_fldr: 空间书签的存储目录
        file_name: 空间书签文件名称

    Returns:
        None
    """
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
        generate_book_mark(name_loc_dict=book_mark_dict, input_fldr=out_fldr, prj_name=file_name, _mode='replace')
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


def dense_res_based_on_net(net: Net, match_res_df: pd.DataFrame, lng_field: str = 'prj_lng',
                           lat_field: str = 'prj_lat', dis_threshold: float = 3,
                           time_format: str = '%Y-%m-%d %H:%M:%S',
                           time_unit: str = 's', plain_crs: str = 'EPSG:3857') -> pd.DataFrame:
    """路径匹配结果路径增密函数 - dense_res_based_on_net：

    - 对MapMatch后输出的匹配结果进行路径增密

    Args:
        net: Net路网对象
        match_res_df: 匹配结果表
        lng_field: 经度字段名称
        lat_field: 纬度字段名称
        dis_threshold: 合并距离阈值(米)
        time_format: 匹配结果表时间列字符串格式化模板
        time_unit: 匹配结果表时间列单位
        plain_crs: 平面投影坐标系

    Returns:
        增密后的匹配结果表
    """
    if match_res_df is not None and not match_res_df.empty:
        build_time_col(df=match_res_df, time_format=time_format, time_unit=time_unit, time_field=gps_field.TIME_FIELD)

        single_link_gdf = net.get_link_data()
        _crs = single_link_gdf.crs
        single_link_gdf = single_link_gdf[
            single_link_gdf[net_field.LINK_ID_FIELD].isin(set(match_res_df[net_field.LINK_ID_FIELD]))][
            [net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD, net_field.GEOMETRY_FIELD,
             net_field.LENGTH_FIELD]].copy()

        cache_prj_gdf = net.split_segment(single_link_gdf, del_loc=False)
        del single_link_gdf
        cache_prj_gdf[net_field.GEOMETRY_FIELD] = gpd.points_from_xy(cache_prj_gdf['t_x'], cache_prj_gdf['t_y'])
        cache_prj_gdf = gpd.GeoDataFrame(cache_prj_gdf, geometry=net_field.GEOMETRY_FIELD, crs=_crs)
        cache_prj_gdf = cache_prj_gdf.to_crs('EPSG:4326')
        cache_prj_gdf['t_x'], cache_prj_gdf['t_y'] = cache_prj_gdf[net_field.GEOMETRY_FIELD].x, \
            cache_prj_gdf[net_field.GEOMETRY_FIELD].y
        cache_prj_gdf.rename(columns={'t_x': lng_field, 't_y': lat_field}, inplace=True)
        del cache_prj_gdf['f_x'], cache_prj_gdf['f_y']
        del cache_prj_gdf['lv_dx'], cache_prj_gdf['lv_dy'], cache_prj_gdf['lvl']
        cache_prj_gdf['topo_seq'] += 1

        non_dup_ft = \
            cache_prj_gdf.drop_duplicates(subset=[net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD], inplace=False)[
                [net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD]]

        node_gdf = net.get_node_data()
        node_gdf.reset_index(inplace=True, drop=True)
        non_dup_ft = pd.merge(non_dup_ft, node_gdf, left_on=net_field.FROM_NODE_FIELD, right_on=net_field.NODE_ID_FIELD)
        del non_dup_ft[net_field.NODE_ID_FIELD], node_gdf
        non_dup_ft['topo_seq'] = 0
        non_dup_ft = gpd.GeoDataFrame(non_dup_ft, geometry=net_field.GEOMETRY_FIELD, crs=_crs)
        non_dup_ft = non_dup_ft.to_crs('EPSG:4326')
        non_dup_ft[lng_field], non_dup_ft[lat_field] = non_dup_ft[net_field.GEOMETRY_FIELD].x, non_dup_ft[
            net_field.GEOMETRY_FIELD].y
        cache_prj_gdf = pd.concat([cache_prj_gdf, non_dup_ft])
        del non_dup_ft
        cache_prj_gdf[net_field.SEG_ACCU_LENGTH] = cache_prj_gdf[net_field.SEG_ACCU_LENGTH].fillna(0)
        cache_prj_gdf.sort_values(by=[net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD, 'topo_seq'], inplace=True)
        cache_prj_gdf.reset_index(inplace=True, drop=True)
        match_res_df = label_local_group(df=match_res_df[[gps_field.AGENT_ID_FIELD, net_field.FROM_NODE_FIELD,
                                                          net_field.TO_NODE_FIELD, gps_field.TIME_FIELD, lng_field,
                                                          lat_field, markov_field.DRIVING_L]],
                                         subset=[gps_field.AGENT_ID_FIELD, net_field.FROM_NODE_FIELD,
                                                 net_field.TO_NODE_FIELD])
        match_res_df[gps_field.GEOMETRY_FIELD] = gpd.points_from_xy(match_res_df[lng_field], match_res_df[lat_field])
        match_res_df = gpd.GeoDataFrame(match_res_df, geometry=gps_field.GEOMETRY_FIELD, crs='EPSG:4326')
        non_dup_interpolate_res_df = \
            match_res_df.drop_duplicates(subset=["__group__"], keep='first', inplace=False)

        interpolate_res_df = pd.merge(non_dup_interpolate_res_df[[gps_field.AGENT_ID_FIELD, net_field.FROM_NODE_FIELD,
                                                                  net_field.TO_NODE_FIELD, "__group__"]], cache_prj_gdf,
                                      on=[net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD])
        del non_dup_interpolate_res_df, cache_prj_gdf
        del interpolate_res_df['topo_seq']
        interpolate_res_df.rename(columns={net_field.SEG_ACCU_LENGTH: markov_field.DRIVING_L}, inplace=True)
        dense_res = pd.concat([match_res_df, interpolate_res_df], ignore_index=True)
        del interpolate_res_df
        dense_res.sort_values(by=['__group__', markov_field.DRIVING_L], inplace=True)
        dense_res.reset_index(inplace=True, drop=True)
        dense_res = dense_res.to_crs(plain_crs)
        dense_res['__gap__'] = dense_res[gps_field.GEOMETRY_FIELD].shift(-1).fillna(
            dense_res[gps_field.GEOMETRY_FIELD]).distance(
            dense_res[gps_field.GEOMETRY_FIELD])
        del dense_res[net_field.GEOMETRY_FIELD], dense_res["__group__"]
        dense_res.loc[
            dense_res[gps_field.AGENT_ID_FIELD].shift(-1).fillna(dense_res[gps_field.AGENT_ID_FIELD]) != dense_res[
                gps_field.AGENT_ID_FIELD],
            '__gap__'] = 1
        dense_res.loc[dense_res.tail(1).index, '__gap__'] = 1
        dense_res.drop(index=dense_res[dense_res['__gap__'] <= dis_threshold].index, inplace=True, axis=0)
        dense_res.reset_index(inplace=True, drop=True)
        dense_res[gps_field.TIME_FIELD] = dense_res[gps_field.TIME_FIELD].interpolate(method='linear')
        dense_res.dropna(subset=gps_field.TIME_FIELD, inplace=True)
        dense_res.reset_index(inplace=True, drop=True)
        return dense_res
    else:
        return pd.DataFrame()


def label_local_group(df=None, subset: list[str] = None):
    _df = df[subset].copy()
    _df.drop_duplicates(subset=subset, inplace=True, keep='first')
    _df['__seq__'] = [i for i in range(1, len(_df) + 1)]
    df = pd.merge(df, _df, on=subset, how='left')
    df['__group__'] = (df['__seq__'].shift(1).bfill().astype(int) != df['__seq__']).astype(int).cumsum()
    del df['__seq__']
    return df


class MatchResAna(object):
    def __init__(self):
        """匹配结果处理类-MatchResAna

        - 初始化
        """
        pass

    @staticmethod
    def del_dup_links(match_res_df: pd.DataFrame | gpd.GeoDataFrame,
                      time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's', keep: str = 'first',
                      use_time: bool = True) -> pd.DataFrame | gpd.GeoDataFrame:
        """MatchResAna类静态方法 - del_dup_links

        - 对匹配结果中的局部重复路段进行剔除

        Args:
            match_res_df: 匹配结果表
            use_time: 是否启用时间字段
            time_format: 时间列字符串模板, use_time=True时起效
            time_unit: 时间列单位, use_time=True时起效
            keep: 去重类型, first: 保留最早时刻进入某link的记录, last: 保留最晚时刻进入某link的记录

        Returns:
            局部去重后的匹配结果表
        """
        return del_dup_links(match_res_df=match_res_df, time_format=time_format, time_unit=time_unit, keep=keep,
                             use_time=use_time)

    @staticmethod
    def dense_res_based_on_net(net: Net, match_res_df: pd.DataFrame, lng_field: str = 'prj_lng',
                               lat_field: str = 'prj_lat', dis_threshold: float = 3,
                               time_format: str = '%Y-%m-%d %H:%M:%S',
                               time_unit: str = 's', plain_crs: str = 'EPSG:3857') -> pd.DataFrame:
        """MatchResAna类静态方法 - dense_res_based_on_net：

        - 对MapMatch后输出的匹配结果进行路径增密

        Args:
            net: Net路网对象
            match_res_df: 匹配结果表
            lng_field: 经度字段名称
            lat_field: 纬度字段名称
            dis_threshold: 合并距离阈值(米)
            time_format: 匹配结果表时间列字符串格式化模板
            time_unit: 匹配结果表时间列单位
            plain_crs: 平面投影坐标系

        Returns:
            增密后的匹配结果表
        """
        return dense_res_based_on_net(net=net, match_res_df=match_res_df, lat_field=lat_field, lng_field=lng_field,
                                      dis_threshold=dis_threshold, time_unit=time_unit, time_format=time_format,
                                      plain_crs=plain_crs)
def del_dup_links(match_res_df: pd.DataFrame | gpd.GeoDataFrame,
                  time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's', keep: str = 'first',
                  use_time: bool = True) -> pd.DataFrame | gpd.GeoDataFrame:
    """del_dup_links

    - 对匹配结果中的局部重复路段进行剔除

    Args:
        match_res_df: 匹配结果表
        use_time: 是否启用时间字段
        time_format: 时间列字符串模板, use_time=True时起效
        time_unit: 时间列单位, use_time=True时起效
        keep: 去重类型, first: 保留最早时刻进入的记录, last: 保留最晚时刻进入的记录

    Returns:
        局部去重后的匹配结果表
    """
    match_res_df = match_res_df.reset_index(drop=True)
    if use_time:
        build_time_col(df=match_res_df, time_unit=time_unit, time_format=time_format)
        match_res_df.sort_values(by=[agent_id_field, time_field], ascending=[True, True], inplace=True)

    rename_dict = avoid_duplicate_cols(df=match_res_df,
                                       built_in_col_list=['ft', 'single_link_id', 'next_single', 'label'])
    match_res_df['ft'] = match_res_df[from_node_field].astype(str) + '_' + match_res_df[to_node_field].astype(str)
    non_unique_ft = set(match_res_df['ft'])
    ft_single = {ft: s for ft, s in zip(non_unique_ft, [i for i in range(len(non_unique_ft))])}
    match_res_df['single_link_id'] = match_res_df['ft'].map(ft_single)
    if keep == 'first':
        match_res_df['next_single'] = match_res_df['single_link_id'].shift(1).fillna(-1).astype(int)
    else:
        match_res_df['next_single'] = match_res_df['single_link_id'].shift(-1).fillna(-1).astype(int)
    match_res_df['label'] = match_res_df['single_link_id'] - match_res_df['next_single']
    GpsPointsGdf.del_consecutive_zero(df=match_res_df, col='label', n=0)
    match_res_df.reset_index(inplace=True, drop=True)
    del match_res_df['label'], match_res_df['ft'], match_res_df['single_link_id'], match_res_df['next_single']

    # 恢复冲突字段
    rename_dict_reverse = dict((v, k) for k, v in rename_dict.items())
    if rename_dict:
        match_res_df.rename(columns=rename_dict_reverse, inplace=True)
    return match_res_df
