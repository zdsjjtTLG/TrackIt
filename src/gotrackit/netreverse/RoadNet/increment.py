# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData


"""依据新的轨迹对已有路网进行增量编辑"""


import os
import pandas as pd
import geopandas as gpd
from itertools import chain
from .save_file import save_file
from .optimize_net import optimize
from shapely.geometry import LineString
from .Split.SplitPath import split_path
from ..PublicTools.GeoProcess import calc_link_angle
from .SaveStreets.streets import generate_node_from_link

node_id_field = 'node_id'
direction_field = 'dir'
link_id_field = 'link_id'
from_node_id_field = 'from_node'
to_node_id_field = 'to_node'
_link_id_field = '_link_id'
length_field = 'length'
road_name_field = 'road_name'
geometry_field = 'geometry'


def increment(link_gdf=None, node_gdf=None, path_gdf_dict=None, plain_crs='EPSG:32649', out_fldr=None,
              overlap_buffer_size: float = 0.3, save_times=20, save_new_split_link=True,
              limit_col_name: str = 'road_name', cover_ratio_threshold: float = 60.0,
              cover_angle_threshold: float = 6.5, net_file_type: str = 'shp') -> \
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    input: EPSG:4326, same output crs as input
    :param link_gdf:
    :param node_gdf:
    :param path_gdf_dict:
    :param plain_crs:
    :param out_fldr:
    :param overlap_buffer_size:
    :param save_times:
    :param save_new_split_link:
    :param cover_ratio_threshold:
    :param cover_angle_threshold:
    :param net_file_type:
    :param limit_col_name:
    :return:
    """
    if len(path_gdf_dict) <= 0:
        return link_gdf, node_gdf

    link_gdf = link_gdf.to_crs(plain_crs)
    node_gdf = node_gdf.to_crs(plain_crs)

    link_gdf['true_ft'] = link_gdf.apply(
        lambda x: tuple(sorted([x[to_node_id_field], x[from_node_id_field]])), axis=1)
    link_gdf['link_geo'] = link_gdf[geometry_field]

    _count = 0
    all_export_split_link = gpd.GeoDataFrame()

    for k in path_gdf_dict.keys():

        path_gdf = path_gdf_dict[k]

        print(rf'append od_id.{k}')
        new_link_gdf, new_node_gdf, to_be_del_link_list, to_be_remain_node_list, double_ft_list, export_split_link = \
            modify_net(
                link_gdf=link_gdf,
                node_gdf=node_gdf,
                plain_crs=plain_crs,
                path_gdf=path_gdf,
                limit_col_name=limit_col_name,
                overlap_buffer_size=overlap_buffer_size,
                cover_angle_threshold=cover_angle_threshold, cover_ratio_threshold=cover_ratio_threshold)
        _count += 1
        if save_new_split_link:
            export_split_link['od_id'] = k
            all_export_split_link = pd.concat([all_export_split_link, export_split_link])

        if new_link_gdf is not None:
            # new_link_gdf = new_link_gdf.to_crs(origin_crs)
            # new_node_gdf = new_node_gdf.to_crs(origin_crs)
            # link_gdf = link_gdf.to_crs(origin_crs)
            # node_gdf = node_gdf.to_crs(origin_crs)
            # new_link_gdf.to_file(os.path.join(out_fldr, rf'new_link.geojson'),
            #                      encoding='gbk', driver=rf'GeoJSON')
            # new_node_gdf.to_file(os.path.join(out_fldr, 'new_node.geojson'),
            #                      encoding='gbk', driver='GeoJSON')

            # 将要被删除的node是新link关联到的原来的哪些路网的from_node和to_node
            to_be_del_link_index = link_gdf[link_id_field].isin(to_be_del_link_list)
            to_be_del_link_gdf = link_gdf[to_be_del_link_index].copy()

            to_be_del_node_list = list((set(to_be_del_link_gdf[from_node_id_field]) |
                                        set(to_be_del_link_gdf[to_node_id_field])) - set(to_be_remain_node_list))
            node_gdf.drop(index=node_gdf[node_gdf[node_id_field].isin(to_be_del_node_list)].index, inplace=True, axis=0)
            node_gdf = pd.concat([node_gdf, new_node_gdf])
            node_gdf.reset_index(inplace=True, drop=True)

            new_link_gdf['true_ft'] = new_link_gdf.apply(
                lambda x: tuple(sorted([x[to_node_id_field], x[from_node_id_field]])), axis=1)

            new_link_gdf.loc[new_link_gdf['true_ft'].isin(double_ft_list), direction_field] = 0
            new_link_gdf['link_geo'] = new_link_gdf['geometry']

            link_gdf.drop(index=link_gdf[to_be_del_link_index].index, axis=0, inplace=True)
            link_gdf = pd.concat([link_gdf, new_link_gdf])
            link_gdf.reset_index(inplace=True, drop=True)
        else:
            link_gdf.loc[link_gdf['true_ft'].isin(double_ft_list), direction_field] = 0

        if _count >= save_times or _count == len(path_gdf_dict):
            export_link_gdf = link_gdf.copy()
            export_node_gdf = node_gdf.copy()
            export_link_gdf.drop(columns=['true_ft', 'link_geo'], axis=1, inplace=True)
            if export_link_gdf.crs != 'EPSG:4326':
                export_link_gdf = export_link_gdf.to_crs('EPSG:4326')
                export_node_gdf = export_node_gdf.to_crs('EPSG:4326')
            if out_fldr is not None:
                save_file(data_item=export_link_gdf, file_type=net_file_type,
                          out_fldr=out_fldr, file_name='increment_link')
                save_file(data_item=export_node_gdf, file_type=net_file_type,
                          out_fldr=out_fldr, file_name='increment_node')

            if _count == len(path_gdf_dict):
                if save_new_split_link:
                    all_export_split_link = gpd.GeoDataFrame(all_export_split_link, geometry=geometry_field,
                                                             crs='EPSG:4326')
                    save_file(data_item=all_export_split_link, file_type=net_file_type,
                              out_fldr=out_fldr, file_name='new_split_path.shp')
                return export_link_gdf, export_node_gdf
            else:
                _count = 0
                del export_link_gdf
                del export_node_gdf

def modify_net(link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None, plain_crs: str = 'EPSG:32649',
               path_gdf: gpd.GeoDataFrame = None, overlap_buffer_size: float = 0.2,
               cover_ratio_threshold: float = 60.0, cover_angle_threshold: float = 6.5,
               limit_col_name: str = None) -> (gpd.GeoDataFrame, gpd.GeoDataFrame, list, list, list, gpd.GeoDataFrame):
    """
    合并新路网
    :param link_gdf:
    :param node_gdf:
    :param plain_crs:
    :param path_gdf:
    :param overlap_buffer_size:
    :param limit_col_name:
    :param cover_angle_threshold:
    :param cover_ratio_threshold:
    :return:
    """
    # 这里的目标有三个, 提取新路径中的部分link和原有路网中的部分link构成reshape_link_gdf, 去生成新的点层和线层

    # 1.依据每个折点进行拆分, 加上_link_id, _from_node, _to_node, 并且转换投影坐标, 并且生成新的buffer面域几何列'path_buffer_geo'
    split_path_gdf = split_path(path_gdf=path_gdf, restrict_region_gdf=None)
    split_path_gdf[_link_id_field] = [i for i in range(1, len(split_path_gdf) + 1)]
    split_path_gdf.drop(columns=['ft_loc'], axis=1, inplace=True)

    export_split_link = split_path_gdf.copy()

    split_path_gdf = split_path_gdf.to_crs(plain_crs)
    split_path_gdf['path_buffer_geo'] = split_path_gdf[geometry_field].apply(lambda geo: geo.buffer(overlap_buffer_size))
    split_path_gdf.set_geometry('path_buffer_geo', inplace=True, crs=plain_crs)
    split_path_gdf[direction_field] = 1

    # 2.关联新路径和原有路网
    sjoin_gdf = gpd.sjoin(split_path_gdf, link_gdf)

    # 完全没有任何交集
    if sjoin_gdf.empty:
        # reshape_link_gdf就是新的路径文件
        print(r'新路径和原路网没有任何交集, 直接新增......')
        reshape_link_gdf = split_path_gdf
        reshape_link_gdf.drop(columns=[_link_id_field, 'path_buffer_geo'], axis=1, inplace=True)
        reshape_link_gdf.set_geometry(geometry_field, inplace=True, crs=plain_crs)
        cor_origin_link_list, double_ft_list = [], []
    else:
        # 计算两者相交的区域面积 、 占比link_buffer的比率
        def calc_inter_ratio(link_geo=None, path_buffer_geo=None):
            try:
                link_buffer_geo = link_geo.buffer(overlap_buffer_size)
            except AttributeError:
                return 0
            inter_polygon = link_buffer_geo.intersection(path_buffer_geo)
            return 100 * inter_polygon.area / path_buffer_geo.area

        sjoin_gdf['inter_ratio'] = sjoin_gdf.apply(
            lambda item: calc_inter_ratio(link_geo=item['link_geo'], path_buffer_geo=item['path_buffer_geo']), axis=1)
        sjoin_gdf.reset_index(inplace=True, drop=True)

        # 这里还要记录完全被cover的路径所关联的原路网的行车方向, 可能要更改原路网的dir为0
        # 这些完全被原有道路cover的split_path, 它还有可能会和其他的线路相交, 但是是点相交
        # 通过下面这一步, 所以要先记录这些split_path的_link_id, 再删除
        # 对于这些被原来link完全cover的path, 要检测其方向
        # 这个角度也用于和inter_ratio结合判断是否是重合link
        sjoin_gdf['angle'] = sjoin_gdf.apply(
            lambda item: calc_link_angle(link_geo1=item['link_geo'],
                                         link_geo2=item['geometry']), axis=1)

        # 相交率超过阈值 或者 相交率超过阈值 - 30 且角度小于阈值, 条件较为宽松, 宁愿少加也不多加
        covered_index = (sjoin_gdf['inter_ratio'] >= cover_ratio_threshold) | (
                    (sjoin_gdf['inter_ratio'] >= cover_ratio_threshold - 30) &
                    (sjoin_gdf['angle'] <= cover_angle_threshold) &
                    (sjoin_gdf['angle'] >= 0)) | (
                                (sjoin_gdf['inter_ratio'] >= cover_ratio_threshold - 30) &
                                (sjoin_gdf['angle'] >= 180 - cover_angle_threshold))
        be_covered_sjoin_df = sjoin_gdf[covered_index].copy()  # 完全被原有道路覆盖的关联对
        if be_covered_sjoin_df.empty:
            double_ft_list = []
        else:
            be_covered_sjoin_df['is_reverse'] = be_covered_sjoin_df['angle'].apply(
                lambda angle: True if angle >= 100 else False)
            be_covered_double_sjoin_df = be_covered_sjoin_df[be_covered_sjoin_df['is_reverse'] == True].copy()
            double_ft_list = list(
                set([tuple(sorted((int(f), int(t)))) for f, t in zip(be_covered_double_sjoin_df[from_node_id_field],
                                                                     be_covered_double_sjoin_df[to_node_id_field])]))
            print(rf'双向路段更新: {double_ft_list}')
        # 删掉所有的被100%被原有道路覆盖的关联对
        all_be_covered_path_link_id = list(sjoin_gdf[covered_index][_link_id_field].unique())
        sjoin_gdf.drop(index=sjoin_gdf[covered_index].index, inplace=True, axis=0)

        # 被100%覆盖的path的_link_id和split_path_gdf的_link_id一样
        if set(all_be_covered_path_link_id) == set(split_path_gdf[_link_id_field]):
            print('新路径完全被原有路网覆盖...')
            return None, None, None, None, double_ft_list, export_split_link
        else:
            not_covered_path = list(set(split_path_gdf[_link_id_field]) - set(all_be_covered_path_link_id))
            print(sjoin_gdf[sjoin_gdf[_link_id_field].isin(not_covered_path)][[link_id_field, _link_id_field,
                                                                               'inter_ratio', 'angle']])
            sjoin_gdf.reset_index(inplace=True, drop=True)
            sjoin_gdf[link_id_field] = sjoin_gdf[link_id_field].fillna(-1)
            sjoin_gdf[link_id_field] = sjoin_gdf[link_id_field].astype(int)

            # 选择那些没有被 原有路网 完全覆盖的path
            split_path_gdf.set_geometry('geometry', inplace=True)
            split_path_gdf.drop(columns=['path_buffer_geo'], axis=1, inplace=True)
            new_split_path_gdf = split_path_gdf[~split_path_gdf[_link_id_field].isin(all_be_covered_path_link_id)].copy()

            # 注意1: 被关联的那些原路网(只要和path有关联就选上)
            cor_origin_link_list = list(sjoin_gdf[sjoin_gdf[link_id_field] != -1][link_id_field].unique())
            origin_link_gdf = link_gdf[link_gdf[link_id_field].isin(cor_origin_link_list)].copy()
            origin_link_gdf.drop(columns='link_geo', inplace=True, axis=1)
            origin_split_link_gdf = split_path(path_gdf=origin_link_gdf)
            origin_split_link_gdf.drop(columns=['ft_loc'], axis=1, inplace=True)

            # 行成reshape_link
            reshape_link_gdf = pd.concat([new_split_path_gdf, origin_split_link_gdf])
            reshape_link_gdf.drop(columns=[from_node_id_field, to_node_id_field, _link_id_field,
                                           link_id_field, length_field],
                                  axis=1, inplace=True)
            reshape_link_gdf.reset_index(inplace=True, drop=True)

    # 生成点线
    new_link_gdf, new_node_gdf, node_group_status_gdf = generate_node_from_link(link_gdf=reshape_link_gdf,
                                                                                update_link_field_list=[
                                                                                    link_id_field,
                                                                                    length_field,
                                                                                    from_node_id_field,
                                                                                    to_node_id_field],
                                                                                using_from_to=False,
                                                                                plain_prj=plain_crs,
                                                                                ignore_merge_rule=True,
                                                                                modify_minimum_buffer=0.2,
                                                                                save_streets_before_modify_minimum=False,
                                                                                save_streets_after_modify_minimum=False)
    # 找出新的点层和原有点层的重叠点
    new_node_gdf['node_buffer'] = new_node_gdf['geometry'].apply(lambda x: x.buffer(0.2))
    new_node_gdf.set_geometry('node_buffer', inplace=True)
    old_new_join_df = gpd.sjoin(new_node_gdf, node_gdf)
    new_node_gdf.set_geometry('geometry', inplace=True)
    new_node_gdf.drop(columns=['node_buffer'], axis=1, inplace=True)
    shadow_new_node_list = set(old_new_join_df['node_id_left'].unique())

    shadow_edge_list = list(chain(*[[[shadow_new_node, -i],
                                     [shadow_new_node, - i - 1]] for shadow_new_node, i in
                                    zip(shadow_new_node_list,
                                        range(3, 2 * len(
                                            shadow_new_node_list) + 3, 2))]))

    shadow_link_gdf = gpd.GeoDataFrame(shadow_edge_list, columns=[from_node_id_field,
                                                                  to_node_id_field],
                                       geometry=[LineString([(0, 0), (1, 1)])] * len(shadow_edge_list), crs=plain_crs)
    shadow_link_gdf[direction_field] = 1
    shadow_link_gdf[link_id_field] = [-i for i in range(1, len(shadow_link_gdf) + 1)]
    new_link_gdf = pd.concat([new_link_gdf, shadow_link_gdf])
    new_link_gdf.reset_index(inplace=True, drop=True)

    # 做拓扑优化
    new_link_gdf, new_node_gdf, _ = optimize(link_gdf=new_link_gdf, node_gdf=new_node_gdf, ignore_dir=False,
                                             limit_col_name=limit_col_name,
                                             allow_ring=False, plain_prj=plain_crs, accu_l_threshold=125,
                                             angle_threshold=20,
                                             restrict_length=True, restrict_angle=True, save_preliminary=False,
                                             out_fldr=None,
                                             min_length=50.0,
                                             is_process_dup_link=False)
    new_link_gdf.drop(index=new_link_gdf[new_link_gdf[link_id_field] < 0].index, inplace=True, axis=0)
    new_link_gdf.reset_index(inplace=True, drop=True)
    # new_node_gdf和node_gdf做sjoin, 新老路网节点映射
    max_node = node_gdf['node_id'].max()
    max_link_id = link_gdf[link_id_field].max()
    new_link_gdf[link_id_field] = [i + max_link_id for i in range(1, len(new_link_gdf) + 1)]

    new_node_gdf['node_id'] = new_node_gdf['node_id'] + max_node
    new_link_gdf[from_node_id_field] = new_link_gdf[from_node_id_field] + max_node
    new_link_gdf[to_node_id_field] = new_link_gdf[to_node_id_field] + max_node

    new_node_gdf['node_buffer'] = new_node_gdf['geometry'].apply(lambda x: x.buffer(0.1))
    new_node_gdf.set_geometry('node_buffer', inplace=True)
    old_new_join_df = gpd.sjoin(new_node_gdf, node_gdf)
    new_node_gdf.set_geometry('geometry', inplace=True)
    new_node_gdf.drop(columns=['node_buffer'], axis=1, inplace=True)

    node_map_dict = {node_id_left: node_id_right
                     for node_id_left, node_id_right in zip(old_new_join_df['node_id_left'],
                                                            old_new_join_df['node_id_right'])}
    # 删除在原有路网中已经存在的node
    new_node_gdf.drop(index=new_node_gdf[new_node_gdf['node_id'].isin(list(node_map_dict.keys()))].index, axis=0,
                      inplace=True)

    # 原路网要保留的
    to_be_remain_node_list = list(node_map_dict.values())

    # 映射from_node和to_node
    new_link_gdf[from_node_id_field] = new_link_gdf[from_node_id_field].apply(
        lambda x: node_map_dict[x] if x in node_map_dict.keys() else x)
    new_link_gdf[to_node_id_field] = new_link_gdf[to_node_id_field].apply(
        lambda x: node_map_dict[x] if x in node_map_dict.keys() else x)
    ####
    return new_link_gdf, new_node_gdf, cor_origin_link_list, to_be_remain_node_list, double_ft_list, export_split_link


if __name__ == '__main__':
    pass



