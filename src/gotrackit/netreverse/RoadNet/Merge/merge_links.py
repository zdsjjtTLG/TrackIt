# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData


"""路网拓扑优化"""

import os
import pandas as pd
from tqdm import tqdm
import networkx as nx
import geopandas as gpd
from itertools import chain
from ...GlobalVal import NetField
from shapely.ops import linemerge
from geopy.distance import distance
from shapely.geometry import Point, LineString
from .get_merged_link_seq import get_merged_link_seq
from .limit.same_head_tail_limit import same_ht_limit
from .limit.same_head_tail_limit import get_head_tail_root

net_field = NetField()

length_field = net_field.LENGTH_FIELD
node_id_field = net_field.NODE_ID_FIELD
link_id_field = net_field.LINK_ID_FIELD
geometry_field = net_field.GEOMETRY_FIELD
to_node_id_field = net_field.TO_NODE_FIELD
direction_field = net_field.DIRECTION_FIELD
from_node_id_field = net_field.FROM_NODE_FIELD
required_field_list = [link_id_field, length_field, direction_field,
                       from_node_id_field, to_node_id_field, geometry_field]


def merge_two_degrees_node(link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None,
                           ignore_dir: bool = False, limit_col_name: str = None,
                           allow_ring: bool = False, plain_prj: str = 'EPSG:32650', accu_l_threshold: float = 200.0,
                           angle_threshold: float = 15.0,
                           restrict_length: bool = True, restrict_angle: bool = True, min_length: float = 50.0) \
        -> (gpd.GeoDataFrame, gpd.GeoDataFrame):
    """
    按照限制规则合并2度节点
    :param link_gdf: 路网线层, EPSG:4326
    :param node_gdf: 路网点层, EPSG:4326
    :param ignore_dir: 是否忽略方向进行路段合并
    :param limit_col_name: 限制字段名称
    :param allow_ring: 是否允许合并后出现环
    :param plain_prj: 平面投影坐标系
    :param accu_l_threshold: 合并后的路段最长不能超过多少m
    :param angle_threshold: 当路段转角超过多少度则划分节点
    :param restrict_length: 是否启用路段长度限制
    :param restrict_angle: 是否启用路段转交限制
    :param min_length: 路段最短不能短于多少米
    :return: crs - EPSG:4326
    """
    # 1.找出可以合并的2度节点组
    for col in [link_id_field, from_node_id_field, to_node_id_field, direction_field]:
        link_gdf[col] = link_gdf[col].astype(int)
    merged_link_df = get_merged_link_seq(link_gdf=link_gdf, judge_col_name=limit_col_name, ignore_dir=ignore_dir,
                                         allow_ring=allow_ring,
                                         node_gdf=node_gdf,
                                         plain_prj=plain_prj, accu_l_threshold=accu_l_threshold,
                                         angle_threshold=angle_threshold,
                                         restrict_length=restrict_length, restrict_angle=restrict_angle,
                                         min_length=min_length)
    # 2.进行合并
    if merged_link_df is not None:
        new_link, new_node, info_dict = merge_links(link_gdf=link_gdf, node_gdf=node_gdf, merge_link_df=merged_link_df)
    else:
        new_link, new_node, info_dict = link_gdf, node_gdf, dict()

    origin_crs = new_link.crs
    if origin_crs == plain_prj:
        pass
    else:
        new_link = new_link.to_crs(plain_prj)
        new_link[length_field] = new_link[geometry_field].apply(lambda x: x.length)
        new_link = new_link.to_crs(origin_crs)
    drop_no_use_nodes(link_gdf=new_link, node_gdf=new_node)
    return new_link, new_node, info_dict


# 逻辑子模块, 只记录合并信息, 并不修改传入的数据
def merge_links(link_gdf=None, node_gdf=None, merge_link_df=None) -> (gpd.GeoDataFrame, gpd.GeoDataFrame, dict):
    """传入可合并的路段组, 直接在线层数据, 点层数据上修改
    :param link_gdf: gpd.GeoDataFrame, 线层数据, EPSG:4326
    :param node_gdf: gpd.GeoDataFrame, 点层数据, EPSG:4326
    :param merge_link_df: pd.oDataFrame, 合并路段信息
    :return: crs - EPSG:4326

    ~~Input~~:
    merge_link_df:
       group              link_seq                                           dir_list         attr_list
        1     [(2, 10), (10, 99), (8, 99)]                                   [1,1,1]      [XX路,XX路,XX路]
        2     [(13, 14), (14, 15), (15, 16), (16, 17)]                       [0,0,0]      [XX路,XX路,XX路]
        3     [(1, 7), (7, 9), (3, 9), (3, 4), (4, 5), (5, 99)]               [0,1,0]      [XX路,XX路,XX路]
    """

    # 直接修改传入的link_gdf
    print(r'##########   Merge Road Sections')
    origin_crs = 'EPSG:4326'
    node_geo_map = {node: geo for node, geo in zip(node_gdf[node_id_field], node_gdf[geometry_field])}
    node_gdf.set_index(node_id_field, inplace=True)
    link_gdf['sorted_ft'] = link_gdf[[from_node_id_field, to_node_id_field]].apply(lambda x: tuple(sorted(x)), axis=1)

    origin_sorted_ft_list = link_gdf['sorted_ft'].to_list()
    # 用于记录合并信息
    merge_info_dict = dict()

    # 计算head_node, tail_node, root_node

    merge_link_df['head_tail_root_ring'] = merge_link_df['link_seq'].apply(lambda x: get_head_tail_root(x))

    # 不允许组内合并后首尾节点一致
    merge_link_df = same_ht_limit(origin_link_sorted_ft_list=origin_sorted_ft_list, merge_link_df=merge_link_df)

    # link表原有的属性
    link_columns_list = list(link_gdf.columns)

    # 未指定的属性置空
    non_specified_field_list = list(set(link_columns_list) - set(required_field_list))

    sum_del_node_list = []
    sum_merge_link_list = []
    sum_link_data_list = []
    for row in tqdm(merge_link_df.itertuples(), total=len(merge_link_df), desc=rf'Merge Road Sections', ncols=100):
        link_seq_list = getattr(row, 'link_seq')
        head_tail_root_ring = getattr(row, 'head_tail_root_ring')
        dir_list = getattr(row, 'dir_list')
        ring = head_tail_root_ring[0]
        root = head_tail_root_ring[1]
        head = head_tail_root_ring[2]
        tail = head_tail_root_ring[3]

        # 选出要合并的线层索引
        to_be_merge_link_gdf = link_gdf[link_gdf['sorted_ft'].isin(link_seq_list)].copy()

        # 修正线型(保证相接的点的坐标一致)
        # to_be_merge_link_gdf['geometry'] = to_be_merge_link_gdf.apply(
        #     lambda item: LineString([node_gdf.at[item[from_node_id_field], geometry_field]] +
        #                             list(item['geometry'].coords)[1:-1] +
        #                             [node_gdf.at[item[to_node_id_field], geometry_field]]), axis=1)
        to_be_merge_link_gdf['geometry'] = [LineString([node_geo_map[f]] +
                                                       list(line_geo.coords)[1:-1] +
                                                       [node_geo_map[t]]) for line_geo, f, t in
                                            zip(to_be_merge_link_gdf['geometry'],
                                                to_be_merge_link_gdf[from_node_id_field],
                                                to_be_merge_link_gdf[to_node_id_field])]
        merge_link_index = list(to_be_merge_link_gdf.index)

        if not ring:
            # 不是环
            if set(dir_list) == {0}:
                new_dir = 0
                # 任取一个首尾节点深度遍历
                assume_from_node = head
                assume_to_node = tail
            else:
                new_dir = 1
                d_g = nx.DiGraph()

                # to_be_merge_link_gdf['_d_from_to_'] = \
                #     to_be_merge_link_gdf.apply(lambda x: [x[from_node_id_field], x[to_node_id_field]], axis=1)
                # d_edge_list = to_be_merge_link_gdf['_d_from_to_'].to_list()
                # d_g.add_edges_from(d_edge_list)
                d_g.add_edges_from([[f, t] for f, t in zip(to_be_merge_link_gdf[from_node_id_field],
                                                           to_be_merge_link_gdf[to_node_id_field])])
                if nx.has_path(d_g, head, tail):
                    assume_from_node = head
                    assume_to_node = tail

                else:
                    assume_from_node = tail
                    assume_to_node = head

        else:
            # 是环, 找到根结点
            assume_from_node, assume_to_node = root, root
            new_dir = dir_list[0]

        # 获取待合并的路段的LineString
        merge_line_list = to_be_merge_link_gdf[geometry_field].to_list()
        merge_line = linemerge(merge_line_list)

        # 获取此时假定的拓扑起点的坐标
        # assumed_from_point = node_gdf.at[assume_from_node, geometry_field]
        assumed_from_point = node_geo_map[assume_from_node]

        # 获取实际的拓扑线段的起终点坐标
        # 合并后有可能出现多段线
        try:
            merge_line_start_point = Point(merge_line.coords[0])
            merge_line_end_point = Point(merge_line.coords[-1])

        # 这个问题的出现是因为如果merge_line_list里面的线型不连续, 会组成一个MultiLineString
        # 部分可以合并的链, 由于在其相接点上坐标有微小差异(小数点后8~9位不一样), 也会导致合并为MultiLineString
        # 所以先修正
        except NotImplementedError as e:
            print(e)
            # to_be_merge_link_gdf.to_csv(r'temp.csv', encoding='utf_8_sig', index=False)
            # to_be_merge_link_gdf.drop(columns=['sorted_ft'], axis=1, inplace=True)
            # to_be_merge_link_gdf.to_file(r'temp.shp')
        else:
            new_from_node, new_to_node = assume_from_node, assume_to_node

            if assumed_from_point.distance(merge_line_start_point) >= assumed_from_point.distance(merge_line_end_point):
                merge_line = LineString(list(merge_line.coords)[::-1])
            else:
                pass

            # 记录删除结点的信息
            if ring:
                del_node_list = list(set(chain(*link_seq_list)) - {root})
                sum_del_node_list.append(del_node_list)
            else:
                del_node_list = list(set(chain(*link_seq_list)) - {head, tail})
                sum_del_node_list.append(del_node_list)

            # 新增一条合并后的link
            new_link_id = link_gdf.at[list(merge_link_index)[0], link_id_field]

            first_link_index = list(to_be_merge_link_gdf.index)[0]
            non_specified_data_dict = {field: to_be_merge_link_gdf.loc[first_link_index, field] for field in
                                       non_specified_field_list}

            length = get_length_from_linestring(linestring_obj=merge_line, crs=origin_crs)

            data_dict = {link_id_field: new_link_id, direction_field: new_dir, length_field: length,
                         from_node_id_field: new_from_node, to_node_id_field: new_to_node, geometry_field: merge_line}

            data_dict.update(non_specified_data_dict)

            sum_link_data_list.append(data_dict)
            sum_merge_link_list.append(merge_link_index)

        merge_info_dict[rf'Merge-{link_seq_list[0]}'] = list(merge_line.coords)[0]

    node_gdf.reset_index(inplace=True, drop=False)
    # 删除节点信息
    sum_del_node_list = list(chain(*sum_del_node_list))
    node_gdf.drop(index=node_gdf[node_gdf[node_id_field].isin(sum_del_node_list)].index, inplace=True)

    # 删除被合并的Links
    sum_merge_link_list = list(chain(*sum_merge_link_list))
    link_gdf.drop(index=sum_merge_link_list, inplace=True)

    # 添加合并后的link信息
    new_link_dict = {field: [] for field in sum_link_data_list[0].keys()}

    for data_dict in sum_link_data_list:
        for key in data_dict.keys():
            new_link_dict[key].append(data_dict[key])
    new_link_df = pd.DataFrame(new_link_dict)
    new_link_gdf = gpd.GeoDataFrame(new_link_df, geometry=geometry_field, crs=origin_crs)
    new_link_gdf = new_link_gdf.astype(link_gdf.dtypes)
    link_gdf = pd.concat([link_gdf, new_link_gdf])
    link_gdf.drop(columns=['sorted_ft'], axis=1, inplace=True)
    link_gdf.reset_index(inplace=True, drop=True)
    link_gdf.drop_duplicates(subset=[from_node_id_field, to_node_id_field], keep='first', inplace=True)
    link_gdf.reset_index(inplace=True, drop=True)

    node_gdf.reset_index(inplace=True, drop=True)
    return link_gdf, node_gdf, merge_info_dict


def get_length_from_linestring(linestring_obj=None, crs='EPSG:4326') -> float:
    """
    在epsg:4326下计算LineString的长度, km
    :param linestring_obj: LineString, 多段线对象
    :param crs:
    :return:
    """
    if crs == 'EPSG:4326':
        coord_list = list(linestring_obj.coords)
        try:
            length_list = [distance(tuple(coord_list[i][::-1]), tuple(coord_list[i + 1][::-1])).m for i in range(0, len(coord_list) - 1)]
            return sum(length_list)
        except ValueError as e:
            # print(r'是平面坐标')
            return linestring_obj.length
    else:
        return linestring_obj.length


def drop_no_use_nodes(link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None):
    # 去除没有link连接的节点
    used_node = set(link_gdf[net_field.FROM_NODE_FIELD]) | set(link_gdf[net_field.TO_NODE_FIELD])
    node_gdf.reset_index(inplace=True, drop=True)
    node_gdf.drop(index=node_gdf[~node_gdf[net_field.NODE_ID_FIELD].isin(used_node)].index, inplace=True, axis=1)
    node_gdf.reset_index(inplace=True, drop=True)
