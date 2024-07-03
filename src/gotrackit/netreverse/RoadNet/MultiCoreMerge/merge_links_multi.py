# -- coding: utf-8 --
# @Time    : 2024/4/8 11:45
# @Author  : TangKai
# @Team    : ZheChengData


"""多进程路网拓扑优化"""
import os
import pandas as pd
import networkx as nx
from tqdm import tqdm
import multiprocessing
import geopandas as gpd
from itertools import chain
from ...GlobalVal import NetField
from shapely.ops import linemerge
from geopy.distance import distance
from ....WrapsFunc import function_time_cost
from shapely.geometry import Point, LineString
from .limit.attr_limit import limit_attr_alpha
from .limit.direction_limit import limit_direction
from .limit.same_head_tail_limit import same_ht_limit
from ....tools.group import cut_group, cut_group_for_df
from .limit.same_head_tail_limit import get_head_tail_root
from .limit.two_degrees_group import get_two_degrees_node_seq

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

@function_time_cost
def merge_links_multi(link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None,
                      allow_ring: bool = False, ignore_dir: bool = False,
                      limit_col_name: str = None,
                      plain_prj: str = 'EPSG:32650',
                      accu_l_threshold: float = 200.0,
                      angle_threshold: float = 15.0,
                      restrict_length: bool = True,
                      restrict_angle: bool = True, min_length: float = 50.0, core_num: int = 3):
    """

    :param link_gdf: 路网线层, EPSG:4326
    :param node_gdf: 路网点层, EPSG:4326
    :param allow_ring: 是否允许合并后出现环
    :param ignore_dir: 是否忽略方向限制
    :param limit_col_name: 限制合并的字段名称
    :param plain_prj: 平面投影坐标系名称
    :param accu_l_threshold: 不允许合并后的长度超过该值(m)
    :param angle_threshold: 两个路段的转角超过该值则不允许合并(°)
    :param restrict_length: 是否开启长度限制
    :param restrict_angle: 是否开启转角限制
    :param min_length: 限制路段的最小长度, m
    :param core_num: 启用的核的数目
    :return:crs - EPSG:4326
    """
    print(rf'multicore merge - {core_num} cores')
    origin_crs = 'EPSG:4326'
    # step1: 建立图, 不会修改link_gdf
    ud_graph, d_graph = build_graph_from_link(link_df=link_gdf[[from_node_id_field, to_node_id_field, direction_field]],
                                              from_col_name=from_node_id_field, to_col_name=to_node_id_field,
                                              ignore_dir=ignore_dir, dir_col=direction_field)

    # 初步确定所有的可以合并的group
    connected_components, two_degrees_sub_graph = get_connected_components(ud_graph=ud_graph, d_graph=d_graph)

    # 对迭代器切片
    connected_components_group = cut_group(obj_list=connected_components, n=core_num)
    del connected_components
    n = len(connected_components_group)
    # 测试
    # merge_link_df = get_merge_df_main(link_gdf, node_gdf, connected_components_group[0], two_degrees_sub_graph,
    #                                   ud_graph, allow_ring, ignore_dir, judge_col_name, plain_prj, accu_l_threshold,
    #                                   angle_threshold, restrict_length, restrict_angle, min_length)
    print('starting filter group')
    pool = multiprocessing.Pool(processes=n)
    result_list = []
    for i in range(0, n):
        result = pool.apply_async(get_merge_df_main,
                                  args=(link_gdf, node_gdf, connected_components_group[i], two_degrees_sub_graph,
                                        ud_graph, allow_ring, ignore_dir, limit_col_name, plain_prj, accu_l_threshold,
                                        angle_threshold, restrict_length, restrict_angle, min_length))
        result_list.append(result)
    pool.close()
    pool.join()

    # 可合并的link组
    merge_link_df = pd.DataFrame()
    for merge_res in result_list:
        merge_link_df = pd.concat([merge_link_df, merge_res.get()])
    merge_link_df.reset_index(inplace=True, drop=True)
    del result_list

    if merge_link_df.empty:
        drop_no_use_nodes(link_gdf=link_gdf, node_gdf=node_gdf)
        return link_gdf, node_gdf

    link_gdf['sorted_ft'] = link_gdf[[from_node_id_field, to_node_id_field]].apply(lambda x: tuple(sorted(x)), axis=1)
    origin_sorted_ft_list = link_gdf['sorted_ft'].to_list()

    # 计算head_node, tail_node, root_node
    merge_link_df['head_tail_root_ring'] = merge_link_df.apply(lambda row: get_head_tail_root(row['link_seq']), axis=1)

    # 不允许组内合并后首尾节点一致
    merge_link_df = same_ht_limit(origin_link_sorted_ft_list=origin_sorted_ft_list, merge_link_df=merge_link_df)

    # 切分
    merge_link_df_group = cut_group_for_df(df=merge_link_df, n=core_num)
    del merge_link_df

    # test
    # merge_link_df = merge_link_df_group[0]
    # target_link_seq = list(chain(*merge_link_df['link_seq'].to_list()))
    # target_link_index = link_gdf['sorted_ft'].isin(target_link_seq)
    # target_link_gdf = link_gdf[target_link_index]
    # target_node_set = set(chain(*target_link_seq))
    # target_node_gdf = node_gdf[node_gdf['node_id'].isin(target_node_set)]
    # new_link_df, sum_del_node_list, merge_info_dict = merge_links(target_link_gdf, target_node_gdf, merge_link_df, link_gdf.crs.srs)
    # node_gdf.set_index('node_id', inplace=True)
    # node_gdf.drop(index=sum_del_node_list, axis=0, inplace=True)
    # node_gdf.reset_index(inplace=True, drop=False)
    # link_gdf.drop(index=link_gdf[target_link_index].index, inplace=True, axis=0)
    # new_link_gdf = gpd.GeoDataFrame(new_link_df, crs=link_gdf.crs, geometry='geometry')
    # del new_link_df

    pool = multiprocessing.Pool(processes=len(connected_components_group))
    result_list = []
    target_link_index = list()
    for i in range(0, len(merge_link_df_group)):
        merge_link_df = merge_link_df_group[i]
        target_link_seq = list(chain(*merge_link_df['link_seq'].to_list()))
        _target_link_index = link_gdf['sorted_ft'].isin(target_link_seq)
        target_link_gdf = link_gdf[_target_link_index]
        target_link_index.extend(list(link_gdf[_target_link_index].index))
        target_node_set = set(chain(*target_link_seq))
        target_node_gdf = node_gdf[node_gdf['node_id'].isin(target_node_set)]
        result = pool.apply_async(merge_links,
                                  args=(target_link_gdf, target_node_gdf, merge_link_df, origin_crs))
        result_list.append(result)
    pool.close()
    pool.join()
    new_link_df, sum_del_node_list, merge_info_dict = pd.DataFrame(), list(), dict()
    for new_res in result_list:
        _new_link_df, _sum_del_node_list, _merge_info_dict = new_res.get()
        new_link_df = pd.concat([new_link_df, _new_link_df])
        sum_del_node_list.extend(_sum_del_node_list)
        merge_info_dict.update(_merge_info_dict)

    node_gdf.set_index('node_id', inplace=True)
    node_gdf.drop(index=list(set(sum_del_node_list)), axis=0, inplace=True)
    node_gdf.reset_index(inplace=True, drop=False)
    link_gdf.drop(index=target_link_index, inplace=True, axis=0)
    new_link_gdf = gpd.GeoDataFrame(new_link_df, crs=link_gdf.crs, geometry='geometry')
    del new_link_df
    new_link_gdf = new_link_gdf.astype(link_gdf.dtypes)
    link_gdf = pd.concat([link_gdf, new_link_gdf])
    link_gdf.reset_index(inplace=True, drop=True)
    link_gdf.drop(columns=['sorted_ft'], axis=1, inplace=True)
    drop_no_use_nodes(link_gdf=link_gdf, node_gdf=node_gdf)
    return link_gdf, node_gdf, merge_info_dict


def build_graph_from_link(link_df=None, from_col_name='from', to_col_name='to', ignore_dir=False, dir_col='dir'):
    """
    根据路网的from_node, to_node, dir字段建立图
    :param link_df: pd.DataFrame, 路网线层数据
    :param from_col_name: str, 拓扑起始节点字段名称
    :param to_col_name: str, 拓扑终到节点字段名称
    :param ignore_dir: bool, 是否忽略方向
    :param dir_col: str, 方向字段名称
    :return:
    """

    edge_list = link_df[[from_col_name, to_col_name]].to_numpy()
    ud_g = nx.Graph()
    ud_g.add_edges_from(edge_list)

    # 忽略方向
    if ignore_dir:
        return ud_g, ud_g
    # 不忽略方向
    else:
        assert dir_col in list(link_df.columns), f'找不到方向字段{dir_col}'
        used_df = link_df.copy()
        zero_df = used_df[used_df[dir_col] == 0].copy()
        edge_list_a, edge_list_b = [], []
        if not used_df.empty:
            used_df['edge'] = \
                used_df.apply(
                    lambda x: [x[from_col_name], x[to_col_name], {'dir': x[dir_col]}] if x[dir_col] == 1 else [
                        x[to_col_name], x[from_col_name], {'dir': x[dir_col]}], axis=1)
            edge_list_a = used_df['edge'].to_list()

        if not zero_df.empty:
            zero_df['edge'] = zero_df.apply(lambda x: [x[from_col_name], x[to_col_name], {'dir': x[dir_col]}], axis=1)
            edge_list_b = zero_df['edge'].to_list()

        edge_list = edge_list_a + edge_list_b
        d_g = nx.DiGraph()
        d_g.add_edges_from(edge_list)
        return ud_g, d_g


def get_connected_components(ud_graph: nx.Graph = None, d_graph: nx.DiGraph = None):
    """
    初步找到所有潜在可以合并的组
    :param ud_graph: nx.net, 无向图
    :param d_graph: nx.net, 有向图
    :return:

      group              link_seq(sorted...)
        1     [(2, 10), (10, 99), (8, 99)]
        2     [(13, 14), (14, 15), (15, 16), (16, 17)]
        3     [(1, 7), (7, 9), (3, 9), (3, 4), (4, 5), (5, 1)]
    """
    # 找出度为2的节点id
    degree_dict = dict(nx.degree(ud_graph))  # 无向图的度
    d_degree_dict = dict(d_graph.degree)  # 有向图的度

    # 找出2度节点
    # 无向图中度为2的节点,(且在有向图中的入度和出度之和为2或者4)
    two_degree_node_list = [node for node in list(degree_dict.keys()) if
                            (degree_dict[node] == 2) and d_degree_dict[node] in [2, 4]]

    # 使用度为2的节点建立子图
    two_degrees_sub_graph = nx.subgraph(ud_graph, two_degree_node_list)

    # 度为2的节点组成子图中也包含很多不连通的子图, all_seq_list: [[12, 23, 34], [1, 234, 2], ...]
    connected_components = list(nx.connected_components(two_degrees_sub_graph))

    return connected_components, two_degrees_sub_graph


def get_merge_df_main(link_gdf=None, node_gdf=None, sub_graph_node_group=None, two_degrees_sub_graph=None,
                      ud_graph=None, allow_ring=False, ignore_dir: bool = False,
                      judge_col_name=None,
                      plain_prj: str = 'EPSG:32650',
                      accu_l_threshold: float = 200.0,
                      angle_threshold: float = 15.0,
                      restrict_length: bool = True,
                      restrict_angle: bool = True, min_length: float = 50.0):
    print('filter merge_link...')
    # 初步筛选拓扑可以合并的组
    merged_df = get_two_degrees_node_seq(ud_graph=ud_graph, allow_ring=allow_ring,
                                         sub_graph_node_group=sub_graph_node_group,
                                         two_degrees_sub_graph=two_degrees_sub_graph)
    # 限制
    if merged_df is None:
        return pd.DataFrame()
    else:
        # 添加一个link_seq_str字段, 需要用到
        # group, link_seq, link_seq_str
        merged_df['link_seq_str'] = merged_df['link_seq']. \
            apply(lambda x: ['_'.join(map(str, x[i])) for i in range(0, len(x))])

        # step4: 依据方向参数, 这里有可能返回None, 则说明没有满足优化条件的路段
        # 如果忽略方向
        if ignore_dir:
            # 则不进入方向限制函数, 直接添加dir_list字段
            # group, link_seq, link_seq_str, dir_list
            merged_df['dir_list'] = merged_df['link_seq_str'].apply(lambda x: [0] * len(x))
        # 如果不忽略方向
        else:

            # 不会修改link_gdf, 会修改merged_df, 增加一个dir_list字段
            # group, link_seq, link_seq_str, dir_list
            merged_df = limit_direction(merged_df=merged_df,
                                        origin_graph_degree_dict=dict(nx.degree(ud_graph)),
                                        link_df=link_gdf)

            # 经过方向限制后可能没有可合并的路段
            if merged_df is None:
                return pd.DataFrame()
            else:
                # 开始考虑属性限制
                if judge_col_name is None:
                    return merged_df
                else:
                    merged_df = limit_attr_alpha(merged_df=merged_df,
                                                 node_gdf=node_gdf,
                                                 link_df=link_gdf,
                                                 attr_col=judge_col_name,
                                                 plain_prj=plain_prj,
                                                 accu_l_threshold=accu_l_threshold,
                                                 angle_threshold=angle_threshold,
                                                 restrict_length=restrict_length,
                                                 restrict_angle=restrict_angle,
                                                 min_length=min_length)
                    if merged_df is None:
                        return pd.DataFrame()
                    else:
                        return merged_df
    return merged_df


# 逻辑子模块, 只记录合并信息, 并不修改传入的数据
def merge_links(link_gdf=None, node_gdf=None, merge_link_df=None, origin_crs: str = 'EPSG:4326') -> \
        (pd.DataFrame, list[int], dict):
    """传入可合并的路段组, 直接在线层数据, 点层数据上修改
    :param link_gdf: 将要被合并的link
    :param node_gdf: 将要被合并的link所涉及的node
    :param merge_link_df
    :param origin_crs
    :return:

    ~~Input~~:
    merge_link_df:
       group              link_seq                                           dir_list         attr_list
        1     [(2, 10), (10, 99), (8, 99)]                                   [1,1,1]      [XX路,XX路,XX路]
        2     [(13, 14), (14, 15), (15, 16), (16, 17)]                       [0,0,0]      [XX路,XX路,XX路]
        3     [(1, 7), (7, 9), (3, 9), (3, 4), (4, 5), (5, 99)]               [0,1,0]      [XX路,XX路,XX路]
    """

    # 直接修改传入的link_gdf
    print(r'##########   Merge Road Sections')
    node_geo_map = {node: geo for node, geo in zip(node_gdf[node_id_field], node_gdf[geometry_field])}
    node_gdf.set_index(node_id_field, inplace=True)

    merge_info_dict = dict()
    sum_del_node_list = []
    sum_link_data_list = []

    # link表原有的属性
    link_columns_list = list(link_gdf.columns)

    # 未指定的属性置空
    non_specified_field_list = list(set(link_columns_list) - set(required_field_list))

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
            # sum_merge_link_list.append(merge_link_index)

        merge_info_dict[rf'Merge-{link_seq_list[0]}'] = list(merge_line.coords)[0]

    node_gdf.reset_index(inplace=True, drop=False)

    # 记录要被删除的节点信息
    sum_del_node_list = list(chain(*sum_del_node_list))

    # 记录合并后的link信息
    new_link_dict = {field: [] for field in sum_link_data_list[0].keys()}
    for data_dict in sum_link_data_list:
        for key in data_dict.keys():
            new_link_dict[key].append(data_dict[key])
    new_link_df = pd.DataFrame(new_link_dict)
    return new_link_df, sum_del_node_list, merge_info_dict

def get_length_from_linestring(linestring_obj=None, crs='EPSG:4326'):
    """
    在epsg:4326下计算LineString的长度, km
    :param linestring_obj: LineString, 多段线对象
    :param crs:
    :return:
    """
    if crs.upper() == 'EPSG:4326':
        coord_list = list(linestring_obj.coords)
        try:
            length_list = [distance(tuple(coord_list[i][::-1]), tuple(coord_list[i + 1][::-1])).m for i in
                           range(0, len(coord_list) - 1)]
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