# -- coding: utf-8 --
# @Time    : 2024/1/25 14:37
# @Author  : TangKai
# @Team    : ZheChengData


"""路网点层生产, 并且做一些空间优化"""
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from ..save_file import save_file
from ...GlobalVal import NetField
from shapely.geometry import Point
from shapely.geometry import LineString
from ....WrapsFunc import function_time_cost


net_field = NetField()

link_required_field_list = [net_field.LINK_ID_FIELD, net_field.DIRECTION_FIELD, net_field.LENGTH_FIELD,
                            net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD]


def generate_node_from_link(link_gdf: gpd.GeoDataFrame = None, update_link_field_list: list[str] = None,
                            using_from_to: bool = False, fill_dir: int = 0, plain_prj: str = 'EPSG:32650',
                            ignore_merge_rule: bool = True, modify_minimum_buffer: float = 0.8,
                            execute_modify: bool = True, auxiliary_judge_field: str = None,
                            out_fldr: str = None, save_streets_before_modify_minimum: bool = False,
                            save_streets_after_modify_minimum: bool = False, net_file_type: str = 'shp',
                            ) -> \
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    same crs as input
    :param link_gdf: gpd.GeoDataFrame, 路网线层gdf数据
    :param using_from_to: bool, 是否使用输入线层中的from_node字段和to_node字段
    :param fill_dir: int, 填充所有的dir为1 or 0
    :param update_link_field_list: List[str], List[str], 需要更新的字段列表(只能从6个必需字段中选取, geometry不可选)
    :param plain_prj: str, 要使用的平面投影坐标系EPSG:32650
    :param execute_modify: 是否执行极小间隔节点优化
    :param modify_minimum_buffer: 极小间隔节点优化的buffer, m
    :param ignore_merge_rule: 是否忽略极小间隔点的合并规则
    :param auxiliary_judge_field: str, 用于判断是否可以合并的线层字段, 只有当ignore_merge_rule为False才起效
    :param out_fldr: 输出文件的存储目录
    :param save_streets_before_modify_minimum: 是否保存优化前的结果
    :param save_streets_after_modify_minimum: 是否保留最终结果
    :param net_file_type: shp or geojson
    :return:
    """
    if save_streets_before_modify_minimum or save_streets_after_modify_minimum:
        assert out_fldr is not None

    if not ignore_merge_rule:
        assert auxiliary_judge_field in link_gdf.columns, \
            'The auxiliary_judge_field must be in the field of the link layer table'

    origin_crs = link_gdf.crs

    # 1.点坐标空间去重(坐标字符严格一致去重)
    node_gdf = generate_node(link_gdf=link_gdf, using_from_to=using_from_to, origin_crs=origin_crs)
    update_link_field_list = [] if update_link_field_list is None else update_link_field_list

    # 2.更新线层数据
    if using_from_to:
        if net_field.FROM_NODE_FIELD in update_link_field_list:
            update_link_field_list.remove(net_field.FROM_NODE_FIELD)
        if net_field.TO_NODE_FIELD in update_link_field_list:
            update_link_field_list.remove(net_field.TO_NODE_FIELD)

    # 3.更新线层数据
    link_gdf = update_link(link_gdf=link_gdf, node_gdf=node_gdf, update_link_field_list=update_link_field_list,
                           origin_crs=origin_crs, plain_prj=plain_prj, fill_dir=fill_dir)

    # 去除没有link连接的节点
    drop_no_use_nodes(link_gdf, node_gdf)

    if save_streets_before_modify_minimum:
        save_file(data_item=link_gdf, file_type=net_file_type, out_fldr=out_fldr, file_name='LinkBeforeModify')
        save_file(data_item=node_gdf, file_type=net_file_type, out_fldr=out_fldr, file_name='NodeBeforeModify')

    # 极小间隔点优化
    node_group_status_gdf = gpd.GeoDataFrame()
    if execute_modify:
        link_gdf, node_gdf, node_group_status_gdf = modify_minimum(plain_prj=plain_prj, node_gdf=node_gdf,
                                                                   link_gdf=link_gdf,
                                                                   buffer=modify_minimum_buffer,
                                                                   ignore_merge_rule=ignore_merge_rule)

    if save_streets_after_modify_minimum:
        save_file(data_item=link_gdf, file_type=net_file_type, out_fldr=out_fldr, file_name='LinkAfterModify')
        save_file(data_item=node_gdf, file_type=net_file_type, out_fldr=out_fldr, file_name='NodeAfterModify')
        save_file(data_item=node_group_status_gdf, file_name='MergeNodeLabel', out_fldr=out_fldr,
                  file_type=net_file_type)

    return link_gdf, node_gdf, node_group_status_gdf


@function_time_cost
def generate_node(link_gdf: gpd.GeoDataFrame = None, using_from_to: bool = False,
                  origin_crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """
    生产点层
    :param link_gdf:
    :param using_from_to:
    :param origin_crs:
    :return:
    """
    # 如果已经标注了from_node和to_node
    if using_from_to:
        assert net_field.FROM_NODE_FIELD in link_gdf.columns, f'输入路网线层缺少{net_field.FROM_NODE_FIELD}字段!'
        assert net_field.TO_NODE_FIELD in link_gdf.columns, f'输入路网线层缺少{net_field.TO_NODE_FIELD}字段!'

        # 取出操作列
        used_link_geo = link_gdf[[net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD, net_field.GEOMETRY_FIELD]].copy()
        used_link_geo['from_point'] = used_link_geo[net_field.GEOMETRY_FIELD].apply(lambda x: Point(list(x.coords)[0]))
        used_link_geo['to_point'] = used_link_geo[net_field.GEOMETRY_FIELD].apply(lambda x: Point(list(x.coords)[-1]))

        from_node_gdf = used_link_geo[[net_field.FROM_NODE_FIELD, 'from_point']].rename(
            columns={net_field.FROM_NODE_FIELD: net_field.NODE_ID_FIELD, 'from_point': net_field.GEOMETRY_FIELD})

        to_node_gdf = used_link_geo[[net_field.TO_NODE_FIELD, 'to_point']].rename(
            columns={net_field.TO_NODE_FIELD: net_field.NODE_ID_FIELD, 'to_point': net_field.GEOMETRY_FIELD})
        del used_link_geo
        node_gdf = pd.concat([from_node_gdf, to_node_gdf]).reset_index(drop=True)
        del from_node_gdf, to_node_gdf
        node_gdf.drop_duplicates(subset=[net_field.NODE_ID_FIELD], keep='first', inplace=True)
        node_gdf.reset_index(inplace=True, drop=True)
        node_gdf = gpd.GeoDataFrame(node_gdf, geometry=net_field.GEOMETRY_FIELD, crs=origin_crs)
    else:
        # 取出操作列
        used_link_geo = link_gdf[[net_field.GEOMETRY_FIELD]].copy()

        # 取出两端坐标
        used_link_geo['f_coords'] = used_link_geo[net_field.GEOMETRY_FIELD].apply(lambda geo: list(geo.coords)[0])
        used_link_geo['t_coords'] = used_link_geo[net_field.GEOMETRY_FIELD].apply(lambda geo: list(geo.coords)[-1])
        non_dup_coords_list = used_link_geo['f_coords'].to_list() + used_link_geo['t_coords'].to_list()
        del used_link_geo
        node_coords_list = list(set(non_dup_coords_list))
        node_gdf = gpd.GeoDataFrame({net_field.NODE_ID_FIELD: [x for x in range(1, len(node_coords_list) + 1)]},
                                    geometry=[Point(cor) for cor in node_coords_list], crs=origin_crs)
    return node_gdf


@function_time_cost
def update_link(link_gdf=None, node_gdf=None, update_link_field_list=None, origin_crs='EPSG:4326',
                plain_prj='EPSG:32650', fill_dir: int = 0) -> gpd.GeoDataFrame:
    """
    根据link的地理信息和节点的地理信息生成from_node和to_node字段, 在传入的gdf上直接修改, epsg:4326, 同时更新length字段
    :param link_gdf: gpd.GeoDataFrame, 线层数据
    :param node_gdf: gpd.GeoDataFrame, 点层数据
    :param update_link_field_list: List[str], 需要更新的字段列表(只能从6个必需字段中选取, geometry不可选)
    :param origin_crs: str
    :param plain_prj: str
    :param  fill_dir: int
    :return:
    """
    avoid_duplicate_cols(df=link_gdf, update_col_name_list=update_link_field_list)
    link_gdf.reset_index(inplace=True, drop=True)
    col_list = list(link_gdf.columns)

    # from_node和to_node一定要更新
    link_gdf['___idx'] = [i for i in range(0, len(link_gdf))]
    # link_gdf['__TopologyFromCoord__'] = link_gdf.apply(lambda x: Point(x[net_field.GEOMETRY_FIELD].coords[0]), axis=1)
    # link_gdf['__TopologyToCoord___'] = link_gdf.apply(lambda x: Point(x[net_field.GEOMETRY_FIELD].coords[-1]), axis=1)
    link_gdf['__TopologyFromCoord__'] = link_gdf[net_field.GEOMETRY_FIELD].apply(lambda l: Point(l.coords[0]))
    link_gdf['__TopologyToCoord___'] = link_gdf[net_field.GEOMETRY_FIELD].apply(lambda l: Point(l.coords[-1]))

    from_to_point_gdf = gpd.GeoDataFrame(link_gdf[['___idx', '__TopologyFromCoord__', '__TopologyToCoord___']].copy(),
                                         geometry='__TopologyFromCoord__', crs=origin_crs)

    join_data = gpd.sjoin(from_to_point_gdf, node_gdf, how='left', predicate='intersects')
    join_data.drop_duplicates(subset=['___idx'], keep='first', inplace=True)
    join_data.reset_index(inplace=True, drop=True)
    link_gdf[net_field.FROM_NODE_FIELD] = join_data[net_field.NODE_ID_FIELD].values

    from_to_point_gdf.set_geometry('__TopologyToCoord___', inplace=True)
    from_to_point_gdf.crs = origin_crs
    join_data = gpd.sjoin(from_to_point_gdf, node_gdf, how='left', predicate='intersects')
    del from_to_point_gdf
    join_data.drop_duplicates(subset=['___idx'], keep='first', inplace=True)
    join_data.reset_index(inplace=True, drop=True)
    link_gdf[net_field.TO_NODE_FIELD] = join_data[net_field.NODE_ID_FIELD].values

    link_gdf.dropna(subset=[net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD], axis=0, how='any', inplace=True)
    link_gdf[net_field.FROM_NODE_FIELD] = link_gdf[net_field.FROM_NODE_FIELD].astype(int)
    link_gdf[net_field.TO_NODE_FIELD] = link_gdf[net_field.TO_NODE_FIELD].astype(int)
    del join_data

    link_gdf.drop(['__TopologyFromCoord__', '__TopologyToCoord___', '___idx'], inplace=True, axis=1)

    if net_field.LENGTH_FIELD in update_link_field_list:
        # 更新length
        link_gdf = link_gdf.to_crs(plain_prj)
        link_gdf[net_field.LENGTH_FIELD] = np.around(link_gdf[net_field.GEOMETRY_FIELD].length, 2)
        link_gdf = link_gdf.to_crs(origin_crs)
    else:
        assert net_field.LENGTH_FIELD in col_list, \
            f'The {net_field.LENGTH_FIELD} field is missing in link layer, but no update is specified'

    if net_field.LINK_ID_FIELD in update_link_field_list:
        # 更新link_id
        link_gdf[net_field.LINK_ID_FIELD] = [x for x in range(1, len(link_gdf) + 1)]
    else:
        assert net_field.LINK_ID_FIELD in col_list, \
            f'The {net_field.LINK_ID_FIELD} field is missing in link layer, but no update is specified'

    if net_field.DIRECTION_FIELD in update_link_field_list:
        # 更新dir
        link_gdf[net_field.DIRECTION_FIELD] = fill_dir
    else:
        assert net_field.DIRECTION_FIELD in col_list, \
            f'The {net_field.DIRECTION_FIELD} field is missing in link layer, but no update is specified'

    non_required_col_list = list(set(col_list) - set(link_required_field_list + [net_field.GEOMETRY_FIELD]))

    link_gdf.drop(index=link_gdf[(link_gdf[net_field.FROM_NODE_FIELD] == link_gdf[net_field.TO_NODE_FIELD]) &
                                 (link_gdf[net_field.LENGTH_FIELD] <= 1e-7)].index,
                  inplace=True, axis=0)
    link_gdf.reset_index(inplace=True, drop=True)
    return link_gdf[link_required_field_list + non_required_col_list + [net_field.GEOMETRY_FIELD]]


@function_time_cost
def modify_minimum(plain_prj: str = 'EPSG:32650', node_gdf: gpd.GeoDataFrame = None,
                   link_gdf: gpd.GeoDataFrame = None, buffer: float = 1.0, auxiliary_judge_field: str = 'road_name',
                   ignore_merge_rule: bool = False) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    极小间隔节点合并, same crs as input
    :param plain_prj:
    :param node_gdf:
    :param link_gdf:
    :param buffer:
    :param auxiliary_judge_field:
    :param ignore_merge_rule:
    :return:
    """
    origin_crs = link_gdf.crs

    u_g = nx.Graph()
    u_g.add_edges_from([(f, t) for f, t in zip(link_gdf[net_field.FROM_NODE_FIELD], link_gdf[net_field.TO_NODE_FIELD])])
    # node_degrees_dict = {node: u_g.degree[node] for node in u_g.nodes}
    node_degrees_dict = dict(u_g.degree)

    # 对节点做buffer
    if origin_crs == plain_prj:
        pass
    else:
        node_gdf = node_gdf.to_crs(plain_prj)

    buffer_node_gdf = node_gdf.copy()

    buffer_node_gdf[net_field.GEOMETRY_FIELD] = buffer_node_gdf[net_field.GEOMETRY_FIELD].buffer(buffer)
    join_df = gpd.sjoin(node_gdf, buffer_node_gdf)

    node_gdf = node_gdf.to_crs(origin_crs)

    join_df.reset_index(inplace=True, drop=True)

    join_df.drop(
        index=join_df[join_df[net_field.NODE_ID_FIELD + '_left'] == join_df[net_field.NODE_ID_FIELD + '_right']].index,
        inplace=True, axis=0)

    node_group_status_list = []
    if join_df.empty:
        node_map_dict = dict()
        node_gdf = node_gdf.to_crs(origin_crs)
        # 去除没有link连接的节点
        drop_no_use_nodes(link_gdf, node_gdf)
        return link_gdf, node_gdf, gpd.GeoDataFrame()
    else:
        # 建立图
        g = nx.Graph()
        g.add_edges_from([(f, t) for f, t in zip(join_df[net_field.NODE_ID_FIELD + '_left'],
                                                 join_df[net_field.NODE_ID_FIELD + '_right'])])

        # delete_node: remain_node
        node_map_dict = {}
        all_group_node_list = []
        for node_group in nx.connected_components(g):
            # 必然有 >= 2 个元素
            node_group_list = list(node_group)
            all_group_node_list.extend(node_group_list)
            # 依据联通关系以及路名信息判断是否可以合并
            if ignore_merge_rule:
                if_merge = True
            else:
                if_merge, if_records = judge_if_same_node(link_gdf=link_gdf, attr_name_field=auxiliary_judge_field,
                                                          node_group_list=node_group_list)
            # 应该删除谁呢? 优先删除度小的
            node_degrees_map = {node: node_degrees_dict[node] for node in node_group_list}
            node_group_list = [item[0] for item in sorted(node_degrees_map.items(), key=lambda x: x[1], reverse=True)]
            if if_merge:
                node_map_dict.update({origin_node: node_group_list[0] for origin_node in node_group_list[1:]})
            node_group_status_list.append([node_group_list[0], if_merge])

        # 映射新的node_id
        node_gdf['new_' + net_field.NODE_ID_FIELD] = node_gdf[net_field.NODE_ID_FIELD].map(node_map_dict)
        node_gdf['new_' + net_field.NODE_ID_FIELD] = node_gdf['new_' + net_field.NODE_ID_FIELD].fillna(
            node_gdf[net_field.NODE_ID_FIELD])
        node_gdf['new_' + net_field.NODE_ID_FIELD] = node_gdf['new_' + net_field.NODE_ID_FIELD].astype(int)
        node_gdf.drop(columns=[net_field.NODE_ID_FIELD], axis=1, inplace=True)
        node_gdf.rename(columns={'new_' + net_field.NODE_ID_FIELD: net_field.NODE_ID_FIELD}, inplace=True)
        node_gdf.drop_duplicates(subset=[net_field.NODE_ID_FIELD], keep='first', inplace=True)
        node_gdf.set_index(net_field.NODE_ID_FIELD, inplace=True)

        # 更新线层
        alter_index = (link_gdf[net_field.FROM_NODE_FIELD].isin(all_group_node_list)) | \
                      (link_gdf[net_field.TO_NODE_FIELD].isin(all_group_node_list))
        alter_link_gdf = link_gdf[alter_index].copy()

        def get_val(map_dict=None, k=None):
            try:
                return map_dict[k]
            except KeyError:
                return k

        alter_link_gdf[net_field.FROM_NODE_FIELD] = alter_link_gdf[net_field.FROM_NODE_FIELD].apply(
            lambda x: get_val(map_dict=node_map_dict,
                              k=x))
        alter_link_gdf[net_field.TO_NODE_FIELD] = alter_link_gdf[net_field.TO_NODE_FIELD].apply(
            lambda x: get_val(map_dict=node_map_dict,
                              k=x))
        alter_link_gdf.drop_duplicates(subset=[net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD], inplace=True,
                                       keep='first')
        alter_link_gdf.drop(index=alter_link_gdf[alter_link_gdf[net_field.FROM_NODE_FIELD] ==
                                                 alter_link_gdf[net_field.TO_NODE_FIELD]].index, inplace=True, axis=0)
        link_gdf = link_gdf.drop(index=link_gdf[alter_index].index, axis=0, inplace=False)

        # 修改起终点坐标
        # 可能一条link退化为点
        if not alter_link_gdf.empty:
            # alter_link_gdf[net_field.GEOMETRY_FIELD] = alter_link_gdf.apply(
            #     lambda item: LineString([(node_gdf.at[item[net_field.FROM_NODE_FIELD], net_field.GEOMETRY_FIELD].x,
            #                               node_gdf.at[item[net_field.FROM_NODE_FIELD], net_field.GEOMETRY_FIELD].y)] +
            #                             list(item[net_field.GEOMETRY_FIELD].coords)[1:-1] +
            #                             [(node_gdf.at[item[net_field.TO_NODE_FIELD], net_field.GEOMETRY_FIELD].x,
            #                               node_gdf.at[item[net_field.TO_NODE_FIELD], net_field.GEOMETRY_FIELD].y)]), axis=1)
            alter_link_gdf[net_field.GEOMETRY_FIELD] = [
                LineString([(node_gdf.at[fn, net_field.GEOMETRY_FIELD].x,
                             node_gdf.at[fn, net_field.GEOMETRY_FIELD].y)] +
                           list(geo.coords)[1:-1] +
                           [(node_gdf.at[tn, net_field.GEOMETRY_FIELD].x,
                             node_gdf.at[tn, net_field.GEOMETRY_FIELD].y)]) for fn, tn, geo in
                zip(alter_link_gdf[net_field.FROM_NODE_FIELD],
                    alter_link_gdf[net_field.TO_NODE_FIELD],
                    alter_link_gdf[net_field.GEOMETRY_FIELD])]

            link_gdf = pd.concat([link_gdf, alter_link_gdf])

        link_gdf = gpd.GeoDataFrame(link_gdf, geometry=net_field.GEOMETRY_FIELD, crs=origin_crs)

        link_gdf.reset_index(inplace=True, drop=True)
        node_gdf.reset_index(inplace=True, drop=False)

        link_gdf.drop_duplicates(subset=[net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD], keep='first',
                                 inplace=True)
        link_gdf.reset_index(inplace=True, drop=True)

        # 去除没有link连接的节点
        drop_no_use_nodes(link_gdf, node_gdf)

        node_group_status_df = pd.DataFrame(node_group_status_list, columns=[net_field.NODE_ID_FIELD, 'status'])
        node_group_status_df = pd.merge(node_group_status_df,
                                        node_gdf[[net_field.NODE_ID_FIELD, net_field.GEOMETRY_FIELD]],
                                        on=net_field.NODE_ID_FIELD,
                                        how='left')
        node_group_status_gdf = gpd.GeoDataFrame(node_group_status_df, geometry=net_field.GEOMETRY_FIELD,
                                                 crs=origin_crs)
        return link_gdf, node_gdf, node_group_status_gdf


def judge_if_same_node(link_gdf: gpd.GeoDataFrame = None, node_group_list: list[int] = None,
                       attr_name_field: str = 'road_name') -> tuple[bool, bool]:
    # node_group_list: 潜在的可合并节点组

    # 先看潜在合并节点所关联的link
    corr_link_gdf = link_gdf[(link_gdf[net_field.FROM_NODE_FIELD].isin(node_group_list)) |
                             (link_gdf[net_field.TO_NODE_FIELD].isin(node_group_list))].copy()

    # 看可以构成几个图
    g = nx.Graph()
    g.add_edges_from([(f, t) for f, t in zip(corr_link_gdf[net_field.FROM_NODE_FIELD],
                                             corr_link_gdf[net_field.TO_NODE_FIELD])])
    conn_graph_len = len(list(nx.connected_components(g)))

    if conn_graph_len == 1:
        # 意味这是联通的
        # 再看道路名称是否一致
        if len(corr_link_gdf[attr_name_field].unique()) == 1:
            # 道路名称也一样
            return True, False
        else:
            # 联通的但是道路名称不一样, 也合并, 但是记录可能错误
            return True, True
    else:
        if len(corr_link_gdf[attr_name_field].unique()) == 1:
            # 道路名称一样
            return True, True
        else:
            # 不联通, 且道路名称也不一样
            return False, False


def avoid_duplicate_cols(df=None, update_col_name_list=None) -> None:
    """
    给出需要更新的列名称, 判断这些新名称是否在原来的df中已经存在, 若存在则先改名
    :param df:
    :param update_col_name_list:
    :return:
    """
    if update_col_name_list is None or len(update_col_name_list) < 1:
        return None
    rename_dict = {}
    for _update_col in update_col_name_list:
        if _update_col in df.columns:
            for i in range(0, 100000):
                _modify_col = _update_col + '_' + str(i)
                if _modify_col in df.columns:
                    continue
                else:
                    rename_dict[_update_col] = _modify_col
                    break
        else:
            pass
    df.rename(columns=rename_dict, inplace=True)


def get_dup_node(node_gdf: gpd.GeoDataFrame = None, buffer: float = 0.5) -> dict[int, int]:
    """
    input crs: 平面
    :param node_gdf:
    :param buffer:
    :return:
    """
    buffer_node_gdf = node_gdf.copy()

    buffer_node_gdf[net_field.GEOMETRY_FIELD] = \
        buffer_node_gdf[net_field.GEOMETRY_FIELD].apply(lambda x: x.buffer(buffer))
    join_df = gpd.sjoin(node_gdf, buffer_node_gdf)

    join_df.reset_index(inplace=True, drop=True)

    join_df.drop(
        index=join_df[join_df[net_field.NODE_ID_FIELD + '_left'] == join_df[net_field.NODE_ID_FIELD + '_right']].index,
        inplace=True, axis=0)

    node_group_status_list = []
    if join_df.empty:
        return dict()
    else:
        # 建立图
        g = nx.Graph()
        g.add_edges_from([(f, t) for f, t in zip(join_df[net_field.NODE_ID_FIELD + '_left'],
                                                 join_df[net_field.NODE_ID_FIELD + '_right'])])

        # delete_node: remain_node
        node_map_dict = {}
        for node_group in nx.connected_components(g):
            # 必然有 >= 2 个元素
            node_group_list = list(node_group)
            node_map_dict.update({origin_node: node_group_list[0] for origin_node in node_group_list[1:]})


def drop_no_use_nodes(link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None):
    # 去除没有link连接的节点
    used_node = set(link_gdf[net_field.FROM_NODE_FIELD]) | set(link_gdf[net_field.TO_NODE_FIELD])
    node_gdf.reset_index(inplace=True, drop=True)
    node_gdf.drop(index=node_gdf[~node_gdf[net_field.NODE_ID_FIELD].isin(used_node)].index, inplace=True, axis=1)
    node_gdf.reset_index(inplace=True, drop=True)
