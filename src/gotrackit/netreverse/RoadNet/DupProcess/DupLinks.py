# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData

"""消除重叠LINK"""

import pandas as pd
import networkx as nx
import geopandas as gpd
from itertools import chain
from ...PublicTools.GeoProcess import prj_xfer
from ...PublicTools.MapProcess import get_val
from shapely.geometry import Point, LineString
from ..SaveStreets.streets import modify_minimum
from ...PublicTools.GeoProcess import calc_link_angle
from ...PublicTools.GeoProcess import cut_line_in_nearest_point


# 线层数据、点层数据必需字段
length_field = 'length'  # 线层的长度, km
direction_field = 'dir'  # 线层的方向, 0, 1, -1
link_id_field = 'link_id'  # 线层的id
from_node_id_field = 'from_node'  # 线层的拓扑起始结点
to_node_id_field = 'to_node'  # 线层的拓扑终到结点
node_id_field = 'node_id'  # 点层的id
geometry_field = 'geometry'  # 几何属性字段
required_field_list = [link_id_field, length_field, direction_field,
                       from_node_id_field, to_node_id_field, geometry_field]


def process_dup_link(link_gdf: gpd.GeoDataFrame = None,
                     node_gdf: gpd.GeoDataFrame = None, buffer: float = 0.45,
                     dup_link_buffer_ratio: float = 60.0, modify_minimum_buffer: float = 0.6) -> \
        (gpd.GeoDataFrame, gpd.GeoDataFrame, dict):
    """
    去除重叠link, 输入的必须是平面投影坐标系
    :param link_gdf: 必须是平面投影坐标
    :param node_gdf:
    :param buffer:
    :param dup_link_buffer_ratio:
    :param modify_minimum_buffer
    :return: PlainPrj
    """
    plain_crs = link_gdf.crs

    dup_info_dict = dict()
    link_gdf.drop(index=link_gdf[link_gdf[from_node_id_field] == link_gdf[to_node_id_field]].index,
                  inplace=True, axis=0)
    link_gdf.reset_index(inplace=True, drop=True)
    used_node = list(set(link_gdf[from_node_id_field]) | set(link_gdf[to_node_id_field]))
    node_gdf = node_gdf[node_gdf[node_id_field].isin(used_node)].copy()
    node_gdf.reset_index(inplace=True, drop=True)

    # 建无向图, 获取节点的度
    u_g = nx.Graph()
    u_g.add_edges_from([(f, t) for f, t in zip(link_gdf[from_node_id_field],
                                               link_gdf[to_node_id_field])])
    node_degrees_dict = {node: u_g.degree[node] for node in node_gdf[node_id_field]}

    link_gdf['link_buffer'] = link_gdf[geometry_field].buffer(buffer)
    link_gdf.set_geometry('link_buffer', inplace=True, crs=plain_crs)

    right_link_gdf = link_gdf.copy()
    right_link_gdf['right_link_buffer'] = right_link_gdf['link_buffer']

    # 自相交
    join_df = gpd.sjoin(link_gdf[[link_id_field, 'link_buffer']],
                        right_link_gdf[[link_id_field, 'link_buffer', 'right_link_buffer']])
    join_df.reset_index(inplace=True, drop=True)
    join_df.drop(index=join_df[join_df[rf'{link_id_field}_left'] == join_df[rf'{link_id_field}_right']].index,
                 inplace=True, axis=0)
    join_df.reset_index(inplace=True, drop=True)

    # 计算相交率
    join_df['inter_ratio'] = \
        join_df.apply(lambda x: x['link_buffer'].intersection(x['right_link_buffer']).area / x['link_buffer'].area, axis=1)

    join_df = join_df[join_df['inter_ratio'] >= dup_link_buffer_ratio / 100].copy()
    join_df.reset_index(inplace=True, drop=True)
    if join_df.empty:
        link_gdf.drop(columns=['link_buffer'], inplace=True, axis=1)
        link_gdf.set_geometry(geometry_field, crs=plain_crs, inplace=True)
        final_link_gdf = link_gdf
        final_node_gdf = node_gdf
    else:
        # 建立图
        g = nx.Graph()
        g.add_edges_from([(f, t) for f, t in zip(join_df[rf'{link_id_field}_left'], join_df[rf'{link_id_field}_right'])])
        node_gdf.set_index(node_id_field, inplace=True)

        new_link_gdf = gpd.GeoDataFrame()
        node_map_dict = dict()
        to_be_del_link_list = []
        renew_node_gdf = gpd.GeoDataFrame([])
        for link_group in nx.connected_components(g):
            # print(link_group)
            # 计算合并后的link
            dup_link_gdf = link_gdf[link_gdf[link_id_field].isin(list(link_group))].copy()
            dup_link_gdf.reset_index(inplace=True, drop=True)
            _new_link_gdf, _node_map_dict, _renew_node_gdf = classify_dup(dup_link_gdf=dup_link_gdf,
                                                                          node_gdf=node_gdf,
                                                                          plain_crs=plain_crs,
                                                                          node_degrees_dict=node_degrees_dict, u_g=u_g)
            renew_node_gdf = pd.concat([renew_node_gdf, _renew_node_gdf])
            new_link_gdf = pd.concat([new_link_gdf, _new_link_gdf])
            node_map_dict.update(_node_map_dict)
            to_be_del_link_list += list(link_group)

            info_key = ','.join(list(map(str, link_group)))
            p = Point(list(dup_link_gdf.at[0, geometry_field].coords)[0])  # 是投影坐标系
            p = prj_xfer(from_crs=link_gdf.crs, to_crs='EPSG:4326', origin_p=p)
            dup_info_dict[rf'DupLink-{info_key}'] = (p.x, p.y)

        if node_map_dict:
            # 先映射from_node, to_node
            link_gdf[from_node_id_field] = link_gdf[from_node_id_field].apply(lambda origin_node:
                                                                              get_val(map_dict=node_map_dict,
                                                                                      k=origin_node))
            link_gdf[to_node_id_field] = link_gdf[to_node_id_field].apply(lambda origin_node:
                                                                          get_val(map_dict=node_map_dict,
                                                                                  k=origin_node))
        else:
            pass

        # 新的线层(由重叠link生成)
        new_link_gdf.reset_index(inplace=True, drop=True)
        new_link_gdf = gpd.GeoDataFrame(new_link_gdf, geometry=geometry_field, crs=link_gdf.crs)

        # 从原来的link_gdf中删除重叠的link
        link_gdf.set_geometry(geometry_field, inplace=True, crs=plain_crs)
        link_gdf.drop(columns=['link_buffer'], axis=1, inplace=True)
        link_gdf.drop(index=link_gdf[link_gdf[link_id_field].isin(to_be_del_link_list)].index, axis=0, inplace=True)
        link_gdf.reset_index(inplace=True, drop=True)

        # 将新的link添加到link_gdf
        max_link_id = link_gdf[link_id_field].max()
        new_link_gdf[link_id_field] = [i + max_link_id for i in range(1, len(new_link_gdf) + 1)]
        final_link_gdf = pd.concat([link_gdf, new_link_gdf])
        final_link_gdf.reset_index(inplace=True, drop=True)

        node_gdf.reset_index(inplace=True, drop=False)  # 恢复node_id
        # 有更新的点层, 将新的node替换原来的node
        if renew_node_gdf.empty:
            renew_node_list = []
            final_node_gdf = node_gdf
        else:
            renew_node_gdf.reset_index(inplace=True, drop=True)
            renew_node_gdf.drop_duplicates(subset=[node_id_field], inplace=True, keep='first')
            renew_node_list = renew_node_gdf[node_id_field].to_list()
            node_gdf.drop(index=node_gdf[node_gdf[node_id_field].isin(renew_node_list)].index, axis=0, inplace=True)
            # final_node_gdf = node_gdf._append(renew_node_gdf)
            final_node_gdf = pd.concat([node_gdf, renew_node_gdf])
            final_node_gdf.reset_index(inplace=True, drop=True)

        used_node_list = set(final_link_gdf[from_node_id_field]) | set(final_link_gdf[to_node_id_field])

        final_node_gdf = final_node_gdf[final_node_gdf[node_id_field].isin(used_node_list)].copy()
        final_node_gdf.reset_index(inplace=True, drop=True)

        # final_link_gdf中线型可能不准确(起终点在renew_node_list中的)
        alter_link_index = (final_link_gdf[from_node_id_field].isin(renew_node_list)) | \
                           (final_link_gdf[to_node_id_field].isin(renew_node_list))
        alter_link_gdf = final_link_gdf[alter_link_index].copy()

        if alter_link_gdf.empty:
            pass
        else:
            final_node_gdf.set_index(node_id_field, inplace=True)
            alter_link_gdf[geometry_field] = \
                alter_link_gdf.apply(
                    lambda item: LineString([final_node_gdf.at[item[from_node_id_field], geometry_field]] +
                                            list(item[geometry_field].coords)[1:-1] +
                                            [final_node_gdf.at[item[to_node_id_field], geometry_field]]), axis=1)
            final_link_gdf.drop(index=final_link_gdf[alter_link_index].index, axis=0, inplace=True)
            final_node_gdf.reset_index(drop=False, inplace=True)
            final_link_gdf = pd.concat([final_link_gdf, alter_link_gdf])
            final_link_gdf.reset_index(inplace=True, drop=True)

    # 再进行一次极小间隔点优化
    final_link_gdf, final_node_gdf, _ = modify_minimum(plain_prj=final_link_gdf.crs, node_gdf=final_node_gdf,
                                                       link_gdf=final_link_gdf,
                                                       buffer=modify_minimum_buffer, auxiliary_judge_field='road_name',
                                                       ignore_merge_rule=True)

    # 去除重复的link
    final_link_gdf.drop_duplicates(subset=[from_node_id_field, to_node_id_field], keep='first', inplace=True)
    final_link_gdf.reset_index(inplace=True, drop=True)
    return final_link_gdf, final_node_gdf, dup_info_dict


def classify_dup(dup_link_gdf=None, node_gdf=None, node_degrees_dict=None, u_g=None, plain_crs=None):
    dup_link_gdf.sort_values(by=length_field, inplace=True, ascending=False)
    dup_link_gdf.reset_index(inplace=True, drop=True)
    dup_link_gdf[direction_field] = dup_link_gdf[direction_field].astype(int)

    drop_dup_gdf = dup_link_gdf.drop_duplicates(subset=[from_node_id_field, to_node_id_field],
                                                keep='first', inplace=False)
    # from_node, to_node 重复
    if len(drop_dup_gdf) == 1:
        # 直接取最长的base_link, 不会生成新节点
        new_link_gdf, node_map_dict, renew_node_gdf = merge_dup_links_beta(dup_link_gdf=dup_link_gdf)
    # 不是完全重复
    else:
        # 查看除开base_link的两端节点后其他节点的度
        base_link_from_node, base_link_to_node = dup_link_gdf.at[0, from_node_id_field], dup_link_gdf.at[0, to_node_id_field]
        node_list_except_base = list((set(dup_link_gdf[from_node_id_field]) | set(dup_link_gdf[to_node_id_field])) - \
                                     {base_link_from_node, base_link_to_node})

        # 看看这些点除了连接dup_link里面的node外还连接了哪些点
        dup_node_set = set(dup_link_gdf[from_node_id_field]) | set(dup_link_gdf[to_node_id_field])
        adj_list = [set(list(u_g.adj[node_except_base])) - dup_node_set for node_except_base in node_list_except_base]
        adj_len_list = [len(item) for item in adj_list]
        if adj_list and max(adj_len_list) >= 1:
            # 以最长的base_link作为基准, 会生成新的线, 还会有原有节点的映射(node_map_dict)
            new_link_gdf, node_map_dict, renew_node_gdf = merge_dup_links_alpha(dup_link_gdf=dup_link_gdf,
                                                                                node_gdf=node_gdf,
                                                                                node_degrees_dict=node_degrees_dict,
                                                                                plain_crs=plain_crs)
        else:
            # 直接取最长的base_link, 不会生成新节点
            new_link_gdf, node_map_dict, renew_node_gdf = merge_dup_links_beta(dup_link_gdf=dup_link_gdf)

    if 'link_buffer' in new_link_gdf.columns:
        new_link_gdf.drop(columns=['link_buffer'], axis=1, inplace=True)

    return new_link_gdf, node_map_dict, renew_node_gdf


def merge_dup_links_alpha(dup_link_gdf=None, node_gdf=None, node_degrees_dict=None, plain_crs=None):
    """"""
    # print(rf'alpha方法切割BaseLink...')
    origin_crs = dup_link_gdf.crs

    # 选出最长的link
    base_link_geo = dup_link_gdf.at[0, 'geometry']
    base_link_dir = dup_link_gdf.at[0, 'dir']
    try:
        base_link_road_name = dup_link_gdf.at[0, 'road_name']
    except KeyError:
        base_link_road_name = '无名道路'
    from_node_list = []
    to_node_list = []

    # 涉及到的node
    node_id_list = list(set(dup_link_gdf['from_node']) | set(dup_link_gdf['to_node']))

    # 候选的参与生成新link的节点, 去除距离特别近的点组中, 只保留一个点
    candidate_node_gdf = node_gdf.loc[node_id_list, :].copy()
    candidate_node_gdf.reset_index(inplace=True, drop=False)
    candidate_node_gdf, node_map_dict = del_dup_node(node_gdf=candidate_node_gdf,
                                                     node_degrees_dict=node_degrees_dict)
    candidate_node_gdf.set_index('node_id', inplace=True)
    candidate_node_id_list = list(candidate_node_gdf.index)
    node_geo_list = [candidate_node_gdf.at[node, 'geometry'] for node in candidate_node_id_list]

    sort_df = calc_prj_dis_from_start(node_geo_list=node_geo_list,
                                      base_link_geo=base_link_geo,
                                      node_id_list=candidate_node_id_list)

    # 生成新的link, 我们需要最大程度的保留原有线型
    initial_geo = base_link_geo  # 用于继承的线型
    new_link_geo_list = []
    new_node_id_list = []
    new_node_geo_list = []

    for i in range(0, len(sort_df) - 1):
        from_p_type = sort_df.at[i, 'type']
        to_p_type = sort_df.at[i + 1, 'type']
        # print(from_p_type, to_p_type)
        # 两个点都是头部外侧或者尾部外侧, 则直接连接
        if (from_p_type, to_p_type) in [('head_beyond', 'head_beyond'), ('tail_beyond', 'tail_beyond')]:
            new_link_geo_list.append(LineString([node_gdf.at[sort_df.at[i, 'node_id'], 'geometry'],
                                                 node_gdf.at[sort_df.at[i + 1, 'node_id'], 'geometry']]))
            from_node_list.append(sort_df.at[i, 'node_id'])
            to_node_list.append(sort_df.at[i + 1, 'node_id'])

            new_node_id_list.append(sort_df.at[i, 'node_id'])
            new_node_id_list.append(sort_df.at[i + 1, 'node_id'])

            new_node_geo_list.append(node_gdf.at[sort_df.at[i, 'node_id'], 'geometry'])
            new_node_geo_list.append(node_gdf.at[sort_df.at[i + 1, 'node_id'], 'geometry'])

        else:
            # 投影点来切割线型
            cut_line_list = cut_line_in_nearest_point(initial_geo,
                                                      node_gdf.at[sort_df.at[i + 1, 'node_id'], geometry_field])

            if cut_line_list is None:
                # print(initial_geo)
                # print(node_gdf.at[sort_df.at[i + 1, 'node_id'], geometry_field])

                _l_gdf = gpd.GeoDataFrame([], geometry=[initial_geo], crs=origin_crs)
                _n_gdf = gpd.GeoDataFrame([], geometry=[node_gdf.at[sort_df.at[i + 1, 'node_id'], geometry_field]],
                                          crs=plain_crs)
                # _l_gdf.to_file(r'l_gdf.shp', endcoding='gbk')
                # _n_gdf.to_file(r'n_gdf.shp', endcoding='gbk')

            if len(cut_line_list) == 1:
                initial_geo = cut_line_list[0]
            else:
                initial_geo = cut_line_list[1]

            get_line_geo = cut_line_list[0]
            if (from_p_type, to_p_type) == ('head_beyond', 'tail_beyond'):
                get_line_coords = list(get_line_geo.coords)
                start_p, end_p = node_gdf.at[sort_df.at[i, 'node_id'], 'geometry'], \
                    node_gdf.at[sort_df.at[i + 1, 'node_id'], 'geometry']
                coords_list = [(start_p.x, start_p.y)] + get_line_coords + [(end_p.x, end_p.y)]
                new_link_geo_list.append(LineString(coords_list))

                new_node_geo_list.append(Point((start_p.x, start_p.y)))
                new_node_geo_list.append(Point((end_p.x, end_p.y)))

            else:
                new_link_geo_list.append(get_line_geo)

                new_node_geo_list.append(Point(list(get_line_geo.coords)[0]))
                new_node_geo_list.append(Point(list(get_line_geo.coords)[-1]))

            from_node_list.append(sort_df.at[i, 'node_id'])
            to_node_list.append(sort_df.at[i + 1, 'node_id'])

            new_node_id_list.append(sort_df.at[i, 'node_id'])
            new_node_id_list.append(sort_df.at[i + 1, 'node_id'])

    new_link_gdf = gpd.GeoDataFrame({'from_node': from_node_list,
                                     'to_node': to_node_list}, geometry=new_link_geo_list, crs=origin_crs)
    new_link_gdf['dir'] = base_link_dir
    new_link_gdf['road_name'] = base_link_road_name
    new_link_gdf['length'] = new_link_gdf['geometry'].apply(lambda x: x.length)

    renew_node_gdf = gpd.GeoDataFrame({'node_id': new_node_id_list}, geometry=new_node_geo_list,
                                      crs=origin_crs)
    renew_node_gdf.drop_duplicates(subset=[node_id_field], keep='first', inplace=True)
    renew_node_gdf.reset_index(inplace=True, drop=True)
    return new_link_gdf, node_map_dict, renew_node_gdf


def del_dup_node(node_gdf=None, node_degrees_dict=None):
    """
    """
    right_node_gdf = node_gdf.copy()
    # 右边是buffer
    right_node_gdf['geometry'] = right_node_gdf['geometry'].buffer(0.50)

    join_df = gpd.sjoin(node_gdf, right_node_gdf)
    join_df.reset_index(inplace=True, drop=True)
    join_df.drop(index=join_df[join_df['node_id_left'] == join_df['node_id_right']].index, axis=0, inplace=True)

    # 没有距离很近的点
    if join_df.empty:
        return node_gdf, []
    else:
        g = nx.Graph()
        g.add_edges_from([(f, t) for f, t in zip(join_df['node_id_left'], join_df['node_id_right'])])

        not_consider_node_list = []
        # 同一个组内不能随机删除, 优先删除度为1的
        map_dict = dict()
        for group_item in nx.connected_components(g):
            group_item_degrees_dict = {node: node_degrees_dict[node] for node in group_item}
            # 按照度排序
            group_item_degrees_dict = dict(sorted(group_item_degrees_dict.items(), key=lambda x: x[1], reverse=True))
            # print(group_item_degrees_dict)
            now_sorted_node_list = list(group_item_degrees_dict.keys())
            not_consider_node_list.append(now_sorted_node_list[1:])
            map_dict.update({del_node: now_sorted_node_list[0] for del_node in now_sorted_node_list[1:]})

        not_consider_node_list = list(chain(*not_consider_node_list))
        # print('map:')
        # print(map_dict)
        node_gdf.drop(index=node_gdf[node_gdf['node_id'].isin(not_consider_node_list)].index, inplace=True, axis=0)
        node_gdf.reset_index(inplace=True, drop=True)
        return node_gdf, map_dict


def calc_prj_dis_from_start(node_geo_list=None, base_link_geo=None, node_id_list=None):
    base_link_l = base_link_geo.length
    base_link_coords = list(base_link_geo.coords)
    base_link_start_p, base_link_end_p = Point(base_link_coords[0]), Point(base_link_coords[-1])

    # 计算node在base_link上的投影点到拓扑起点的距离
    prj_dis = [base_link_geo.project(node_geo) for node_geo in node_geo_list]

    prj_dis_new = [
        prj_dis[i] if 0 < prj_dis[i] < base_link_l
        else -node_geo_list[i].distance(base_link_start_p) if prj_dis[i] == 0 else node_geo_list[i].distance(base_link_start_p) + base_link_l
        for i in range(0, len(prj_dis))]

    sort_df = pd.DataFrame({'node_id': node_id_list, 'dis': prj_dis_new})
    sort_df['type'] = sort_df['dis'].apply(
        lambda x: 'within' if 0 < x < base_link_l else 'head_beyond' if x <= 0 else 'tail_beyond')
    sort_df.sort_values(by='dis', ascending=True, inplace=True)
    sort_df.reset_index(inplace=True, drop=True)
    return sort_df


def merge_dup_links_beta(dup_link_gdf=None):
    """"""
    # print(rf'beta方法保留BaseLink...')
    # 选出最长的link
    base_link_geo = dup_link_gdf.at[0, 'geometry']
    new_link_gdf = dup_link_gdf.loc[[0], :].copy() # 只取第一个base_link保留

    if 0 in dup_link_gdf[direction_field].to_list():
        new_link_gdf[direction_field] = 0
    else:
        # 计算base_link和重叠组别其他link的方向夹角
        base_other_angle_list = [calc_link_angle(link_geo1=base_link_geo, link_geo2=other_geo) for
                                 other_geo in dup_link_gdf.loc[1:, :]['geometry']]
        calc_link_angle(link_geo1=base_link_geo, link_geo2=dup_link_gdf.at[1, 'geometry'])
        if max(base_other_angle_list) >= 120:
            new_link_gdf[direction_field] = 0
    # 将被删除的
    return new_link_gdf, dict(), gpd.GeoDataFrame()


if __name__ == '__main__':
    adj_list = [{1, 1,12}]
    adj_len_list = [len(item) for item in adj_list]
    print(adj_len_list)
    if adj_list and max(adj_len_list) > 1:
        print('aaa')
