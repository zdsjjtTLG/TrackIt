# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData

"""路网拓扑优化, 计算可以合并的路段组"""

import networkx as nx
from .limit.attr_limit import limit_attr_alpha
from .limit.direction_limit import limit_direction
from .limit.two_degrees_group import get_two_degrees_node_seq


# 线层数据、点层数据必需字段
length_field = 'length'
direction_field = 'dir'
link_id_field = 'link_id'
from_node_id_field = 'from_node'
to_node_id_field = 'to_node'
node_id_field = 'node_id'
geometry_field = 'geometry'

required_field_list = [link_id_field, length_field, direction_field,
                       from_node_id_field, to_node_id_field, geometry_field]


# 路网拓扑优化主函数:
def get_merged_link_seq(link_gdf=None, judge_col_name=None, ignore_dir=False, allow_ring=False,
                        node_gdf=None, plain_prj=None, accu_l_threshold=3000, angle_threshold=17,
                        restrict_length=True, restrict_angle=True, min_length: float = 50.0):
    """

    获取可合并路段的信息
    :param link_gdf: gpd.GeoDataFrame, 线层文件
    :param node_gdf: gpd.GeoDataFrame, 点层文件
    :param judge_col_name: str, 用于限制合并的字段名称
    :param ignore_dir: bool, 是否考虑方向限制
    :param allow_ring: bool, 是否允许合并后出现环
    :param plain_prj: str
    :param accu_l_threshold: float
    :param angle_threshold: float
    :param restrict_length: bool
    :param restrict_angle: bool
    :param min_length
    :return:
      group              link_seq                                           dir_list         attr_list
        1     [(2, 10), (10, 99), (8, 99)]                                   [1,1,1]      [XX路,XX路,XX路]
        2     [(13, 14), (14, 15), (15, 16), (16, 17)]                       [0,0,0]      [XX路,XX路,XX路]
        3     [(1, 7), (7, 9), (3, 9), (3, 4), (4, 5), (5, 1)]               [0,1,0]      [XX路,XX路,XX路]
    """
    link_gdf = link_gdf.copy()

    # step2: 建立无向图, 不会修改link_gdf
    ud_graph, d_graph = build_graph_from_link(link_df=link_gdf[[from_node_id_field, to_node_id_field, direction_field]],
                                              from_col_name=from_node_id_field, to_col_name=to_node_id_field,
                                              ignore_dir=ignore_dir, dir_col=direction_field)

    # step3: 依据环参数获取最初始的2度结点路段组, 这里有可能返回None, 则说明没有满足优化条件的路段
    # group, link_seq
    # link_seq的元素是list, 一个元素代表一条link, [(node1, node2), (node2, node5)...], 升序!!! 不一定对应实际可行走的行车方向
    merged_df = get_two_degrees_node_seq(ud_graph=ud_graph, allow_ring=allow_ring,
                                         d_graph=d_graph)

    if merged_df is None:
        return None, None
    else:
        # 添加一个link_seq_str字段, 需要用到
        # group, link_seq, link_seq_str
        merged_df['link_seq_str'] = merged_df['link_seq'].\
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
                return None, None
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
                        return None, None
                    else:
                        return merged_df

    return merged_df


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
                used_df.apply(lambda x: [x[from_col_name], x[to_col_name], {'dir': x[dir_col]}] if x[dir_col] == 1 else [x[to_col_name], x[from_col_name], {'dir': x[dir_col]}], axis=1)
            edge_list_a = used_df['edge'].to_list()

        if not zero_df.empty:
            zero_df['edge'] = zero_df.apply(lambda x: [x[from_col_name], x[to_col_name], {'dir': x[dir_col]}], axis=1)
            edge_list_b = zero_df['edge'].to_list()

        edge_list = edge_list_a + edge_list_b
        d_g = nx.DiGraph()
        d_g.add_edges_from(edge_list)
        return ud_g, d_g








