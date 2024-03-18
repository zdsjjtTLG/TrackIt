# -- coding: utf-8 --
# @Time    : 2023/12/20 17:03
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
import networkx as nx


"""找出拓扑上可以合并的link"""


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


# # 逻辑子模块: 找出拓扑上可以合并的edge_seq序列
# def get_topology_merged_node_seq(link_df=None, ignore_dir=False, allow_ring=False):
#     """
#     找出拓扑上可以合并的2度节点序列, 边的节点按照节点编号升序排列
#     :param link_df: pd.DataFrame, 包含from_node, to_node, direction三个关键字段
#     :param ignore_dir: bool, 是否考虑方向限制
#     :param allow_ring: bool, 是否允许合并后出现环
#     :return: pd.DataFrame
#
#       group              link_seq                    link_str               dir_list
#         1     [[2, 10], [10, 99], [8, 99]]      [2_10, 10_99, 8_99]         [1, 1, 1]
#         2     [[13, 14], [14, 16], [16, 17]]    [13_14, 14_16, 16_17]       [0, 0, 0]
#         3     [[1, 7], [7, 5], [5, 1]]          [[1_7], [7_5], [5_1]]       [1, 1, 1]
#     """
#
#     link_df = link_df.copy()
#
#     # 建立无向图
#     ud_graph = build_graph_from_link(link_df=link_df,
#                                      from_col_name=from_node_id_field, to_col_name=to_node_id_field,
#                                      ignore_dir=True)
#
#     # 找到可以合并的2度节点组
#     merged_df = get_two_degrees_node_seq(ud_graph=ud_graph, allow_ring=allow_ring)
#
#     if merged_df is None:
#         return None
#     else:
#
#         # 计算link_seq_str用作键
#         merged_df['link_seq_str'] = merged_df['link_seq'].apply(
#             lambda x: ['_'.join(map(str, x[i])) for i in range(0, len(x))])
#
#         # 如果启用了方向限制
#         if not ignore_dir:
#
#             # 将每条link的dir存为字典
#             link_df['_from_to_'] = link_df[[from_node_id_field, to_node_id_field]]. \
#                 apply(lambda x: '_'.join(map(str, sorted(x))), axis=1)
#
#             dir_dict = {k: v for k, v in zip(link_df['_from_to_'].to_list(), link_df[direction_field].to_list())}
#
#             merged_df['dir_list'] = merged_df['link_seq_str'].apply(lambda x: list(map(lambda i: dir_dict[i], x)))
#
#             merged_df = limit_direction(merged_df=merged_df,
#                                         origin_graph_degree_dict=dict(nx.degree(ud_graph)),
#                                         link_df=link_df)
#
#             link_df.drop(columns='_from_to_', axis=1, inplace=True)
#
#             if merged_df is None:
#                 return None
#             else:
#                 return merged_df
#         else:
#             merged_df['dir_list'] = merged_df['link_seq'].apply(lambda x: [0] * len(x))
#             return merged_df


# 主模块, 找出2度路段组
def get_two_degrees_node_seq(ud_graph=None, allow_ring=False, d_graph=None):
    """
    找出2度节点组
    :param ud_graph: nx.net, 无向图
    :param d_graph: nx.net, 有向图
    :param allow_ring: bool, 是否允许出现环
    :return:

      group              link_seq(sorted...)
        1     [(2, 10), (10, 99), (8, 99)]
        2     [(13, 14), (14, 15), (15, 16), (16, 17)]
        3     [(1, 7), (7, 9), (3, 9), (3, 4), (4, 5), (5, 1)]
    """
    # 找出度为2的节点id
    degree_dict = dict(nx.degree(ud_graph))  # 无向图的度
    d_degree_dict = dict(d_graph.degree)  # 有向图的度
    # 用于存储所有的链组
    all_seq_list = []

    # 用于存储所有的环组
    all_cycle_list = []

    # 找出2度节点
    # 无向图中度为2的节点,(且在有向图中的入度和出度之和为2或者4)
    two_degree_node_list = [node for node in list(degree_dict.keys()) if
                            (degree_dict[node] == 2) and d_degree_dict[node] in [2, 4]]

    # 使用度为2的节点建立子图
    two_degrees_sub_graph = nx.subgraph(ud_graph, two_degree_node_list)

    # 度为2的节点组成子图中也包含很多不连通的子图, all_seq_list: [[12, 23, 34], [1, 234, 2], ...]
    for sub_graph_node in nx.connected_components(two_degrees_sub_graph):
        seq_list = []

        # 如果大于等于两个节点
        if len(sub_graph_node) >= 2:
            sub_graph = nx.subgraph(two_degrees_sub_graph, list(sub_graph_node))

            # 先找度节点
            degree_dict = dict(nx.degree(sub_graph))

            # 找到起始节点
            start_end_node = [x for x in list(degree_dict.keys()) if degree_dict[x] == 1]

            # 非环是存在一个度的节点的
            if start_end_node:
                seq_list = list(nx.dfs_postorder_nodes(sub_graph, source=start_end_node[0]))
            # 环是不存在一个度的节点的
            else:
                # 如果允许合并后出现环
                if allow_ring:
                    cycle_list = list(nx.dfs_postorder_nodes(sub_graph, source=list(sub_graph_node)[0]))
                    cycle_list.append(cycle_list[0])
                    all_cycle_list.append(cycle_list)
                else:
                    cycle_list = list(nx.dfs_postorder_nodes(sub_graph, source=list(sub_graph_node)[0]))
                    all_cycle_list.append(cycle_list)

        else:
            seq_list = list(sub_graph_node)

        # 进行一次筛选
        if seq_list:
            # 处理一个2度节点的情况
            if len(seq_list) == 1:
                node_neighbor_list = list(ud_graph.neighbors(seq_list[0]))
                if node_neighbor_list[0] == node_neighbor_list[1]:
                    if allow_ring:
                        seq_list.insert(0, node_neighbor_list[0])
                        seq_list.append(seq_list[0])
                        all_cycle_list.append(seq_list)
                    else:
                        pass
                else:
                    seq_list.insert(0, node_neighbor_list[0])
                    seq_list.append(node_neighbor_list[1])
                    all_seq_list.append(seq_list)
            else:
                # 首结点的邻接结点(除去seq_list中的那个邻接点)
                a_node_neighbor_set = set(list(ud_graph.neighbors(seq_list[0])))
                a_node_neighbor = list(a_node_neighbor_set - set(seq_list))[0]

                # 尾结点的邻接结点(除去seq_list中的那个邻接点)
                b_node_neighbor_set = set(list(ud_graph.neighbors(seq_list[-1])))
                b_node_neighbor = list(b_node_neighbor_set - set(seq_list))[0]

                # 环路判断
                if a_node_neighbor == b_node_neighbor:
                    if allow_ring:
                        seq_list.insert(0, a_node_neighbor)
                        seq_list.append(seq_list[0])
                        all_cycle_list.append(seq_list)
                    else:
                        seq_list.insert(0, a_node_neighbor)
                        if len(seq_list) >= 3:
                            all_seq_list.append(seq_list)
                        else:
                            pass
                else:
                    seq_list.insert(0, a_node_neighbor)
                    seq_list.append(b_node_neighbor)
                    all_seq_list.append(seq_list)

    merged_node_list = all_cycle_list + all_seq_list

    if not merged_node_list:
        print('无可合并2度结点组!')
        return None
    else:
        merged_df = pd.DataFrame({'link_seq': merged_node_list})
        # merged_df['fact_link_seq'] = merged_df['link_seq']. \
        #     apply(lambda x: [x[i:i + 2] for i in range(0, len(x) - 1)])
        merged_df['link_seq'] = merged_df['link_seq'].\
            apply(lambda x: [tuple(sorted(x[i:i+2])) for i in range(0, len(x) - 1)])
        merged_df.reset_index(inplace=True, drop=True)
        merged_df['group'] = [x for x in range(1, len(merged_df) + 1)]
        return merged_df


if __name__ == '__main__':
    pass
    # d_graph = nx.DiGraph()
    # ud_graph = nx.Graph()
    # edge_list = [[1,2], [2,3], [3,4], [4,5], [5,4]]
    # d_graph.add_edges_from(edge_list)
    # ud_graph.add_edges_from(edge_list)
    # degree_dict = dict(nx.degree(ud_graph))  # 无向图的度
    # d_degree_dict = dict(d_graph.degree)  # 有向图的度
    #
    # two_degree_node_list = [node for node in list(degree_dict.keys()) if
    #                         (degree_dict[node] == 2) and d_degree_dict[node] in [2, 4]]
    # print(two_degree_node_list)
    a = nx.DiGraph()

    edge_list = [(1, 9), (9, 8), (8,7), (7, 5), (9, 5), (5,6)]
    a.add_edges_from(edge_list)
    b = a.to_undirected()
    x = get_two_degrees_node_seq(ud_graph=b, allow_ring=False, d_graph=a)
    print(x)