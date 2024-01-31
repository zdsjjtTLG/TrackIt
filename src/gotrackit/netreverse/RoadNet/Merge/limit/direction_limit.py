# -- coding: utf-8 --
# @Time    : 2023/12/20 17:03
# @Author  : TangKai
# @Team    : ZheChengData


"""方向限制合并"""

import math
import pandas as pd
import networkx as nx
from itertools import groupby


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


def limit_direction(merged_df=None, origin_graph_degree_dict=None, link_df=None):
    """
    方向限制, 限制可合并的组中所有的link的direction必须全为1或者全为0, 全为1的组还需要进一步检查
    :param merged_df:
    :param origin_graph_degree_dict: 原网络无向图的结点的度的字典
    :param link_df:
    :return:
      group              link_seq
        1     [(2, 10), (10, 99), (8, 99)]
        2     [(13, 14), (14, 15), (15, 16), (16, 17)]
        3     [(1, 7), (7, 9), (3, 9), (3, 4), (4, 5), (5, 1)]

    返回:
     group              link_seq                                             dir_list
        1     [(2, 10), (10, 99), (8, 99)]                                   [1,1,1]
        2     [(13, 14), (14, 15), (15, 16), (16, 17)]                       [0,0,0]
        3     [(1, 7), (7, 9), (3, 9), (3, 4), (4, 5), (5, 1)]               [0,1,0]
    """

    # 选出方向为1的集合后面要用

    link_df = link_df.copy()

    # 路网建立辅助字典, {(f, t): dir}, 这里认为一个升序排列的ft能够唯一确定一个link, 这对路网有预处理要求
    link_df['sorted_ft'] = link_df.apply(lambda x: tuple(sorted([x[from_node_id_field], x[to_node_id_field]])), axis=1)
    dir_dict = {k: int(v) for k, v in zip(link_df['sorted_ft'], link_df[direction_field])}

    # 为merge_df添加dir_list
    merged_df['dir_list'] = merged_df['link_seq'].apply(lambda x: list(map(lambda i: dir_dict[i], x)))

    # 选出dir为1的
    pos_link_df = link_df[link_df[direction_field] == 1]

    # 使用方向限制对all_seq_list里面的线路进行初步筛选
    merged_df = initial_limit_direction(merged_df=merged_df)

    if merged_df.empty:
        return None
    else:
        # 初步筛选后, 选出组内方向均为1的进一步进行判断
        merged_df['dir'] = merged_df['dir_list'].apply(lambda dir_list: dir_list[0])
        used_merged_df = merged_df[merged_df['dir'] == 1].copy()

        # 组内dir全是为0的
        if used_merged_df.empty:
            pass
        else:
            # 先删除dir为1的
            merged_df.drop(index=merged_df[merged_df['dir'] == 1].index, inplace=True, axis=0)

            # 对dir为1的组进行检查
            for row in used_merged_df.itertuples():
                link_seq_list = getattr(row, 'link_seq')  # [(f1, t1), (f2, t2), ...]

                # 选出边, 构造有向图
                select_df = pos_link_df[pos_link_df['sorted_ft'].isin(link_seq_list)]
                from_list = select_df[from_node_id_field].to_list()
                to_list = select_df[to_node_id_field].to_list()
                edge_list = [[from_node, to_node] for from_node, to_node in zip(from_list, to_list)]

                d_graph = nx.DiGraph()
                d_graph.add_edges_from(edge_list)

                new_group = find_match_dir_link_seq(d_graph=d_graph, origin_graph_degree_dict=origin_graph_degree_dict)
                if new_group is not None:
                    for new_node_list in new_group:
                        if new_node_list is None:
                            pass
                        else:
                            new_link_seq = [tuple(sorted(new_node_list[i: i + 2])) for i in range(0, len(new_node_list) - 1)]
                            try:
                                merged_df = merged_df._append({'link_seq': new_link_seq,
                                                               'dir_list': [1] * len(new_link_seq)},
                                                              ignore_index=True)
                            except:
                                merged_df.loc[len(merged_df), :] = {'link_seq': new_link_seq,
                                                                    'dir_list': [1] * len(new_link_seq)}

        merged_df['group'] = [x for x in range(1, len(merged_df) + 1)]
        merged_df['group'] = merged_df['group'].astype(int)
        merged_df.drop(columns=['dir'], inplace=True, axis=1)

        # 重设索引
        merged_df.reset_index(inplace=True, drop=True)

        return merged_df


# 若启用方向限制, 则要求组内所有可合并的Link的dir必须都是0或者1
def initial_limit_direction(merged_df=None):
    """初筛方向, 一组内可合并的link的dir必须都是1或者0
    :param merged_df:
    :return:
    输入和返回结构:
     group              link_seq                                             dir_list
        1     [(2, 10), (10, 99), (8, 99)]                                   [1,1,1]
        2     [(13, 14), (14, 15), (15, 16), (16, 17)]                       [0,0,0]
        3     [(1, 7), (7, 9), (3, 9), (3, 4), (4, 5), (5, 1)]               [0,1,0]

    """
    new_group_id = 1
    new_group_list = []
    new_link_seq_list = []
    new_dir_list = []

    for row in merged_df.itertuples():
        origin_dir_list = getattr(row, 'dir_list')
        origin_link_seq_list = getattr(row, 'link_seq')

        continue_dir_index = find_continuous_repeat_index(origin_dir_list)

        if continue_dir_index is not None:
            for index in continue_dir_index:
                new_group_list.append(new_group_id)
                new_dir_list.append(origin_dir_list[index[0]:index[-1] + 1])
                new_link_seq_list.append(origin_link_seq_list[index[0]:index[-1] + 1])
                new_group_id += 1

    merged_df = pd.DataFrame({'group': new_group_list,
                              'link_seq': new_link_seq_list,
                              'dir_list': new_dir_list})
    # 可能返回空值
    return merged_df


# 每两个结点之间只有一条边连接的有向链图, 找出可合并的序列
def find_match_dir_link_seq(d_graph=None, origin_graph_degree_dict=None):
    """
    传入原网络无向图的结点度的字典, 传入一个环或者一条链(有向图), 判断哪些可以合并
    :param d_graph:
    :param origin_graph_degree_dict:
    :return:
    """
    # 先看是不是环
    # 构建无向图
    d_graph = d_graph.copy()
    ud_graph = d_graph.to_undirected()
    degree_dict = dict(nx.degree(ud_graph))
    head_tail_list = [node for node in list(degree_dict.keys()) if degree_dict[node] == 1]

    # 是环的话, 转化为链, 将根节点拆为两个新节点
    if not head_tail_list:

        # 找到根节点
        root_node_list = [node for node in list(d_graph.nodes) if origin_graph_degree_dict[node] > 2]

        # 原图中独立的环
        if not root_node_list:
            # 任意找一个不平衡的节点作为根节点
            un_ba_node_list = [node for node in d_graph.nodes
                               if int(math.fabs(d_graph.out_degree(node) - d_graph.in_degree(node))) == 2]
            if not un_ba_node_list:
                root_node = list(d_graph.nodes)[0]

                res_list = list(nx.dfs_preorder_nodes(ud_graph, source=root_node))
                res_list.append(root_node)
                return [res_list]
            else:
                root_node = un_ba_node_list[0]
        else:
            root_node = root_node_list[0]

        # 将环转化为非环
        # 找到和根节点相连的边
        neighbor_list = list(ud_graph.neighbors(root_node))
        add_edges_list = [list(item) for item in list(d_graph.edges()) if (root_node in item and neighbor_list[0])
                          or (root_node in item and neighbor_list[1])]

        flag = 1
        for add_edge in add_edges_list:
            if root_node in add_edge:
                index = add_edge.index(root_node)
                add_edge[index] = 'add' + str(flag) + '_' + str(root_node)
                flag += 1

        d_graph.add_edges_from(add_edges_list)
        d_graph.remove_node(root_node)

    # 统计每个结点的度平衡值: |入度 - 出度|
    # 结点的度不平衡的被删除
    ba_node_list = [node for node in d_graph.nodes
                    if int(math.fabs(d_graph.out_degree(node) - d_graph.in_degree(node))) == 2]

    # 构建无向图
    ud_graph = d_graph.to_undirected()
    degree_dict = dict(nx.degree(ud_graph))
    head_tail_list = [node for node in list(degree_dict.keys()) if degree_dict[node] == 1]
    dfs_order = list(nx.dfs_preorder_nodes(ud_graph, source=head_tail_list[0]))
    match_node_list = [match_node for match_node in dfs_order if match_node in ba_node_list]
    match_node_list.append(head_tail_list[-1])
    match_node_list.insert(0, head_tail_list[0])
    res = [dfs_order[dfs_order.index(match_node_list[i]): dfs_order.index(match_node_list[i + 1]) + 1]
           for i in range(0, len(match_node_list) - 1)]
    new_res = [item for item in res if len(item) > 2]

    if not new_res:
        return None
    else:
        # 还原根节点的名称
        def recover(item):
            res_list = []
            for i in item:
                if isinstance(i, str):
                    res_list.append(int(i.split('_')[1]))
                else:
                    res_list.append(i)
            return res_list

        new_res = list(map(recover, new_res))
        return new_res


# 逻辑子模块
def find_continuous_repeat_index(input_list):
    """
    找出一个list中连续重复元素的索引
    :param input_list: list
    :return: 返回None代表没有连续重复的元素, 否则返回一个列表

    Examples
    --------
    >> a = [1, 2, 3]
    >> print(find_continuous_repeat_index(input_list=a))
    None

    >> b = [1, 1, 1, 2, 1, 1]
    >> print(find_continuous_repeat_index(input_list=b))
    [[0, 1, 2], [4, 5]]

    >> c = [1, 1, 1, 1, 1, 1]
    >> print(find_continuous_repeat_index(input_list=c))
    [[0, 1, 2, 3, 4, 5]]
    """

    # 所有元素都一样
    if len(list(set(input_list))) == 1:
        return [[index for index in range(0, len(input_list))]]

    # 所有元素都不一样
    elif len(list(set(input_list))) == len(input_list):
        return None

    # 有部分元素一样
    else:
        dup_list = []

        index = 0
        for item_group in groupby(input_list):

            index_list = []

            for item in item_group[1]:
                index_list.append(index)
                index += 1

            if len(index_list) >= 2:
                dup_list.append(index_list)
            else:
                pass

        if not dup_list:
            return None
        else:
            return dup_list


if __name__ == '__main__':
    print(find_continuous_repeat_index(input_list=[(1,2), ()]))


