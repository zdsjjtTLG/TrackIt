# -- coding: utf-8 --
# @Time    : 2023/12/20 17:03
# @Author  : TangKai
# @Team    : ZheChengData

"""路段属性限制合并"""

import numpy as np
import pandas as pd
import networkx as nx
from itertools import chain
from shapely.geometry import LineString
from ....PublicTools.GeoProcess import calc_link_angle
from ....PublicTools.IndexAna import find_continuous_repeat_index


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


def find_one_degree_node(ud_g):
    # 找出度为1的节点id
    degree_dict = dict(nx.degree(ud_g))
    one_degree_node_list = [node for node in list(degree_dict.keys()) if degree_dict[node] == 1]
    return one_degree_node_list


def generate_attr_by_cum_val(val_attr_list=None, prefix='a', cum_val_threshold=100.0, start_num=1, min_length=50.0):
    """
    依据值列表属性, 依据累计值规则生成抽象属性
    :param val_attr_list:
    :param prefix:
    :param cum_val_threshold:
    :param min_length:
    :param start_num:
    :return:
    """
    val_attr_list = list(map(float, val_attr_list))
    cum_val_series = pd.Series(np.cumsum(val_attr_list)) / cum_val_threshold
    if len(cum_val_series[cum_val_series >= 1]) >= 1:
        gap_index_list = []
        cum_sum = 0
        for i in range(0, len(val_attr_list)):
            cum_sum += val_attr_list[i]
            if cum_sum >= cum_val_threshold:
                gap_index_list.append(i)
                cum_sum = 0
            else:
                pass
        # 检查从i开始往后的link长度总和够不够
        if gap_index_list[-1] == len(val_attr_list) - 1:
            pass
        else:
            if sum(val_attr_list[gap_index_list[-1] + 1:]) <= min_length:
                gap_index_list[-1] = len(val_attr_list) - 1

        gap_index_list = [-1] + gap_index_list + [len(val_attr_list) - 1]

        abstract_attr_list = [[prefix + rf'{i + start_num}'] * (gap_index_list[i + 1] - gap_index_list[i])
                              for i in range(0, len(gap_index_list) - 1)]
        return list(chain(*abstract_attr_list))
    else:
        return [prefix + rf'1' for i in range(1, len(val_attr_list) + 1)]


def generate_attr_by_cum_val_alpha(val_attr_list=None, prefix='a', cum_val_threshold=100, min_length: float = 50):
    """
    依据值列表属性, 依据累计值规则生成抽象属性
    :param val_attr_list:
    :param prefix:
    :param cum_val_threshold:
    :param min_length:
    :return:
    """
    _len = len(val_attr_list) # 列表长度
    val_attr_list = list(map(float, val_attr_list))
    cum_val_series = pd.Series(np.cumsum(val_attr_list)) / cum_val_threshold
    used_cum_val_threshold = cum_val_threshold - 25
    if len(cum_val_series[cum_val_series >= 1]) >= 1:
        head_name_list = []
        cum_sum = 0
        # 先找出来那头
        restart_index = 0
        for i in range(0, len(val_attr_list)):
            cum_sum += val_attr_list[i]
            if cum_sum >= used_cum_val_threshold:
                cum_sum = 0
                restart_index = i + 1
                head_name_list = [prefix + '1'] * (i + 1)
                break
            else:
                pass

        # 判断剩下的长度是否够一个cum_val_threshold
        tail_name_list = []
        reverse_val_list = val_attr_list[restart_index:][::-1]
        remain_l = sum(reverse_val_list)
        if remain_l >= cum_val_threshold:
            # 还够, 反过来再找
            end_index = -1
            n = 0
            for i in range(0, len(reverse_val_list)):
                cum_sum += reverse_val_list[i]
                if cum_sum >= used_cum_val_threshold:
                    tail_name_list = [prefix + '2'] * (i + 1)
                    n = i
                    break
                else:
                    pass
            remain_num = _len - restart_index - (n + 1)
            if remain_num <= 0:
                return head_name_list + tail_name_list
            else:
                mid_val_list = val_attr_list[restart_index: restart_index + remain_num]
                mid_name_list = generate_attr_by_cum_val(val_attr_list=mid_val_list,
                                                         prefix=prefix, cum_val_threshold=cum_val_threshold, start_num=3,
                                                         min_length=min_length)
                return head_name_list + mid_name_list + tail_name_list
        else:
            # 剩下的link加起来不够一个cum_val_threshold
            # 但是大于min_length
            if remain_l > min_length:
                head_name_list.extend([prefix + '2'] * (_len - restart_index))
            else:
                # 但是小于min_length, 直接和前面的组合并
                head_name_list.extend([head_name_list[-1]] * (_len - restart_index))
            return head_name_list
    else:
        return [prefix + rf'1' for i in range(1, len(val_attr_list) + 1)]


def generate_attr_by_val(val_attr_list=None, prefix='a', val_threshold=10):
    val_series = pd.Series(val_attr_list)
    target_index_list = list(val_series[val_series >= val_threshold].index)

    if target_index_list:
        target_index_list = [-1] + target_index_list + [len(val_attr_list) - 1]
        res = [[prefix + rf'{i}'] * (target_index_list[i + 1] - target_index_list[i]) for
               i in range(0, len(target_index_list) - 1)]
        return list(chain(*res))
    else:
        return [prefix + rf'1' for i in range(1, len(val_attr_list) + 1)]


def limit_attr(merged_df=None, link_df=None, attr_col=None, utm_prj='EPSG:32650', node_gdf=None):
    """

    :param merged_df:
    :param link_df:
    :param attr_col:
    :param utm_prj:
    :param node_gdf:

    :return:
    """
    # 辅助字典
    node_df = node_gdf.copy()
    node_df = node_df.to_crs(utm_prj)
    node_df.set_index(node_id_field, inplace=True)

    link_df['_from_to_'] = link_df[[from_node_id_field, to_node_id_field]].apply(lambda x: [x[0], x[1]], axis=1)
    link_df['_from_to_str_'] = link_df['_from_to_'].apply(lambda x: '_'.join(map(str, sorted(x))))
    _from_to_str_list = link_df['_from_to_str_'].to_list()
    _attr_list = link_df[attr_col].to_list()
    attr_dict = {k: v for k, v in zip(_from_to_str_list, _attr_list)}
    merged_df['attr_list'] = merged_df['link_seq_str'].apply(lambda x: list(map(lambda i: attr_dict[i], x)))

    group_id = 1
    new_group_list = []
    new_link_seq_list = []
    new_attr_list = []
    new_dir_list = []

    for row in merged_df.itertuples():
        attr_list = getattr(row, 'attr_list')
        link_seq_list = getattr(row, 'link_seq') # 不一定是首尾相接
        dir_list = getattr(row, 'dir_list')

        if [10012, 59170] in link_seq_list:
            a = 1

        ud_g = nx.Graph()
        ud_g.add_edges_from(link_seq_list)
        s = list(set(find_one_degree_node(ud_g=ud_g)) & set(link_seq_list[0]))[0]
        fact_link_seq_list = list(nx.dfs_preorder_nodes(ud_g, s))
        fact_link_seq_list = [[fact_link_seq_list[i], fact_link_seq_list[i+1]] for i in range(0, len(fact_link_seq_list) - 1)]
        continue_dir_index = find_continuous_repeat_index(attr_list)

        if continue_dir_index is not None:
            for index in continue_dir_index:
                # 计算组内的方向角度(相邻路段的)
                dir_vec_list = [calc_link_angle(LineString([node_df.at[fact_link_seq_list[index[i]][0], 'geometry'],
                                                            node_df.at[fact_link_seq_list[index[i]][1], 'geometry']]),
                                                LineString([node_df.at[fact_link_seq_list[index[i + 1]][0], 'geometry'],
                                                            node_df.at[fact_link_seq_list[index[i + 1]][1], 'geometry']])
                                                ) for i in
                                range(0, len(index) - 1)]
                dir_vec_series = pd.Series(dir_vec_list)
                target_split_index_list = list(dir_vec_series[dir_vec_series > 20].index)

                if target_split_index_list:
                    if len(target_split_index_list) >= len(index) - 1:
                        # 说明组内link均不可合并
                        pass
                    else:
                        last_target_split_index = 0
                        for target_split_index in target_split_index_list:
                            if (target_split_index - last_target_split_index) < 1:
                                last_target_split_index = target_split_index + 1
                                pass
                            else:
                                new_group_list.append(group_id)
                                new_attr_list.append(
                                    attr_list[index[last_target_split_index]:index[target_split_index + 1]])
                                new_link_seq_list.append(
                                    link_seq_list[index[last_target_split_index]:index[target_split_index + 1]])
                                new_dir_list.append(
                                    dir_list[index[last_target_split_index]:index[target_split_index + 1]])

                                group_id += 1
                                last_target_split_index = target_split_index + 1

                        if last_target_split_index <= len(index) - 2:
                            e = index[-1] + 1
                            new_group_list.append(group_id)
                            new_attr_list.append(
                                attr_list[index[last_target_split_index]:e])
                            new_link_seq_list.append(link_seq_list[index[last_target_split_index]:e])
                            new_dir_list.append(dir_list[index[last_target_split_index]:e])
                            group_id += 1
                        else:
                            pass
                else:
                    new_group_list.append(group_id)
                    new_attr_list.append(attr_list[index[0]:index[-1] + 1])
                    new_link_seq_list.append(link_seq_list[index[0]:index[-1] + 1])
                    new_dir_list.append(dir_list[index[0]:index[-1] + 1])
                    group_id += 1

    merged_df = pd.DataFrame({'group': new_group_list, 'link_seq': new_link_seq_list, 'attr_list': new_attr_list,
                              'dir_list': new_dir_list})

    if merged_df.empty:
        return None
    else:
        # 更新link_seq_str
        merged_df['link_seq_str'] = merged_df['link_seq'].apply(
            lambda x: ['_'.join(map(str, x[i])) for i in range(0, len(x))])

        return merged_df


def limit_attr_alpha(merged_df=None, link_df=None, attr_col=None, plain_prj='EPSG:32650', node_gdf=None,
                     accu_l_threshold=3000, angle_threshold=10, restrict_length=False,
                     restrict_angle=False, min_length: float = 50.0):
    """
    :param merged_df:
    :param link_df:
    :param attr_col:
    :param plain_prj:
    :param node_gdf:
    :param accu_l_threshold:
    :param angle_threshold:
    :param restrict_length:
    :param restrict_angle:
    :param min_length:
    :return:
       group              link_seq                                           dir_list         attr_list
        1     [(2, 10), (10, 99), (8, 99)]                                   [1,1,1]      [XX路,XX路,XX路]
        2     [(13, 14), (14, 15), (15, 16), (16, 17)]                       [0,0,0]      [XX路,XX路,XX路]
        3     [(1, 7), (7, 9), (3, 9), (3, 4), (4, 5), (5, 1)]               [0,1,0]      [XX路,XX路,XX路]
    """
    # 辅助字典
    origin_crs = node_gdf.crs

    node_df = node_gdf.copy()
    if origin_crs == plain_prj:
        pass
    else:
        node_df = node_df.to_crs(plain_prj)
    node_df.set_index(node_id_field, inplace=True)

    link_df['sorted_ft'] = link_df.apply(lambda ft: tuple(sorted([ft[from_node_id_field], ft[to_node_id_field]])), axis=1)
    sorted_ft_list = link_df['sorted_ft'].to_list()

    # 指定属性的attr_list
    for col in [attr_col, 'length']:

        _attr_list = link_df[col].to_list()
        attr_dict = {k: v for k, v in zip(sorted_ft_list, _attr_list)}
        merged_df[col + '_list'] = merged_df['link_seq'].apply(lambda link_seq:
                                                               list(map(lambda i: attr_dict[i], link_seq)))

    group_id = 1
    new_group_list = []
    new_link_seq_list = []
    new_attr_list = []
    new_dir_list = []
    for row in merged_df.itertuples():
        attr_list = getattr(row, attr_col + '_list')  # 获取实际属性列表
        link_seq_list = getattr(row, 'link_seq')  # [(f1, t1), (f2, t2), (f3, t3), ...]不一定是首尾相接
        dir_list = getattr(row, 'dir_list')
        # print(attr_list)
        # 是否开启长度限制
        abstract_length_attr_list = ['' for _ in range(0, len(attr_list))]
        if restrict_length:
            length_attr_list = getattr(row, 'length_list')  # 获取长度属性列表
            # 利用组内的长度累加列表得到长度虚拟属性
            abstract_length_attr_list = generate_attr_by_cum_val_alpha(val_attr_list=length_attr_list,
                                                                       cum_val_threshold=accu_l_threshold,
                                                                       prefix='l', min_length=min_length)

        # 是否开启角度限制
        abstract_angle_attr_list = ['' for _ in range(0, len(attr_list))]
        used_angle_threshold = angle_threshold
        if restrict_angle:
            ud_g = nx.Graph()
            ud_g.add_edges_from(link_seq_list)
            s = list(set(find_one_degree_node(ud_g=ud_g)) & set(link_seq_list[0]))[0]
            fact_link_seq_list = list(nx.dfs_preorder_nodes(ud_g, s))
            fact_link_seq_list = [[fact_link_seq_list[i], fact_link_seq_list[i + 1]] for i in
                                  range(0, len(fact_link_seq_list) - 1)]

            # 利用组内角度转向值得到角度虚拟值
            # 先计算组内角度转向
            dir_vec_list = [calc_link_angle(LineString([node_df.at[fact_link_seq_list[i][0], 'geometry'],
                                                        node_df.at[fact_link_seq_list[i][1], 'geometry']]),
                                            LineString([node_df.at[fact_link_seq_list[i + 1][0], 'geometry'],
                                                        node_df.at[fact_link_seq_list[i + 1][1], 'geometry']])
                                            ) for i in range(0, len(fact_link_seq_list) - 1)]
            dir_vec_list = dir_vec_list + [0]
            # 计算组内第一条Link和其余link的夹角, 如果最大值超过100度, 很有可能是匝道\掉头路
            cum_dir_vec_list = [calc_link_angle(LineString([node_df.at[fact_link_seq_list[0][0], 'geometry'],
                                                            node_df.at[fact_link_seq_list[0][1], 'geometry']]),
                                                LineString([node_df.at[fact_link_seq_list[i][0], 'geometry'],
                                                            node_df.at[fact_link_seq_list[i][1], 'geometry']])
                                                ) for i in range(1, len(fact_link_seq_list))]
            # print(cum_dir_vec_list)
            if max(cum_dir_vec_list) >= 100:
                used_angle_threshold = angle_threshold + 18
                # print('匝道\掉头路 识别...')
                # print(rf'angle_threshold: {used_angle_threshold}')
            abstract_angle_attr_list = generate_attr_by_val(val_attr_list=dir_vec_list,
                                                            val_threshold=used_angle_threshold,
                                                            prefix='angle')

        # 组合实际限制属性, 角度转向虚拟属性, 长度累计限制属性
        combined_attr_list = [''.join([attr, abstract_attr_a, abstract_attr_b]) for
                              attr, abstract_attr_a, abstract_attr_b in
                              zip(attr_list, abstract_angle_attr_list,
                                  abstract_length_attr_list)]
        continue_dir_index = find_continuous_repeat_index(combined_attr_list)
        if continue_dir_index is not None:
            for index in continue_dir_index:
                new_group_list.append(group_id)
                new_attr_list.append(attr_list[index[0]:index[-1] + 1])
                new_link_seq_list.append(link_seq_list[index[0]:index[-1] + 1])
                new_dir_list.append(dir_list[index[0]:index[-1] + 1])
                group_id += 1
    merged_df = pd.DataFrame({'group': new_group_list, 'link_seq': new_link_seq_list, 'attr_list': new_attr_list,
                              'dir_list': new_dir_list})
    return merged_df


if __name__ == '__main__':

    pass
    # val_list = [np.random.randint(5, 40) for i in range(0, 5)]
    # print(val_list)
    # val_list = [71, 75, 73, 72, 16, 100, 125, 365]
    #
    # z = generate_attr_by_cum_val(val_attr_list=val_list,
    #                              cum_val_threshold=45, prefix='e')

    # x = generate_attr_by_cum_val(val_attr_list=val_list,
    #                              prefix='a', cum_val_threshold=120.0, start_num=3, min_length=50.0)
    # print(x)

    # a = []
    # for i in range(0, np.random.randint(6, 12)):
    #     a.append(np.random.randint(5, 90))
    # x = generate_attr_by_cum_val_alpha(val_attr_list=a,
    #                                    cum_val_threshold=200, prefix='e', min_length=50)
    # print(a)
    # print(x)
    #
    # assert len(a) == len(x)
