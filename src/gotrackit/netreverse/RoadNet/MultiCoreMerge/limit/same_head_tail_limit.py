# -- coding: utf-8 --
# @Time    : 2023/12/20 17:03
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd


def get_head_tail_root(_link_seq_list):
    root_list = list(set(_link_seq_list[0]) & set(_link_seq_list[-1]))

    if root_list and len(_link_seq_list) > 2:
        _root = root_list[0]
        _head = -2
        _tail = -2
        _ring = True
    else:
        _root = -2
        _head = list(set(_link_seq_list[0]) - (set(_link_seq_list[0]) & set(_link_seq_list[1])))[0]
        _tail = list(set(_link_seq_list[-1]) - (set(_link_seq_list[-1]) & set(_link_seq_list[-2])))[0]
        _ring = False
    return [_ring, _root, _head, _tail]


def same_ht_limit(merge_link_df: pd.DataFrame, origin_link_sorted_ft_list: list[tuple]) -> pd.DataFrame:
    """

    :param merge_link_df:
    :param origin_link_sorted_ft_list:
    :return:
      group              link_seq                                            dir_list
        1     [(2, 10), (10, 99), (8, 99)]                                   [1,1,1]
        2     [(13, 14), (14, 15), (15, 16), (16, 17)]                       [0,0,0]
        3     [(1, 7), (7, 9), (3, 9), (3, 4), (4, 5), (5, 1)]               [0,1,0]
    """

    # 取组内合并后的首尾节点
    merge_link_df['head_tail'] = merge_link_df['head_tail_root_ring'].apply(lambda x: tuple(sorted((x[2], x[3]))))
    merge_link_df['dup'] = 0

    # 找出合并后首尾相接的
    merge_link_df.loc[merge_link_df['head_tail'].duplicated(), 'dup'] = 1

    # 找出合并后会和原来link的ft重合的
    merge_link_df.loc[merge_link_df['head_tail'].isin(origin_link_sorted_ft_list), 'dup'] = 1

    # 更新link_seq
    target_index = merge_link_df['dup'] == 1
    merge_link_df.loc[target_index, 'link_seq'] = merge_link_df.loc[target_index, :]['link_seq'].apply(lambda item: item[0][:-1])
    merge_link_df['seq_len'] = merge_link_df['link_seq'].apply(lambda link_seq: len(link_seq))

    merge_link_df.drop(index=merge_link_df[merge_link_df['seq_len'] == 1].index, inplace=True, axis=0)
    merge_link_df.reset_index(inplace=True, drop=True)

    # 更新head_tail_root_ring
    target_index = merge_link_df['dup'] == 1
    merge_link_df.loc[target_index, 'head_tail_root_ring'] = merge_link_df.loc[target_index, :]['link_seq'].apply(
        lambda x: get_head_tail_root(x))
    merge_link_df.drop(columns=['dup', 'seq_len'], axis=1, inplace=True)

    return merge_link_df
    

if __name__ == '__main__':
    # df = pd.DataFrame({'a': [1,2,3]})
    # print(df)
    # df.loc[df['a'].duplicated(), 'a'] = 12
    # print(df)

    df = pd.DataFrame({'a': [1, 2, 3, 45], 'b': [1, 1, 0, 1]})
    print(df)
    used_index = df['b'] == 1
    df.loc[used_index, 'a'] = df.loc[used_index, :]['b'].apply(lambda x: x ** 2 + 99)
    print(df)
