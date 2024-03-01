# -- coding: utf-8 --
# @Time    : 2024/2/25 9:45
# @Author  : TangKai
# @Team    : ZheChengData


from timeit import timeit
import networkx as nx
import time
import numba
import pandas as pd
from numba import jit
from src.gotrackit.WrapsFunc import function_time_cost
import swifter
import itertools

a = dict()


class Test(object):
    def __init__(self):
        self.done_prj = dict()

    def add(self, a, b):
        x = a + b
        return x

    @function_time_cost
    def fuc(self, df: pd.DataFrame = None):
        df['c'] = df.swifter.apply(lambda item: self.add(item['a'], item['b']), axis=1)
        # df['c'] = df.apply(lambda item: self.add(item['a'], item['b']), axis=1)
        print(df)


def get_l():
    pass

@function_time_cost
def trans():
    a = {j: [i for i in range(1, 60)] for j in range(0, 500)}
    k_list = sorted(list(a.keys()))

    # t = dict()
    # for i in range(0, len(k_list) - 1):
    #     _ = {(k_list[i], k_list[i + 1]): [(f, t) for f in a[k_list[i]] for t in a[k_list[i + 1]]]}
    #     t.update(_)

    # t = {(k_list[i], k_list[i + 1]): [(f, t) for f in a[k_list[i]] for t in a[k_list[i + 1]]] for i in range(0, len(k_list) - 1)}

    t = {(k_list[i], k_list[i + 1]): [list(itertools.product(a[k_list[i]], a[k_list[i + 1]]))] for i in range(0, len(k_list) - 1)}
    df = pd.DataFrame(t).T.reset_index(drop=False).rename(columns={'level_0': 'f', 'level_1': 't', 0: 'iter'})
    df = df.explode(column=['iter'], ignore_index=True)
    # df['ff'] = df.apply(lambda item: item['iter'][0], axis=1)
    df['ff'] = df.swifter.apply(lambda item: item['iter'][0], axis=1)
    # df['link_list'] = df.swifter.apply(lambda item: list(item['iter']), axis=1)
    # df['link_list'] = df.apply(lambda item: list(item['iter']), axis=1)
    print(df)

if __name__ == '__main__':
    # n = 2500
    # test_df = pd.DataFrame({'a': [i for i in range(n)], 'b': [i for i in range(n)]})
    # t = Test()
    # t.fuc(test_df)
    #
    # print(len(t.done_prj))

    # a = {1: [2,3,4,4], 2: [23,12,1211,34], 3: [12,89,13,12,4444,12]}
    # print(pd.DataFrame(a))
    # trans()
    # transit_list = [[seq_list[i], seq_list[i + 1],
    #                  gps_candidate_link[gps_candidate_link[gps_field.POINT_SEQ_FIELD] == seq_list[i]][
    #                      net_field.SINGLE_LINK_ID_FIELD].to_list(),
    #                  gps_candidate_link[gps_candidate_link[gps_field.POINT_SEQ_FIELD] == seq_list[i + 1]][
    #                      net_field.SINGLE_LINK_ID_FIELD].to_list()] for i in range(0, len(seq_list) - 1)]
    import numpy as np
    df = pd.DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
        'C': np.random.randn(8),
        'D': np.random.randn(8)
    })

    # 对'A'列进行分组，然后对'C'和'D'列进行聚合，并将结果转换为numpy数组
    grouped = df.groupby('A')
    agg_result = grouped[['C']].agg(np.array)

    print(agg_result)