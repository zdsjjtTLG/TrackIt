# -- coding: utf-8 --
# @Time    : 2024/5/11 22:20
# @Author  : TangKai
# @Team    : ZheChengData
import time

import numba
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from src.gotrackit.WrapsFunc import function_time_cost
import pyproj
from shapely.ops import transform
import shapely
@function_time_cost
def t1(gdf):
    gdf[['x', 'y']] = gdf.apply(lambda x: (x['geometry'].x, x['geometry'].y), result_type='expand', axis=1)


@function_time_cost
def t2(gdf):
    x_dict = {i: geo.x for i, geo in zip(gdf['id'], gdf['geometry'])}
    y_dict = {i: geo.y for i, geo in zip(gdf['id'], gdf['geometry'])}
    gdf['x'] = gdf['id'].map(x_dict)
    gdf['y'] = gdf['id'].map(y_dict)



@numba.jit
def f_plain(x):
    return x * (x - 1)


@numba.jit
def integrate_f_numba(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_plain(a + i * dx)
    return s * dx


@numba.jit
def apply_integrate_f_numba(col_a, col_b, col_N):
    n = len(col_N)
    result = np.empty(n, dtype="float64")
    assert len(col_a) == len(col_b) == n
    for i in range(n):
        result[i] = integrate_f_numba(col_a[i], col_b[i], col_N[i])
    return result


@function_time_cost
def compute_numba(df):
    result = apply_integrate_f_numba(
        df["a"].to_numpy(), df["b"].to_numpy(), df["N"].to_numpy()
    )
    return pd.Series(result, index=df.index, name="result")



def integrate_f_plain(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += (a + i * dx) ** 2
    return s * dx

@function_time_cost
def compute_without_numba(df):

    df['res2'] = df.apply(lambda row: integrate_f_plain(row["a"], row["b"], row["N"]), axis=1)



def sample_fill_non():
    """"""
    od_df = pd.DataFrame({'o_time': ['--', '1615', '--', '1700', '2115', '1615', '--', '--', '1515', '1000'],
                          'd_time': ['--', '1715', '1715', '1745', '--', '1645', '--', '1920', '1715', '1715'],
                          'dis': [123, 123, 234, 124, 12, 56, 79, 12, 13, 63]})

    # # 仅仅o_time缺失的行索引
    # only_o_time_none_idx = (od_df['o_time'] == '--') & (od_df['d_time'] != '--')
    #
    # # 仅仅o_time缺失的行有多少行
    # only_o_time_none_len = len(od_df[only_o_time_none_idx])
    #
    # # 找出仅o_time缺失的od_df中d_time的值有哪些
    # only_o_none_d_time = set(od_df.loc[only_o_time_none_idx, :]['d_time'])
    #
    # # 然后筛选样本(o_time不为空且d_time在only_o_none_d_time内)
    # target_df = od_df[(od_df['o_time'] != '--') & (od_df['d_time'].isin(only_o_none_d_time))]
    #
    # # 从target中抽样only_o_time_none_len抽样only_o_time_none_lent条填充
    # od_df.loc[only_o_time_none_idx, 'o_time'] = target_df.sample(n=only_o_time_none_len)['o_time'].to_list()
    #
    # print(od_df)
    bin_list = [i for i in range(0, 300, 5)]
    print(bin_list)
    od_df['dis_label'] = list(pd.cut(od_df['dis'], bins=bin_list, labels=[i for i in range(len(bin_list) - 1)]))
    print(od_df)

    # od都为空的
    od_none_idx = (od_df['o_time'] == '--') & (od_df['d_time'] == '--')

    # od都不为空的
    od_right_idx = (od_df['o_time'] != '--') & (od_df['d_time'] != '--')

    all_target_od = od_df[od_right_idx]

    # print(od_df[od_none_idx])
    all_none_df = od_df[od_none_idx].copy().reset_index(drop=True)
    # print(all_none_df)
    for dis_label, none_df in all_none_df.groupby(['dis_label']):
        print(none_df)

        # 找到target
        now_target_df = all_target_od[all_target_od['dis_label'] == dis_label]

        if len(none_df) > len(now_target_df):
            pass
        else:
            now_sample_df = now_target_df.sample(n=len(none_df))

            o_time, d_time = now_sample_df['o_time'].to_list(), now_sample_df['d_time'].to_list()

            none_df['o_time'] = o_time
            none_df['d_time'] = d_time

            # od_df.loc[, 'o_time'] = o_time

@function_time_cost
def a1(adf: gpd.GeoDataFrame = None):
    adf = adf.to_crs('EPSG:32650')
    adf['px'] = adf['geometry'].apply(lambda x: x.x)
    adf['py'] = adf['geometry'].apply(lambda x: x.y)
    print(adf)

@function_time_cost
def a3(adf: gpd.GeoDataFrame = None):
    adf = adf.to_crs('EPSG:32650')
    adf[['px', 'py']] = adf.apply(lambda row: (row['geometry'], row['geometry'].y), axis=1, result_type='expand')
    print(adf)


@function_time_cost
def a2(adf: gpd.GeoDataFrame = None):
    adf = adf.to_crs('EPSG:32650')

    x_map = {i: geo.x for i, geo in zip(adf['id'], adf['geometry'])}
    y_map = {i: geo.y for i, geo in zip(adf['id'], adf['geometry'])}

    adf['px'] = adf['id'].map(x_map)
    adf['py'] = adf['id'].map(y_map)



@function_time_cost
def a4(adf: gpd.GeoDataFrame = None):
    adf = adf.to_crs('EPSG:32650')
    adf['geo'] = adf['geometry'].shift(1)

    adf.dropna(subset=['geo'], inplace=True)
    adf['l'] = adf['geo'].distance(adf['geometry'])


@function_time_cost
def a5(adf: gpd.GeoDataFrame = None):
    adf = adf.to_crs('EPSG:32650')
    adf['px'] = adf['geometry'].apply(lambda x: x.x)
    adf['py'] = adf['geometry'].apply(lambda x: x.y)

    adf['pre_px'] = adf['px'].shift(1)
    adf['pre_py'] = adf['py'].shift(1)

    adf.dropna(subset=['pre_px'], inplace=True)
    a = time.time()
    adf['l'] = np.sqrt((adf['px'] - adf['pre_px']) ** 2 + (adf['py'] - adf['pre_py']) ** 2)
    print(time.time() - a)



def prj_xfer(from_crs='EPSG:4326', to_crs='EPSG:32650', origin_p: shapely.geometry = None) -> shapely.geometry:
    before = pyproj.CRS(from_crs)
    after = pyproj.CRS(to_crs)
    project = pyproj.Transformer.from_crs(before, after, always_xy=True).transform
    utm_geo = transform(project, origin_p)
    return utm_geo


def merge_t():
    dfa = pd.DataFrame({'a': [0] * 300 + [1] * 400 + [2] * 500})
    dfa['vala'] = [np.random.randint(0, 10) for i in range(len(dfa))]
    dfa.set_index('a', inplace=True)

    dfb = pd.DataFrame({'a': [0] * 300 + [1] * 400 + [2] * 500})
    dfb['valb'] = [np.random.randint(0, 10) for i in range(len(dfb))]
    dfb.set_index('a', inplace=True)

    t1 = time.time()
    x = dfa.join(dfb, how='outer').reset_index(drop=False)
    t2 = time.time()

    dfa.reset_index(inplace=True, drop=False)
    dfb.reset_index(inplace=True, drop=False)

    t3 = time.time()
    z = pd.merge(dfa, dfb, how='outer', on='a')
    t4 = time.time()

    print(x)
    print(z)

    print(t2 - t1, t4 - t3)



def at_dict():
    link = gpd.read_file(r'./data/input/net/test/0402BUG/load/new_link.shp')

    a = time.time()
    _map = {_ :geo for _, geo in zip(link['link_id'], link['geometry'])}
    b = time.time()
    link.set_index('link_id', inplace=True)

    c = time.time()
    x = [link.at[i, 'geometry'] for i in link.index]
    d = time.time()
    y = [_map[i] for i in link.index]
    e = time.time()

    print(b - a)
    print(d - c)
    print(e - d)






if __name__ == '__main__':
    # N = 30000
    # df = pd.DataFrame({'geometry': N * [Point(111352.365, 236.336)],
    #                    'id': [i for i in range(N)]})
    # print(df)
    # t1(df)
    # t2(df)
    # sample_fill_non()
    # x = pd.DataFrame({"a": np.random.randn(100000),
    #                   "b": np.random.randn(100000),
    #                   "N": np.random.randint(100, 1000, (100000)),
    #                   "x": "x"})
    #
    #
    # x['res1'] = compute_numba(x)
    #
    # compute_without_numba(x)
    #
    # print(x)
    # N = 400000
    # df = gpd.GeoDataFrame({'id': [i for i in range(N)], 'geometry': [Point(np.random.randint(113, 115), 23.111) for i in range(N)]},
    #                       geometry='geometry', crs='EPSG:4326')
    #
    # a1(df.copy())
    # a3(df.copy())
    # a4(df.copy())
    # a5(df.copy())
    # merge_t()
    at_dict()
