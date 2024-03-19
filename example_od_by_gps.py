# -- coding: utf-8 --
# @Time    : 2024/3/7 13:55
# @Author  : TangKai
# @Team    : ZheChengData

import numpy as np
import pandas as pd
import geopandas as gpd
from src.gotrackit.netreverse.PublicTools.od import extract_od_by_gps
from shapely.geometry import Point
from itertools import chain

def get_sample_data():
    sz_gps_df = pd.read_csv(r'./data/output/gps/real_sz/TaxiData2.csv')
    sample_car = list(set(sz_gps_df.sample(n=20)['VehicleNum']))
    sz_test_gps_df = sz_gps_df[sz_gps_df['VehicleNum'].isin(sample_car)]
    sz_test_gps_df['geometry'] = sz_test_gps_df.apply(lambda row: Point((row['longitude'], row['latitude'])), axis=1)
    sz_test_gps_gdf = gpd.GeoDataFrame(sz_test_gps_df, geometry='geometry', crs='EPSG:4326')
    sz_test_gps_gdf.to_file(r'./data/output/gps/real_sz/sample.shp', encoding='gbk')


def test_simplify():
    sz_test_gps_gdf = gpd.read_file(r'./data/output/gps/real_sz/sample.shp', encoding='gbk')
    sz_test_gps_gdf = sz_test_gps_gdf.to_crs('EPSG:32650')
    sz_test_gps_gdf['time'] = pd.to_datetime(sz_test_gps_gdf['timestamp'], unit='s')
    sz_test_gps_gdf.sort_values(by=['VehicleNum', 'time'], ascending=[True, True], inplace=True)
    gap_threshold = 6 * 60
    min_speed = 2
    min_l = 100
    min_time_gap = 80

    def get_v(l, t) -> float:
        try:
            return l / t
        except ZeroDivisionError:
            return 2.0

    for car_id, df in sz_test_gps_gdf.groupby('VehicleNum'):

        # 时间差和距离差
        df['next_time'] = df['time'].shift(-1)
        df['next_time'] = df['next_time'].fillna(df['time'])
        df['next_p'] = df['geometry'].shift(-1)
        df['next_p'] = df['next_p'].fillna(df['geometry'])
        df['time_gap'] = df.apply(lambda row: (row['next_time'] - row['time']).seconds, axis=1)
        df['adj_l'] = df.apply(lambda row: row['next_p'].distance(row['geometry']), axis=1)

        # 切分主行程
        df['label'] = df.apply(lambda row: 1 if row['time_gap'] > gap_threshold else 0, axis=1)
        split_by_row(label_field='label', df=df)
        df.sort_values(by='time', ascending=True, inplace=True)

        for _, _df in df.groupby('__group__'):
            _df['adj_speed'] = _df.apply(lambda row: get_v(row['adj_l'], row['time_gap']), axis=1)
            _df['speed_label'] = _df.apply(
                lambda row: 0 if row['adj_speed'] < min_speed and row['adj_l'] < min_l and row[
                    'time_gap'] < min_time_gap else 1, axis=1)
            del_consecutive_zero(df=_df, col='speed_label')




def split_by_row(label_field: str = 'label', df: pd.DataFrame or gpd.GeoDataFrame = None):
    group = [-1] + list(df[df[label_field] == 1].index)
    group = [[i] * (group[i] - group[i - 1]) for i in range(1, len(group))]
    seq = [[i for i in range(0, len(item))] for item in group]
    group = list(chain(*group))
    group = group + [max(group) + 1] * (len(df) - len(group))
    seq = list(chain(*seq))
    seq.extend([i for i in range(0, len(group) - len(seq))])
    df['__group__'] = group
    df['seq'] = seq
    # return list(zip(*df.groupby('__group__')))[-1]


def del_consecutive_zero(df: pd.DataFrame = None, col: str = None, n: int = 3) -> None:
    """标记超过连续n行为0的行"""
    m = df[col].ne(0)

    df['__del__'] = (df.groupby(m.cumsum())[col]
                     .transform('count').gt(n + 1)
                     & (~m)
                     )

    df['__aaaa__'] = df['__del__'].ne(1).cumsum()
    df['__cut__'] = df['__aaaa__'] & df['__del__']
    df.drop_duplicates(subset=['__aaaa__'], keep='last', inplace=True)

    print('aaaa')


def first_in_index():
    pass

if __name__ == '__main__':
    # get_sample_data()
    test_simplify()

    df = pd.DataFrame({'label': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]})
    print(df)
    del_consecutive_zero(df=df, n=5, col='label')
    print(df)
    # split_by_row(df=df)
    # dfs = list(
    #     zip(
    #         *df.groupby(
    #             (1 * (df["Case"] == "B"))
    #             .cumsum()
    #             .rolling(window=3, min_periods=1)
    #             .median()
    #         )
    #     )
    # )[-1]
