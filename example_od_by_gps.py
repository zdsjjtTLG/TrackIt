# -- coding: utf-8 --
# @Time    : 2024/3/7 13:55
# @Author  : TangKai
# @Team    : ZheChengData

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from itertools import chain

from src.gotrackit.gps.GpsTrip import GpsTrip


def get_sample_data():
    sz_gps_df = pd.read_csv(r'../../../data/output/gps/real_sz/TaxiData2.csv')
    sample_car = list(set(sz_gps_df.sample(n=20)['VehicleNum']))
    sz_test_gps_df = sz_gps_df[sz_gps_df['VehicleNum'].isin(sample_car)]
    sz_test_gps_df['geometry'] = sz_test_gps_df.apply(lambda row: Point((row['longitude'], row['latitude'])), axis=1)
    sz_test_gps_gdf = gpd.GeoDataFrame(sz_test_gps_df, geometry='geometry', crs='EPSG:4326')
    sz_test_gps_gdf.to_file(r'./data/output/gps/real_sz/sample.shp', encoding='gbk')


def simplify():
    sz_test_gps_gdf = gpd.read_file(r'./data/output/gps/real_sz/sample.shp', encoding='gbk')
    sz_test_gps_gdf.drop(columns=['geometry'], axis=1, inplace=True)
    sz_test_gps_gdf.rename(columns={'VehicleNum': 'agent_id',
                                    'longitude': 'lng', 'latitude': 'lat',  'timestamp': 'time'}, inplace=True)

    gtp = GpsTrip(gps_df=sz_test_gps_gdf, time_unit='s')
    gtp.add_main_group()
    res = gtp.clean_res()
    res.to_file(r'./data/output/split_gps/res.geojson', encoding='gbk', driver='GeoJSON')


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
    simplify()

