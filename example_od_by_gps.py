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


def simplify_0402():
    from src.gotrackit.tools.coord_trans import LngLatTransfer
    con = LngLatTransfer()

    sz_test_gps_gdf = pd.read_csv(r'./data/input/net/test/0402BUG/gps/gps.csv')
    sz_test_gps_gdf[['lng', 'lat']] = sz_test_gps_gdf.apply(lambda row: con.loc_convert(lng=row['lng'],
                                                                                        lat=row['lat'], con_type='84-gc'), axis=1, result_type='expand')
    gtp = GpsTrip(gps_df=sz_test_gps_gdf, min_distance_threshold=20, n=5, way_points_num=5)
    gtp.add_main_group()
    od_df = gtp.generate_od()
    od_df.to_csv(r'./data/input/net/test/0402BUG/gps/gps_diy_od.csv', index=False)

    from shapely.geometry import LineString
    od_df['geometry'] = \
        od_df.apply(lambda row: LineString(
            [(float(row['o_x']), float(row['o_y']))] + [tuple(map(float, item.split(','))) for item in
                                                        row['way_points'].split(';')] + [
                (float(row['d_x']), float(row['d_y']))]), axis=1)

    od_line = gpd.GeoDataFrame(od_df, geometry='geometry', crs='EPSG:4326')
    del od_line['way_points']

    od_line.to_file(r'./data/input/net/test/0402BUG/gps/gps_od.shp',encoding='gbk')



if __name__ == '__main__':
    # get_sample_data()
    # simplify()
    simplify_0402()

