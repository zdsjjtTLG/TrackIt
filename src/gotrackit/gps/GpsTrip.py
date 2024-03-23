# -- coding: utf-8 --
# @Time    : 2024/3/7 13:55
# @Author  : TangKai
# @Team    : ZheChengData

import numpy as np
import pandas as pd
import geopandas as gpd
from itertools import chain
from ..GlobalVal import GpsField
from shapely.geometry import Point

gps_field = GpsField()

lng_field = gps_field.LNG_FIELD
lat_field = gps_field.LAT_FIELD
next_p_field = gps_field.NEXT_P
time_field = gps_field.TIME_FIELD
group_field = gps_field.GROUP_FIELD
sub_group_field = gps_field.SUB_GROUP_FIELD
next_time_field = gps_field.NEXT_TIME
agent_field = gps_field.AGENT_ID_FIELD
geometry_field = gps_field.GEOMETRY_FIELD
time_gap_field = gps_field.ADJ_TIME_GAP
dis_gap_field = gps_field.ADJ_DIS
adj_speed_field = gps_field.ADJ_DIS


class GpsTrip(object):
    def __init__(self, gps_df: pd.DataFrame = None, time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's',
                 plain_crs: str = 'EPSG:32650', group_gap_threshold: float = 360.0,
                 min_speed_threshold: float = 2.0, min_distance_threshold: float = 100.0,
                 min_time_gap: float = 80):

        self.plain_crs = plain_crs
        self.group_gap_threshold = group_gap_threshold  # s, 相邻GPS的时间超过这个阈值则被切分行程
        self.min_speed_threshold = min_speed_threshold  # m/s, 相邻GPS的速度小于这个阈值
        self.min_distance_threshold = min_distance_threshold  # m, 相邻GPS的直线距离
        self.min_time_gap = min_time_gap  # s, 相邻GPS的时间小于这个阈值

        self.gps_gdf = gps_df.copy()
        self.check()
        self.gps_gdf[gps_field.GEOMETRY_FIELD] = self.gps_gdf.apply(
            lambda item: Point(item[lng_field], item[lat_field]), axis=1)
        self.gps_gdf = gpd.GeoDataFrame(self.gps_gdf, geometry=geometry_field, crs='EPSG:4326')
        try:
            self.gps_gdf[gps_field.TIME_FIELD] = \
                pd.to_datetime(self.gps_gdf[gps_field.TIME_FIELD], format=time_format)
        except Exception as e:
            self.gps_gdf[gps_field.TIME_FIELD] = \
                pd.to_datetime(self.gps_gdf[gps_field.TIME_FIELD], unit=time_unit)

        self.gps_gdf.sort_values(by=[agent_field, time_field], ascending=[True, True], inplace=True)
        self.gps_gdf = self.gps_gdf.to_crs(self.plain_crs)

        self.__clean_gps_gdf = gpd.GeoDataFrame()
    def check(self):
        _gap = {agent_field, lng_field, lat_field, time_field} - set(self.gps_gdf.columns)
        assert _gap == set(), rf'GPS数据缺少{_gap}字段'

    def add_main_group(self):

        def get_v(l, t) -> float:
            try:
                return l / t
            except ZeroDivisionError:
                return 2.0

        for agent_id, group_gps_gdf in self.gps_gdf.groupby(agent_field):
            group_gps_gdf.sort_values(by=time_field, ascending=True, inplace=True)

            # 时间差和距离差
            group_gps_gdf[next_time_field] = group_gps_gdf[time_field].shift(-1).fillna(group_gps_gdf[time_field])
            group_gps_gdf[next_p_field] = group_gps_gdf[geometry_field].shift(-1).fillna(group_gps_gdf[geometry_field])
            group_gps_gdf[time_gap_field] = group_gps_gdf.apply(
                lambda row: (row[next_time_field] - row[time_field]).seconds, axis=1)
            group_gps_gdf[dis_gap_field] = group_gps_gdf.apply(
                lambda row: row[next_p_field].distance(row[geometry_field]), axis=1)

            # 切分主行程
            group_gps_gdf['main_label'] = group_gps_gdf.apply(
                lambda row: 1 if row[time_gap_field] > self.group_gap_threshold else 0, axis=1)
            self.add_group(label_field='main_label', df=group_gps_gdf)
            group_gps_gdf.drop(columns=['main_label'], axis=1, inplace=True)

            for _, _gps_df in group_gps_gdf.groupby(group_field):
                _gps_df[adj_speed_field] = _gps_df.apply(lambda row: get_v(row[dis_gap_field], row[time_gap_field]),
                                                         axis=1)
                _gps_df['speed_label'] = _gps_df.apply(
                    lambda row: 0 if row[adj_speed_field] < self.min_speed_threshold and row[
                        dis_gap_field] < self.min_distance_threshold and row[
                                         time_gap_field] < self.min_time_gap else 1, axis=1)
                self.del_consecutive_zero(df=_gps_df, col='speed_label')
                self.__clean_gps_gdf = pd.concat([self.__clean_gps_gdf, _gps_df])
                break
            break

    @staticmethod
    def add_group(df: pd.DataFrame = None, label_field: str = 'label'):
        """
        基于0/1列的label_field添加group
        :param df:
        :param label_field:
        :return:
        """
        if group_field in df.columns:
            try:
                df.rename(columns={group_field: '__' + group_field}, inplace=True)
            except Exception as e:
                print(repr(e))
        df[group_field] = df[label_field].cumsum()

    @staticmethod
    def del_consecutive_zero(df: pd.DataFrame = None, col: str = None, n: int = 3) -> None:
        """标记超过连续n行为0的行"""

        m = df[col].ne(0)
        df['__del__'] = (df.groupby(m.cumsum())[col]
                         .transform('count').gt(n + 1)
                         & (~m)
                         )
        df['__a__'] = df['__del__'].ne(1).cumsum()
        df['__cut__'] = df['__a__'] & df['__del__']
        df.drop_duplicates(subset=['__a__'], keep='last', inplace=True)
        df[sub_group_field] = df['__cut__'].ne(0).cumsum()
        df.drop(columns=['__del__', '__a__', '__cut__'], axis=1, inplace=True)

    def clean_res(self) -> gpd.GeoDataFrame:
        export_res = self.__clean_gps_gdf.to_crs('EPSG:4326')
        if next_p_field in export_res.columns:
            export_res.drop(columns=[next_p_field], axis=1, inplace=True)

        export_res['final'] = export_res.apply(lambda row: '-'.join([str(row[group_field]), str(row[sub_group_field])]),
                                               axis=1)
        return export_res
