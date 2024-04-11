# -- coding: utf-8 --
# @Time    : 2024/3/7 13:55
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
import geopandas as gpd
from .GpsArray import GpsArray
from ..GlobalVal import GpsField
from shapely.geometry import LineString

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
adj_speed_field = gps_field.ADJ_SPEED


class GpsTrip(GpsArray):
    def __init__(self, gps_df: pd.DataFrame = None, time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's',
                 plain_crs: str = 'EPSG:32650', group_gap_threshold: float = 360.0, n: int = 5,
                 min_distance_threshold: float = 10.0, way_points_num: int = 5, dwell_accu_time: float = 150.0):

        GpsArray.__init__(self, gps_points_df=gps_df, time_unit=time_unit, time_format=time_format,
                          plane_crs=plain_crs, geo_crs='EPSG:4326')

        # 主行程时间阈值
        self.group_gap_threshold = group_gap_threshold  # s, 相邻GPS的时间超过这个阈值则被切分行程

        # 子行程距离阈值(停留点)
        self.min_distance_threshold = min_distance_threshold  # m, 相邻GPS的直线距离小于这个值就被切分子行程
        self.n = n  # 连续n个GPS点的距离小于min_distance_threshold则被初步认为是停留点
        self.dwell_accu_time = dwell_accu_time  # 连续n个GPS点的停留时间大于该值则会被切分子行程

        self.gps_points_gdf.sort_values(by=[agent_field, time_field], ascending=[True, True], inplace=True)
        self.__clean_gps_gdf = gpd.GeoDataFrame()

        # 构造OD的途径点数量
        assert way_points_num <= 9
        self.way_points_num = way_points_num

    def add_main_group(self):
        car_num = len(self.gps_points_gdf[agent_field].unique())
        print(rf'{car_num} vehicles, cutting group...')
        for agent_id, group_gps_gdf in self.gps_points_gdf.groupby(agent_field):
            group_gps_gdf.sort_values(by=time_field, ascending=True, inplace=True)

            # 时间差和距离差
            group_gps_gdf[next_time_field] = group_gps_gdf[time_field].shift(-1).fillna(group_gps_gdf[time_field])
            group_gps_gdf[next_p_field] = group_gps_gdf[geometry_field].shift(-1).fillna(group_gps_gdf[geometry_field])
            group_gps_gdf[time_gap_field] = group_gps_gdf.apply(
                lambda row: (row[next_time_field] - row[time_field]).seconds, axis=1)
            group_gps_gdf[dis_gap_field] = group_gps_gdf.apply(
                lambda row: row[next_p_field].distance(row[geometry_field]), axis=1)

            # 切分主行程
            group_gps_gdf['main_label'] = (group_gps_gdf[time_gap_field] > self.group_gap_threshold).astype(int)
            self.add_group(label_field='main_label', df=group_gps_gdf, agent_id=agent_id)
            group_gps_gdf.drop(columns=['main_label'], axis=1, inplace=True)

            for _, _gps_df in group_gps_gdf.groupby(group_field):
                _gps_df['sub_label'] = (_gps_df[dis_gap_field] >= self.min_distance_threshold).astype(int)
                self.del_consecutive_zero(df=_gps_df, col='sub_label', n=self.n)
                self.__clean_gps_gdf = pd.concat([self.__clean_gps_gdf, _gps_df])
        self.__clean_gps_gdf.reset_index(inplace=True, drop=True)

    @staticmethod
    def add_group(df: pd.DataFrame = None, label_field: str = 'label', agent_id: str = None):
        """
        基于0/1列的label_field添加group
        :param df:
        :param label_field:
        :param agent_id
        :return:
        """
        if group_field in df.columns:
            try:
                df.rename(columns={group_field: '__' + group_field}, inplace=True)
            except Exception as e:
                print(repr(e))
        df[group_field] = df[label_field].cumsum()
        df[group_field] = df.apply(lambda x: str(agent_id) + '_' + str(x[label_field]), axis=1)

    def del_consecutive_zero(self, df: pd.DataFrame = None, col: str = None, n: int = 3) -> None:
        """标记超过连续n行为0的行, 并且只保留最后一行"""

        m = df[col].ne(0)
        df['__del__'] = (df.groupby(m.cumsum())[col]
                         .transform('count').gt(n + 1)
                         & (~m)
                         )
        df['__a__'] = df['__del__'].ne(1).cumsum()

        # 停留点的累计停留时间
        df['accu_time'] = df.groupby('__a__')[time_gap_field].transform('sum')

        df['__cut__'] = df['__a__'].ne(0) & df['__del__'] & (df['accu_time'] > self.dwell_accu_time)
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

    def generate_od(self) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        def generate_way_point(df=None):
            df.reset_index(inplace=True, drop=True)
            _l = len(df)
            if len(df) <= self.way_points_num:
                pass
            else:
                o_x, o_y, d_x, d_y = df.at[0, 'lng'], df.at[0, 'lat'], df.at[_l - 1, 'lng'], df.at[_l - 1, 'lat']
                df.drop(index=[0, _l - 1], axis=0, inplace=True)
                df.reset_index(inplace=True, drop=True)
                gap = int(len(df) / self.way_points_num)
                _sle = df.loc[[gap * i + int(gap / 2) for i in range(self.way_points_num)], :].copy()
                del df
                _sle['loc'] = _sle.apply(lambda row: ','.join([str(row['lng']), str(row['lat'])]), axis=1)
                return o_x, o_y, d_x, d_y, ';'.join(_sle['loc'].to_list())

        res_df = self.clean_res()
        res_df.rename(columns={'final': 'trip_id'}, inplace=True)

        od_df = res_df.groupby('trip_id').apply(lambda df:
                                                generate_way_point(df)).reset_index(drop=False).rename(
            columns={0: 'item'})
        if od_df.empty:
            return pd.DataFrame(), gpd.GeoDataFrame()
        else:
            od_df.dropna(subset=['item'], inplace=True)
            od_df[['o_x', 'o_y', 'd_x', 'd_y', 'way_points']] = od_df.apply(lambda row: row['item'], axis=1,
                                                                            result_type='expand')
            od_df['od_id'] = [i for i in range(1, len(od_df) + 1)]
            del od_df['item']

            od_df['geometry'] = \
                od_df.apply(lambda row: LineString(
                    [(float(row['o_x']), float(row['o_y']))] + [tuple(map(float, item.split(','))) for item in
                                                                row['way_points'].split(';')] + [
                        (float(row['d_x']), float(row['d_y']))]), axis=1)

            od_line = gpd.GeoDataFrame(od_df, geometry='geometry', crs='EPSG:4326')
            del od_line['way_points']
            return od_df, od_line
