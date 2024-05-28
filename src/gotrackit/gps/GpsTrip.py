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
pre_p_field = gps_field.PRE_P
time_field = gps_field.TIME_FIELD
group_field = gps_field.GROUP_FIELD
sub_group_field = gps_field.SUB_GROUP_FIELD
pre_time_field = gps_field.PRE_TIME
agent_field = gps_field.AGENT_ID_FIELD
ori_agent_field = gps_field.ORIGIN_AGENT_ID_FIELD
pre_agent_field = gps_field.PRE_AGENT_ID_FIELD
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

    def cut_group(self):

        self.gps_points_gdf.sort_values(by=[agent_field, time_field], ascending=[True, True], inplace=True)
        self.gps_points_gdf.reset_index(inplace=True, drop=True)
        origin_agent_set = set(self.gps_points_gdf[agent_field])
        car_num = len(origin_agent_set)
        print(rf'{car_num} vehicles, cutting group...')
        # 记录原始agent_id
        self.gps_points_gdf[ori_agent_field] = self.gps_points_gdf[agent_field]

        # 时间差和距离差
        self.gps_points_gdf[pre_time_field] = self.gps_points_gdf[time_field].shift(1).fillna(
            self.gps_points_gdf[time_field])
        self.gps_points_gdf[pre_p_field] = self.gps_points_gdf[geometry_field].shift(1).fillna(
            self.gps_points_gdf[geometry_field])
        self.gps_points_gdf[time_gap_field] = self.gps_points_gdf[time_field] - self.gps_points_gdf[pre_time_field]
        self.gps_points_gdf[time_gap_field] = self.gps_points_gdf[time_gap_field].apply(lambda t: t.seconds)
        self.gps_points_gdf[dis_gap_field] = self.gps_points_gdf[pre_p_field].distance(
            self.gps_points_gdf[geometry_field])

        # 前序agent_id
        self.gps_points_gdf[pre_agent_field] = self.gps_points_gdf[agent_field].shift(1).fillna(
            self.gps_points_gdf[agent_field]).astype(int)

        # 切分主行程
        self.gps_points_gdf['main_label'] = (self.gps_points_gdf[time_gap_field] > self.group_gap_threshold).astype(int)
        _ = self.gps_points_gdf[agent_field] != self.gps_points_gdf[pre_agent_field]
        self.gps_points_gdf.loc[_, 'main_label'] = 1
        self.gps_points_gdf[group_field] = self.gps_points_gdf['main_label'].cumsum()
        self.gps_points_gdf.drop(columns=['main_label'], axis=1, inplace=True)
        del self.gps_points_gdf[agent_field]
        # 主行程ID, 再次更新pre_agent
        self.gps_points_gdf.rename(columns={group_field: agent_field}, inplace=True)
        self.gps_points_gdf[pre_agent_field] = self.gps_points_gdf[agent_field].shift(1).fillna(
            self.gps_points_gdf[agent_field]).astype(int)

        # 切分子行程
        # 找出距离小于dis_gap_field的行
        self.gps_points_gdf['sub_label'] = (self.gps_points_gdf[dis_gap_field] >= self.min_distance_threshold).astype(
            int)
        _ = self.gps_points_gdf[agent_field] != self.gps_points_gdf[pre_agent_field]
        self.gps_points_gdf.loc[_, 'sub_label'] = 1
        self.del_consecutive_zero(df=self.gps_points_gdf, col='sub_label', n=self.n)

        self.gps_points_gdf[agent_field] = self.gps_points_gdf[agent_field].astype(str) + '-' + self.gps_points_gdf[
            sub_group_field].astype(str)
        unique_id = set(self.gps_points_gdf[agent_field])
        _m = {s:i for s, i in zip(unique_id, [i for i in range(1, len(unique_id) + 1)])}
        self.gps_points_gdf[agent_field] = self.gps_points_gdf[agent_field].map(_m)
        del self.gps_points_gdf['sub_label'], self.gps_points_gdf[pre_time_field], self.gps_points_gdf[pre_p_field], \
            self.gps_points_gdf[time_gap_field], self.gps_points_gdf[dis_gap_field], self.gps_points_gdf['pre_agent'], \
            self.gps_points_gdf[sub_group_field]

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
        df.drop(columns=['__del__', '__a__', '__cut__', 'accu_time'], axis=1, inplace=True)

    def clean_res(self) -> gpd.GeoDataFrame:
        export_res = self.gps_points_gdf.to_crs('EPSG:4326')
        return export_res

    def generate_od(self) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        res_df = self.clean_res()
        res_df['trip_len'] = res_df.groupby(agent_field)[gps_field.TIME_FIELD].transform('count')
        res_df['gap'] = (res_df['trip_len'] / self.way_points_num).astype(int)
        res_df['rk'] = res_df.groupby(agent_field)[gps_field.TIME_FIELD].rank(method='min').astype(int)

        choose_idx = ((res_df['rk'] == 1) | (res_df['rk'] == res_df['trip_len']) | (
                res_df['rk'] % res_df['gap'] == 0)) & (res_df['trip_len'] >= 2)
        res_df = res_df.loc[choose_idx]
        res_df.reset_index(inplace=True, drop=True)

        # od_line
        od_line_gdf = pd.DataFrame(res_df).groupby(agent_field).agg({geometry_field: list}).reset_index(drop=False)
        od_line_gdf[geometry_field] = od_line_gdf[geometry_field].apply(lambda pl: LineString(pl))
        od_line_gdf = gpd.GeoDataFrame(od_line_gdf, geometry=geometry_field, crs='EPSG:4326')
        od_line_gdf.rename(columns={agent_field: 'od_id'}, inplace=True)

        # od
        res_df['str_xy'] = res_df[geometry_field].apply(lambda geo: list(map(str, (geo.x, geo.y))))
        od_res = res_df.groupby(agent_field).agg({'str_xy': list}).reset_index(drop=False)
        od_res['o_xy'] = od_res['str_xy'].apply(lambda x: x[0])
        od_res['d_xy'] = od_res['str_xy'].apply(lambda x: x[-1])
        od_res['way_points'] = od_res['str_xy'].apply(lambda x: x[1:-1])
        del od_res['str_xy']
        od_res['o_x'] = od_res['o_xy'].apply(lambda x: x[0])
        od_res['o_y'] = od_res['o_xy'].apply(lambda x: x[1])
        od_res['d_x'] = od_res['d_xy'].apply(lambda x: x[0])
        od_res['d_y'] = od_res['d_xy'].apply(lambda x: x[1])
        del od_res['o_xy'], od_res['d_xy']
        od_res['way_points'] = od_res['way_points'].apply(lambda x: ';'.join([','.join(item) for item in x]))
        od_res.rename(columns={agent_field: 'od_id'}, inplace=True)
        return od_res, od_line_gdf

    def execute_gps_od(self) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        self.cut_group()
        od_df, od_line = self.generate_od()
        return od_df, od_line


def _generate_od_by_gps(gps_df: pd.DataFrame = None, time_format: str = '%Y-%m-%d %H:%M:%S',
                        time_unit: str = 's',
                        plain_crs: str = 'EPSG:32650', group_gap_threshold: float = 360.0, n: int = 5,
                        min_distance_threshold: float = 10.0, way_points_num: int = 5,
                        dwell_accu_time: float = 60.0) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    gtp = GpsTrip(gps_df=gps_df, time_unit=time_unit, time_format=time_format, plain_crs=plain_crs,
                  group_gap_threshold=group_gap_threshold, n=n, min_distance_threshold=min_distance_threshold,
                  way_points_num=way_points_num, dwell_accu_time=dwell_accu_time)
    od_df, od_line = gtp.execute_gps_od()
    return od_df, od_line
