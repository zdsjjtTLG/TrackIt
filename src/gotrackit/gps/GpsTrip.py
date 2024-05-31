# -- coding: utf-8 --
# @Time    : 2024/3/7 13:55
# @Author  : TangKai
# @Team    : ZheChengData

import os
import pandas as pd
import multiprocessing
import geopandas as gpd
from .GpsArray import GpsArray
from ..tools.group import cut_group
from shapely.geometry import LineString
from ..GlobalVal import GpsField, OdField

gps_field = GpsField()
od_field = OdField()

agent_field = gps_field.AGENT_ID_FIELD
ori_agent_field = gps_field.ORIGIN_AGENT_ID_FIELD
pre_agent_field = gps_field.PRE_AGENT_ID_FIELD
geometry_field = gps_field.GEOMETRY_FIELD
lng_field, lat_field = gps_field.LNG_FIELD, gps_field.LAT_FIELD
pre_p_field, pre_time_field = gps_field.PRE_P, gps_field.PRE_TIME
time_field = gps_field.TIME_FIELD
group_field, sub_group_field = gps_field.GROUP_FIELD, gps_field.SUB_GROUP_FIELD
time_gap_field, dis_gap_field = gps_field.ADJ_TIME_GAP, gps_field.ADJ_DIS
od_id_field = od_field.OD_ID_FIELD
waypoints_field = od_field.WAYPOINTS_FIELD
ox_field, oy_field = od_field.OX_FIELD, od_field.OY_FIELD
dx_field, dy_field = od_field.DX_FIELD, od_field.DY_FIELD


class GpsPreProcess(object):
    def __init__(self, gps_df: pd.DataFrame = None, use_multi_core: bool = False, used_core_num: int = 2):
        self.gps_df = gps_df
        self.use_multi_core = use_multi_core
        self.used_core_num = used_core_num

    def create_pool(self):
        core_num = os.cpu_count() if self.used_core_num > os.cpu_count() else self.used_core_num
        all_agent = list(set(self.gps_df[gps_field.AGENT_ID_FIELD]))
        agent_group = cut_group(obj_list=all_agent, n=core_num)
        print(f'using multiprocessing - {len(agent_group)} cores')
        pool = multiprocessing.Pool(processes=len(agent_group))
        fact_core_num = len(agent_group)
        return pool, fact_core_num, agent_group

    def sampling_waypoints_od(self, way_points_num: int = 5) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        if self.use_multi_core:
            od_df, od_line = pd.DataFrame(), gpd.GeoDataFrame()
            result_list = []
            pool, core_num, agent_group = self.create_pool()
            for i in range(0, core_num):
                _gps_df = self.gps_df[self.gps_df[gps_field.AGENT_ID_FIELD].isin(agent_group[i])].copy()
                result = pool.apply_async(self._sampling_waypoints_od,
                                          args=(_gps_df, way_points_num))
                result_list.append(result)
            pool.close()
            pool.join()
            for res in result_list:
                _od_df, _od_line = res.get()
                od_df = pd.concat([od_df, _od_df])
                od_line = pd.concat([od_line, _od_line])
            od_df.reset_index(inplace=True, drop=True)
            od_line.reset_index(inplace=True, drop=True)
        else:
            od_df, od_line = self._sampling_waypoints_od(gps_df=self.gps_df, way_points_num=way_points_num)
        return od_df, od_line

    def generate_od_by_gps(self, time_format: str = '%Y-%m-%d %H:%M:%S',
                           time_unit: str = 's',
                           plain_crs: str = 'EPSG:32650', group_gap_threshold: float = 1800.0, n: int = 5,
                           min_distance_threshold: float = 10.0,
                           dwell_accu_time: float = 60.0, way_points_num: int = 5) \
            -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        if self.use_multi_core:
            od_df, od_line = pd.DataFrame(), gpd.GeoDataFrame()
            result_list = []
            pool, core_num, agent_group = self.create_pool()
            for i in range(0, core_num):
                _gps_df = self.gps_df[self.gps_df[gps_field.AGENT_ID_FIELD].isin(agent_group[i])].copy()
                result = pool.apply_async(self._generate_od_by_gps,
                                          args=(_gps_df, time_format, time_unit, plain_crs, group_gap_threshold, n,
                                                min_distance_threshold, way_points_num, dwell_accu_time))
                result_list.append(result)
            pool.close()
            pool.join()

            max_od_id = 0
            for res in result_list:
                _od_df, _od_line = res.get()
                _od_df[od_id_field] = _od_df[od_id_field] + max_od_id
                _od_line[od_id_field] = _od_line[od_id_field] + max_od_id
                od_df = pd.concat([od_df, _od_df])
                od_line = pd.concat([od_line, _od_line])
                max_od_id = od_df[od_id_field].max()
            od_df.reset_index(inplace=True, drop=True)
            od_line.reset_index(inplace=True, drop=True)
        else:
            od_df, od_line = self._generate_od_by_gps(gps_df=self.gps_df, time_format=time_format,
                                                      time_unit=time_unit, plain_crs=plain_crs,
                                                      group_gap_threshold=group_gap_threshold, n=n,
                                                      min_distance_threshold=min_distance_threshold,
                                                      way_points_num=way_points_num, dwell_accu_time=dwell_accu_time)
        return od_df, od_line

    def trip_segmentations(self, time_format: str = '%Y-%m-%d %H:%M:%S',
                           time_unit: str = 's',
                           plain_crs: str = 'EPSG:32650', group_gap_threshold: float = 1800.0, n: int = 5,
                           min_distance_threshold: float = 10.0,
                           dwell_accu_time: float = 60.0) -> pd.DataFrame:
        if self.use_multi_core:
            trip_df = pd.DataFrame()
            result_list = []
            pool, core_num, agent_group = self.create_pool()
            for i in range(0, core_num):
                _gps_df = self.gps_df[self.gps_df[gps_field.AGENT_ID_FIELD].isin(agent_group[i])].copy()
                result = pool.apply_async(self._trip_segmentation,
                                          args=(_gps_df, time_format, time_unit, plain_crs, group_gap_threshold, n,
                                                min_distance_threshold, dwell_accu_time))
                result_list.append(result)
            pool.close()
            pool.join()
            max_agent_id = 0
            for res in result_list:
                _trip_df = res.get()
                _trip_df[agent_field] = _trip_df[agent_field] + max_agent_id
                trip_df = pd.concat([trip_df, _trip_df])
                max_agent_id = trip_df[agent_field].max()
            trip_df.reset_index(inplace=True, drop=True)
        else:
            trip_df = self._trip_segmentation(gps_df=self.gps_df, time_format=time_format,
                                              time_unit=time_unit, plain_crs=plain_crs,
                                              group_gap_threshold=group_gap_threshold, n=n,
                                              min_distance_threshold=min_distance_threshold,
                                              dwell_accu_time=dwell_accu_time)
        trip_df = pd.DataFrame(trip_df)
        try:
            del trip_df[geometry_field]
        except:
            pass
        return trip_df

    @staticmethod
    def _generate_od_by_gps(gps_df: pd.DataFrame = None, time_format: str = '%Y-%m-%d %H:%M:%S',
                            time_unit: str = 's', plain_crs: str = 'EPSG:32650', group_gap_threshold: float = 1800.0,
                            n: int = 5, min_distance_threshold: float = 10.0, way_points_num: int = 5,
                            dwell_accu_time: float = 60.0) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        gtp = GpsTrip(gps_df=gps_df, time_unit=time_unit, time_format=time_format, plain_crs=plain_crs,
                      group_gap_threshold=group_gap_threshold, n=n, min_distance_threshold=min_distance_threshold,
                      way_points_num=way_points_num, dwell_accu_time=dwell_accu_time)
        od_df, od_line = gtp.execute_gps_od()
        return od_df, od_line

    @staticmethod
    def _sampling_waypoints_od(gps_df: pd.DataFrame = None, way_points_num: int = 5) -> \
            tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """
        从已经切分好行程的GPS数据中抽取带途径点的OD, 要求数据已经按照agent_id排列好了顺序, 且同一行程中已经按照时间排序
        :param way_points_num:
        :return:
        """
        assert {agent_field, lng_field, lat_field}.issubset(set(gps_df.columns)), \
            rf'请确保GPS数据中有以下字段: {agent_field, lng_field, lat_field}'
        gps_df['seq'] = [i for i in range(len(gps_df))]
        gps_df[geometry_field] = gps_df[[lng_field, lat_field]].apply(lambda row: list(row), axis=1)
        od_df, od_line = GpsTrip.sampling_waypoints_od(gps_df=gps_df, way_points_num=way_points_num,
                                                       seq_field='seq')
        return od_df, od_line

    @staticmethod
    def _trip_segmentation(gps_df: pd.DataFrame = None, time_format: str = '%Y-%m-%d %H:%M:%S',
                           time_unit: str = 's',
                           plain_crs: str = 'EPSG:32650', group_gap_threshold: float = 1800.0, n: int = 5,
                           min_distance_threshold: float = 10.0,
                           dwell_accu_time: float = 60.0) -> pd.DataFrame or gpd.GeoDataFrame:
        gtp = GpsTrip(gps_df=gps_df, time_unit=time_unit, time_format=time_format, plain_crs=plain_crs,
                      group_gap_threshold=group_gap_threshold, n=n, min_distance_threshold=min_distance_threshold,
                      dwell_accu_time=dwell_accu_time)
        trip_df = gtp.cut_group()
        return trip_df


class GpsTrip(GpsArray):
    def __init__(self, gps_df: pd.DataFrame = None, time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's',
                 plain_crs: str = 'EPSG:32650', group_gap_threshold: float = 1800.0, n: int = 5,
                 min_distance_threshold: float = 10.0, way_points_num: int = 5, dwell_accu_time: float = 150.0):
        GpsArray.__init__(self, gps_points_df=gps_df, time_unit=time_unit, time_format=time_format,
                          plane_crs=plain_crs, geo_crs='EPSG:4326')

        # 主行程时间阈值
        self.group_gap_threshold = group_gap_threshold  # s, 相邻GPS的时间超过这个阈值则被切分行程

        # 子行程距离阈值(停留点)
        self.min_distance_threshold = min_distance_threshold  # m, 相邻GPS的直线距离小于这个值就被切分子行程
        self.n = n  # 连续n个GPS点的距离小于min_distance_threshold则被初步认为是停留点
        self.dwell_accu_time = dwell_accu_time  # 连续n个GPS点的停留时间大于该值则会被切分子行程

        self.__clean_gps_gdf = gpd.GeoDataFrame()

        # 构造OD的途径点数量
        assert way_points_num <= 10
        self.way_points_num = way_points_num

    def cut_group(self):
        """行程切分"""
        origin_agent_set = set(self.gps_points_gdf[agent_field])
        car_num = len(origin_agent_set)
        print(rf'{car_num} vehicles, cutting group...')

        # 记录原始agent_id
        self.gps_points_gdf[ori_agent_field] = self.gps_points_gdf[agent_field]
        try:
            self.gps_points_gdf[agent_field] = self.gps_points_gdf[agent_field].astype(int)
        except ValueError:
            str_map = {str_agent: int_agent for str_agent, int_agent in
                       zip(origin_agent_set, range(1, len(origin_agent_set) + 1))}
            self.gps_points_gdf[agent_field] = self.gps_points_gdf[ori_agent_field].map(str_map)
        # 和前序点位的: 时间差\距离差
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
        self.gps_points_gdf[agent_field] = self.gps_points_gdf['main_label'].cumsum() + 1 # 主行程ID更新
        del self.gps_points_gdf['main_label']
        self.gps_points_gdf[pre_agent_field] = self.gps_points_gdf[agent_field].shift(1).fillna(
            self.gps_points_gdf[agent_field]).astype(int)  # 再次更新pre_agent

        # 切分子行程
        self.gps_points_gdf['sub_label'] = \
            (self.gps_points_gdf[dis_gap_field] >= self.min_distance_threshold).astype(int)
        self.gps_points_gdf.loc[0, 'sub_label'] = 1
        if len(self.gps_points_gdf[self.gps_points_gdf['sub_label'].eq(0)]) == 0:
            del self.gps_points_gdf['sub_label'], self.gps_points_gdf[pre_time_field], \
                self.gps_points_gdf[pre_p_field], self.gps_points_gdf[time_gap_field], \
                self.gps_points_gdf[dis_gap_field], self.gps_points_gdf[pre_agent_field]
            return self.gps_points_gdf.to_crs('EPSG:4326')
        _ = self.gps_points_gdf[agent_field] != self.gps_points_gdf[pre_agent_field]
        self.gps_points_gdf.loc[_, 'sub_label'] = 1
        self.del_consecutive_zero(df=self.gps_points_gdf, col='sub_label', n=self.n)  # add sub_group

        self.gps_points_gdf[agent_field] = self.gps_points_gdf[agent_field].astype(str) + '-' + self.gps_points_gdf[
            sub_group_field].astype(str)
        unique_id = set(self.gps_points_gdf[agent_field])
        _m = {s:i for s, i in zip(unique_id, [i for i in range(1, len(unique_id) + 1)])}
        self.gps_points_gdf[agent_field] = self.gps_points_gdf[agent_field].map(_m)
        del self.gps_points_gdf['sub_label'], self.gps_points_gdf[pre_time_field], self.gps_points_gdf[pre_p_field], \
            self.gps_points_gdf[time_gap_field], self.gps_points_gdf[dis_gap_field], self.gps_points_gdf[pre_agent_field], \
            self.gps_points_gdf[sub_group_field]
        return self.gps_points_gdf.to_crs('EPSG:4326')

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

    @staticmethod
    def sampling_waypoints_od(gps_df: gpd.GeoDataFrame or pd.DataFrame = None, seq_field: str = 'time',
                              way_points_num: int = 5) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """
        从已经切分好行程的GPS数据中抽样获得途径点OD
        :param gps_df:
        :param seq_field: 一个行程内标定时序的字段
        :param way_points_num:
        :return:
        """
        gps_df['trip_len'] = gps_df.groupby(agent_field)[seq_field].transform('count')
        gps_df['gap'] = (gps_df['trip_len'] / way_points_num).astype(int)
        gps_df['rk'] = gps_df.groupby(agent_field)[seq_field].rank(method='min').astype(int)

        choose_idx = ((gps_df['rk'] == 1) | (gps_df['rk'] == gps_df['trip_len']) | (
                gps_df['rk'] % gps_df['gap'] == 0)) & (gps_df['trip_len'] >= 2)
        res_df = gps_df.loc[choose_idx].copy()
        del gps_df
        res_df.reset_index(inplace=True, drop=True)

        # od_line
        od_line_gdf = pd.DataFrame(res_df).groupby(agent_field).agg({geometry_field: list}).reset_index(drop=False)
        od_line_gdf[geometry_field] = od_line_gdf[geometry_field].apply(lambda pl: LineString(pl))
        od_line_gdf = gpd.GeoDataFrame(od_line_gdf, geometry=geometry_field, crs='EPSG:4326')
        od_line_gdf.rename(columns={agent_field: od_id_field}, inplace=True)

        # od
        try:
            res_df['str_xy'] = res_df[geometry_field].apply(lambda geo: list(map(str, (geo.x, geo.y))))
        except AttributeError:
            res_df['str_xy'] = res_df[geometry_field].apply(lambda geo: list(map(str, geo)))
        od_res = res_df.groupby(agent_field).agg({'str_xy': list}).reset_index(drop=False)
        od_res['o_xy'] = od_res['str_xy'].apply(lambda x: x[0])
        od_res['d_xy'] = od_res['str_xy'].apply(lambda x: x[-1])
        od_res[waypoints_field] = od_res['str_xy'].apply(lambda x: x[1:-1])
        del od_res['str_xy']
        od_res[ox_field] = od_res['o_xy'].apply(lambda x: x[0])
        od_res[oy_field] = od_res['o_xy'].apply(lambda x: x[1])
        od_res[dx_field] = od_res['d_xy'].apply(lambda x: x[0])
        od_res[dy_field] = od_res['d_xy'].apply(lambda x: x[1])
        del od_res['o_xy'], od_res['d_xy']
        od_res[waypoints_field] = od_res[waypoints_field].apply(lambda x: ';'.join([','.join(item) for item in x]))
        od_res.rename(columns={agent_field: od_id_field}, inplace=True)
        return od_res, od_line_gdf

    def execute_gps_od(self) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """切分行程且计算带途径的OD数据"""
        done_cut_df = self.cut_group()
        od_df, od_line = self.sampling_waypoints_od(gps_df=done_cut_df, seq_field=gps_field.TIME_FIELD,
                                                    way_points_num=self.way_points_num)
        return od_df, od_line
