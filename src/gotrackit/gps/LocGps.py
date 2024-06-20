# -- coding: utf-8 --
# @Time    : 2023/12/10 20:12
# @Author  : TangKai
# @Team    : ZheChengData

"""车辆GPS数据的相关方法和属性"""

import numpy as np
import pandas as pd
import geopandas as gpd
from ..map.Net import Net
from datetime import timedelta
from ..tools.geo_process import prj_inf
from ..tools.geo_process import segmentize
from shapely.geometry import Point, Polygon, LineString
from ..GlobalVal import GpsField, NetField, PrjConst, MarkovField


gps_field = GpsField()
net_field = NetField()
prj_const = PrjConst()
markov_field = MarkovField()


lng_field = gps_field.LNG_FIELD
lat_field = gps_field.LAT_FIELD
next_p_field = gps_field.NEXT_P
next_seq_field = gps_field.NEXT_SEQ
pre_p_field = gps_field.PRE_P
time_field = gps_field.TIME_FIELD
next_time_field = gps_field.NEXT_TIME
agent_field = gps_field.AGENT_ID_FIELD
geometry_field = gps_field.GEOMETRY_FIELD
time_gap_field = gps_field.ADJ_TIME_GAP
dis_gap_field = gps_field.ADJ_DIS
adj_speed_field = gps_field.ADJ_SPEED
dense_geo_field = gps_field.DENSE_GEO
n_seg_field = gps_field.N_SEGMENTS
geo_crs = prj_const.PRJ_CRS
sub_group_field = gps_field.SUB_GROUP_FIELD
ori_seq_field = gps_field.ORIGIN_POINT_SEQ_FIELD


class GpsPointsGdf(object):

    def __init__(self, gps_points_df: pd.DataFrame = None, buffer: float = 200.0,
                 time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's',
                 plane_crs: str = 'EPSG:32649', dense_gps: bool = True, dense_interval: float = 25.0,
                 dwell_l_length: float = 10.0, dwell_n: int = 3, user_filed_list: list[str] = None):
        """

        :param gps_points_df: gps数据dataframe, agent_id, lng, lat, time
        :param buffer: GPS点的buffer半径大小(用于生成候选路段), m
        :param time_format: 时间列的字符格式
        :param plane_crs: 平面投影坐标系
        :param dense_gps: 是否加密GPS点
        :param dwell_l_length: 停留点识别距离阈值
        :param dwell_n: 连续dwell_n个点的距离小于dwell_l_length就被识别为停留点
        :param dense_interval: 加密间隔(相邻GPS点的直线距离小于dense_interval即会进行加密)
        :param user_filed_list
        """
        if user_filed_list:
            assert set(user_filed_list).issubset(gps_points_df.columns), '指定的用户字段不存在!'
        self.geo_crs = geo_crs
        self.buffer = buffer
        self.__crs = self.geo_crs
        self.plane_crs = plane_crs
        self.dense_gps = dense_gps
        self.dense_interval = dense_interval
        self.dwell_l_length = dwell_l_length
        self.dwell_n = dwell_n
        self.__gps_point_dis_dict = dict()
        gps_points_df.reset_index(inplace=True, drop=True)
        self.agent_id = gps_points_df.at[0, gps_field.AGENT_ID_FIELD]
        self.check(gps_points_df=gps_points_df)
        gps_points_df[gps_field.GEOMETRY_FIELD] = \
            gps_points_df[[gps_field.LNG_FIELD, gps_field.LAT_FIELD]].apply(lambda p: Point(p), axis=1)
        gps_points_gdf = gpd.GeoDataFrame(gps_points_df, geometry=gps_field.GEOMETRY_FIELD, crs=self.geo_crs)
        del gps_points_df
        try:
            gps_points_gdf[gps_field.TIME_FIELD] = \
                pd.to_datetime(gps_points_gdf[gps_field.TIME_FIELD], format=time_format)
        except ValueError:
            print(rf'time column does not match format {time_format}, try using time-unit: {time_unit}')
            if gps_points_gdf[time_field].dtype == object:
                gps_points_gdf[time_field] = gps_points_gdf[time_field].astype(float)
            gps_points_gdf[gps_field.TIME_FIELD] = \
                pd.to_datetime(gps_points_gdf[gps_field.TIME_FIELD], unit=time_unit)

        gps_points_gdf.sort_values(by=[gps_field.TIME_FIELD], ascending=[True], inplace=True)
        gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(gps_points_gdf))]
        gps_points_gdf.reset_index(inplace=True, drop=True)

        self.__source_gps_points_gdf = gps_points_gdf.copy()  # 存储最原始的GPS信息

        self.__user_gps_info = pd.DataFrame()
        user_filed_list = list() if user_filed_list is None else user_filed_list
        self.user_filed_list = list(
            set(user_filed_list) - {agent_field, gps_field.POINT_SEQ_FIELD, gps_field.SUB_SEQ_FIELD,
                                    time_field, gps_field.LOC_TYPE, net_field.LINK_ID_FIELD,
                                    net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD, net_field.DIRECTION_FIELD,
                                    lng_field, lat_field, geometry_field, 'prj_lng', 'prj_lat', markov_field.PRJ_GEO,
                                    markov_field.DIS_TO_NEXT})

        self.__user_gps_info = gps_points_gdf[
            [gps_field.POINT_SEQ_FIELD] + self.user_filed_list].copy()  # user_diy_info
        self.__user_gps_info[gps_field.LOC_TYPE] = 'source'

        self.__gps_points_gdf = gps_points_gdf[
            [gps_field.POINT_SEQ_FIELD, agent_field, time_field, geometry_field]].copy()

        self.to_plane_prj()
        self.generate_plain_xy()

        self.gps_adj_dis_map = dict()
        self.gps_seq_time_map = dict()
        self.gps_seq_geo_map = dict()
        self.gps_rou_buffer = None

        self.done_diff_heading = False

    @staticmethod
    def check(gps_points_df: pd.DataFrame = None):
        assert {gps_field.LNG_FIELD, gps_field.LAT_FIELD,
                gps_field.AGENT_ID_FIELD, gps_field.TIME_FIELD}.issubset(
            set(gps_points_df.columns)), \
            rf'GPS数据字段有误, 请至少包含如下字段: {gps_field.AGENT_ID_FIELD, gps_field.LNG_FIELD, gps_field.LAT_FIELD, gps_field.TIME_FIELD}'

    def generate_plain_xy(self):
        self.__gps_points_gdf[gps_field.PLAIN_X] = self.__gps_points_gdf[net_field.GEOMETRY_FIELD].apply(
            lambda geo: geo.x)
        self.__gps_points_gdf[gps_field.PLAIN_Y] = self.__gps_points_gdf[net_field.GEOMETRY_FIELD].apply(
            lambda geo: geo.y)

    def dense(self) -> None:
        if len(self.__gps_points_gdf) <= 1:
            return None
        # 时间差和距离差
        self.calc_adj_dis_gap()
        self.calc_adj_time_gap()
        should_be_dense_gdf = self.__gps_points_gdf[self.__gps_points_gdf[dis_gap_field] > self.dense_interval].copy()
        self.__gps_points_gdf.drop(columns=[next_time_field, next_p_field, time_gap_field, dis_gap_field], axis=1,
                                   inplace=True)
        if should_be_dense_gdf.empty:
            return None

        should_be_dense_gdf[n_seg_field] = (0.001 + should_be_dense_gdf[dis_gap_field] / self.dense_interval).astype(
            int) + 1

        dense_geo, dense_time = list(), list()
        _ = [[dense_geo.append(segmentize(s_loc=(p.x, p.y), e_loc=(next_p.x, next_p.y), n=n)),
              dense_time.append([t + timedelta(seconds=i * t_gap / n) for i in range(1, n)])]
             for p, next_p, n, t, t_gap in zip(should_be_dense_gdf[geometry_field],
                                               should_be_dense_gdf[next_p_field],
                                               should_be_dense_gdf[n_seg_field], should_be_dense_gdf[time_field],
                                               should_be_dense_gdf[time_gap_field])]
        del _
        should_be_dense_gdf[dense_geo_field] = dense_geo
        should_be_dense_gdf[time_field] = dense_time

        should_be_dense_gdf.drop(columns=[geometry_field], axis=1, inplace=True)
        should_be_dense_gdf = pd.DataFrame(should_be_dense_gdf).explode(column=[time_field, dense_geo_field],
                                                                        ignore_index=True)
        should_be_dense_gdf.rename(columns={dense_geo_field: geometry_field}, inplace=True)
        should_be_dense_gdf[geometry_field] = should_be_dense_gdf[geometry_field].apply(lambda x: Point(x))
        should_be_dense_gdf.drop(columns=[next_time_field, next_p_field, time_gap_field, dis_gap_field, n_seg_field],
                                 axis=1, inplace=True)
        # must be plain
        should_be_dense_gdf = gpd.GeoDataFrame(should_be_dense_gdf, geometry=geometry_field, crs=self.crs)
        should_be_dense_gdf[gps_field.PLAIN_X] = should_be_dense_gdf[geometry_field].apply(lambda geo: geo.x)
        should_be_dense_gdf[gps_field.PLAIN_Y] = should_be_dense_gdf[geometry_field].apply(lambda geo: geo.y)
        self.__gps_points_gdf[ori_seq_field] = self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD]
        self.__gps_points_gdf = pd.concat([self.__gps_points_gdf, should_be_dense_gdf])
        self.__gps_points_gdf.sort_values(by=time_field, ascending=True, inplace=True)
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(self.__gps_points_gdf))]

        if not self.__user_gps_info.empty:
            self.__gps_points_gdf[ori_seq_field] = self.__gps_points_gdf[ori_seq_field].fillna(-1).astype(int)
            ori_now_map = {_ori: _now for _ori, _now in zip(self.__gps_points_gdf[ori_seq_field],
                                                            self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD])}
            self.__user_gps_info[gps_field.POINT_SEQ_FIELD] = self.__user_gps_info[gps_field.POINT_SEQ_FIELD].map(
                ori_now_map).fillna(-1).astype(int)
        del self.__gps_points_gdf[ori_seq_field]

    def calc_adj_dis_gap(self) -> None:
        # 距离差
        self.__gps_points_gdf[next_p_field] = self.__gps_points_gdf[geometry_field].shift(-1).fillna(
            self.__gps_points_gdf[geometry_field])
        self.__gps_points_gdf[dis_gap_field] = self.__gps_points_gdf[next_p_field].distance(
            self.__gps_points_gdf[geometry_field])

    def calc_adj_time_gap(self) -> None:
        # 时间差
        self.__gps_points_gdf[next_time_field] = self.__gps_points_gdf[time_field].shift(-1).fillna(
            self.__gps_points_gdf[time_field])
        self.__gps_points_gdf[time_gap_field] = self.__gps_points_gdf[next_time_field] - self.__gps_points_gdf[
            time_field]
        self.__gps_points_gdf[time_gap_field] = self.__gps_points_gdf[time_gap_field].apply(lambda x: x.seconds)

    def calc_pre_next_dis(self) -> pd.DataFrame():
        self.calc_adj_dis_gap()
        # next_seq
        res = self.__gps_points_gdf.copy()
        self.gps_adj_dis_map = {seq: adj_dis for seq, adj_dis in zip(res[gps_field.POINT_SEQ_FIELD],
                                                                     res[gps_field.ADJ_DIS])}

    def lower_frequency(self, n: int = 5):
        """
        GPS数据降频
        :param n: 降频倍数
        :return:
        """
        self.__gps_points_gdf['label'] = pd.Series([i for i in range(len(self.__gps_points_gdf))]) % n
        self.__gps_points_gdf = self.__gps_points_gdf[self.__gps_points_gdf['label'].eq(0)].copy()
        del self.__gps_points_gdf['label']

    def rolling_average(self, window: int = 2):
        """
        滑动窗口降噪, 执行该操作后, 不支持匹配结果表中输出用于自定义字段
        :param window: 窗口大小
        :return:
        """
        # 滑动窗口执行后会重置所有的gps的seq字段
        if len(self.__gps_points_gdf) <= window:
            return None

        self.__gps_points_gdf[gps_field.TIME_FIELD] = self.__gps_points_gdf[gps_field.TIME_FIELD].apply(
            lambda t: t.timestamp())
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        rolling_average_df = self.__gps_points_gdf[
            [gps_field.PLAIN_X, gps_field.PLAIN_Y, gps_field.TIME_FIELD]].rolling(window=window).mean()

        rolling_average_df.dropna(subset=[gps_field.PLAIN_X, gps_field.PLAIN_Y, gps_field.TIME_FIELD], inplace=True,
                                  how='any')
        rolling_average_df[gps_field.AGENT_ID_FIELD] = self.__gps_points_gdf.at[0, gps_field.AGENT_ID_FIELD]

        self.__gps_points_gdf = pd.concat([self.__gps_points_gdf.loc[[0], :],
                                           rolling_average_df,
                                           self.__gps_points_gdf.loc[[len(self.__gps_points_gdf) - 1], :]])
        del rolling_average_df
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(self.__gps_points_gdf))]
        self.__gps_points_gdf[net_field.GEOMETRY_FIELD] = \
            self.__gps_points_gdf[[gps_field.PLAIN_X, gps_field.PLAIN_Y]].apply(lambda item: Point(item), axis=1)
        self.__gps_points_gdf[gps_field.TIME_FIELD] = \
            pd.to_datetime(self.__gps_points_gdf[gps_field.TIME_FIELD], unit='s')
        self.__gps_points_gdf = gpd.GeoDataFrame(self.__gps_points_gdf, geometry=gps_field.GEOMETRY_FIELD, crs=self.crs)
        self.__user_gps_info = self.__gps_points_gdf[[gps_field.POINT_SEQ_FIELD]].copy()
        self.__user_gps_info[gps_field.LOC_TYPE] = 'source'
        self.user_filed_list = []

    def calc_diff_heading(self):
        if self.done_diff_heading:
            return None
        self.__gps_points_gdf['next_x'] = self.__gps_points_gdf[gps_field.PLAIN_X].shift(-1).fillna(
            self.__gps_points_gdf[gps_field.PLAIN_X])
        self.__gps_points_gdf['next_y'] = self.__gps_points_gdf[gps_field.PLAIN_Y].shift(-1).fillna(
            self.__gps_points_gdf[gps_field.PLAIN_Y])
        self.__gps_points_gdf['pre_x'] = self.__gps_points_gdf[gps_field.PLAIN_X].shift(1).fillna(
            self.__gps_points_gdf[gps_field.PLAIN_X])
        self.__gps_points_gdf['pre_y'] = self.__gps_points_gdf[gps_field.PLAIN_Y].shift(1).fillna(
            self.__gps_points_gdf[gps_field.PLAIN_Y])
        self.__gps_points_gdf[gps_field.X_DIFF] = self.__gps_points_gdf['next_x'] - self.__gps_points_gdf['pre_x']
        self.__gps_points_gdf[gps_field.Y_DIFF] = self.__gps_points_gdf['next_y'] - self.__gps_points_gdf['pre_y']
        del self.__gps_points_gdf['next_x'], self.__gps_points_gdf['next_y'], \
            self.__gps_points_gdf['pre_x'], self.__gps_points_gdf['pre_y']

        self.__gps_points_gdf[gps_field.VEC_LEN] = np.sqrt(
            self.__gps_points_gdf[gps_field.X_DIFF] ** 2 + self.__gps_points_gdf[gps_field.Y_DIFF] ** 2)
        self.done_diff_heading = True

    @property
    def gps_gdf(self) -> gpd.GeoDataFrame:
        return self.__gps_points_gdf.copy()

    @property
    def user_info(self) -> pd.DataFrame:
        return self.__user_gps_info.copy()

    @property
    def gps_seq_time(self) -> dict:
        if self.gps_seq_time_map:
            return self.gps_seq_time_map
        else:
            self.gps_seq_time_map = {seq: t for seq, t in
                                     zip(self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD],
                                         self.__gps_points_gdf[gps_field.TIME_FIELD])}
            return self.gps_seq_time_map

    @property
    def gps_seq_geo(self) -> dict:
        if self.gps_seq_geo_map:
            return self.gps_seq_geo_map
        else:
            self.gps_seq_geo_map = {seq: t for seq, t in
                                    zip(self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD],
                                        self.__gps_points_gdf[gps_field.GEOMETRY_FIELD])}
            return self.gps_seq_geo_map

    @property
    def crs(self):
        return self.__crs

    def get_gps_array_buffer(self, buffer: float = 200.0, dup_threshold: float = 10.0) -> Polygon or None:
        """输出gps路径的buffer范围面域"""
        gps_route_l = gpd.GeoSeries(LineString(self.__gps_points_gdf[gps_field.GEOMETRY_FIELD].to_list()))
        try:
            simplify_gps_route_l = gps_route_l.remove_repeated_points(dup_threshold)
        except:
            simplify_gps_route_l = gps_route_l.simplify(dup_threshold / 10.0)
        gps_array_buffer = simplify_gps_route_l[0].buffer(buffer)
        self.gps_rou_buffer = gps_array_buffer
        return gps_array_buffer

    def generate_candidate_link(self, net: Net = None, is_hierarchical: bool = False) -> \
            tuple[pd.DataFrame, list[int]]:
        """
        计算GPS观测点的候选路段
        :param net:
        :param is_hierarchical
        :return: GPS候选路段信息, 未匹配到候选路段的gps点id
        """
        gps_buffer_gdf = self.__gps_points_gdf[[gps_field.POINT_SEQ_FIELD, gps_field.GEOMETRY_FIELD]].copy()
        if gps_buffer_gdf.crs.srs.upper() != self.plane_crs:
            gps_buffer_gdf = gps_buffer_gdf.to_crs(self.plane_crs)
        gps_buffer_gdf['gps_buffer'] = gps_buffer_gdf[net_field.GEOMETRY_FIELD].buffer(self.buffer)
        gps_buffer_gdf.set_geometry('gps_buffer', inplace=True, crs=gps_buffer_gdf.crs)
        origin_seq = set(gps_buffer_gdf[gps_field.POINT_SEQ_FIELD])

        single_link_gdf = net.get_link_data()[[net_field.SINGLE_LINK_ID_FIELD, net_field.LINK_ID_FIELD,
                                               net_field.FROM_NODE_FIELD,
                                               net_field.TO_NODE_FIELD, net_field.DIRECTION_FIELD,
                                               net_field.LENGTH_FIELD, net_field.GEOMETRY_FIELD]]
        if not net.is_sub_net and is_hierarchical:
            try:
                pre_filter_link = net.calc_pre_filter(gps_rou_buffer_gdf=gps_buffer_gdf)
                single_link_gdf = single_link_gdf[single_link_gdf[net_field.LINK_ID_FIELD].isin(pre_filter_link)]
            except Exception as e:
                print(repr(e), '空间分层关联失效')
        single_link_gdf.reset_index(inplace=True, drop=True)
        candidate_link = gpd.sjoin(gps_buffer_gdf, single_link_gdf)
        remain_gps_list = list(origin_seq - set(candidate_link[gps_field.POINT_SEQ_FIELD]))

        if not candidate_link.empty:
            candidate_link.drop(columns=['index_right', 'gps_buffer'], axis=1, inplace=True)
            # add link geo
            single_link_gdf.rename(columns={net_field.GEOMETRY_FIELD: 'single_link_geo'}, inplace=True)
            candidate_link = pd.merge(candidate_link,
                                      single_link_gdf[[net_field.SINGLE_LINK_ID_FIELD, 'single_link_geo']],
                                      on=net_field.SINGLE_LINK_ID_FIELD, how='left')
            candidate_link.reset_index(inplace=True, drop=True)
        return candidate_link, remain_gps_list

    def to_plane_prj(self) -> None:
        if self.__gps_points_gdf.crs.srs.upper() == self.plane_crs:
            self.__crs = self.plane_crs
        else:
            self.__gps_points_gdf = self.__gps_points_gdf.to_crs(self.plane_crs)
            self.__crs = self.plane_crs

    def to_geo_prj(self) -> None:
        if self.__gps_points_gdf.crs.srs.upper() == self.geo_crs:
            self.__crs = self.geo_crs
        else:
            self.__gps_points_gdf = self.__gps_points_gdf.to_crs(self.geo_crs)
            self.__crs = self.geo_crs

    def get_point(self, seq: int = 0):
        return self.__gps_points_gdf.at[seq, gps_field.TIME_FIELD], \
            self.__gps_points_gdf.at[seq, gps_field.GEOMETRY_FIELD]

    @property
    def source_gps(self) -> gpd.GeoDataFrame:
        if self.__source_gps_points_gdf is None:
            return self.__gps_points_gdf.copy()
        else:
            return self.__source_gps_points_gdf.copy()

    def get_prj_inf(self, line: LineString, seq: int = 0) -> tuple[Point, float, float, float, float, float]:
        """
        计算当前gps点实例在指定线对象上的投影信息
        :param line:
        :param seq:
        :return:
        """
        (prj_p, prj_dis, route_dis, l_length, dx, dy) = self._get_prj_inf(self.get_point(seq)[1], line)
        return prj_p, prj_dis, route_dis, l_length, dx, dy

    def delete_target_gps(self, target_seq_list: list[int]) -> None:
        self.__gps_points_gdf.drop(
            index=self.__gps_points_gdf[self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].isin(target_seq_list)].index,
            inplace=True, axis=0)

    @staticmethod
    def _get_prj_inf(gps_point: Point = None, line: LineString = None) -> tuple[Point, float, float, float, float, float]:
        """
        # 返回 (GPS投影点坐标, GPS点到投影点的直线距离, GPS投影点到line拓扑起点的路径距离, line的长度)
        :param gps_point:
        :param line:
        :return: (GPS投影点坐标, GPS点到投影点的直线距离, GPS投影点到line拓扑起点的路径距离, line的长度)
        """
        prj_p, p_prj_l, prj_route_l, line_length, _, dx, dy = prj_inf(p=gps_point, line=line)
        return prj_p, p_prj_l, prj_route_l, line_length, dx, dy

    @property
    def gps_list_length(self) -> int:
        return len(self.__gps_points_gdf)

    @property
    def used_observation_seq_list(self) -> list[int]:
        return self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].to_list()

    def del_dwell_points(self) -> None:
        # add field = dis_gap_field
        self.calc_adj_dis_gap()
        del self.__gps_points_gdf[next_p_field]
        self.__gps_points_gdf['dwell_label'] = \
            (self.__gps_points_gdf[dis_gap_field] > self.dwell_l_length).astype(int)
        del self.__gps_points_gdf[dis_gap_field]
        self.__gps_points_gdf.loc[len(self.__gps_points_gdf) - 1, 'dwell_label'] = 1
        self.__gps_points_gdf = self.del_consecutive_zero(df=self.__gps_points_gdf, col='dwell_label', n=self.dwell_n)
        del self.__gps_points_gdf['dwell_label']
        try:
            self.__gps_points_gdf.drop(columns=[sub_group_field], axis=1, inplace=True)
        except KeyError:
            pass

    @staticmethod
    def del_consecutive_zero(df: pd.DataFrame or gpd.GeoDataFrame = None,
                             col: str = None, n: int = 3,
                             del_all_dwell: bool = True) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        标记超过连续n行为0的行, 并且只保留最后一行
        :param df:
        :param col:
        :param n:
        :param del_all_dwell
        :return:
        """
        m = df[col].ne(0)
        df['__del__'] = (df.groupby(m.cumsum())[col]
                         .transform('count').gt(n + 1)
                         & (~m)
                         )
        if del_all_dwell:
            df.drop(index=df[df['__del__']].index, inplace=True, axis=0)
            df.drop(columns=['__del__'], axis=1, inplace=True)
        else:
            df['__a__'] = df['__del__'].ne(1).cumsum()
            df['__cut__'] = df['__a__'] & df['__del__']
            df.drop_duplicates(subset=['__a__'], keep='last', inplace=True)
            df[sub_group_field] = df['__cut__'].ne(0).cumsum()
            df.drop(columns=['__del__', '__a__', '__cut__'], axis=1, inplace=True)
        return df



