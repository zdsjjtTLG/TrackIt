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
from ..tools.kf import OffLineTrajectoryKF
from ..tools.time_build import build_time_col
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
                 plane_crs: str = 'EPSG:32649', user_filed_list: list[str] = None,
                 already_plain: bool = False, multi_agents: bool = False):
        """

        :param gps_points_df: gps数据dataframe, agent_id, lng, lat, time
        :param buffer: GPS点的buffer半径大小(用于生成候选路段), m
        :param time_format: 时间列的字符格式
        :param plane_crs: 平面投影坐标系
        :param user_filed_list:
        :param already_plain: 原始lng, lat是否 已经是平面投影坐标系
        :param multi_agents: 是否包含多个agents
        """
        self.geo_crs = geo_crs
        self.buffer = buffer
        self.__crs = self.geo_crs
        self.plane_crs = plane_crs
        self.multi_agents = multi_agents
        self.__gps_point_dis_dict = dict()
        gps_points_df.reset_index(inplace=True, drop=True)
        self.agent_id = gps_points_df.at[0, gps_field.AGENT_ID_FIELD]
        self.already_plain = already_plain
        gps_points_df[gps_field.GEOMETRY_FIELD] = \
            gps_points_df[[gps_field.LNG_FIELD, gps_field.LAT_FIELD]].apply(lambda p: Point(p), axis=1)
        if not already_plain:
            gps_points_gdf = gpd.GeoDataFrame(gps_points_df, geometry=gps_field.GEOMETRY_FIELD, crs=self.geo_crs)
        else:
            gps_points_gdf = gpd.GeoDataFrame(gps_points_df, geometry=gps_field.GEOMETRY_FIELD, crs=self.plane_crs)
        del gps_points_df
        build_time_col(df=gps_points_gdf, time_unit=time_unit, time_format=time_format, time_field=gps_field.TIME_FIELD)

        gps_points_gdf.sort_values(by=[agent_field, gps_field.TIME_FIELD], ascending=[True, True], inplace=True)

        self.add_seq_field(gps_points_gdf=gps_points_gdf, multi_agents=self.multi_agents)
        gps_points_gdf.reset_index(inplace=True, drop=True)

        self.__source_gps_points_gdf = gps_points_gdf.copy()  # 存储最原始的GPS信息

        self.__user_gps_info = pd.DataFrame()
        self.user_filed_list = list() if user_filed_list is None else user_filed_list

        self.__user_gps_info = gps_points_gdf[
            [agent_field, gps_field.POINT_SEQ_FIELD] + self.user_filed_list].copy()  # user_diy_info
        self.__user_gps_info[gps_field.LOC_TYPE] = 's'

        self.__gps_points_gdf = gps_points_gdf[
            [agent_field, gps_field.POINT_SEQ_FIELD, time_field, geometry_field]].copy()

        if not already_plain:
            self.to_plane_prj()
            self.generate_plain_xy()
        else:
            self.__crs = self.plane_crs
            self.__gps_points_gdf[gps_field.PLAIN_X] = gps_points_gdf[gps_field.LNG_FIELD]
            self.__gps_points_gdf[gps_field.PLAIN_Y] = gps_points_gdf[gps_field.LAT_FIELD]

        self.gps_adj_dis_map = dict()
        self.gps_seq_time_map = dict()
        self.gps_seq_geo_map = dict()
        self.gps_rou_buffer = None

        self.done_diff_heading = False

    @staticmethod
    def check(gps_points_df: pd.DataFrame = None, user_field_list: list[str] = None):
        user_field_list = list() if user_field_list is None else user_field_list
        all_gps_field_set = set(gps_points_df.columns)
        assert {agent_field, time_field, lng_field, lat_field}.issubset(set(gps_points_df.columns)), \
            rf'''the GPS data field is incorrect, please include at least the following fields: 
            {agent_field, time_field, lng_field, lat_field}'''
        init_used_set = {agent_field, time_field, lng_field, lat_field}
        sys_field_set = {agent_field, net_field.SINGLE_LINK_ID_FIELD, gps_field.POINT_SEQ_FIELD,
                         gps_field.SUB_SEQ_FIELD,
                         time_field, gps_field.LOC_TYPE, net_field.LINK_ID_FIELD, 'prj_x', 'prj_y',
                         net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD, net_field.DIRECTION_FIELD,
                         lng_field, lat_field, geometry_field, 'prj_lng', 'prj_lat', markov_field.PRJ_GEO,
                         markov_field.DIS_TO_NEXT, net_field.X_DIFF, net_field.Y_DIFF, net_field.VEC_LEN,
                         markov_field.MATCH_HEADING, markov_field.DRIVING_L}

        if user_field_list:
            user_field_set = set(user_field_list)
            to_del_fields = all_gps_field_set - {lng_field, lat_field, agent_field, time_field} - user_field_set
            if to_del_fields:
                gps_points_df.drop(columns=list(to_del_fields), inplace=True, axis=1)
            assert user_field_set.issubset(all_gps_field_set), 'user input field does not exist in GPS data table'
            dup_fields = user_field_set & sys_field_set
            if dup_fields:
                ori_fields = [ori for ori in dup_fields]
                rename_fields = ['_' + ori for ori in dup_fields]
                gps_points_df[rename_fields] = gps_points_df[ori_fields]
                can_del_fields = set(ori_fields) - init_used_set
                if can_del_fields:
                    gps_points_df.drop(columns=list(can_del_fields), axis=1, inplace=True)
                user_field_list = list((user_field_set - dup_fields) | set(rename_fields))
        return user_field_list

    def generate_plain_xy(self):
        self.__gps_points_gdf[gps_field.PLAIN_X] = self.__gps_points_gdf[net_field.GEOMETRY_FIELD].apply(
            lambda geo: geo.x)
        self.__gps_points_gdf[gps_field.PLAIN_Y] = self.__gps_points_gdf[net_field.GEOMETRY_FIELD].apply(
            lambda geo: geo.y)

    def dense(self, dense_interval: float = 120.0):
        """

        :param dense_interval:
        :return:
        """
        if len(self.__gps_points_gdf) <= 1:
            return self
        # 时间差和距离差
        self.calc_adj_dis_gap()
        self.calc_adj_time_gap()
        not_same_agent_idx = self.not_same_agent_idx()
        should_be_dense_gdf = self.__gps_points_gdf[
            (self.__gps_points_gdf[dis_gap_field] > dense_interval) & ~not_same_agent_idx].copy()
        self.__gps_points_gdf.drop(columns=[next_time_field, next_p_field, time_gap_field, dis_gap_field], axis=1,
                                   inplace=True)
        if should_be_dense_gdf.empty:
            return self

        should_be_dense_gdf[n_seg_field] = (0.001 + should_be_dense_gdf[dis_gap_field] / dense_interval).astype(
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
        should_be_dense_gdf = should_be_dense_gdf.astype(self.__gps_points_gdf.dtypes)
        self.__gps_points_gdf[ori_seq_field] = self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD]
        should_be_dense_gdf[ori_seq_field] = -1
        self.__gps_points_gdf = pd.concat([self.__gps_points_gdf, should_be_dense_gdf])
        self.__gps_points_gdf.sort_values(by=[agent_field, time_field], ascending=[True, True], inplace=True)
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)

        self.add_seq_field(gps_points_gdf=self.__gps_points_gdf, multi_agents=self.multi_agents)

        if not self.__user_gps_info.empty:
            self.__user_gps_info[ori_seq_field] = self.__user_gps_info[gps_field.POINT_SEQ_FIELD]
            del self.__user_gps_info[gps_field.POINT_SEQ_FIELD]
            self.__user_gps_info = pd.merge(self.__user_gps_info, self.__gps_points_gdf[
                [ori_seq_field, gps_field.POINT_SEQ_FIELD, agent_field]], on=[ori_seq_field, agent_field],
                                            how='left')
            self.__user_gps_info[gps_field.POINT_SEQ_FIELD] = self.__user_gps_info[gps_field.POINT_SEQ_FIELD].fillna(
                -1).astype(int)
            del self.__user_gps_info[ori_seq_field]
        del self.__gps_points_gdf[ori_seq_field]
        return self

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

    def lower_frequency(self, lower_n: int = 2, multi_agents: bool = True):
        """
        data downscaling, reduce the amount of data
        :param lower_n: 降频倍数
        :param multi_agents:
        :return:
        """
        if multi_agents:
            self.__gps_points_gdf['label'] = self.__gps_points_gdf.groupby(agent_field).cumcount() % lower_n
            self.__gps_points_gdf = self.__gps_points_gdf[self.__gps_points_gdf['label'].eq(0)].copy()
        else:
            self.__gps_points_gdf['label'] = pd.Series([i for i in range(len(self.__gps_points_gdf))]) % lower_n
            self.__gps_points_gdf = self.__gps_points_gdf[self.__gps_points_gdf['label'].eq(0)].copy()
        del self.__gps_points_gdf['label']
        return self

    def kf_smooth(self, p_noise_std: list or float = 0.01, o_noise_std: list or float = 0.1):
        """
        use kalman filter to smooth the trajectory, fields: agent_id, time, lng, lat
        :param p_noise_std: standard deviation of process noise
        :param o_noise_std: standard deviation of observation noise
        the smaller o_noise_std is, the closer the smoothing result is to the observed trajectory
        :return:
        """
        tks = OffLineTrajectoryKF(trajectory_df=self.__gps_points_gdf,
                                  x_field=gps_field.PLAIN_X, y_field=gps_field.PLAIN_Y)
        self.__gps_points_gdf = tks.execute(p_noise_std=p_noise_std, o_noise_std=o_noise_std)
        self.__gps_points_gdf[geometry_field] = self.__gps_points_gdf[[gps_field.PLAIN_X, gps_field.PLAIN_Y]].apply(
            lambda x: Point(x), axis=1)
        return self

    def rolling_average(self, multi_agents: bool = True, rolling_window: int = 2):
        """
        sliding window noise reduction.
        after performing this operation, the output of the matching result table for custom fields is not supported.
        :param multi_agents:
        :param rolling_window:
        :return:
        """
        # 滑动窗口执行后会重置所有的gps的seq字段
        if len(self.__gps_points_gdf) <= rolling_window:
            return None

        self.__gps_points_gdf[gps_field.TIME_FIELD] = self.__gps_points_gdf[gps_field.TIME_FIELD].apply(
            lambda t: t.timestamp())
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)

        self.__gps_points_gdf = \
            self.__gps_points_gdf.groupby(agent_field)[
                [gps_field.PLAIN_X, gps_field.PLAIN_Y,
                 gps_field.TIME_FIELD]].rolling(window=rolling_window, min_periods=1).mean().reset_index(drop=False)
        try:
            del self.__gps_points_gdf['level_1']
        except Exception as e:
            print(repr(e))

        self.__gps_points_gdf.dropna(subset=[gps_field.PLAIN_X, gps_field.PLAIN_Y,
                                             gps_field.TIME_FIELD], inplace=True, how='any')
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)

        self.add_seq_field(gps_points_gdf=self.__gps_points_gdf, multi_agents=multi_agents)

        self.__gps_points_gdf[net_field.GEOMETRY_FIELD] = \
            self.__gps_points_gdf[[gps_field.PLAIN_X, gps_field.PLAIN_Y]].apply(lambda item: Point(item), axis=1)
        self.__gps_points_gdf[gps_field.TIME_FIELD] = \
            pd.to_datetime(self.__gps_points_gdf[gps_field.TIME_FIELD], unit='s')
        self.__gps_points_gdf = gpd.GeoDataFrame(self.__gps_points_gdf, geometry=gps_field.GEOMETRY_FIELD, crs=self.crs)
        self.__user_gps_info = self.__gps_points_gdf[[gps_field.AGENT_ID_FIELD, gps_field.POINT_SEQ_FIELD]].copy()
        self.__user_gps_info[gps_field.LOC_TYPE] = 's'
        self.user_filed_list = []
        return self

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
        try:
            gps_array_buffer = simplify_gps_route_l[0].buffer(buffer)
        except Exception as e:
            print(repr(e))
            return None
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
                print(repr(e), 'spatial layered association failure')
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
        self.__gps_points_gdf = self.__gps_points_gdf.to_crs(self.plane_crs)
        self.__crs = self.plane_crs

    def to_geo_prj(self) -> None:
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
        """only support one agent"""
        return self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].to_list()

    def del_dwell_points(self, dwell_l_length: float = 5.0, dwell_n: int = 2):
        """support multi agents"""
        # add field = dis_gap_field
        self.calc_adj_dis_gap()
        del self.__gps_points_gdf[next_p_field]
        self.__gps_points_gdf['dwell_label'] = \
            (self.__gps_points_gdf[dis_gap_field] > dwell_l_length).astype(int)
        del self.__gps_points_gdf[dis_gap_field]

        not_same_agent_idx = self.not_same_agent_idx()
        self.__gps_points_gdf.loc[not_same_agent_idx, 'dwell_label'] = 1
        self.__gps_points_gdf.loc[self.__gps_points_gdf.tail(1).index, 'dwell_label'] = 1
        self.__gps_points_gdf = self.del_consecutive_zero(df=self.__gps_points_gdf, col='dwell_label', n=dwell_n)
        del self.__gps_points_gdf['dwell_label']
        try:
            self.__gps_points_gdf.drop(columns=[sub_group_field], axis=1, inplace=True)
        except KeyError:
            pass
        return self

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

    @property
    def max_seq(self) -> int:
        return self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].max()

    def last_row(self, n) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[int]]:
        last_n_row = self.__gps_points_gdf.tail(n)
        last_seq = list(last_n_row[gps_field.POINT_SEQ_FIELD])

        return self.__gps_points_gdf[self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].isin(last_seq)][
            [gps_field.AGENT_ID_FIELD, gps_field.POINT_SEQ_FIELD, gps_field.TIME_FIELD,
             gps_field.PLAIN_X, gps_field.PLAIN_Y, gps_field.GEOMETRY_FIELD]], \
            self.__user_gps_info[self.__user_gps_info[gps_field.POINT_SEQ_FIELD].isin(last_seq)], last_seq

    def merge_gps(self, gps_obj=None, depth: int = 3, dis_gap_threshold: float = 600.0,
                  time_gap_threshold: float = 1800.0) -> tuple[bool, list[int]]:
        """only support one agent"""
        cr, ur, last_seq_list = gps_obj.last_row(depth)
        if not last_seq_list:
            return False, last_seq_list

        _idx = last_seq_list[-1]
        last_loc, last_time = cr.at[_idx, gps_field.GEOMETRY_FIELD], cr.at[_idx, gps_field.TIME_FIELD]
        _idx = list(self.__gps_points_gdf.index)[0]
        first_loc, first_time = self.__gps_points_gdf.at[_idx, gps_field.GEOMETRY_FIELD], self.__gps_points_gdf.at[
            _idx, gps_field.TIME_FIELD]
        time_gap, dis_gap = np.abs((first_time - last_time).seconds), first_loc.distance(last_loc)
        if time_gap >= time_gap_threshold or dis_gap >= dis_gap_threshold:
            return False,  last_seq_list
        else:
            max_seq = gps_obj.max_seq + 1
            self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = self.__gps_points_gdf[
                                                                   gps_field.POINT_SEQ_FIELD] + max_seq
            self.__user_gps_info[gps_field.POINT_SEQ_FIELD] = self.__user_gps_info[gps_field.POINT_SEQ_FIELD] + max_seq
            self.__source_gps_points_gdf[gps_field.POINT_SEQ_FIELD] = self.__source_gps_points_gdf[
                                                                          gps_field.POINT_SEQ_FIELD] + max_seq
            cr[time_field] = cr[time_field].astype(self.__gps_points_gdf[time_field].dtype)
            self.__gps_points_gdf = pd.concat([cr, self.__gps_points_gdf])
            self.__user_gps_info = pd.concat([ur, self.__user_gps_info])
            # self.__user_gps_info.reset_index(inplace=True, drop=True)
            # self.__gps_points_gdf.reset_index(inplace=True, drop=True)
            self.__gps_points_gdf.index = self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].values
            self.__user_gps_info.index = self.__user_gps_info[gps_field.POINT_SEQ_FIELD].values
            return True, last_seq_list

    def first_seq(self) -> int:
        """only support one agent"""
        return self.__gps_points_gdf.at[self.__gps_points_gdf.head(1).index[0], gps_field.POINT_SEQ_FIELD]

    def not_same_agent_idx(self) -> bool:
        next_agent_field = '__next'
        # 前序agent_id
        self.__gps_points_gdf[next_agent_field] = self.__gps_points_gdf[agent_field].shift(-1).fillna(
            self.__gps_points_gdf[agent_field]).astype(self.__gps_points_gdf[agent_field].dtype)
        not_same_agent = self.__gps_points_gdf[agent_field] != self.__gps_points_gdf[next_agent_field]
        del self.__gps_points_gdf[next_agent_field]
        return not_same_agent

    @staticmethod
    def add_seq_field(gps_points_gdf: gpd.GeoDataFrame or pd.DataFrame = None, multi_agents: bool = False):
        if not multi_agents:
            gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(gps_points_gdf))]
        else:
            gps_points_gdf[gps_field.POINT_SEQ_FIELD] = \
                gps_points_gdf.groupby(agent_field)[gps_field.TIME_FIELD].rank(method='min').astype(int) - 1

    def simplify_trajectory(self, l_threshold: float = 5.0):
        """
        simplify trajectories using Douglas Pecker's algorithm
        """
        agent_count = self.__gps_points_gdf.groupby(agent_field)[[time_field]].count().rename(columns={time_field: 'c'})
        one_agent = list(agent_count[agent_count['c'] <= 1].index)

        process_trajectory = self.__gps_points_gdf[~self.__gps_points_gdf[agent_field].isin(one_agent)].copy()
        process_trajectory.reset_index(inplace=True, drop=True)

        no_process_gps = self.__gps_points_gdf[self.__gps_points_gdf[agent_field].isin(one_agent)].copy()

        origin_crs = self.__gps_points_gdf.crs
        del self.__gps_points_gdf

        line_gdf = process_trajectory.groupby(agent_field)[[geometry_field]].agg(
            {geometry_field: list}).reset_index(drop=False)
        line_gdf[geometry_field] = line_gdf[geometry_field].apply(lambda p: LineString(p))
        line_gdf = gpd.GeoDataFrame(line_gdf, geometry=geometry_field, crs=origin_crs)
        line_gdf[geometry_field] = line_gdf[geometry_field].simplify(l_threshold)

        p_simplify_line = pd.merge(process_trajectory[[gps_field.AGENT_ID_FIELD]],
                                   line_gdf, on=gps_field.AGENT_ID_FIELD, how='left')
        p_simplify_line = gpd.GeoSeries(p_simplify_line[geometry_field])

        # origin point prj to simplify
        prj_p_array = p_simplify_line.project(process_trajectory[geometry_field])

        process_trajectory[geometry_field] = p_simplify_line.interpolate(prj_p_array)

        if not no_process_gps.empty:
            self.__gps_points_gdf = pd.concat([process_trajectory, no_process_gps])
        self.__gps_points_gdf = process_trajectory
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        return self

    def trajectory_data(self, export_crs: str = 'EPSG:4326', _type: str = "gdf") -> gpd.GeoDataFrame or pd.DataFrame:
        """get the """
        export_trajectory = self.__gps_points_gdf.copy()
        try:
            del export_trajectory[gps_field.PLAIN_X], export_trajectory[gps_field.PLAIN_Y]
        except Exception as e:
            print(repr(e))
        export_trajectory = export_trajectory.to_crs(export_crs)
        export_trajectory = pd.merge(export_trajectory, self.__user_gps_info,
                                     on=[agent_field, gps_field.POINT_SEQ_FIELD], how='left')
        export_trajectory[gps_field.LOC_TYPE] = export_trajectory[gps_field.LOC_TYPE].fillna('d')
        export_trajectory[lng_field] = export_trajectory[geometry_field].apply(lambda g: g.x)
        export_trajectory[lat_field] = export_trajectory[geometry_field].apply(lambda g: g.y)
        if _type == 'df':
            del export_trajectory[geometry_field]
            return export_trajectory
        return export_trajectory
