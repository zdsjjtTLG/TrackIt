# -- coding: utf-8 --
# @Time    : 2023/12/10 20:12
# @Author  : TangKai
# @Team    : ZheChengData

"""车辆GPS数据的相关方法和属性"""

import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from ..map.Net import Net
from datetime import timedelta
from ..tools.geo_process import prj_inf
from ..tools.geo_process import segmentize
from ..GlobalVal import GpsField, NetField, PrjConst
from shapely.geometry import Point, Polygon, LineString


gps_field = GpsField()
net_field = NetField()
prj_const = PrjConst()

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
diff_vec = gps_field.DIFF_VEC
geo_crs = prj_const.PRJ_CRS
sub_group_field = gps_field.SUB_GROUP_FIELD


class GpsPointsGdf(object):

    def __init__(self, gps_points_df: pd.DataFrame = None,
                 buffer: float = 200.0, increment_buffer: float = 20.0, max_increment_times: int = 10,
                 time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's',
                 plane_crs: str = 'EPSG:32649', dense_gps: bool = True, dense_interval: float = 25.0,
                 dwell_l_length: float = 10.0, dwell_n: int = 3):
        """

        :param gps_points_df: gps数据dataframe, agent_id, lng, lat, time
        :param buffer: GPS点的buffer半径大小(用于生成候选路段), m
        :param increment_buffer: 使用buffer进行关联, 可能会存在部分GPS点仍然关联不到任何路段, 对于这部分路段, 将启用增量buffer进一步关联
        :param max_increment_times: 增量搜索的最大次数
        :param time_format: 时间列的字符格式
        :param plane_crs: 平面投影坐标系
        :param dense_gps: 是否加密GPS点
        :param dwell_l_length: 停留点识别距离阈值
        :param dwell_n: 连续dwell_n个点的距离小于dwell_l_length就被识别为停留点
        :param dense_interval: 加密间隔(相邻GPS点的直线距离小于dense_interval即会进行加密)
        """
        self.geo_crs = geo_crs
        self.buffer = buffer
        self.__crs = self.geo_crs
        self.plane_crs = plane_crs
        self.increment_buffer = increment_buffer
        self.dense_gps = dense_gps
        self.dense_interval = dense_interval
        self.max_increment_times = 1 if max_increment_times <= 0 else max_increment_times
        self.dwell_l_length = dwell_l_length
        self.dwell_n = dwell_n
        self.__gps_point_dis_dict = dict()
        gps_points_df.reset_index(inplace=True, drop=True)
        self.__gps_points_gdf = gps_points_df
        self.agent_id = self.__gps_points_gdf.at[0, gps_field.AGENT_ID_FIELD]
        self.check()
        if gps_field.HEADING_FIELD not in self.__gps_points_gdf.columns:
            self.__gps_points_gdf[gps_field.HEADING_FIELD] = 0.0
        self.__gps_points_gdf[gps_field.GEOMETRY_FIELD] = self.__gps_points_gdf.apply(
            lambda item: Point(item[gps_field.LNG_FIELD], item[gps_field.LAT_FIELD]), axis=1)
        self.__gps_points_gdf = gpd.GeoDataFrame(self.__gps_points_gdf, geometry=gps_field.GEOMETRY_FIELD,
                                                 crs=self.geo_crs)
        try:
            self.__gps_points_gdf[gps_field.TIME_FIELD] = \
                pd.to_datetime(self.__gps_points_gdf[gps_field.TIME_FIELD], format=time_format)
        except ValueError:
            self.__gps_points_gdf[gps_field.TIME_FIELD] = \
                pd.to_datetime(self.__gps_points_gdf[gps_field.TIME_FIELD], unit=time_unit)
        self.__gps_points_gdf.sort_values(by=[gps_field.TIME_FIELD], ascending=[True], inplace=True)
        self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(self.__gps_points_gdf))]
        self.__gps_points_gdf[gps_field.ORIGIN_POINT_SEQ_FIELD] = self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD]
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        self.to_plane_prj()
        # self.calc_gps_point_dis()

        # 存储最原始的GPS信息
        self.__source_gps_points_gdf = self.__gps_points_gdf.copy()

        self.done_diff_heading = False

    def check(self):
        assert {gps_field.LNG_FIELD, gps_field.LAT_FIELD,
                gps_field.AGENT_ID_FIELD, gps_field.TIME_FIELD}.issubset(
            set(self.__gps_points_gdf.columns)), \
            rf'GPS数据字段有误, 请至少包含如下字段: {gps_field.AGENT_ID_FIELD, gps_field.LNG_FIELD, gps_field.LAT_FIELD, gps_field.TIME_FIELD}'

    def dense(self) -> None:
        if len(self.__gps_points_gdf) <= 1:
            return None
        # 时间差和距离差
        self.calc_adj_dis_gap()
        self.calc_adj_time_gap()

        should_be_dense_gdf = self.__gps_points_gdf[self.__gps_points_gdf[dis_gap_field] > self.dense_interval].copy()
        if should_be_dense_gdf.empty:
            return None
        self.__gps_points_gdf.drop(columns=[next_time_field, next_p_field, time_gap_field, dis_gap_field], axis=1,
                                   inplace=True)

        should_be_dense_gdf[n_seg_field] = should_be_dense_gdf.apply(
            lambda row: int(0.001 + row[dis_gap_field] / self.dense_interval) + 1, axis=1)

        should_be_dense_gdf[[dense_geo_field, time_field]] = \
            should_be_dense_gdf.apply(
                lambda row: [
                    list(segmentize(LineString([row[geometry_field], row[next_p_field]]), n=row[n_seg_field]).coords[
                         1:-1]),
                    [row[time_field] + timedelta(seconds=i * row[time_gap_field] / row[n_seg_field]) for i in
                     range(1, row[n_seg_field])]],
                axis=1, result_type='expand')
        should_be_dense_gdf.drop(columns=[geometry_field], axis=1, inplace=True)
        should_be_dense_gdf = pd.DataFrame(should_be_dense_gdf).explode(column=[time_field, dense_geo_field],
                                                                        ignore_index=True)
        should_be_dense_gdf.rename(columns={dense_geo_field: geometry_field}, inplace=True)
        should_be_dense_gdf[geometry_field] = should_be_dense_gdf.apply(lambda row: Point(row[geometry_field]), axis=1)
        should_be_dense_gdf.drop(columns=[next_time_field, next_p_field, time_gap_field, dis_gap_field], axis=1,
                                 inplace=True)
        should_be_dense_gdf = gpd.GeoDataFrame(should_be_dense_gdf, geometry=geometry_field, crs=self.crs)
        should_be_dense_gdf = should_be_dense_gdf.to_crs(self.geo_crs)
        should_be_dense_gdf[[lng_field, lat_field]] = should_be_dense_gdf.apply(
            lambda row: (row[geometry_field].x, row[geometry_field].y), axis=1, result_type='expand')
        should_be_dense_gdf = should_be_dense_gdf.to_crs(self.plane_crs)

        self.__gps_points_gdf = pd.concat([self.__gps_points_gdf, should_be_dense_gdf])
        self.__gps_points_gdf.sort_values(by=time_field, ascending=True, inplace=True)
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(self.__gps_points_gdf))]
        self.__gps_points_gdf[gps_field.ORIGIN_POINT_SEQ_FIELD] = self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD]
        # x = self.__gps_points_gdf.to_crs('EPSG:4326')
        # x[['lng', 'lat']] = x.apply(lambda row: (row['geometry'].x, row['geometry'].y), result_type='expand', axis=1)
        # x[[agent_field, time_field, lng_field, lat_field]].to_csv(r'dense_sz.csv', encoding='utf_8_sig', index=False)

    def calc_adj_dis_gap(self) -> None:
        # 距离差
        self.__gps_points_gdf[next_p_field] = self.__gps_points_gdf[geometry_field].shift(-1).fillna(
            self.__gps_points_gdf[geometry_field])
        # self.__gps_points_gdf[dis_gap_field] = self.__gps_points_gdf.apply(
        #     lambda row: row[next_p_field].distance(row[geometry_field]), axis=1)
        self.__gps_points_gdf[dis_gap_field] = self.__gps_points_gdf[next_p_field].distance(
            self.__gps_points_gdf[geometry_field])

    def calc_adj_time_gap(self) -> None:
        # 时间差
        self.__gps_points_gdf[next_time_field] = self.__gps_points_gdf[time_field].shift(-1).fillna(
            self.__gps_points_gdf[time_field])
        # self.__gps_points_gdf[time_gap_field] = self.__gps_points_gdf.apply(
        #     lambda row: (row[next_time_field] - row[time_field]).seconds, axis=1)
        self.__gps_points_gdf[time_gap_field] = self.__gps_points_gdf[next_time_field] - self.__gps_points_gdf[
            time_field]
        self.__gps_points_gdf[time_gap_field] = self.__gps_points_gdf[time_gap_field].apply(lambda x: x.seconds)

    def calc_pre_next_dis(self) -> pd.DataFrame():
        self.calc_adj_dis_gap()
        # next_seq
        res = self.__gps_points_gdf.copy()
        res[next_seq_field] = res[gps_field.POINT_SEQ_FIELD].shift(-1)
        res.dropna(subset=[next_seq_field], inplace=True)
        res[next_seq_field] = res[next_seq_field].astype(int)
        return res[[gps_field.POINT_SEQ_FIELD, next_seq_field, gps_field.ADJ_DIS]]

    def lower_frequency(self, n: int = 5):
        """
        GPS数据降频
        :param n: 降频倍数
        :return:
        """
        self.__gps_points_gdf['label'] = pd.Series([i for i in range(len(self.__gps_points_gdf))]) % n
        self.__gps_points_gdf = self.__gps_points_gdf[self.__gps_points_gdf['label'].eq(0)].copy()
        self.__gps_points_gdf.drop(columns=['label'], axis=1, inplace=True)

    def rolling_average(self, window: int = 2):
        """
        滑动窗口降噪
        :param window: 窗口大小
        :return:
        """
        if len(self.__gps_points_gdf) <= window:
            return None
        self.__gps_points_gdf[gps_field.TIME_FIELD] = self.__gps_points_gdf[gps_field.TIME_FIELD].apply(
            lambda t: t.timestamp())
        self.__gps_points_gdf[gps_field.LNG_FIELD] = self.__gps_points_gdf[net_field.GEOMETRY_FIELD].apply(
            lambda geo: geo.x)
        self.__gps_points_gdf[gps_field.LAT_FIELD] = self.__gps_points_gdf[net_field.GEOMETRY_FIELD].apply(
            lambda geo: geo.y)

        # 滑动窗口执行后会重置所有的gps的seq字段
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)

        rolling_num_df = self.__gps_points_gdf[
            [gps_field.LNG_FIELD, gps_field.LAT_FIELD, gps_field.TIME_FIELD]].rolling(window=window).mean()
        rolling_heading_df = self.__gps_points_gdf[[gps_field.HEADING_FIELD]].rolling(window=window).median()

        rolling_heading_df.dropna(subset=[gps_field.HEADING_FIELD], inplace=True)
        rolling_num_df.dropna(subset=[gps_field.LNG_FIELD, gps_field.LAT_FIELD, gps_field.TIME_FIELD], inplace=True,
                              how='any')

        rolling_average_df = pd.concat([rolling_num_df, rolling_heading_df], axis=1)
        rolling_average_df[gps_field.AGENT_ID_FIELD] = self.__gps_points_gdf.at[0, gps_field.AGENT_ID_FIELD]
        del rolling_heading_df, rolling_num_df

        self.__gps_points_gdf = pd.concat([self.__gps_points_gdf.loc[[0], :],
                                           rolling_average_df,
                                           self.__gps_points_gdf.loc[[len(self.__gps_points_gdf) - 1], :]])
        del rolling_average_df
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(self.__gps_points_gdf))]

        self.__gps_points_gdf[net_field.GEOMETRY_FIELD] = self.__gps_points_gdf[
            [gps_field.LNG_FIELD, gps_field.LAT_FIELD]].apply(
            lambda item: Point(item), axis=1)
        self.__gps_points_gdf[gps_field.TIME_FIELD] = pd.to_datetime(self.__gps_points_gdf[gps_field.TIME_FIELD],
                                                                     unit='s')
        self.__gps_points_gdf[gps_field.TIME_FIELD] = pd.to_datetime(self.__gps_points_gdf[gps_field.TIME_FIELD],
                                                                     format='%Y-%m-%d %H:%M:%S')

    def calc_gps_point_dis(self) -> None:
        seq_list = self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].to_list()
        # {(from_seq, to_seq): dis, ...}
        self.__gps_point_dis_dict = {
            (seq_list[i], seq_list[i + 1]): self.__gps_points_gdf.at[seq_list[i], gps_field.GEOMETRY_FIELD].distance(
                self.__gps_points_gdf.at[seq_list[i + 1], gps_field.GEOMETRY_FIELD]) for i in
            range(len(self.__gps_points_gdf) - 1)}

    def get_gps_point_dis(self, adj_gps_seq: tuple = None) -> float:
        try:
            dis = self.__gps_point_dis_dict[adj_gps_seq]
        # some gps points do not have any candidate links
        except KeyError:
            dis = self.__gps_points_gdf.at[adj_gps_seq[0], gps_field.GEOMETRY_FIELD].distance(
                self.__gps_points_gdf.at[adj_gps_seq[1], gps_field.GEOMETRY_FIELD])
            self.__gps_point_dis_dict[adj_gps_seq] = dis
        return dis

    def get_adj_gps_dis_df(self):
        return self.__gps_point_dis_dict

    def plot_point(self):
        pass

    def get_gps_buffer_gdf(self):
        pass

    def calc_diff_heading(self):
        if self.done_diff_heading:
            return None

        self.__gps_points_gdf[next_p_field] = self.__gps_points_gdf[geometry_field].shift(-1).fillna(
            self.__gps_points_gdf[geometry_field])
        self.__gps_points_gdf[pre_p_field] = self.__gps_points_gdf[geometry_field].shift(1).fillna(
            self.__gps_points_gdf[geometry_field])

        self.__gps_points_gdf['next_loc'] = self.__gps_points_gdf.apply(
            lambda row: np.array([row[next_p_field].x, row[next_p_field].y]), axis=1)
        self.__gps_points_gdf['pre_loc'] = self.__gps_points_gdf.apply(
            lambda row: np.array([row[pre_p_field].x, row[pre_p_field].y]), axis=1)

        # self.__gps_points_gdf['loc'] = self.__gps_points_gdf.apply(
        #     lambda row: np.array([row[geometry_field].x, row[geometry_field].y]), axis=1)
        # self.__gps_points_gdf[diff_vec] = self.__gps_points_gdf.apply(
        #     lambda row: (row['next_loc'] - row['loc'] + row['loc'] - row['pre_loc']) / 2,
        #     axis=1)

        self.__gps_points_gdf[diff_vec] = (self.__gps_points_gdf['next_loc'] - self.__gps_points_gdf['pre_loc']) / 2

        self.__gps_points_gdf.drop(
            columns=[next_p_field, pre_p_field, 'next_loc', 'pre_loc'], axis=1, inplace=True)
        self.done_diff_heading = True

    @property
    def gps_gdf(self) -> gpd.GeoDataFrame:
        return self.__gps_points_gdf.copy()

    @property
    def crs(self):
        return self.__crs

    def get_gps_array_buffer(self, buffer: float = 200.0, dup_threshold: float = 10.0) -> Polygon or None:
        """输出gps路径的buffer范围面域"""
        gps_route_l = gpd.GeoSeries(LineString(self.__gps_points_gdf[gps_field.GEOMETRY_FIELD].to_list()))
        simplify_gps_route_l = gps_route_l.remove_repeated_points(dup_threshold)
        gps_array_buffer = simplify_gps_route_l[0].buffer(buffer)
        return gps_array_buffer

    def generate_candidate_link(self, net: Net = None) -> tuple[pd.DataFrame, list[int]]:
        """
        计算GPS观测点的候选路段
        :param net:
        :return: GPS候选路段信息, 未匹配到候选路段的gps点id
        """
        gps_buffer_gdf = self.__gps_points_gdf[[gps_field.POINT_SEQ_FIELD, gps_field.GEOMETRY_FIELD]].copy()
        if gps_buffer_gdf.crs.srs.upper() != self.plane_crs:
            gps_buffer_gdf = gps_buffer_gdf.to_crs(self.plane_crs)

        single_link_gdf = net.get_link_data()[[net_field.SINGLE_LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                                               net_field.TO_NODE_FIELD, net_field.DIRECTION_FIELD,
                                               net_field.LENGTH_FIELD, net_field.GEOMETRY_FIELD]]
        single_link_gdf.reset_index(inplace=True, drop=True)
        candidate_link = pd.DataFrame()
        remain_gps_list = []
        for i in [i for i in range(0, self.max_increment_times)]:
            now_buffer = self.buffer + i * self.increment_buffer
            print(rf'buffer: {now_buffer}m...')

            gps_buffer_gdf['gps_buffer'] = gps_buffer_gdf[net_field.GEOMETRY_FIELD].apply(lambda p: p.buffer(now_buffer))
            gps_buffer_gdf.set_geometry('gps_buffer', inplace=True, crs=gps_buffer_gdf.crs)
            join_df = gpd.sjoin(gps_buffer_gdf, single_link_gdf, how='left')

            _candidate_link = join_df[~join_df[net_field.SINGLE_LINK_ID_FIELD].isna()]
            candidate_link = pd.concat([candidate_link, _candidate_link])
            del _candidate_link
            remain_gps_list = list(join_df[join_df[net_field.SINGLE_LINK_ID_FIELD].isna()][gps_field.POINT_SEQ_FIELD].unique())
            if not remain_gps_list:
                break

            gps_buffer_gdf = gps_buffer_gdf[gps_buffer_gdf[gps_field.POINT_SEQ_FIELD].isin(remain_gps_list)].copy()
        if not candidate_link.empty:
            candidate_link.drop(columns=['index_right', 'gps_buffer'], axis=1, inplace=True)
            for col in [net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD,
                        net_field.SINGLE_LINK_ID_FIELD, net_field.DIRECTION_FIELD]:
                candidate_link[col] = candidate_link[col].astype(int)
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
            pass
        else:
            self.__gps_points_gdf = self.__gps_points_gdf.to_crs(self.plane_crs)
            self.__crs = self.plane_crs

    def to_geo_prj(self) -> None:
        if self.__gps_points_gdf.crs.srs.upper() == self.geo_crs:
            self.__crs = self.geo_crs
            pass
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

    def get_prj_inf(self, line: LineString, seq: int = 0) -> tuple[Point, float, float, float, np.ndarray]:
        """
        计算当前gps点实例在指定线对象上的投影信息
        :param line:
        :param seq:
        :return:
        """
        (prj_p, prj_dis, route_dis, l_length, p_vec) = self._get_prj_inf(self.get_point(seq)[1], line)
        return prj_p, prj_dis, route_dis, l_length, p_vec

    def delete_target_gps(self, target_seq_list: list[int]) -> None:
        self.__gps_points_gdf.drop(
            index=self.__gps_points_gdf[self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].isin(target_seq_list)].index,
            inplace=True, axis=0)

    @staticmethod
    def _get_prj_inf(gps_point: Point = None, line: LineString = None) -> tuple[Point, float, float, float, np.ndarray]:
        """
        # 返回 (GPS投影点坐标, GPS点到投影点的直线距离, GPS投影点到line拓扑起点的路径距离, line的长度)
        :param gps_point:
        :param line:
        :return: (GPS投影点坐标, GPS点到投影点的直线距离, GPS投影点到line拓扑起点的路径距离, line的长度)
        """
        prj_p, p_prj_l, prj_route_l, line_length, _, prj_vec = prj_inf(p=gps_point, line=line)
        return prj_p, p_prj_l, prj_route_l, line_length, prj_vec

    @property
    def gps_list_length(self) -> int:
        return len(self.__gps_points_gdf)

    @property
    def used_observation_seq_list(self) -> list[int]:
        return self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].to_list()

    def del_dwell_points(self) -> None:
        # add field = dis_gap_field
        self.calc_adj_dis_gap()
        self.__gps_points_gdf['dwell_label'] = \
            (self.__gps_points_gdf[dis_gap_field] > self.dwell_l_length).astype(int)
        self.__gps_points_gdf.loc[len(self.__gps_points_gdf) - 1, 'dwell_label'] = 1
        self.__gps_points_gdf = self.del_consecutive_zero(df=self.__gps_points_gdf, col='dwell_label', n=self.dwell_n)
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



