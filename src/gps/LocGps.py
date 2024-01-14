# -- coding: utf-8 --
# @Time    : 2023/12/10 20:12
# @Author  : TangKai
# @Team    : ZheChengData

"""车辆GPS数据的相关方法和属性"""
import datetime

import pandas as pd
import geopandas as gpd
from src.map.Net import Net
from shapely.geometry import LineString
from shapely.geometry import Point, Polygon
from src.GlobalVal import GpsField, NetField
from src.WrapsFunc import function_time_cost

gps_field = GpsField()
net_field = NetField()


class GpsPointsGdf(object):

    def __init__(self, gps_points_df: pd.DataFrame = None, lng_field: str = None, lat_field: str = None,
                 buffer: float = 200.0, time_format: str = '%Y-%m-%d %H:%M:%S',
                 geo_crs: str = 'EPSG:4326', plane_crs: str = 'EPSG:32649'):
        self.geo_crs = geo_crs
        self.buffer = buffer
        self.__crs = self.geo_crs
        self.plane_crs = plane_crs
        self.__gps_point_dis_dict = dict()
        self.__gps_points_gdf = gps_points_df
        if gps_field.HEADING_FIELD not in self.__gps_points_gdf.columns:
            self.__gps_points_gdf[gps_field.HEADING_FIELD] = 0.0
        self.__gps_points_gdf['geometry'] = self.__gps_points_gdf.apply(
            lambda item: Point(item[lng_field], item[lat_field]), axis=1)
        self.__gps_points_gdf = gpd.GeoDataFrame(self.__gps_points_gdf, geometry='geometry', crs=self.geo_crs)

        self.__gps_points_gdf[gps_field.TIME_FIELD] = \
            pd.to_datetime(self.__gps_points_gdf[gps_field.TIME_FIELD], format=time_format)
        self.__gps_points_gdf.sort_values(by=[gps_field.TIME_FIELD], ascending=[True], inplace=True)
        self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(self.__gps_points_gdf))]
        self.__gps_points_gdf[gps_field.ORIGIN_POINT_SEQ_FIELD] = self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD]

        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        self.to_plane_prj()
        self.calc_gps_point_dis()

        # 存储最原始的GPS信息(未经过降噪)
        self.__source_gps_points_gdf = None

    def lower_frequency(self, n: int = 5):
        """
        GPS数据降频
        :param n:
        :return:
        """
        if self.__source_gps_points_gdf is None:
            self.__source_gps_points_gdf = self.__gps_points_gdf.copy()
        self.__gps_points_gdf['label'] = self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].apply(lambda x: x % n)
        self.__gps_points_gdf = self.__gps_points_gdf[self.__gps_points_gdf['label'] == 0].copy()
        self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(self.__gps_points_gdf))]
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(self.__gps_points_gdf))]
        self.__gps_points_gdf.drop(columns=['label'], axis=1, inplace=True)
        self.calc_gps_point_dis()

    def neighboring_average(self):
        """滑动窗口降噪"""
        if self.__source_gps_points_gdf is None:
            self.__source_gps_points_gdf = self.__gps_points_gdf.copy()
        self.__gps_points_gdf = pd.concat([self.__gps_points_gdf.loc[[0], :],
                                           self.__gps_points_gdf,
                                           self.__gps_points_gdf.loc[[len(self.__gps_points_gdf) - 1], :]])
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        self.__gps_points_gdf[gps_field.LNG_FIELD] = self.__gps_points_gdf['geometry'].apply(lambda geo: geo.x)
        self.__gps_points_gdf[gps_field.LAT_FIELD] = self.__gps_points_gdf['geometry'].apply(lambda geo: geo.y)

        self.__gps_points_gdf[['next_x', 'next_y', 'next_time']] = \
            self.__gps_points_gdf[
                [gps_field.LNG_FIELD, gps_field.LAT_FIELD, gps_field.TIME_FIELD]].shift(-1)
        self.__gps_points_gdf.dropna(subset='next_x', inplace=True)

        self.__gps_points_gdf[gps_field.LNG_FIELD] = (self.__gps_points_gdf['next_x'] + self.__gps_points_gdf[
            gps_field.LNG_FIELD]) / 2
        self.__gps_points_gdf[gps_field.LAT_FIELD] = (self.__gps_points_gdf['next_y'] + self.__gps_points_gdf[
            gps_field.LAT_FIELD]) / 2
        self.__gps_points_gdf[gps_field.TIME_FIELD] = self.__gps_points_gdf[gps_field.TIME_FIELD] + (
                    self.__gps_points_gdf['next_time'] - self.__gps_points_gdf[gps_field.TIME_FIELD]) / 2
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(self.__gps_points_gdf))]
        self.__gps_points_gdf['geometry'] = self.__gps_points_gdf[[gps_field.LNG_FIELD, gps_field.LAT_FIELD]].apply(
            lambda item: Point(item), axis=1)
        self.__gps_points_gdf.drop(columns=['next_x', 'next_y', 'next_time'], axis=1, inplace=True)
        self.calc_gps_point_dis()

    def rolling_average(self, window: int = 2):
        """滑动窗口降噪"""
        if self.__source_gps_points_gdf is None:
            self.__source_gps_points_gdf = self.__gps_points_gdf.copy()
        self.__gps_points_gdf[gps_field.TIME_FIELD] = self.__gps_points_gdf[gps_field.TIME_FIELD].apply(
            lambda t: t.timestamp())
        self.__gps_points_gdf[gps_field.LNG_FIELD] = self.__gps_points_gdf[net_field.GEOMETRY_FIELD].apply(
            lambda geo: geo.x)
        self.__gps_points_gdf[gps_field.LAT_FIELD] = self.__gps_points_gdf[net_field.GEOMETRY_FIELD].apply(
            lambda geo: geo.y)

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
        print(self.__gps_points_gdf.crs)
        self.calc_gps_point_dis()

    def dwell_point_processing(self, buffer: float = 25.0):
        """识别停留点, 去除多余的停留点GPS信息"""
        if self.__source_gps_points_gdf is None:
            self.__source_gps_points_gdf = self.__gps_points_gdf.copy()
        # TO DO ......
        self.calc_gps_point_dis()
        pass

    def calc_gps_point_dis(self) -> None:
        seq_list = self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].to_list()
        self.__gps_point_dis_dict = {
            (seq_list[i], seq_list[i + 1]): self.__gps_points_gdf.at[seq_list[i], net_field.GEOMETRY_FIELD].distance(
                self.__gps_points_gdf.at[seq_list[i + 1], net_field.GEOMETRY_FIELD]) for i in
            range(len(self.__gps_points_gdf) - 1)}

    def get_gps_point_dis(self, adj_gps_seq: tuple = None):
        try:
            dis = self.__gps_point_dis_dict[adj_gps_seq]
        # some gps points do not have any candidate links
        except KeyError:
            dis = self.__gps_points_gdf.at[adj_gps_seq[0], net_field.GEOMETRY_FIELD].distance(
                self.__gps_points_gdf.at[adj_gps_seq[1], net_field.GEOMETRY_FIELD])
            self.__gps_point_dis_dict[adj_gps_seq] = dis
        return dis

    def plot_point(self):
        pass

    def get_gps_buffer_gdf(self):
        pass

    @property
    def gps_gdf(self) -> gpd.GeoDataFrame:
        return self.__gps_points_gdf.copy()

    @property
    def crs(self):
        return self.__crs

    def get_gps_array_buffer(self, buffer: float = 200.0) -> Polygon:
        """输出gps路径的buffer范围面域"""
        gps_array_buffer = LineString(self.__gps_points_gdf['geometry'].to_list()).buffer(buffer)
        return gps_array_buffer

    @function_time_cost
    def generate_candidate_link(self, net: Net = None):
        gps_buffer_gdf = self.__gps_points_gdf[[gps_field.POINT_SEQ_FIELD, 'geometry']].copy()
        if gps_buffer_gdf.crs != self.plane_crs:
            gps_buffer_gdf = gps_buffer_gdf.to_crs(self.plane_crs)
        gps_buffer_gdf['geometry'] = gps_buffer_gdf['geometry'].apply(lambda point_geo: point_geo.buffer(self.buffer))
        candidate_link = gpd.sjoin(gps_buffer_gdf,
                                   net.get_link_data()[[net_field.SINGLE_LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                                                        net_field.TO_NODE_FIELD, net_field.DIRECTION_FIELD,
                                                        net_field.LENGTH_FIELD, net_field.GEOMETRY_FIELD]])
        candidate_link.drop(columns=['index_right'], axis=1, inplace=True)
        candidate_link.reset_index(inplace=True, drop=True)
        return candidate_link

    def to_plane_prj(self) -> None:
        if self.__gps_points_gdf.crs == self.plane_crs:
            self.__crs = self.plane_crs
            pass
        else:
            self.__gps_points_gdf = self.__gps_points_gdf.to_crs(self.plane_crs)
            self.__crs = self.plane_crs

    def to_geo_prj(self) -> None:
        if self.__gps_points_gdf.crs == self.geo_crs:
            self.__crs = self.geo_crs
            pass
        else:
            self.__gps_points_gdf = self.__gps_points_gdf.to_crs(self.geo_crs)
            self.__crs = self.geo_crs

    def get_point(self, seq: int = 0):
        return self.__gps_points_gdf.at[seq, gps_field.TIME_FIELD], self.__gps_points_gdf.at[seq, 'geometry']

    @property
    def source_gps(self):
        if self.__source_gps_points_gdf is None:
            return self.__gps_points_gdf.copy()
        else:
            return self.__source_gps_points_gdf.copy()

    def get_prj_inf(self, line: LineString, seq: int = 0) -> tuple[Point, float, float, float]:
        """
        计算当前gps点实例在指定线对象上的投影信息
        :param line:
        :param seq:
        :return:
        """
        (prj_p, prj_dis, route_dis, l_length) = self._get_prj_inf(self.get_point(seq)[1], line)
        return prj_p, prj_dis, route_dis, l_length

    @staticmethod
    def _get_prj_inf(gps_point: Point = None, line: LineString = None) -> tuple[Point, float, float, float]:
        """
        # 返回 (GPS投影点坐标, GPS点到投影点的直线距离, GPS投影点到line拓扑起点的路径距离, line的长度)
        :param gps_point:
        :param line:
        :return: (GPS投影点坐标, GPS点到投影点的直线距离, GPS投影点到line拓扑起点的路径距离)
        """

        distance = line.project(gps_point)

        if distance <= 0.0:
            prj_p = Point(list(line.coords)[0])
            return prj_p, prj_p.distance(gps_point), distance, line.length
        elif distance >= line.length:
            prj_p = Point(list(line.coords)[-1])
            return prj_p, prj_p.distance(gps_point), distance, line.length
        else:
            coords = list(line.coords)
            for i, p in enumerate(coords):
                xd = line.project(Point(p))
                if xd == distance:
                    prj_p = Point(coords[i])
                    return prj_p, prj_p.distance(gps_point), distance, line.length
                if xd > distance:
                    cp = line.interpolate(distance)
                    prj_p = Point((cp.x, cp.y))
                    return prj_p, prj_p.distance(gps_point), distance, line.length

    @property
    def gps_list_length(self):
        return len(self.__gps_points_gdf)


if __name__ == '__main__':
    from datetime import timedelta

    df = pd.DataFrame({'val': [1,2,3,4,5,6,7]})
    df['val1'] = [1,2,3,4,5,6,7]
    df['time'] = [datetime.datetime.now() + timedelta(seconds=i * 10) for i in range(1, len(df) + 1)]
    result = df[['val', 'val1']].rolling(window=2).mean()
    print(result)
    print(result.at[1, 'val'])
    # print(df)
    # df['time'] = df['time'].apply(lambda x: x.timestamp())
    # result = df['val'].rolling(window=2).mean()
    # result = df['time'].rolling(window=2).mean()
    # print(result)
    #
    # df['time'] = result
    # df['time'] = pd.to_datetime(df['time'], unit='s')
    # print(df)


