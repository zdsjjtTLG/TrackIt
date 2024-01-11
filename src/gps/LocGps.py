# -- coding: utf-8 --
# @Time    : 2023/12/10 20:12
# @Author  : TangKai
# @Team    : ZheChengData

"""车辆GPS数据的相关方法和属性"""

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
                 geo_crs: str = 'EPSG:4326', plane_crs: str = 'EPSG:32650'):
        self.geo_crs = geo_crs
        self.buffer = buffer
        self.plane_crs = plane_crs
        self.__gps_point_dis_dict = dict()
        self.__gps_points_gdf = gps_points_df
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

    def lower_frequency(self, n: int = 5):
        self.__gps_points_gdf['label'] = self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].apply(lambda x: x % n)
        self.__gps_points_gdf = self.__gps_points_gdf[self.__gps_points_gdf['label'] == 0].copy()
        self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(self.__gps_points_gdf))]
        self.__gps_points_gdf.reset_index(inplace=True, drop=True)
        self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD] = [i for i in range(len(self.__gps_points_gdf))]
        self.calc_gps_point_dis()
    def calc_gps_point_dis(self) -> None:
        seq_list = self.__gps_points_gdf[gps_field.POINT_SEQ_FIELD].to_list()
        self.__gps_point_dis_dict = {
            (seq_list[i], seq_list[i + 1]): self.__gps_points_gdf.at[seq_list[i], net_field.GEOMETRY_FIELD].distance(
                self.__gps_points_gdf.at[seq_list[i + 1], net_field.GEOMETRY_FIELD]) for i in
            range(len(self.__gps_points_gdf) - 1)}

    def get_gps_point_dis(self, adj_gps_seq: tuple = None):
        return self.__gps_point_dis_dict[adj_gps_seq]

    def plot_point(self):
        pass

    @property
    def gps_gdf(self) -> gpd.GeoDataFrame:
        return self.__gps_points_gdf.copy()

    def get_gps_buffer_gdf(self):
        pass

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
            pass
        else:
            self.__gps_points_gdf = self.__gps_points_gdf.to_crs(self.plane_crs)

    def to_geo_prj(self) -> None:
        if self.__gps_points_gdf.crs == self.geo_crs:
            pass
        else:
            self.__gps_points_gdf = self.__gps_points_gdf.to_crs(self.geo_crs)

    def get_point(self, seq: int = 0):
        return self.__gps_points_gdf.at[seq, gps_field.TIME_FIELD], self.__gps_points_gdf.at[seq, 'geometry']

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
    pass
