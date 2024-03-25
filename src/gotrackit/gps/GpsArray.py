# -- coding: utf-8 --
# @Time    : 2024/3/25 15:05
# @Author  : TangKai
# @Team    : ZheChengData


import pandas as pd
import geopandas as gpd
from ..tools.geo_process import prj_inf
from ..GlobalVal import GpsField, NetField
from shapely.geometry import Point, LineString

gps_field = GpsField()
net_field = NetField()

lng_field = gps_field.LNG_FIELD
lat_field = gps_field.LAT_FIELD
time_field = gps_field.TIME_FIELD
agent_field = gps_field.AGENT_ID_FIELD
geometry_field = gps_field.GEOMETRY_FIELD


class GpsArray(object):

    def __init__(self, gps_points_df: pd.DataFrame = None,
                 time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's', geo_crs: str = 'EPSG:4326',
                 plane_crs: str = 'EPSG:32649'):
        """
        :param gps_points_df: gps数据dataframe, agent_id, lng, lat, time
        :param time_format: 时间列的字符格式
        :param geo_crs: 地理坐标系
        :param plane_crs: 平面投影坐标系
        """
        self.geo_crs = geo_crs
        self.__crs = self.geo_crs
        self.plane_crs = plane_crs
        self.gps_point_dis_dict = dict()
        self.gps_points_gdf = gps_points_df
        self.check()
        if gps_field.HEADING_FIELD not in self.gps_points_gdf.columns:
            self.gps_points_gdf[gps_field.HEADING_FIELD] = 0.0
        self.gps_points_gdf[gps_field.GEOMETRY_FIELD] = self.gps_points_gdf.apply(
            lambda item: Point(item[gps_field.LNG_FIELD], item[gps_field.LAT_FIELD]), axis=1)
        self.gps_points_gdf = gpd.GeoDataFrame(self.gps_points_gdf, geometry=gps_field.GEOMETRY_FIELD,
                                               crs=self.geo_crs)
        try:
            self.gps_points_gdf[gps_field.TIME_FIELD] = \
                pd.to_datetime(self.gps_points_gdf[gps_field.TIME_FIELD], format=time_format)
        except Exception as e:
            print(repr(e))
            self.gps_points_gdf[gps_field.TIME_FIELD] = \
                pd.to_datetime(self.gps_points_gdf[gps_field.TIME_FIELD], unit=time_unit)
        self.gps_points_gdf.sort_values(by=[gps_field.TIME_FIELD], ascending=[True], inplace=True)
        self.gps_points_gdf.reset_index(inplace=True, drop=True)
        self.to_plane_prj()

        # 存储最原始的GPS信息
        self.__source_gps_points_gdf = self.gps_points_gdf.copy()

    def check(self):
        _gap = {agent_field, lng_field, lat_field, time_field} - set(self.gps_gdf.columns)
        assert _gap == set(), rf'GPS数据缺少{_gap}字段'

    @property
    def gps_gdf(self) -> gpd.GeoDataFrame:
        return self.gps_points_gdf.copy()

    @property
    def crs(self):
        return self.__crs

    def to_plane_prj(self) -> None:
        if self.gps_points_gdf.crs == self.plane_crs:
            self.__crs = self.plane_crs
            pass
        else:
            self.gps_points_gdf = self.gps_points_gdf.to_crs(self.plane_crs)
            self.__crs = self.plane_crs

    def to_geo_prj(self) -> None:
        if self.gps_points_gdf.crs == self.geo_crs:
            self.__crs = self.geo_crs
            pass
        else:
            self.gps_points_gdf = self.gps_points_gdf.to_crs(self.geo_crs)
            self.__crs = self.geo_crs

    @property
    def source_gps(self) -> gpd.GeoDataFrame:
        if self.__source_gps_points_gdf is None:
            return self.gps_points_gdf.copy()
        else:
            return self.__source_gps_points_gdf.copy()

    @staticmethod
    def _get_prj_inf(gps_point: Point = None, line: LineString = None) -> tuple[Point, float, float, float]:
        """
        # 返回 (GPS投影点坐标, GPS点到投影点的直线距离, GPS投影点到line拓扑起点的路径距离, line的长度)
        :param gps_point:
        :param line:
        :return: (GPS投影点坐标, GPS点到投影点的直线距离, GPS投影点到line拓扑起点的路径距离, line的长度)
        """
        prj_p, p_prj_l, prj_route_l, line_length, _ = prj_inf(p=gps_point, line=line)
        return prj_p, p_prj_l, prj_route_l, line_length

    @property
    def gps_list_length(self) -> int:
        return len(self.gps_points_gdf)
