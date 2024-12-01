# -- coding: utf-8 --
# @Time    : 2023/12/13 15:31
# @Author  : TangKai
# @Team    : ZheChengData

"""GCJ-O2、百度、84坐标互转"""

import math
import os.path
import shapely
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon, LinearRing

geometry_field = 'geometry'

class LngLatTransfer(object):

    def __init__(self):
        self.x_pi = 3.14159265358979324 * 3000.0 / 180.0
        self.pi = math.pi
        self.a = 6378245.0  # 长半轴
        self.es = 0.00669342162296594323  # 偏心率平方

    def gcj_to_bd(self, lng: float or np.ndarray, lat: float or np.ndarray) -> tuple[float, float]:
        """
        :param lng:
        :param lat:
        :return:
        """
        z = np.sqrt(lng * lng + lat * lat) + 0.00002 * np.sin(lat * self.x_pi)
        theta = np.arctan2(lat, lng) + 0.000003 * np.cos(lng * self.x_pi)
        bd_lng = z * np.cos(theta) + 0.0065
        bd_lat = z * np.sin(theta) + 0.006
        return bd_lng, bd_lat

    def bd_to_gcj(self, lng: float or np.ndarray, lat: float or np.ndarray) -> tuple[float, float]:
        """
        :param lng:
        :param lat:
        :return:
        """
        x = lng - 0.0065
        y = lat - 0.006
        z = np.sqrt(x * x + y * y) - 0.00002 * np.sin(y * self.x_pi)
        theta = np.arctan2(y, x) - 0.000003 * np.cos(x * self.x_pi)
        gcj_lng = z * np.cos(theta)
        gcj_lat = z * np.sin(theta)
        return gcj_lng, gcj_lat

    def wgs84_to_gcj(self, lng: float or np.ndarray, lat: float or np.ndarray) -> tuple[float, float]:
        """

        :param lng:
        :param lat:
        :return:
        """
        d_lat = self.transform_lat(lng - 105.0, lat - 35.0)
        d_lng = self.transform_lng(lng - 105.0, lat - 35.0)
        rad_lat = lat / 180.0 * self.pi
        magic = np.sin(rad_lat)
        magic = 1 - self.es * magic * magic
        magic_sqrt = np.sqrt(magic)
        d_lat = (d_lat * 180.0) / ((self.a * (1 - self.es)) / (magic * magic_sqrt) * self.pi)
        d_lng = (d_lng * 180.0) / (self.a / magic_sqrt * np.cos(rad_lat) * self.pi)
        gcj_lat = lat + d_lat
        gcj_lng = lng + d_lng
        return gcj_lng, gcj_lat

    def gcj_to_wgs84(self, lng: float or np.ndarray, lat: float or np.ndarray) -> tuple[float, float]:
        """

        :param lng:
        :param lat:
        :return:
        """
        d_lat = self.transform_lat(lng - 105.0, lat - 35.0)
        d_lng = self.transform_lng(lng - 105.0, lat - 35.0)
        rad_lat = lat / 180.0 * self.pi
        magic = np.sin(rad_lat)
        magic = 1 - self.es * magic * magic
        magic_sqrt = np.sqrt(magic)
        d_lat = (d_lat * 180.0) / ((self.a * (1 - self.es)) / (magic * magic_sqrt) * self.pi)
        d_lng = (d_lng * 180.0) / (self.a / magic_sqrt * np.cos(rad_lat) * self.pi)
        wgs_lng = lng * 2 - lng - d_lng
        wgs_lat = lat * 2 - lat - d_lat
        return wgs_lng, wgs_lat

    def bd_to_wgs84(self, bd_lng, bd_lat) -> tuple[float, float]:
        """

        :param bd_lng:
        :param bd_lat:
        :return:
        """
        lng, lat = self.bd_to_gcj(bd_lng, bd_lat)
        return self.gcj_to_wgs84(lng, lat)

    def wgs84_to_bd(self, lng, lat):
        """

        :param lng:
        :param lat:
        :return:
        """
        lng, lat = self.wgs84_to_gcj(lng, lat)
        return self.gcj_to_bd(lng, lat)

    def transform_lat(self, lng: float or np.ndarray, lat: float or np.ndarray) -> float:
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * self.pi) + 20.0 *
                np.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lat * self.pi) + 40.0 *
                np.sin(lat / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (160.0 * np.sin(lat / 12.0 * self.pi) + 320 *
                np.sin(lat * self.pi / 30.0)) * 2.0 / 3.0
        return ret

    def transform_lng(self, lng: float or np.ndarray, lat: float or np.ndarray) -> float:
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * self.pi) + 20.0 *
                np.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lng * self.pi) + 40.0 *
                np.sin(lng / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (150.0 * np.sin(lng / 12.0 * self.pi) + 300.0 *
                np.sin(lng / 30.0 * self.pi)) * 2.0 / 3.0
        return ret

    def loc_convert(self, lng: float or np.ndarray, lat: float or np.ndarray,
                    con_type: str = 'gc-bd') -> tuple[float, float]:
        """地理坐标坐标转换
        支持百度、WGS-84、GCJ-02(高德、火星)坐标之间的相互转换

        Args:
            lng: 经度
            lat: 纬度
            con_type: 转换类型, gc-bd或者gc-84或者84-gc或者84-bd或者bd-84或者bd-gc

        Returns:
            转换后的经度, 转换后的纬度
        """
        if con_type == 'gc-bd':
            return self.gcj_to_bd(lng, lat)
        elif con_type == 'gc-84':
            return self.gcj_to_wgs84(lng, lat)
        elif con_type == '84-bd':
            return self.wgs84_to_bd(lng, lat)
        elif con_type == '84-gc':
            return self.wgs84_to_gcj(lng, lat)
        elif con_type == 'bd-84':
            return self.bd_to_wgs84(lng, lat)
        elif con_type == 'bd-gc':
            return self.bd_to_gcj(lng, lat)
        else:
            return lng, lat

    def obj_convert(self, geo_obj: shapely.geometry, con_type: str = None, ignore_z: bool = True) -> shapely.geometry:
        """几何对象的地理坐标坐标转换
        支持几何对象在百度、WGS-84、GCJ-02(高德、火星)坐标之间的相互转换

        Args:
            geo_obj: 几何对象(点, 线, 面)
            con_type: 转换类型, gc-bd或者gc-84或者84-gc或者84-bd或者bd-84或者bd-gc
            ignore_z: 是否忽略Z坐标

        Returns:
            转换后的几何对象
        """
        if isinstance(geo_obj, (MultiPolygon, MultiLineString, MultiPoint)):
            convert_obj_list = [self.obj_convert(geo, con_type=con_type) for geo in geo_obj.geoms]
            if isinstance(geo_obj, MultiPolygon):
                return MultiPolygon(convert_obj_list) if len(convert_obj_list) > 1 else convert_obj_list[0]
            elif isinstance(geo_obj, MultiLineString):
                return MultiLineString(convert_obj_list) if len(convert_obj_list) > 1 else convert_obj_list[0]
            elif isinstance(geo_obj, MultiPoint):
                return MultiPoint(convert_obj_list) if len(convert_obj_list) > 1 else convert_obj_list[0]
            else:
                raise ValueError(r'MultiTypeGeoms only allow: MultiLineString or MultiPolygon')
        else:
            if isinstance(geo_obj, (LineString, LinearRing)):
                coords_list = self.get_coords(obj=geo_obj, ignore_z=ignore_z)
                return LineString(self.xfer_coords(coords_list=coords_list[0], con_type=con_type, ignore_z=ignore_z))
            elif isinstance(geo_obj, Polygon):
                coords_list = self.get_coords(obj=geo_obj, ignore_z=ignore_z)
                if len(coords_list) > 1:
                    return Polygon(self.xfer_coords(coords_list=coords_list[0], con_type=con_type, ignore_z=ignore_z),
                                   holes=[self.xfer_coords(coords_list=ring_coord, con_type=con_type, ignore_z=ignore_z)
                                          for ring_coord in coords_list[1:]])
                else:
                    return Polygon(self.xfer_coords(coords_list=coords_list[0], con_type=con_type, ignore_z=ignore_z))
            elif isinstance(geo_obj, Point):
                if ignore_z:
                    return Point(self.loc_convert(geo_obj.x, geo_obj.y, con_type))
                else:
                    return Point(self.loc_convert(geo_obj.x, geo_obj.y, con_type) + (geo_obj.z,))
            else:
                raise ValueError(r'Only LineString or Polygon or Point or LineRing are allowed')

    def xfer_coords(self, coords_list: list = None, con_type: str = None, ignore_z: bool = True):
        if ignore_z:
            return [self.loc_convert(x, y, con_type) for (x, y) in coords_list]
        else:
            return [self.loc_convert(x, y, con_type) + (z,) for (x, y, z) in coords_list]

    @staticmethod
    def get_coords(obj=None, ignore_z: bool = True):
        """获取单个line或者polygon的坐标序列

        Args:
            obj:
            ignore_z:

        Returns:

        """
        if isinstance(obj, (LineString, LinearRing)):
            return [process_z(list(obj.coords), ignore_z=ignore_z)]
        elif isinstance(obj, Polygon):
            if judge_hole(p=obj):
                holes = list(obj.interiors)
                return [process_z(list(obj.exterior.coords), ignore_z=ignore_z)] + \
                    [process_z(list(ring.coords), ignore_z=ignore_z) for ring in holes]
            else:
                return [process_z(list(obj.exterior.coords), ignore_z=ignore_z)]
        else:
            raise ValueError('Only LineString or LinearRing or Polygon are allowed.')

    def gdf_convert(self, gdf: gpd.GeoDataFrame, con_type: str = 'gc-84', ignore_z: bool = True) -> gpd.GeoDataFrame:
        """GeoDataFrame地理坐标转换
        支持GeoDataFrame在百度、WGS-84、GCJ-02(高德、火星)坐标之间的相互转换

        Args:
            gdf: GeoDataFrame
            con_type: 转换类型, gc-bd或者gc-84或者84-gc或者84-bd或者bd-84或者bd-gc
            ignore_z: 是否忽略Z坐标

        Returns:

        """
        if gdf is None or gdf.empty:
            return gpd.GeoDataFrame()
        else:
            gdf = gdf.explode(ignore_index=True)
            g = gdf.at[0, geometry_field]
            if isinstance(g, Point):
                nx, ny = self.loc_convert(lng=gdf[geometry_field].x, lat=gdf[geometry_field].y,
                                          con_type=con_type)
                if ignore_z:
                    gdf[geometry_field] = gpd.points_from_xy(nx, ny, crs=gdf.crs)
                else:
                    try:
                        z = gdf[geometry_field].z
                    except Exception as e:
                        z = 0
                    gdf[geometry_field] = gpd.points_from_xy(nx, ny, z, crs=gdf.crs)
            elif isinstance(g, (LineString, LinearRing, Polygon)):
                gdf[geometry_field] = gdf[geometry_field].apply(
                    lambda l: self.obj_convert(geo_obj=l, con_type=con_type, ignore_z=ignore_z))
            else:
                raise ValueError('Unknown Geo-Type')
            return gdf

    def file_convert(self, file_path: str = None, con_type: str = 'gc-84',
                     out_fldr: str = r'./', out_file_name: str = 'transfer', file_type: str = 'shp',
                     ignore_z: bool = True):
        """文件-地理坐标转换
        支持地理文件在百度、WGS-84、GCJ-02(高德、火星)坐标之间的相互转换

        Args:
            file_path: 文件路径(能够被geopandas读取的文件)
            con_type: 转换类型, gc-bd或者gc-84或者84-gc或者84-bd或者bd-84或者bd-gc
            out_fldr: 输出路径
            out_file_name: 转换结束后输出文件的名称
            file_type: 输出文件的存储类型, 支持shp或者geojson
            ignore_z: 是否忽略Z坐标

        Returns:
            None
        """

        gdf = gpd.read_file(file_path)
        gdf = self.gdf_convert(gdf=gdf, con_type=con_type, ignore_z=ignore_z)
        if file_type == 'shp':
            gdf.to_file(os.path.join(out_fldr, out_file_name + '.shp'))
        else:
            gdf.to_file(os.path.join(out_fldr, out_file_name + '.geojson'), driver='GeoJSON')

def process_z(coords_list: list = None, ignore_z: bool = True) -> list[tuple]:
    return [(item[0], item[1]) for item in coords_list] if ignore_z else coords_list


def judge_hole(p=None) -> bool:
    """判断一个polygon对象内部是否含有ring"""
    if len(list(p.interiors)) >= 1:
        return True
    else:
        return False
