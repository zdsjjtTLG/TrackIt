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
        self.pi = math.pi  # π
        self.a = 6378245.0  # 长半轴
        self.es = 0.00669342162296594323  # 偏心率平方

    def GCJ02_to_BD09(self, gcj_lng: float or np.ndarray, gcj_lat: float or np.ndarray) -> tuple[float, float]:
        """
        实现GCJ02向BD09坐标系的转换
        :param gcj_lng: GCJ02坐标系下的经度
        :param gcj_lat: GCJ02坐标系下的纬度
        :return:
        """
        z = np.sqrt(gcj_lng * gcj_lng + gcj_lat * gcj_lat) + 0.00002 * np.sin(gcj_lat * self.x_pi)
        theta = np.arctan2(gcj_lat, gcj_lng) + 0.000003 * np.cos(gcj_lng * self.x_pi)
        bd_lng = z * np.cos(theta) + 0.0065
        bd_lat = z * np.sin(theta) + 0.006
        return bd_lng, bd_lat

    def BD09_to_GCJ02(self, bd_lng: float or np.ndarray, bd_lat: float or np.ndarray) -> tuple[float, float]:
        '''
        实现BD09坐标系向GCJ02坐标系的转换
        :param bd_lng: BD09坐标系下的经度
        :param bd_lat: BD09坐标系下的纬度
        :return: 转换后的GCJ02下经纬度
        '''
        x = bd_lng - 0.0065
        y = bd_lat - 0.006
        z = np.sqrt(x * x + y * y) - 0.00002 * np.sin(y * self.x_pi)
        theta = np.arctan2(y, x) - 0.000003 * np.cos(x * self.x_pi)
        gcj_lng = z * np.cos(theta)
        gcj_lat = z * np.sin(theta)
        return gcj_lng, gcj_lat

    def WGS84_to_GCJ02(self, lng: float or np.ndarray, lat: float or np.ndarray) -> tuple[float, float]:
        '''
        实现WGS84坐标系向GCJ02坐标系的转换
        :param lng: WGS84坐标系下的经度
        :param lat: WGS84坐标系下的纬度
        :return: 转换后的GCJ02下经纬度
        '''
        dlat = self._transformlat(lng - 105.0, lat - 35.0)
        dlng = self._transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * self.pi
        magic = np.sin(radlat)
        magic = 1 - self.es * magic * magic
        sqrtmagic = np.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.es)) / (magic * sqrtmagic) * self.pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * np.cos(radlat) * self.pi)
        gcj_lat = lat + dlat
        gcj_lng = lng + dlng
        return gcj_lng, gcj_lat

    def GCJ02_to_WGS84(self, gcj_lng: float or np.ndarray, gcj_lat: float or np.ndarray) -> tuple[float, float]:
        '''
        实现GCJ02坐标系向WGS84坐标系的转换
        :param gcj_lng: GCJ02坐标系下的经度
        :param gcj_lat: GCJ02坐标系下的纬度
        :return: 转换后的WGS84下经纬度
        '''
        dlat = self._transformlat(gcj_lng - 105.0, gcj_lat - 35.0)
        dlng = self._transformlng(gcj_lng - 105.0, gcj_lat - 35.0)
        radlat = gcj_lat / 180.0 * self.pi
        magic = np.sin(radlat)
        magic = 1 - self.es * magic * magic
        sqrtmagic = np.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.es)) / (magic * sqrtmagic) * self.pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * np.cos(radlat) * self.pi)
        mglat = gcj_lat + dlat
        mglng = gcj_lng + dlng
        lng = gcj_lng * 2 - mglng
        lat = gcj_lat * 2 - mglat
        return lng, lat

    def BD09_to_WGS84(self, bd_lng, bd_lat) -> tuple[float, float]:
        '''
        实现BD09坐标系向WGS84坐标系的转换
        :param bd_lng: BD09坐标系下的经度
        :param bd_lat: BD09坐标系下的纬度
        :return: 转换后的WGS84下经纬度
        '''
        lng, lat = self.BD09_to_GCJ02(bd_lng, bd_lat)
        return self.GCJ02_to_WGS84(lng, lat)

    def WGS84_to_BD09(self, lng, lat):
        '''
        实现WGS84坐标系向BD09坐标系的转换
        :param lng: WGS84坐标系下的经度
        :param lat: WGS84坐标系下的纬度
        :return: 转换后的BD09下经纬度
        '''
        lng, lat = self.WGS84_to_GCJ02(lng, lat)
        return self.GCJ02_to_BD09(lng, lat)

    def _transformlat(self, lng: float or np.ndarray, lat: float or np.ndarray) -> float:
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * self.pi) + 20.0 *
                np.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lat * self.pi) + 40.0 *
                np.sin(lat / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (160.0 * np.sin(lat / 12.0 * self.pi) + 320 *
                np.sin(lat * self.pi / 30.0)) * 2.0 / 3.0
        return ret

    def _transformlng(self, lng: float or np.ndarray, lat: float or np.ndarray) -> float:
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * np.sqrt(np.fabs(lng))
        ret += (20.0 * np.sin(6.0 * lng * self.pi) + 20.0 *
                np.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lng * self.pi) + 40.0 *
                np.sin(lng / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (150.0 * np.sin(lng / 12.0 * self.pi) + 300.0 *
                np.sin(lng / 30.0 * self.pi)) * 2.0 / 3.0
        return ret

    def WGS84_to_WebMercator(self, lng: float or np.ndarray, lat: float or np.ndarray) -> tuple[float, float]:
        '''
        实现WGS84向web墨卡托的转换
        :param lng: WGS84经度
        :param lat: WGS84纬度
        :return: 转换后的web墨卡托坐标
        '''
        x = lng * 20037508.342789 / 180
        y = np.log(np.tan((90 + lat) * self.pi / 360)) / (self.pi / 180)
        y = y * 20037508.34789 / 180
        return x, y

    def WebMercator_to_WGS84(self, x: float or np.ndarray, y: float or np.ndarray) -> tuple[float, float]:
        '''
        实现web墨卡托向WGS84的转换
        :param x: web墨卡托x坐标
        :param y: web墨卡托y坐标
        :return: 转换后的WGS84经纬度
        '''
        lng = x / 20037508.34 * 180
        lat = y / 20037508.34 * 180
        lat = 180 / self.pi * (2 * np.arctan2(np.exp(lat * self.pi / 180)) - self.pi / 2)
        return lng, lat

    def loc_convert(self, lng: float or np.ndarray, lat: float or np.ndarray,
                    con_type: str = 'gc-bd') -> tuple[float, float]:
        if con_type == 'gc-bd':
            return self.GCJ02_to_BD09(lng, lat)
        elif con_type == 'gc-84':
            return self.GCJ02_to_WGS84(lng, lat)
        elif con_type == '84-bd':
            return self.WGS84_to_BD09(lng, lat)
        elif con_type == '84-gc':
            return self.WGS84_to_GCJ02(lng, lat)
        elif con_type == 'bd-84':
            return self.BD09_to_WGS84(lng, lat)
        elif con_type == 'bd-gc':
            return self.BD09_to_GCJ02(lng, lat)
        else:
            return lng, lat

    def obj_convert(self, geo_obj: shapely.geometry, con_type: str, ignore_z: bool = True) -> shapely.geometry:
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
                coords_list = self.get_coords(obj=geo_obj)
                if len(coords_list) > 1:
                    return Polygon(self.xfer_coords(coords_list=coords_list[0], con_type=con_type, ignore_z=ignore_z),
                                   holes=[self.xfer_coords(coords_list=ring_coord, con_type=con_type, ignore_z=ignore_z)
                                          for ring_coord in coords_list[1:]])
                else:
                    return Polygon(self.xfer_coords(coords_list=coords_list[0], con_type=con_type, ignore_z=ignore_z))
            elif isinstance(geo_obj, Point):
                return Point(self.loc_convert(geo_obj.x, geo_obj.y, con_type))
            else:
                raise ValueError(r'Single类型只允许LineString or Polygon or Point or LineRing')

    def xfer_coords(self, coords_list: list = None, con_type: str = None, ignore_z: bool = True):
        if ignore_z:
            return [self.loc_convert(x, y, con_type) for (x, y) in coords_list]
        else:
            return [self.loc_convert(x, y, con_type) + (z,) for (x, y, z) in coords_list]

    @staticmethod
    def get_coords(obj=None, ignore_z: bool = True):
        """
        获取单个line或者polygon的坐标序列
        :param obj:
        :param ignore_z:
        :return: [coords_list]
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
            raise ValueError('Only LineString or LinearRing or Polygon is allowed.')

    def geo_convert(self, gdf: gpd.GeoDataFrame, con_type: str = 'gc-84', ignore_z: bool = True) -> gpd.GeoDataFrame:
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
        gdf = gpd.read_file(file_path)
        gdf = self.geo_convert(gdf=gdf, con_type=con_type, ignore_z=ignore_z)
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


if __name__ == '__main__':
    # import pandas as pd
    handler = LngLatTransfer()
    # transData = pd.DataFrame({'x': [114.2361, 114.669, 113.695], 'y': [22.22,22.36, 36.44]})
    # transData['geometry'] = gpd.points_from_xy(transData.x, transData.y, crs='EPSG:4326')
    # transData['geometry'] = transData['geometry'].apply(lambda row: handler.obj_convert(geo_obj=row, con_type='gc-84'))
    # transData["lng"], transData['lat'] = handler.loc_convert(transData['x'], transData['y'], con_type='gc-84')
    #
    #
    # # 火星坐标系 转换为 wgs84坐标系：GCJ02_to_WGS84 (lng, lat)
    # x, y = handler.loc_convert(113.695, 36.44, con_type='gc-bd')
    #
    # print(x)
    # print(y)

    # link = gpd.read_file(r'F:\PyPrj\TrackIt\data\input\QuickStart-Match-1\modifiedConn_link.shp')
    #
    # link['geometry'] = link['geometry'].apply(
    #     lambda row: handler.obj_convert(geo_obj=row, con_type='gc-84', ignore_z=True))

    # node = gpd.read_file(r'F:\PyPrj\TrackIt\data\input\QuickStart-Match-1\modifiedConn_node.shp')

    handler.file_convert(file_path=r'F:\PyPrj\TrackIt\data\input\QuickStart-Match-1\modifiedConn_node.shp',
                         out_fldr=r'C:\Users\Administrator\Desktop\temp',
                         con_type='84-gc')
