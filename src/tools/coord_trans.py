# -- coding: utf-8 --
# @Time    : 2023/12/13 15:31
# @Author  : TangKai
# @Team    : ZheChengData

"""GCJO2、百度、84坐标互转"""
import datetime
import math
import pyproj
import shapely
import pandas as pd
from shapely.ops import transform
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon, LinearRing

class LngLatTransfer(object):

    def __init__(self):
        self.x_pi = 3.14159265358979324 * 3000.0 / 180.0
        self.pi = math.pi  # π
        self.a = 6378245.0  # 长半轴
        self.es = 0.00669342162296594323  # 偏心率平方

    def GCJ02_to_BD09(self, gcj_lng, gcj_lat) -> tuple[float, float]:
        """
        实现GCJ02向BD09坐标系的转换
        :param gcj_lng: GCJ02坐标系下的经度
        :param gcj_lat: GCJ02坐标系下的纬度
        :return:
        """
        z = math.sqrt(gcj_lng * gcj_lng + gcj_lat * gcj_lat) + 0.00002 * math.sin(gcj_lat * self.x_pi)
        theta = math.atan2(gcj_lat, gcj_lng) + 0.000003 * math.cos(gcj_lng * self.x_pi)
        bd_lng = z * math.cos(theta) + 0.0065
        bd_lat = z * math.sin(theta) + 0.006
        return bd_lng, bd_lat

    def BD09_to_GCJ02(self, bd_lng, bd_lat) -> tuple[float, float]:
        '''
        实现BD09坐标系向GCJ02坐标系的转换
        :param bd_lng: BD09坐标系下的经度
        :param bd_lat: BD09坐标系下的纬度
        :return: 转换后的GCJ02下经纬度
        '''
        x = bd_lng - 0.0065
        y = bd_lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * self.x_pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * self.x_pi)
        gcj_lng = z * math.cos(theta)
        gcj_lat = z * math.sin(theta)
        return gcj_lng, gcj_lat

    def WGS84_to_GCJ02(self, lng, lat) -> tuple[float, float]:
        '''
        实现WGS84坐标系向GCJ02坐标系的转换
        :param lng: WGS84坐标系下的经度
        :param lat: WGS84坐标系下的纬度
        :return: 转换后的GCJ02下经纬度
        '''
        dlat = self._transformlat(lng - 105.0, lat - 35.0)
        dlng = self._transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * self.pi
        magic = math.sin(radlat)
        magic = 1 - self.es * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.es)) / (magic * sqrtmagic) * self.pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * math.cos(radlat) * self.pi)
        gcj_lat = lat + dlat
        gcj_lng = lng + dlng
        return gcj_lng, gcj_lat

    def GCJ02_to_WGS84(self, gcj_lng, gcj_lat) -> tuple[float, float]:
        '''
        实现GCJ02坐标系向WGS84坐标系的转换
        :param gcj_lng: GCJ02坐标系下的经度
        :param gcj_lat: GCJ02坐标系下的纬度
        :return: 转换后的WGS84下经纬度
        '''
        dlat = self._transformlat(gcj_lng - 105.0, gcj_lat - 35.0)
        dlng = self._transformlng(gcj_lng - 105.0, gcj_lat - 35.0)
        radlat = gcj_lat / 180.0 * self.pi
        magic = math.sin(radlat)
        magic = 1 - self.es * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.es)) / (magic * sqrtmagic) * self.pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * math.cos(radlat) * self.pi)
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

    def _transformlat(self, lng, lat) -> float:
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
                math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * self.pi) + 40.0 *
                math.sin(lat / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * self.pi) + 320 *
                math.sin(lat * self.pi / 30.0)) * 2.0 / 3.0
        return ret

    def _transformlng(self, lng, lat) -> float:
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
                math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * self.pi) + 40.0 *
                math.sin(lng / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * self.pi) + 300.0 *
                math.sin(lng / 30.0 * self.pi)) * 2.0 / 3.0
        return ret

    def WGS84_to_WebMercator(self, lng, lat) -> tuple[float, float]:
        '''
        实现WGS84向web墨卡托的转换
        :param lng: WGS84经度
        :param lat: WGS84纬度
        :return: 转换后的web墨卡托坐标
        '''
        x = lng * 20037508.342789 / 180
        y = math.log(math.tan((90 + lat) * self.pi / 360)) / (self.pi / 180)
        y = y * 20037508.34789 / 180
        return x, y

    def WebMercator_to_WGS84(self, x, y) -> tuple[float, float]:
        '''
        实现web墨卡托向WGS84的转换
        :param x: web墨卡托x坐标
        :param y: web墨卡托y坐标
        :return: 转换后的WGS84经纬度
        '''
        lng = x / 20037508.34 * 180
        lat = y / 20037508.34 * 180
        lat = 180 / self.pi * (2 * math.atan(math.exp(lat * self.pi / 180)) - self.pi / 2)
        return lng, lat

    def loc_convert(self, lng, lat, con_type='gc2bd') -> tuple[float, float]:
        assert con_type in ['gc-bd', 'gc-84', '84-bd', '84-gc', 'bd-84']
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
        else:
            return self.BD09_to_GCJ02(lng, lat)

    def obj_convert(self, geo_obj: shapely.geometry, con_type: str) -> shapely.geometry:
        if isinstance(geo_obj, (MultiPolygon, MultiLineString, MultiPoint)):
            convert_obj_list = [self.obj_convert(geo, con_type=con_type) for geo in geo_obj.geoms]
            if isinstance(geo_obj, MultiPolygon):
                return MultiPolygon(convert_obj_list) if len(convert_obj_list) > 1 else convert_obj_list[0]
            elif isinstance(geo_obj, MultiLineString):
                return MultiLineString(convert_obj_list) if len(convert_obj_list) > 1 else convert_obj_list[0]
            elif isinstance(geo_obj, MultiPoint):
                return MultiPoint(convert_obj_list) if len(convert_obj_list) > 1 else convert_obj_list[0]
            else:
                raise ValueError(r'Multi类型只允许MultiLineString or MultiPolygon')
        else:
            if isinstance(geo_obj, (LineString, LinearRing)):
                coords_list = self.get_coords(obj=geo_obj)
                return LineString([self.loc_convert(x, y, con_type) for (x, y) in coords_list[0]])
            elif isinstance(geo_obj, Polygon):
                coords_list = self.get_coords(obj=geo_obj)
                if len(coords_list) > 1:
                    return Polygon([self.loc_convert(x, y, con_type) for (x, y) in coords_list[0]],
                                   holes=[[self.loc_convert(x, y, con_type) for x, y in ring_coord] for ring_coord
                                          in
                                          coords_list[1:]])
                else:
                    return Polygon([self.loc_convert(x, y, con_type) for (x, y) in coords_list[0]])
            elif isinstance(geo_obj, Point):
                return Point(self.loc_convert(geo_obj.x, geo_obj.y, con_type))
            else:
                raise ValueError(r'Single类型只允许LineString or Polygon or Point or LineRing')

    @staticmethod
    def get_coords(obj=None):
        """
        获取单个line或者polygon的坐标序列
        :param obj:
        :return: [coords_list]
        """
        if isinstance(obj, (LineString, LinearRing)):
            return [remove_z(list(obj.coords))]
        elif isinstance(obj, Polygon):
            if judge_hole(p=obj):
                holes = list(obj.interiors)
                return [remove_z(list(obj.exterior.coords))] + [remove_z(list(ring.coords)) for ring in holes]
            else:
                return [remove_z(list(obj.exterior.coords))]
        else:
            raise ValueError('obj只能为 LineString or LinearRing or Polygon')

def remove_z(coords_list=None):
    return [(item[0], item[1]) for item in coords_list]

def judge_hole(p=None):
    """判断一个polygon对象内部是否含有ring"""
    if len(list(p.interiors)) >= 1:
        return True
    else:
        return False

def prj_convert(from_crs: str = None, to_crs: str = None, point_obj: Point = None) -> Point:
    """
    地理坐标系和平面投影坐标系之间的转换
    :param from_crs:
    :param to_crs:
    :param point_obj
    :return:
    """
    before = pyproj.CRS(from_crs)
    after = pyproj.CRS(to_crs)
    project = pyproj.Transformer.from_crs(before, after, always_xy=True).transform
    return transform(project, point_obj)


if __name__ == '__main__':
    # fileName = r'F:\武汉轨迹数据\交通事故(2018年)\accidentFileLocations.csv'
    # transData = pd.read_csv(fileName, engine='python')
    # transData["WGS84lng"] = None
    # transData["WGS84lat"] = None
    # # 火星坐标系 转换为 wgs84坐标系：GCJ02_to_WGS84 (lng, lat)
    # handler = LngLatTransfer()
    # # transData[["WGS84lng", "WGS84lat"]] = transData.apply(lambda x: handler.GCJ02_to_WGS84(x["LON"], x["LAT"]), axis=1,
    # #                                                       result_type="expand")
    #
    # x, y = handler.loc_convert(120.002458,30.288055, con_type='gc-84')
    # print(x, y)

    x = datetime.datetime.now()
    print(x)
    print(x.timestamp())




