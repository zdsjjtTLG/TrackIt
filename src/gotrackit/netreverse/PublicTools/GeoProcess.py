# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData


import pyproj
import shapely
import numpy as np
import geopandas as gpd
from shapely.ops import transform
from shapely.ops import unary_union
from geopy.distance import distance
from shapely.geometry import LineString, Point, Polygon


def cut_line_in_nearest_point(line, point):
    xd = line.project(point)
    return cut(line, xd)


def cut(line, distance) -> list[LineString]:
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        xd = line.project(Point(p))
        if xd == distance:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        if xd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


def calc_link_angle(link_geo1=None, link_geo2=None):
    coord_list_a = list(link_geo1.coords)
    coord_list_b = list(link_geo2.coords)

    vec_a = np.array(coord_list_a[-1]) - np.array(coord_list_a[0])
    vec_b = np.array(coord_list_b[-1]) - np.array(coord_list_b[0])
    cos_res = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    if cos_res > 0:
        if cos_res >= 1:
            cos_res = 0.99999
        return 180 * np.arccos(cos_res) / np.pi
    elif cos_res < 0:
        if cos_res < -1.0:
            cos_res = -0.99999
    return 180 * np.arccos(cos_res) / np.pi


def judge_epsg(region_gdf=None):

    all_geo = unary_union(region_gdf['geometry'].to_list())
    cen_lng = all_geo.centroid.x
    # 经纬度范围和6度投影带的映射关系
    utm_prj_dict = {(72 + 6 * i, 78 + 6 * i): rf'EPSG:326{43 + i}' for i in range(0, 11)}
    utm_prj = 'EPSG:32650'

    for lng_range in utm_prj_dict.keys():
        if lng_range[0] <= cen_lng <= lng_range[1]:
            utm_prj = utm_prj_dict[lng_range]

    return utm_prj


def prj_xfer(from_crs='EPSG:4326', to_crs='EPSG:32650', origin_p: shapely.geometry = None) -> shapely.geometry:

    before = pyproj.CRS(from_crs)
    after = pyproj.CRS(to_crs)
    project = pyproj.Transformer.from_crs(before, after, always_xy=True).transform
    utm_geo = transform(project, origin_p)
    return utm_geo


def point_distance(p1=None, p2=None, crs_type='plane'):
    p1 = Point(p1) if isinstance(p1, (tuple, list)) else p1
    p2 = Point(p2) if isinstance(p2, (tuple, list)) else p2
    if crs_type == 'plane':
        return p1.distance(p2)
    elif crs_type == 'geo':
        return distance((p1.y, p1.x), (p2.y, p2.x)).m


def generate_region(min_lng: float = None, min_lat: float = None, w: float = 2000, h: float = 2000,
                    plain_crs: str = None) -> gpd.GeoDataFrame:
    geo_p = Point((min_lng, min_lat))
    plain_p = prj_xfer(from_crs='EPSG:4326', to_crs=plain_crs, origin_p=geo_p)
    min_x, min_y = plain_p.x, plain_p.y
    max_x, max_y = plain_p.x + w, plain_p.y + h
    plain_polygon = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
    geo_polygon = prj_xfer(from_crs=plain_crs, to_crs='EPSG:4326', origin_p=plain_polygon)
    return gpd.GeoDataFrame([], geometry=[geo_polygon], crs='EPSG:4326')


if __name__ == '__main__':
    l1 = LineString([(1, 2), (3,7)])
    l2 = LineString([(3,7), (1,2)])

    print(calc_link_angle(l1, l2))

    # print(180 * np.arccos(0) / np.pi)