# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData


import numpy as np
import pandas as pd
import geopandas as gpd
from ..GlobalVal import NetField
from .coord_trans import LngLatTransfer
from shapely.geometry import LineString, Point
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon, LinearRing


net_field = NetField()
geometry_field = net_field.GEOMETRY_FIELD

def n_equal_points(n, from_loc: tuple = None, to_loc=None,
                   from_point: Point = None, to_point: Point = None, add_noise: bool = False,
                   noise_frac: float = 0.3) -> list[list]:
    """

    :param n:
    :param from_loc:
    :param to_loc:
    :param from_point:
    :param to_point:
    :param add_noise:
    :param noise_frac:
    :return:
    """
    assert n > 1
    if from_point is None or to_point is None:
        line = LineString([from_loc, to_loc])
    else:
        line = LineString([from_point, to_point])
    line_length = line.length

    try:
        dense_line = segmentize(line=line, n=n)
    except AttributeError:
        raise AttributeError(r'请升级geopandas到最新版本0.14.1')

    equal_points = list(dense_line.coords)[1:-1]
    if add_noise:
        base_noise = 0.707106 * noise_frac * line_length / n
        equal_points = [[loc[0] + np.random.normal(loc=0, scale=base_noise),
                         loc[1] + np.random.normal(loc=0, scale=base_noise)] for loc in equal_points]
    return equal_points


def cut_line_in_nearest_point(line, point) -> list[LineString]:
    """

    :param line:
    :param point:
    :return:
    """
    xd = line.project(point)
    return cut(line, xd)


def cut(line: LineString = None, dis: float = None) -> list[LineString]:
    """

    :param line:
    :param dis:
    :return:
    """
    if dis <= 0.0 or dis >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        xd = line.project(Point(p))
        if xd == dis:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        if xd > dis:
            cp = line.interpolate(dis)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


def calc_link_angle(link_geo1=None, link_geo2=None) -> float:
    """

    :param link_geo1:
    :param link_geo2:
    :return:
    """
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


def segmentize(line: LineString = None, n: int = None) -> LineString:
    """
    将直线对象line进行n等分加密
    :param line: 直线
    :param n:
    :return:
    """
    coord_list = list(line.coords)
    s, e = coord_list[0], coord_list[-1]
    try:
        k = (e[1] - s[1]) / (e[0] - s[0])
    except ZeroDivisionError:
        gap = line.length / n
        return LineString([s] + [(s[0], s[1] + (i + 1) * gap) for i in range(n - 1)] + [e])

    b = e[1] - k * e[0]
    gap_x = (e[0] - s[0]) / n
    sample_x_list = [s[0] + (i + 1) * gap_x for i in range(n - 1)]
    return LineString([s] + [(sample_x, k * sample_x + b) for sample_x in sample_x_list] + [e])


def prj_inf(p: Point = None, line: LineString = None) -> tuple[Point, float, float, float, list[LineString]]:
    """
    # 返回 某point到line的(投影点坐标, 点到投影点的直线距离, 投影点到line拓扑起点的路径距离, line的长度, 投影点打断line后的geo list)
    :param p:
    :param line:
    :return: (投影点坐标, 点到投影点的直线距离, 投影点到line拓扑起点的路径距离, line的长度, 投影点打断line后的geo list)
    """

    distance = line.project(p)

    if distance <= 0.0:
        prj_p = Point(list(line.coords)[0])
        return prj_p, prj_p.distance(p), distance, line.length, [LineString(line)]
    elif distance >= line.length:
        prj_p = Point(list(line.coords)[-1])
        return prj_p, prj_p.distance(p), distance, line.length, [LineString(line)]
    else:
        coords = list(line.coords)
        for i, _p in enumerate(coords):
            xd = line.project(Point(_p))
            if xd == distance:
                prj_p = Point(coords[i])
                return prj_p, prj_p.distance(p), distance, line.length, \
                    [LineString(coords[:i + 1]), LineString(coords[i:])]
            if xd > distance:
                cp = line.interpolate(distance)
                prj_p = Point((cp.x, cp.y))
                return prj_p, prj_p.distance(p), distance, line.length, [LineString(coords[:i] + [(cp.x, cp.y)]),
                                                                         LineString([(cp.x, cp.y)] + coords[i:])]


def clean_link_geo(gdf: gpd.GeoDataFrame = None, plain_crs: str = 'EPSG:32649') -> gpd.GeoDataFrame:
    """
    将geometry列中的Multi对象处理为single对象
    :param gdf:
    :param plain_crs
    :return:
    """
    assert geometry_field in gdf.columns
    origin_crs = gdf.crs
    con = LngLatTransfer()

    gdf[geometry_field] = gdf.apply(lambda row: con.obj_convert(geo_obj=row[geometry_field], con_type='None'), axis=1)
    gdf = pd.DataFrame(gdf)
    gdf[[geometry_field, 'is_multi']] = \
        gdf.apply(lambda row:
                  (list(row[geometry_field].geoms), 1)
                  if isinstance(row[geometry_field], (MultiPolygon, MultiLineString, MultiPoint))
                  else (row[geometry_field], 0), axis=1, result_type='expand')

    is_multi_index = gdf['is_multi'] == 1
    multi_gdf = gdf[is_multi_index].copy()
    gdf.drop(index=gdf[is_multi_index].index, axis=0, inplace=True)

    multi_gdf = multi_gdf.explode(column=[geometry_field], ignore_index=True)
    multi_gdf.dropna(subset=[geometry_field], axis=0, inplace=True)

    gdf = pd.concat([gdf, multi_gdf])
    gdf.reset_index(inplace=True, drop=True)

    gdf.drop(columns=['is_multi'], axis=1, inplace=True)
    gdf = gpd.GeoDataFrame(gdf, geometry=geometry_field, crs=origin_crs)
    gdf = gdf.to_crs(plain_crs)
    gdf[geometry_field] = gdf[geometry_field].remove_repeated_points(0.1)
    gdf = gdf.to_crs(origin_crs)
    return gdf


def divide_line_by_l(line_geo: LineString = None, divide_l: float = 50.0, l_min: float = 0.5) -> tuple[
    list[LineString], list[Point], int]:
    """

    :param line_geo:
    :param divide_l:
    :param l_min:
    :return:
    """
    line_l = line_geo.length
    n = int(line_l / divide_l)
    remain_l = line_l % divide_l
    is_remain = False
    if remain_l > l_min:
        is_remain = True
    p_list = [line_geo.interpolate(i * divide_l) for i in range(1, n + 1)]
    used_l = line_geo
    divide_line_list = []
    divide_point_list = []

    for i, p in enumerate(p_list):
        if i + 1 == len(p_list):
            if not is_remain:
                divide_line_list.append(used_l)
                break
        prj_p, _, dis, _, split_line_list = prj_inf(p, used_l)
        used_l = split_line_list[-1]
        if i + 1 == len(p_list):
            if is_remain:
                divide_line_list.extend(split_line_list)
                divide_point_list.append(p)
                break
        divide_line_list.append(split_line_list[0])
        divide_point_list.append(p)

    return divide_line_list, divide_point_list, len(divide_line_list)


def vector_angle(v1: np.ndarray = None, v2: np.ndarray = None):
    # 计算点积
    dot_product = np.dot(v1, v2)
    # 计算两个向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 避免除以零
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    # 计算夹角
    cos_angle = dot_product / (norm_v1 * norm_v2)
    # 防止因浮点数计算问题导致cos_angle超出范围
    cos_angle = min(max(cos_angle, -1), 1)

    # 计算角度值，结果以弧度表示
    angle = np.arccos(cos_angle)

    # 转换为角度制，如果需要转换为度数，乘以180/np.pi
    return np.degrees(angle)


def angle_base_north(v: np.ndarray = None):
    angle = vector_angle(v, np.array([0, 1]))
    if v[0] <= 0:
        return angle
    else:
        return 360 - angle


if __name__ == '__main__':
    pass
    # import geopandas as gpd
    # import matplotlib.pyplot as plt
    #
    # l = LineString([(0,0), (12, 90)])
    # p = gpd.GeoDataFrame({'geometry': [Point(item) for item in l.coords]}, geometry='geometry')
    # p.plot()
    # plt.show()
    #
    # l_1 = segmentize(line=l, n=12)
    # print(len(list(l_1.coords)))
    # p1 = gpd.GeoDataFrame({'geometry': [Point(item) for item in l_1.coords]}, geometry='geometry')
    # p1.plot()
    # plt.show()




