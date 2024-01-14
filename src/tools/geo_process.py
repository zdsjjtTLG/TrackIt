# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData


import pyproj
import numpy as np
from shapely.ops import transform
from shapely.geometry import LineString, Point


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
    dense_line = line.segmentize(line_length / (1.0 * n))
    equal_points = list(dense_line.coords)[1:-1]
    if add_noise:
        base_noise = 0.707106 * noise_frac * line_length / n
        equal_points = [[loc[0] + np.random.normal(loc=0, scale=base_noise),
                         loc[1] + np.random.normal(loc=0, scale=base_noise)] for loc in equal_points]
    return equal_points


def cut_line_in_nearest_point(line, point) -> list[LineString]:
    xd = line.project(point)
    return cut(line, xd)


def cut(line: LineString = None, dis: float = None) -> list[LineString]:
    # Cuts a line in two at a distance from its starting point
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


def prj_xfer(from_crs='EPSG:4326', to_crs='EPSG:32650', origin_p: Point = None):

    before = pyproj.CRS(from_crs)
    after = pyproj.CRS(to_crs)
    project = pyproj.Transformer.from_crs(before, after, always_xy=True).transform
    utm_point = transform(project, origin_p)
    return utm_point


if __name__ == '__main__':
    import geopandas as gpd
    import matplotlib.pyplot as plt
    pass
    # l1 = LineString([(1, 2), (3,7)])
    # l2 = LineString([(3,7), (1,2)])
    #
    # print(calc_link_angle(l1, l2))

    # print(180 * np.arccos(0) / np.pi)

    # l1 = LineString([(1, 2), (3,7)])
    p1 = Point((12.122, 14.109))
    p2 = Point((3,7))

    z = n_equal_points(3, from_point=p1, to_point=p2, add_noise=True, noise_frac=0.5)

    p_gdf = gpd.GeoSeries([p1, p2])
    p1_gdf = gpd.GeoSeries([Point(x) for x in z])
    ax = p_gdf.plot()
    p1_gdf.plot(ax=ax, color='red')
    plt.show()