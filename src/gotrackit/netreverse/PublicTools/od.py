# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData

import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import distance
from ..GlobalVal import RegionField, ODField
from shapely.geometry import Point, LineString


od_field = ODField()
region_field = RegionField()

region_id_field, region_geo_field = region_field.REGION_ID_FIELD, region_field.GEO_FIELD
o_x_field, o_y_field, d_x_field, d_y_field, od_id_field = od_field.O_X_FIELD, od_field.O_Y_FIELD, \
    od_field.D_X_FIELD, od_field.D_Y_FIELD, od_field.OD_ID_FIELD

"""生成请求OD对"""


# 区域内随机OD
def region_rnd_od(polygon_obj=None, flag_name=None,
                  od_num=10000, gap_n=1000, length_limit=1500) -> pd.DataFrame:
    """
    依据区域生产随机OD
    :param polygon_obj: Polygon
    :param flag_name: str, 输出OD文件的命名{flag_name}.csv
    :param od_num: int, 要生成的OD数
    :param gap_n: 栅格个数(横向和纵向)
    :param length_limit, OD之间直线距离的最小值(m)
    :return:
    """
    print(rf'{flag_name} 生成OD......')

    od_loc = [generate_rnd_od(p_geo=polygon_obj, length_limit=length_limit, gap_n=gap_n) for i in range(0, od_num)]
    od_df = gpd.GeoDataFrame(od_loc, columns=['o', 'd'], geometry='o', crs='EPSG:4326')
    od_df[o_x_field] = od_df['o'].apply(lambda g: g.x)
    od_df[o_y_field] = od_df['o'].apply(lambda g: g.y)

    od_df.set_geometry('d', crs='EPSG:4326', inplace=True)
    od_df[d_x_field] = od_df['d'].apply(lambda g: g.x)
    od_df[d_y_field] = od_df['d'].apply(lambda g: g.y)
    od_df[od_id_field] = range(1, len(od_df) + 1)
    od_df.drop(columns=['o', 'd'], axis=1, inplace=True)
    return od_df


def generate_rnd_od(p_geo=None, length_limit=0.001, gap_n=1000) -> tuple[Point, Point]:
    """

    :param p_geo:
    :param length_limit:
    :param gap_n:
    :return:
    """
    max_times = 20000
    (o_x, o_y) = rnd_point_in_polygon(p_geo=p_geo, gap_n=gap_n)
    _count = 0
    while True:
        _count += 1
        (d_x, d_y) = rnd_point_in_polygon(p_geo=p_geo, gap_n=gap_n)

        if distance((o_y, o_x), (d_y, d_x)).m >= length_limit or _count > max_times:
            break
    return Point(o_x, o_y), Point(d_x, d_y)


def rnd_point_in_polygon(p_geo=None, gap_n=1000) -> tuple[float, float]:
    """

    :param p_geo:
    :param gap_n:
    :return:
    """
    (x_min, y_min, x_max, y_max) = p_geo.bounds
    while True:
        (rnd_x, rnd_y) = rand_point(gap_n=gap_n,
                                    x_min=x_min, x_max=x_max,
                                    y_min=y_min, y_max=y_max)
        rnd_p = Point((rnd_x, rnd_y))
        if rnd_p.within(p_geo):
            break

    return rnd_x, rnd_y


def rand_point(gap_n=1000, x_min=None, y_min=None, x_max=None, y_max=None) -> tuple[float, float]:

    x_gap = (x_max - x_min) / gap_n
    y_gap = (y_max - y_min) / gap_n

    rnd_x = (x_min + x_gap * np.random.randint(0, gap_n))
    rnd_y = (y_min + y_gap * np.random.randint(0, gap_n))

    return rnd_x, rnd_y


# 各区域之间形心生成OD
def region_od(region_gdf: gpd.GeoDataFrame = None) -> pd.DataFrame:
    """

    :param region_gdf:
    :return:
    """
    assert region_id_field in region_gdf.columns
    region_id_list = set(region_gdf[region_id_field])
    region_cen_loc_dict = {region_id: geo for region_id, geo in zip(region_gdf[region_id_field],
                                                                    region_gdf[region_geo_field])}
    od_list = [[o, d] for o in region_id_list for d in region_id_list if o != d]
    od_df = pd.DataFrame(od_list, columns=['f', 't'])

    od_df[o_x_field] = od_df['f'].apply(lambda g: region_cen_loc_dict[g].centroid.x)
    od_df[o_y_field] = od_df['f'].apply(lambda g: region_cen_loc_dict[g].centroid.y)
    od_df[d_x_field] = od_df['t'].apply(lambda g: region_cen_loc_dict[g].centroid.x)
    od_df[d_y_field] = od_df['t'].apply(lambda g: region_cen_loc_dict[g].centroid.y)

    od_df[od_id_field] = range(1, len(od_df) + 1)
    return od_df


def extract_od_by_gps(gps_gdf: pd.DataFrame or gpd.GeoDataFrame = None):
    pass
