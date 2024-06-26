# -- coding: utf-8 --
# @Time    : 2024/1/27 16:51
# @Author  : TangKai
# @Team    : ZheChengData


import pandas as pd
import geopandas as gpd
from .GlobalVal import RegionField
from shapely.ops import unary_union
from .PublicTools.od import region_rnd_od, region_od

region_field = RegionField()


class FormatOD(object):
    def __init__(self, plain_crs: str = 'EPSG:32650'):
        self.plain_crs = plain_crs

    @staticmethod
    def format_region_od(region_gdf: gpd.GeoDataFrame = None) -> pd.DataFrame:
        """
        input crs: EPSG:4326
        :param region_gdf:
        :return:
        """
        od_df = region_od(region_gdf=region_gdf)
        return od_df

    def format_region_rnd_od(self, region_gdf: gpd.GeoDataFrame = None, flag_name=None,
                             od_num=10000, gap_n=1000, length_limit=1500,
                             boundary_buffer: float = 2000) -> pd.DataFrame:
        """
        input crs: EPSG:4326
        :param self:
        :param region_gdf:
        :param flag_name:
        :param od_num:
        :param gap_n:
        :param length_limit:
        :param boundary_buffer:
        :return:
        """
        origin_crs = region_gdf.crs
        region_gdf = region_gdf.to_crs(self.plain_crs)
        region_gdf[region_field.GEO_FIELD] = region_gdf[region_field.GEO_FIELD].apply(
            lambda geo: geo.buffer(boundary_buffer))
        region_gdf = region_gdf.to_crs(origin_crs)
        polygon_obj = unary_union(region_gdf[region_field.GEO_FIELD].to_list())
        rnd_od_df = region_rnd_od(polygon_obj=polygon_obj, flag_name=flag_name,
                                  od_num=od_num, gap_n=gap_n, length_limit=length_limit)
        return rnd_od_df

    def format_gps_od(self, gps_df: pd.DataFrame = None) -> pd.DataFrame:
        pass
