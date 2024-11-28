# -- coding: utf-8 --
# @Time    : 2024/4/15 17:02
# @Author  : TangKai
# @Team    : ZheChengData

"""将高德导航路径转化为GPS点"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from ..GlobalVal import RouteField, GpsField

gps_field = GpsField()
route_field = RouteField()

agent_id_field = gps_field.AGENT_ID_FIELD
time_field = gps_field.TIME_FIELD
lng_field, lat_field = gps_field.LNG_FIELD, gps_field.LAT_FIELD
path_id_field = route_field.PATH_ID_FIELD
seq_field = route_field.SEQ_FIELD
time_cost_field = route_field.TIME_COST_FIELD
o_time_field = route_field.O_TIME_FIELD
geometry_field = gps_field.GEOMETRY_FIELD


class Route2Gps(object):

    def __init__(self, path_gdf: gpd.GeoDataFrame = None, path_o_time_df: pd.DataFrame = None,
                 planar_crs: str = 'EPSG:3857'):
        """

        Args:
            path_gdf: path_id, seq, time_cost, geometry, crs must be: EPSG:4326
            path_o_time_df: path_id, o_time(datetime format)
            planar_crs:
        """
        assert len(path_o_time_df) >= 1
        assert {path_id_field, seq_field, time_cost_field, geometry_field}.issubset(set(path_gdf.columns))
        assert {path_id_field, o_time_field}.issubset(set(path_o_time_df.columns))
        self.path_gdf = path_gdf
        self.path_o_time_df = path_o_time_df
        self.path_gdf = self.path_gdf.to_crs(planar_crs)
        self.path_gdf[geometry_field] = self.path_gdf[geometry_field].remove_repeated_points(5.0)
        self.path_gdf = self.path_gdf.to_crs('EPSG:4326')
        self.path_gdf = pd.merge(self.path_gdf, path_o_time_df, on=path_id_field, how='left')
        self.path_o_time_df.reset_index(inplace=True, drop=True)
        self.path_gdf[o_time_field] = self.path_gdf[o_time_field].fillna(self.path_o_time_df.at[0, o_time_field])

    def xfer(self):
        self.path_gdf.sort_values(by=[path_id_field, seq_field], ascending=[True, True], inplace=True)
        self.path_gdf.reset_index(inplace=True, drop=True)

        self.path_gdf['p_list'] = self.path_gdf.apply(
            lambda row: [Point(item) for item in list(row[geometry_field].coords)], axis=1)
        self.path_gdf['cost_frac'] = self.path_gdf.apply(
            lambda row: np.array(
                [0] + [row['p_list'][i].distance(row['p_list'][i + 1])
                       for i in range(0, len(row['p_list']) - 1)]) / row[geometry_field].length, axis=1)
        self.path_gdf[time_cost_field] = self.path_gdf[time_cost_field].astype(float)
        self.path_gdf[time_cost_field] = self.path_gdf[time_cost_field] * self.path_gdf['cost_frac']
        self.path_gdf = pd.DataFrame(self.path_gdf)
        self.path_gdf = self.path_gdf.explode(column=['p_list', time_cost_field], ignore_index=True)
        self.path_gdf[time_cost_field] = self.path_gdf[time_cost_field].astype(float)
        self.path_gdf['accu_cost'] = self.path_gdf.groupby([path_id_field])[time_cost_field].cumsum()
        # self.path_gdf[o_time_field] = self.path_gdf[o_time_field].apply(
        #     lambda t: t.timestamp())
        self.path_gdf[time_field] = self.path_gdf[o_time_field] + pd.to_timedelta(self.path_gdf['accu_cost'], unit='s')
        # self.path_gdf[time_field] = self.path_gdf[o_time_field] + self.path_gdf['accu_cost']
        # self.path_gdf[[lng_field, lat_field]] = self.path_gdf.apply(lambda row: (row['p_list'].x, row['p_list'].y),
        #                                                             axis=1,
        #                                                             result_type='expand')
        p_geo = gpd.GeoSeries(self.path_gdf['p_list'])
        self.path_gdf[lng_field] = p_geo.x
        self.path_gdf[lat_field] = p_geo.y
        self.path_gdf.rename(columns={'path_id': agent_id_field}, inplace=True)
        # self.path_gdf[time_field] = self.path_gdf[time_field].astype(int)
        self.path_gdf.drop_duplicates(subset=[agent_id_field, time_field], inplace=True, keep='first')
        return self.path_gdf[[agent_id_field, lng_field, lat_field, time_field]]
