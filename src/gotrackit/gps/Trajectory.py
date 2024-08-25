# -- coding: utf-8 --
# @Time    : 2024/8/16 16:28
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
from .LocGps import GpsPointsGdf
from ..GlobalVal import GpsField, PrjConst
from ..visualization import generate_point_html

prj_const = PrjConst()
gps_field = GpsField()
lng_field = gps_field.LNG_FIELD
lat_field = gps_field.LAT_FIELD
time_field = gps_field.TIME_FIELD
agent_field = gps_field.AGENT_ID_FIELD
geometry_field = gps_field.GEOMETRY_FIELD


class TrajectoryPoints(GpsPointsGdf):
    def __init__(self, gps_points_df: pd.DataFrame = None, time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's',
                 plane_crs: str = 'EPSG:32649', already_plain: bool = False):
        """
        a class for trajectory process
        :param gps_points_df: pd.DataFrame()
        :param time_format: string
        :param time_unit: string
        :param plane_crs: string
        :param already_plain: bool
        """
        user_field_list = list(set(gps_points_df.columns) - {agent_field, lng_field, lat_field, time_field,
                                                             gps_field.POINT_SEQ_FIELD})
        user_field_list = self.check(gps_points_df=gps_points_df, user_field_list=user_field_list)
        GpsPointsGdf.__init__(self, gps_points_df=gps_points_df, time_format=time_format, time_unit=time_unit,
                              plane_crs=plane_crs, already_plain=already_plain, multi_agents=True,
                              user_filed_list=user_field_list)

    def export_html(self, out_fldr: str = r'./', file_name: str = 'trajectory'):
        if self.already_plain:
            origin_tj_df = self.source_gps.to_crs(prj_const.PRJ_CRS)
            origin_tj_df = origin_tj_df[[agent_field, time_field, geometry_field]].copy()
            origin_tj_df[lng_field] = origin_tj_df[geometry_field].apply(lambda g: g.x)
            origin_tj_df[lat_field] = origin_tj_df[geometry_field].apply(lambda g: g.y)
            del origin_tj_df[geometry_field]
        else:
            origin_tj_df = self.source_gps[[agent_field, lng_field, lat_field, time_field]].copy()

        tj_gdf = self.trajectory_data(export_crs=prj_const.PRJ_CRS, _type='df')
        if {gps_field.X_SPEED_FIELD, gps_field.Y_SPEED_FIELD}.issubset(set(tj_gdf.columns)):
            tj_df = tj_gdf[[agent_field, lng_field, lat_field,
                            gps_field.X_SPEED_FIELD, gps_field.Y_SPEED_FIELD, time_field]].copy()
        else:
            tj_df = tj_gdf[[agent_field, lng_field, lat_field, time_field]].copy()
        del tj_gdf
        tj_df['type'] = 'process'
        origin_tj_df['type'] = 'source'
        tj_df[time_field] = tj_df[time_field].astype(origin_tj_df[time_field].dtype)
        df = pd.concat([tj_df, origin_tj_df]).reset_index(drop=True, inplace=False)
        for agent_id, _df in df.groupby(agent_field):
            try:
                generate_point_html(point_df=pd.DataFrame(_df), out_fldr=out_fldr, file_name=rf'{agent_id}_' + file_name)
            except Exception as e:
                print(repr(e))
