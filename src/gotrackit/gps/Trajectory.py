# -- coding: utf-8 --
# @Time    : 2024/8/16 16:28
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
from .LocGps import GpsPointsGdf
from ..visualization import KeplerVis
from ..GlobalVal import GpsField, PrjConst

prj_const = PrjConst()
gps_field = GpsField()
lng_field = gps_field.LNG_FIELD
lat_field = gps_field.LAT_FIELD
time_field = gps_field.TIME_FIELD
agent_field = gps_field.AGENT_ID_FIELD
geometry_field = gps_field.GEOMETRY_FIELD


class TrajectoryPoints(GpsPointsGdf):
    def __init__(self, gps_points_df: pd.DataFrame, time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's',
                 plain_crs: str = 'EPSG:3857', already_plain: bool = False):
        """轨迹类

        提供了定位数据的相关操作方法(滤波、降频、简化、滑动窗口平均、删除停留点、导出HTML动画)

        Args:
            gps_points_df: 定位数据表
            time_format: 时间列字符串格式模板
            time_unit: 时间单位
            plain_crs: 平面投影坐标系

        """
        self.time_format = time_format
        self.time_unit = time_unit
        user_field_list = list(set(gps_points_df.columns) - {agent_field, lng_field, lat_field, time_field,
                                                             gps_field.POINT_SEQ_FIELD})
        user_field_list = self.check(gps_points_df=gps_points_df, user_field_list=user_field_list)
        GpsPointsGdf.__init__(self, gps_points_df=gps_points_df, time_format=time_format, time_unit=time_unit,
                              plane_crs=plain_crs, already_plain=already_plain, multi_agents=True,
                              user_filed_list=user_field_list)

    def export_html(self, out_fldr: str = r'./', file_name: str = 'trajectory', radius: float = 10.0):
        """导出HTML

        将处理后的轨迹导出为HTML，可动态展示处理前后的轨迹

        Args:
            out_fldr: 存储目录
            file_name: 文件名称
            radius: 点的半径大小

        Returns:
            None
        """
        if self.already_plain:
            origin_tj_df = self.source_gps.to_crs(prj_const.PRJ_CRS)
            origin_tj_df = origin_tj_df[[agent_field, time_field, geometry_field]].copy()
            origin_tj_df[lng_field] = origin_tj_df[geometry_field].x
            origin_tj_df[lat_field] = origin_tj_df[geometry_field].y
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
            vis_df = pd.DataFrame(_df)
            vis_df.sort_values(by='type', ascending=False, inplace=True)
            cen_x, cen_y = vis_df[gps_field.LNG_FIELD].mean(), vis_df[gps_field.LAT_FIELD].mean()
            # vis_df[gps_field.TIME_FIELD] = vis_df[gps_field.TIME_FIELD].astype(str)
            try:
                kv = KeplerVis(cen_loc=[cen_x, cen_y])
                kv.add_point_layer(vis_df, lng_field=lng_field, lat_field=lat_field, time_format=self.time_format,
                                   time_unit=self.time_unit, set_avg_zoom=False, radius=radius,
                                   time_field=time_field, layer_id='trajectory', color=[65, 72, 88],
                                   color_field='type',
                                   color_list=['#438ECD', '#FFC300'])
                kv.export_html(out_fldr=out_fldr, file_name=rf'{agent_id}_' + file_name)
            except Exception as e:
                print(repr(e))
