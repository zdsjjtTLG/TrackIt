# -- coding: utf-8 --
# @Time    : 2024/1/13 17:21
# @Author  : TangKai
# @Team    : ZheChengData

import os
import pandas as pd
import multiprocessing
import geopandas as gpd
from .map.Net import Net
from keplergl import KeplerGl
from .tools.group import cut_group
from .GlobalVal import KeplerConfig
from .model.Markov import HiddenMarkov
from .WrapsFunc import function_time_cost
from .tools.time_build import build_time_col
from .tools.coord_trans import LngLatTransfer
from .GlobalVal import GpsField, NetField, ColorMap

con = LngLatTransfer()
gps_field = GpsField()
net_field = NetField()
color_map = ColorMap()
kepler_config = KeplerConfig()


class KeplerVis(object):
    def __init__(self, zoom: int = 15,
                 cen_loc: list[float, float] or tuple[float, float] = None, show_roads: bool = True,
                 map_style: str = 'dark'):
        """可视化类
        开普勒可视化类，提供了几何对象和路径的可视化方法

        Args:
            zoom: 缩放层级, 默认15
            cen_loc: 地图中心点经纬度坐标(WGS-84坐标系)
            show_roads: 是否在底图上显示路网图层
            map_style: 地图风格, 可选dark, light, muted, muted_night, satellite

        """
        self.user_config = kepler_config.get_glb_map_config()
        if cen_loc is not None:
            self.user_config["config"]["mapState"]["longitude"] = cen_loc[0]
            self.user_config["config"]["mapState"]["latitude"] = cen_loc[1]
        self.user_config["config"]["mapState"]["zoom"] = int(zoom)
        self.user_config["config"]["mapStyle"]["styleType"] = map_style
        if show_roads:
            self.user_config["config"]["mapStyle"]["visibleLayerGroups"]['road'] = True
        self.data_dict = dict()
        self.geo_count = 1
        self.point_count = 1
        self.trip_count = 1

    def add_point_layer(self, data: pd.DataFrame, lng_field: str = 'lng', lat_field: str = 'lat',
                        altitude_field: str = None, layer_id: str = None, color: list or str = None, radius: float = 3,
                        set_avg_zoom: bool = True, time_field: str = None, time_format: str = '%Y-%m-%d %H:%M:%S',
                        time_unit: str = 's', speed: float = 0.3, tooltip_fields: list[str] = None,
                        color_field: str = None, color_list: list = None) -> None:
        """添加点层
        可为底图加上一个点层

        Args:
            data: 点层图层数据
            lng_field: 经度字段
            lat_field: 纬度字段
            altitude_field: 高度字段
            layer_id: 图层ID
            color: 点颜色, RGB数值, 如[125, 241, 33]
            radius: 点的半径大小
            set_avg_zoom: 是否自动定位中心点
            time_field: 时间列字段名称
            time_format: 时间列的格式化字符串模板
            time_unit: 时间列的时间单位
            speed: 动画播放速度
            tooltip_fields: 参数未启用
            color_field: 参数未启用
            color_list: 参数未启用

        Returns:
        """
        layer_config = self.get_base_layer()
        layer_id = layer_id if layer_id is not None else rf'point-{self.point_count}'
        layer_config['id'] = layer_id
        layer_config["type"] = 'point'
        layer_config['config']['dataId'] = layer_id
        layer_config['config']['label'] = layer_id
        layer_config["config"]["columns"] = {"lat": lat_field, "lng": lng_field}
        if altitude_field is not None:
            layer_config["config"]["columns"]["altitude"] = altitude_field
        layer_config['config']['color'] = self.get_rgb_by_name(color, default_rgb=[65, 72, 88])
        layer_config["config"]["visConfig"]["radius"] = radius
        if color_field is not None:
            layer_config['visualChannels']['colorField'] = {'name': color_field,
                                                            'type': 'string'}
            layer_config['config']['visConfig']['colorRange'] = {'name': 'Custom Palette', 'type': 'custom',
                                                                 'category': 'Custom', 'colors': color_list,
                                                                 'reversed': False}

        self.user_config["config"]["visState"]["layers"].append(layer_config)

        if time_field is not None and time_field in data.columns:
            self.user_config['config']['visState']['filters'].append(
                self.__format_time_filter(data=data, time_field=time_field,
                                          time_format=time_format, time_unit=time_unit,
                                          layer_id=layer_id, speed=speed))
        if set_avg_zoom:
            cen_x, cen_y = get_avg_loc(df=data, x_field=lng_field, y_field=lat_field)
            self.user_config["config"]["mapState"]["longitude"] = cen_x
            self.user_config["config"]["mapState"]["latitude"] = cen_y
        self.data_dict[layer_id] = data
        if tooltip_fields is not None:
            self.user_config["config"]["visState"]["interactionConfig"]["tooltip"]["fieldsToShow"][
                layer_id] = self.tooltip_config(field_list=tooltip_fields)
        self.point_count += 1

    def add_geo_layer(self, data: gpd.GeoDataFrame, layer_id: str = None, color: list or str = None,
                      stroke_color: list or str = None,
                      width: float = 0.3, time_field: str = None,
                      time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's',
                      speed: float = 0.3, set_avg_zoom: bool = True, tooltip_fields: list[str] = None,
                      color_field: str = None, color_list: list = None):
        """添加几何图层
        可为底图加上一个几何图层(即含有geometry几何列)

        Args:
            data: 几何图层数据
            layer_id: 图层ID
            color: 填充颜色(RGB色值), 默认[65, 72, 88]
            stroke_color: 边缘填充颜色(RGB色值), 默认[65, 72, 88]
            width: 是否自动定位中心点
            time_field: 时间列字段名称
            time_format: 时间列的格式化字符串模板
            time_unit: 时间列的时间单位
            speed: 动画播放速度
            set_avg_zoom: 是否自动定位中心点
            tooltip_fields: 参数未启用
            color_field: 参数未启用
            color_list: 参数未启用

        Returns:
        """
        layer_config = self.get_base_layer()
        layer_id = layer_id if layer_id is not None else rf'geo-{self.geo_count}'
        layer_config['id'] = layer_id
        layer_config['type'] = 'geojson'
        layer_config['config']['dataId'] = layer_id
        layer_config['config']['label'] = layer_id
        layer_config["config"]["columns"] = {"geojson": 'geometry'}
        layer_config['config']['color'] = self.get_rgb_by_name(color, default_rgb=[100, 100, 100])
        layer_config['config']["visConfig"]['strokeColor'] = self.get_rgb_by_name(color, default_rgb=[100, 100, 100])
        layer_config['config']["visConfig"]['strokeColor'] = stroke_color
        layer_config['config']["visConfig"]['thickness'] = width
        if color_field is not None:
            layer_config['visualChannels']['colorField'] = {'name': color_field,
                                                            'type': 'string'}
            layer_config['config']['visConfig']['colorRange'] = {'name': 'Custom Palette', 'type': 'custom',
                                                                 'category': 'Custom', 'colors': color_list,
                                                                 'reversed': False}

        self.user_config["config"]["visState"]["layers"].append(layer_config)
        if time_field is not None and time_field in data.columns:
            self.user_config['config']['visState']['filters'].append(
                self.__format_time_filter(data=data, time_field=time_field,
                                          time_format=time_format, time_unit=time_unit,
                                          layer_id=layer_id, speed=speed))
        if set_avg_zoom:
            geo = data['geometry'].iloc[int(len(data) / 2)]
            _ = geo.buffer(0.0001).centroid
            self.user_config["config"]["mapState"]["longitude"] = _.x
            self.user_config["config"]["mapState"]["latitude"] = _.y
        self.data_dict[layer_id] = data
        if tooltip_fields is not None:
            self.user_config["config"]["visState"]["interactionConfig"]["tooltip"]["fieldsToShow"][
                layer_id] = self.tooltip_config(field_list=tooltip_fields)
        self.geo_count += 1

    def add_trip_layer(self, data: pd.DataFrame, lng_field: str = 'lng', lat_field: str = 'lat',
                       altitude_field: str = None,
                       time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's', layer_id: str = None,
                       thickness: float = 2.0, set_avg_zoom: bool = True,
                       opacity: float = 0.8, color: list or str = None,
                       trail_length: float = 120.0, tooltip_fields: list[str] = None):
        """添加路径动画图层
        可为底图加上一个路径动画图层(带时间字段的轨迹数据)

        Args:
            data: 轨迹数据
            lng_field: 经度字段
            lat_field: 纬度字段
            altitude_field: 高度字段
            time_format: 时间列的格式化字符串模板
            time_unit: 时间列的时间单位
            layer_id: 图层ID
            set_avg_zoom: 是否自动定位中心点
            thickness: 轨迹的显示宽度
            trail_length: 路径拖尾长度
            opacity: 轨迹的透明度
            color: 轨迹的颜色(RGB色值), 默认[241, 225, 37]
            tooltip_fields: 参数未启用

        Returns:
        """
        layer_config = self.get_base_layer()
        trip_data = generate_trip_layer(match_res_df=data, time_format=time_format, time_unit=time_unit,
                                        lng_field=lng_field, lat_field=lat_field, altitude_field=altitude_field)
        if set_avg_zoom:
            cen_x, cen_y = get_avg_loc(df=data, x_field=lng_field, y_field=lat_field)
            self.user_config["config"]["mapState"]["longitude"] = cen_x
            self.user_config["config"]["mapState"]["latitude"] = cen_y
        del data
        layer_id = layer_id if layer_id is not None else rf'trip-{self.trip_count}'
        layer_config['id'] = layer_id
        layer_config['type'] = 'trip'
        layer_config['config']['dataId'] = layer_id
        layer_config['config']['label'] = layer_id
        layer_config['config']['color'] = self.get_rgb_by_name(color, default_rgb=[241, 225, 37])
        layer_config["config"]["columns"] = {"geojson": '_geojson'}
        layer_config['config']["visConfig"]['thickness'] = thickness
        layer_config['config']["visConfig"]['opacity'] = opacity
        layer_config['config']["visConfig"]['trailLength'] = int(trail_length)
        self.user_config["config"]["visState"]["layers"].append(layer_config)
        self.data_dict[layer_id] = trip_data

        if tooltip_fields is not None:
            self.user_config["config"]["visState"]["interactionConfig"]["tooltip"]["fieldsToShow"][
                layer_id] = self.tooltip_config(field_list=tooltip_fields)
        self.trip_count += 1

    def export_html(self, height: float = 600, out_fldr: str = None, file_name: str = 'map'):
        """HTML输出
        将可视化HTML存储到磁盘且返回Map对象

        Args:
            height: 地图对象的高度
            out_fldr: 存储HTML的目录
            file_name: HTML文件的名称

        Returns:
            Kepler Map
        """
        try:
            user_map = KeplerGl(height=height, data=self.data_dict)  # data以图层名为键，对应的矢量数据为值
        except:
            user_map = KeplerGl(height=height)  # data以图层名为键，对应的矢量数据为值
            for key in self.data_dict.keys():
                user_map.add_data(self.data_dict[key], name=key)
        user_map.config = self.user_config
        if out_fldr is not None:
            user_map.save_to_html(file_name=os.path.join(out_fldr, file_name + '.html'))  # 导出到本地可编辑html文件
        return user_map

    @staticmethod
    def get_base_layer():
        return kepler_config.get_base_layer_config()

    @staticmethod
    def get_time_filter(layer_id: str = 'layer', time_field: str = 'time', s_time: float = None,
                        e_time: float = None, speed: float = 0.3) -> dict:
        time_filter_config = kepler_config.get_time_filter_config()
        time_filter_config['dataId'] = [layer_id]
        time_filter_config['id'] = layer_id
        time_filter_config['name'] = [time_field]
        if s_time is not None and e_time is not None:
            time_filter_config['value'] = [s_time, e_time]
        else:
            time_filter_config['value'] = [1652372040.0, 1652373265.0]
        time_filter_config['speed'] = speed
        return time_filter_config

    def __format_time_filter(self, data: pd.DataFrame or gpd.GeoDataFrame = None,
                             time_field: str = 'time', time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's',
                             layer_id: str = 'layer', speed: float = 0.3) -> dict:
        build_time_col(df=data, time_field=time_field, time_format=time_format, time_unit=time_unit)
        s_time, e_time = data[gps_field.TIME_FIELD].min().timestamp(), data[gps_field.TIME_FIELD].max().timestamp()
        data[gps_field.TIME_FIELD] = data[gps_field.TIME_FIELD].astype(str)
        return self.get_time_filter(layer_id=layer_id, time_field=time_field, s_time=s_time, e_time=e_time, speed=speed)

    @staticmethod
    def get_rgb_by_name(name: str or list, default_rgb: list = None) -> list:
        if isinstance(name, list):
            return name
        else:
            try:
                return color_map.color[name]
            except:
                return default_rgb

    @staticmethod
    def tooltip_config(field_list: list[str] = None) -> list[dict]:
        return [{"name": field, 'format': None} for field in field_list]

@function_time_cost
def export_visualization(hmm_obj_list: list[HiddenMarkov], export_all_agents: bool = False, use_gps_source: bool = True,
                         out_fldr: str = None, gps_radius: float = 6.0,
                         export_geo: bool = True, export_html: bool = True, flag_name: str = 'test',
                         multi_core_save: bool = False, sub_net_buffer: float = 200.0, dup_threshold: float = 10.0):
    """

    :param hmm_obj_list:
    :param export_all_agents:
    :param use_gps_source:
    :param out_fldr:
    :param gps_radius:
    :param export_geo:
    :param export_html:
    :param flag_name
    :param multi_core_save
    :param sub_net_buffer:
    :param dup_threshold
    :return:
    """
    out_fldr = './' if out_fldr is None else out_fldr

    if not multi_core_save or len(hmm_obj_list) <= 10 or os.cpu_count() <= 1:
        # print('single export')
        export_vs(hmm_obj_list=hmm_obj_list, use_gps_source=use_gps_source, gps_radius=gps_radius,
                  out_fldr=out_fldr, export_geo=export_geo, export_html=export_html, flag_name=flag_name,
                  sub_net_buffer=sub_net_buffer, dup_threshold=dup_threshold)
    else:
        # print('multi export')
        core_num = 3 if os.cpu_count() >= 3 else os.cpu_count()
        hmm_group = cut_group(obj_list=hmm_obj_list, n=core_num)
        del hmm_obj_list
        hmm_obj_list = []
        pool = multiprocessing.Pool(processes=len(hmm_group))
        res_list = []
        for i in range(0, len(hmm_group)):
            res = pool.apply_async(export_vs,
                                   args=(hmm_group[i], use_gps_source, gps_radius, out_fldr, export_geo, export_html,
                                         flag_name, sub_net_buffer, dup_threshold))
            res_list.append(res)
        pool.close()
        pool.join()
        del hmm_group
        for res in res_list:
            hmm_obj_list.extend(res.get())

    if export_all_agents and export_html:
        # 初始化一个匹配结果管理器
        all_vc = VisualizationCombination(use_gps_source=use_gps_source)
        all_vc.hmm_obj_list = hmm_obj_list
        try:
            all_vc.visualization(zoom=15, out_fldr=out_fldr, file_name=rf'{flag_name}-all_agents',
                                 gps_radius=gps_radius)
        except Exception as e:
            print(repr(e))
            print(rf'输出HTML可视化文件, 出现某些错误, 输出失败...')


def export_vs(hmm_obj_list: list[HiddenMarkov], use_gps_source: bool = True,
              gps_radius: float = 8.0, out_fldr: str = None, export_geo: bool = True, export_html: bool = True,
              flag_name: str = 'test', sub_net_buffer: float = 200.0, dup_threshold: float = 10.0) -> \
        list[HiddenMarkov]:
    _ = [export_v(hmm_obj=hmm_obj, use_gps_source=use_gps_source, gps_radius=gps_radius,
                  out_fldr=out_fldr, export_geo=export_geo, export_html=export_html, flag_name=flag_name,
                  sub_net_buffer=sub_net_buffer, dup_threshold=dup_threshold) for hmm_obj in hmm_obj_list]
    return _

def export_v(hmm_obj: HiddenMarkov, use_gps_source: bool = True, gps_radius: float = 8.0, out_fldr: str = None,
             export_geo: bool = True, export_html: bool = True, flag_name: str = 'test',
             sub_net_buffer: float = 200.0, dup_threshold: float = 10.0) -> HiddenMarkov:
    vc = VisualizationCombination(use_gps_source=use_gps_source, hmm_obj=hmm_obj)
    file_name = flag_name + '-' + str(hmm_obj.gps_points.agent_id)
    if export_html:
        try:
            vc.visualization(file_name=file_name, out_fldr=out_fldr, zoom=15,
                             gps_radius=gps_radius, sub_net_buffer=sub_net_buffer, dup_threshold=dup_threshold)
        except Exception as e:
            print(repr(e))
            print(rf'Output HTML visualization file, some errors occurred, output failed...')

    if export_geo:
        try:
            hmm_obj.acquire_geo_res(out_fldr=out_fldr,
                                    flag_name=file_name)
        except Exception as e:
            print(repr(e))
            print(rf'Output geojson visualization file, some errors occurred, output failed...')
    return hmm_obj


class VisualizationCombination(object):
    def __init__(self, hmm_obj: HiddenMarkov = None, use_gps_source: bool = False):
        if hmm_obj is None:
            self.__hmm_obj_list = []
        else:
            self.__hmm_obj_list = [hmm_obj]
        self.use_gps_source = use_gps_source

    def collect_hmm(self, hmm_obj: HiddenMarkov = None):
        self.__hmm_obj_list.append(hmm_obj)
    def extend_hmm(self, hmm_obj_list: list[HiddenMarkov] = None):
        self.__hmm_obj_list.extend(hmm_obj_list)

    @property
    def hmm_obj_list(self):
        return self.__hmm_obj_list

    @hmm_obj_list.setter
    def hmm_obj_list(self, hmm_obj_list: list[HiddenMarkov]):
        self.__hmm_obj_list = hmm_obj_list

    def visualization(self, zoom: int = 15, out_fldr: str = None, file_name: str = None,
                      link_width: float = 1.5, node_radius: float = 1.5,
                      match_link_width: float = 5.0, gps_radius: float = 3.0, sub_net_buffer: float = 200.0,
                      dup_threshold: float = 10.0) -> None:
        out_fldr = r'./' if out_fldr is None else out_fldr
        base_link_gdf = gpd.GeoDataFrame()
        base_node_gdf = gpd.GeoDataFrame()
        gps_link_gdf = gpd.GeoDataFrame()
        may_error_gdf = gpd.GeoDataFrame()
        for hmm in self.__hmm_obj_list:
            _gps_link_gdf, _base_link_gdf, _base_node_gdf, _error_gdf = hmm.acquire_visualization_res(
                use_gps_source=self.use_gps_source, link_width=link_width, gps_radius=gps_radius,
                match_link_width=match_link_width, node_radius=node_radius, sub_net_buffer=sub_net_buffer,
                dup_threshold=dup_threshold)
            base_link_gdf = pd.concat([base_link_gdf, _base_link_gdf])
            base_node_gdf = pd.concat([base_node_gdf, _base_node_gdf])
            gps_link_gdf = pd.concat([gps_link_gdf, _gps_link_gdf])
            may_error_gdf = pd.concat([may_error_gdf, _error_gdf])

        gps_link_gdf.reset_index(inplace=True, drop=True)
        base_link_gdf.drop_duplicates(subset=[net_field.LINK_ID_FIELD], keep='first', inplace=True)
        base_node_gdf.drop_duplicates(subset=[net_field.NODE_ID_FIELD], keep='first', inplace=True)

        try:
            del base_link_gdf[net_field.LINK_VEC_FIELD]
        except KeyError:
            pass
        base_link_gdf.reset_index(inplace=True, drop=True)
        base_node_gdf.reset_index(inplace=True, drop=True)

        generate_match_html(mix_gdf=gps_link_gdf, link_gdf=base_link_gdf, node_gdf=base_node_gdf,
                            error_gdf=may_error_gdf,
                            zoom=zoom,
                            out_fldr=out_fldr,
                            file_name=file_name)


def generate_match_html(mix_gdf: gpd.GeoDataFrame = None, out_fldr: str = None, file_name: str = None,
                        link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None,
                        error_gdf: gpd.GeoDataFrame = None, zoom: int = 15) -> None:
    """

    :param mix_gdf:
    :param out_fldr:
    :param file_name:
    :param link_gdf:
    :param node_gdf:
    :param error_gdf:
    :param zoom:
    :return:
    """
    mix_gdf.sort_values(by='type', ascending=True, inplace=True)
    mix_gdf[gps_field.LOC_TYPE] = mix_gdf[gps_field.LOC_TYPE].fillna('l')
    cen_geo = mix_gdf.at[0, net_field.GEOMETRY_FIELD].centroid
    cen_x, cen_y = cen_geo.x, cen_geo.y

    kv = KeplerVis(cen_loc=[cen_x, cen_y], zoom=zoom, show_roads=False)
    if error_gdf is not None and not error_gdf.empty:
        kv.add_geo_layer(data=error_gdf, layer_id=kepler_config.ERROR_XFER, width=0.6, color=[245, 97, 129],
                         set_avg_zoom=False)
    if mix_gdf is not None and not mix_gdf.empty:
        _color_map = dict()
        loc_type_set = mix_gdf[gps_field.LOC_TYPE].unique()
        if 'c' in loc_type_set:
            _color_map['c'] = '#FFFFFF'
            if 'd' in loc_type_set:
                _color_map['d'] = '#11BD2B'
        _color_map['l'] = '#438ECD'
        _color_map['s'] = '#FFC300'
        kv.add_geo_layer(data=mix_gdf, layer_id=kepler_config.MIX_NAME, width=0.1, color=[18, 147, 154],
                         time_field=gps_field.TIME_FIELD, set_avg_zoom=False,
                         tooltip_fields=[gps_field.AGENT_ID_FIELD, gps_field.POINT_SEQ_FIELD, gps_field.TIME_FIELD,
                                         net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD,
                                         gps_field.LOC_TYPE], color_field=gps_field.LOC_TYPE,
                         color_list=list(_color_map.values()))
    if node_gdf is not None:
        kv.add_geo_layer(data=node_gdf, layer_id=kepler_config.BASE_NODE_NAME, width=0.2, color=[100, 100, 100],
                         set_avg_zoom=False)
    if link_gdf is not None:
        kv.add_geo_layer(data=link_gdf, layer_id=kepler_config.BASE_LINK_NAME, width=0.3, color=[65, 72, 88],
                         set_avg_zoom=False)
    kv.export_html(height=600, out_fldr=out_fldr, file_name=file_name)


def generate_trip_layer(match_res_df: pd.DataFrame = None, lng_field: str = 'prj_lng', lat_field: str = 'prj_lat',
                        time_format: str = '%Y-%m-%d %H:%M:%S',
                        altitude_field: str = None,
                        time_unit: str = 's') -> dict:
    """
    Convert the matching results into kepler trip data
    :param match_res_df: pd.DataFrame()
    :param lng_field:
    :param lat_field:
    :param altitude_field:
    :param time_unit:
    :param time_format:
    :return:
    """
    assert {lng_field, lat_field, gps_field.TIME_FIELD}.issubset(match_res_df.columns)
    assert not match_res_df.empty

    def agg_seq(df: pd.DataFrame = None, use_z: bool = False):
        if use_z:
            seq_list = [[float(prj_lng), float(prj_lat), float(prj_h), int(t)] for prj_lng, prj_lat, prj_h, t in
                        zip(df[lng_field], df[lat_field], df[altitude_field], df[gps_field.TIME_FIELD])]
        else:
            seq_list = [[float(prj_lng), float(prj_lat), 0, int(t)] for prj_lng, prj_lat, t in
                        zip(df[lng_field], df[lat_field], df[gps_field.TIME_FIELD])]
        property_obj = {gps_field.AGENT_ID_FIELD: str(df[gps_field.AGENT_ID_FIELD].iloc[0])}
        obj = {"type": "Feature",
               "properties": property_obj,
               "geometry": {
                   "type": "LineString",
                   "coordinates": seq_list}
               }
        return obj
    if altitude_field is not None:
        assert altitude_field in match_res_df.columns, 'missing altitude field'
        match_res_df.dropna(subset=[lng_field, lat_field, altitude_field], how='any', axis=0, inplace=True)
    else:
        match_res_df.dropna(subset=[lng_field, lat_field], how='any', axis=0, inplace=True)

    build_time_col(df=match_res_df, time_format=time_format, time_unit=time_unit)
    match_res_df[gps_field.TIME_FIELD] = match_res_df[gps_field.TIME_FIELD].apply(
        lambda t: t.timestamp())
    trip_df = match_res_df.groupby(gps_field.AGENT_ID_FIELD).apply(lambda df: agg_seq(df)).reset_index(drop=False)
    res = {"type": "FeatureCollection", "features": trip_df[0].to_list()}
    return res

def generate_trip_layer_alpha(net: Net = Net, match_res_df: pd.DataFrame = None):
    single_link_gdf = net.get_link_data()
    com_idx = match_res_df['prj_lng'].isna()
    com_df = match_res_df[com_idx]


def get_avg_loc(df: pd.DataFrame = None, x_field: str = 'lng', y_field: str = 'lat') -> tuple[float, float]:
    if len(df) <= 100:
        return df[x_field].mean(), df[y_field].mean()
    else:
        sample_df = df.sample(n=100)
        return sample_df[x_field].mean(), sample_df[y_field].mean()
