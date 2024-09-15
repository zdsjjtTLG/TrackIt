# -- coding: utf-8 --
# @Time    : 2024/1/13 17:21
# @Author  : TangKai
# @Team    : ZheChengData

import os
import datetime
import pandas as pd
import multiprocessing
import geopandas as gpd
from .map.Net import Net
from keplergl import KeplerGl
from .tools.group import cut_group
from .GlobalVal import KeplerConfig
from .model.Markov import HiddenMarkov
from .GlobalVal import GpsField, NetField
from .WrapsFunc import function_time_cost
from .tools.time_build import build_time_col
from .tools.coord_trans import LngLatTransfer

con = LngLatTransfer()
gps_field = GpsField()
net_field = NetField()
kepler_config = KeplerConfig()


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
            print(rf'输出HTML可视化文件, 出现某些错误, 输出失败...')

    if export_geo:
        try:
            hmm_obj.acquire_geo_res(out_fldr=out_fldr,
                                    flag_name=file_name)
        except Exception as e:
            print(repr(e))
            print(rf'输出geojson可视化文件, 出现某些错误, 输出失败...')
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

        generate_html(mix_gdf=gps_link_gdf, link_gdf=base_link_gdf, node_gdf=base_node_gdf, error_gdf=may_error_gdf,
                      zoom=zoom,
                      out_fldr=out_fldr,
                      file_name=file_name)


def generate_html(mix_gdf: gpd.GeoDataFrame = None, out_fldr: str = None, file_name: str = None,
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
    # 生成KeplerGl对象
    mix_gdf.sort_values(by='type', ascending=True, inplace=True)
    data_item = dict()

    # 起点经纬度
    cen_geo = mix_gdf.at[0, net_field.GEOMETRY_FIELD].centroid
    cen_x, cen_y = cen_geo.x, cen_geo.y
    s_time, e_time = mix_gdf[gps_field.TIME_FIELD].min().timestamp(), mix_gdf[gps_field.TIME_FIELD].max().timestamp()
    mix_gdf[gps_field.TIME_FIELD] = mix_gdf[gps_field.TIME_FIELD].astype(str)

    user_config = kepler_config.get_base_config()
    user_config["config"]["visState"]["filters"][0]["value"] = [s_time, e_time]
    user_config["config"]["mapState"]["latitude"] = cen_y
    user_config["config"]["mapState"]["longitude"] = cen_x
    user_config["config"]["mapState"]["zoom"] = int(zoom)

    if error_gdf is not None and not error_gdf.empty:
        error_item = generate_polygon_layer(color=[245, 97, 129], layer_id=kepler_config.ERROR_XFER, width=0.6)
        user_config["config"]["visState"]["layers"].append(error_item)
        data_item[kepler_config.ERROR_XFER] = error_gdf

    if node_gdf is not None:
        node_item = generate_polygon_layer(color=[100, 100, 100], layer_id=kepler_config.BASE_NODE_NAME)
        user_config["config"]["visState"]["layers"].append(node_item)
        data_item[kepler_config.BASE_NODE_NAME] = node_gdf

    if link_gdf is not None:
        link_item = generate_polygon_layer(color=[65, 72, 88], layer_id=kepler_config.BASE_LINK_NAME)
        user_config["config"]["visState"]["layers"].append(link_item)
        data_item[kepler_config.BASE_LINK_NAME] = link_gdf

    data_item[kepler_config.MIX_NAME] = mix_gdf

    try:
        user_map = KeplerGl(height=600, data=data_item)  # data以图层名为键，对应的矢量数据为值
    except:
        user_map = KeplerGl(height=600)  # data以图层名为键，对应的矢量数据为值
        if kepler_config.ERROR_XFER in data_item.keys():
            user_map.add_data(data_item[kepler_config.ERROR_XFER], name=kepler_config.ERROR_XFER)
        user_map.add_data(data_item[kepler_config.BASE_NODE_NAME], name=kepler_config.BASE_NODE_NAME)
        user_map.add_data(data_item[kepler_config.BASE_LINK_NAME], name=kepler_config.BASE_LINK_NAME)
        user_map.add_data(data_item[kepler_config.MIX_NAME], name=kepler_config.MIX_NAME)
    user_map.config = user_config
    user_map.save_to_html(file_name=os.path.join(out_fldr, file_name + '.html'))  # 导出到本地可编辑html文件


def generate_point_html(point_df: pd.DataFrame = None, out_fldr: str = None, file_name: str = None,
                        zoom: int = 15) -> None:
    """

    :param point_df
    :param out_fldr:
    :param file_name:
    :param zoom:
    :return:
    """
    if point_df is None or point_df.empty:
        return None
    # 生成KeplerGl对象
    point_df.sort_values(by='type', ascending=False, inplace=True)
    cen_x, cen_y = point_df[gps_field.LNG_FIELD].mean(), point_df[gps_field.LAT_FIELD].mean(),
    s_time, e_time = point_df[gps_field.TIME_FIELD].min().timestamp(), point_df[gps_field.TIME_FIELD].max().timestamp()
    point_df[gps_field.TIME_FIELD] = point_df[gps_field.TIME_FIELD].astype(str)

    user_config = kepler_config.get_base_config()
    user_config["config"]["visState"]["filters"][0]["dataId"] = kepler_config.TRAJECTORY_NAME
    user_config["config"]["visState"]["filters"][0]["value"] = [s_time, e_time]
    user_config["config"]["mapState"]["latitude"] = cen_y
    user_config["config"]["mapState"]["longitude"] = cen_x
    user_config["config"]["mapState"]["zoom"] = int(zoom)
    user_config["config"]["mapStyle"]["visibleLayerGroups"]["road"] = True

    point_item = generate_point_layer(color=[65, 72, 88], layer_id=kepler_config.TRAJECTORY_NAME)
    user_config["config"]["visState"]["layers"][0] = point_item

    try:
        user_map = KeplerGl(height=600, data={kepler_config.TRAJECTORY_NAME: point_df})  # data以图层名为键，对应的矢量数据为值
    except:
        user_map = KeplerGl(height=600)  # data以图层名为键，对应的矢量数据为值
        user_map.add_data(point_df, name=kepler_config.TRAJECTORY_NAME)

    user_map.config = user_config
    user_map.save_to_html(file_name=os.path.join(out_fldr, file_name + '.html'))  # 导出到本地可编辑html文件


def generate_polygon_layer(color: list = None, layer_id: str = None, width:float = 0.3) -> dict:
    polygon_item = kepler_config.get_polygon_config()
    polygon_item['id'] = layer_id
    polygon_item['config']['dataId'] = layer_id
    polygon_item['config']['label'] = layer_id
    polygon_item['config']["visConfig"]['strokeColor'] = color
    polygon_item['config']["visConfig"]['thickness'] = width
    return polygon_item

def generate_point_layer(color: list = None, layer_id: str = None) -> dict:
    polygon_item = kepler_config.get_point_config()
    polygon_item['id'] = layer_id
    polygon_item['config']['dataId'] = layer_id
    polygon_item['config']['label'] = layer_id
    polygon_item['config']['color'] = color
    return polygon_item


def generate_trip_layer(match_res_df: pd.DataFrame = None, time_format: str = '%Y-%m-%d %H:%M:%S',
                        time_unit: str = 's', user_field_list: list = None, out_fldr: str = r'./',
                        file_name: str = 'trip'):
    def agg_seq(df: pd.DataFrame = None):
        seq_list = [[float(prj_lng), float(prj_lat), 0, int(t)] for prj_lng, prj_lat, t in zip(df['prj_lng'], df['prj_lat'], df['time'])]
        property_obj = {'agent_id': str(df[gps_field.AGENT_ID_FIELD].iloc[0])}
        obj = {"type": "Feature",
               "properties": property_obj,
               "geometry": {
                   "type": "LineString",
                   "coordinates": seq_list}
               }
        return obj
    match_res_df.dropna(subset=['prj_lng'], axis=0, inplace=True)
    if match_res_df['time'].dtype not in ['datetime64[ns]', 'datetime64[ms]', 'datetime64[s]']:
        build_time_col(df=match_res_df, time_format=time_format, time_unit=time_unit)
    match_res_df[gps_field.TIME_FIELD] = match_res_df[gps_field.TIME_FIELD].apply(
        lambda t: t.timestamp())

    trip_df = match_res_df.groupby(gps_field.AGENT_ID_FIELD).apply(lambda df: agg_seq(df)).reset_index(drop=False)
    res = {"type": "FeatureCollection", "features": trip_df[0].to_list()}

    kv = KeplerVis()
    kv.add_trip_layer(layer_id='trip', data=res, thickness=2, trail_length=15, color=[241, 225, 37])
    kv.export_html(out_fldr=out_fldr, file_name=file_name)
    # import json
    # with open(os.path.join(out_fldr, file_name + '.geojson'), 'w') as f:
    #     json.dump(res, f)

def generate_trip_layer_alpha(net: Net = Net, match_res_df: pd.DataFrame = None):
    single_link_gdf = net.get_link_data()
    com_idx = match_res_df['prj_lng'].isna()
    com_df = match_res_df[com_idx]


class KeplerVis(object):
    def __init__(self, zoom: int = 15, s_time: float = None, e_time: float = None,
                 cen_loc: list[float] or tuple[float] = None):
        self.user_config = kepler_config.get_glb_map_config()
        if s_time is not None and e_time is not None:
            self.user_config["config"]["visState"]["filters"][0]["value"] = [s_time, e_time]
        if cen_loc is not None:
            self.user_config["config"]["mapState"]["longitude"] = cen_loc[0]
            self.user_config["config"]["mapState"]["latitude"] = cen_loc[1]
        self.user_config["config"]["mapState"]["zoom"] = int(zoom)
        self.data_dict = dict()

    def add_point_layer(self, data: pd.DataFrame = None, lng_field: str = 'lng', lat_field: str = 'lat',
                        altitude_field: str = None, layer_id: str = 'point', color: list = None, radius: float = 3,
                        set_avg_zoom: bool = False, time_field: str = None, s_time: float = None, e_time: float = None,
                        speed: float = 0.3):
        layer_config = self.get_base_layer()
        layer_config['id'] = layer_id
        layer_config["type"] = 'point'
        layer_config['config']['dataId'] = layer_id
        layer_config['config']['label'] = layer_id
        layer_config["config"]["columns"]["lat"] = lat_field
        layer_config["config"]["columns"]["lng"] = lng_field
        if altitude_field is not None:
            layer_config["config"]["columns"]["altitude"] = altitude_field
        if color is None:
            layer_config['config']['color'] = [65, 72, 88]
        else:
            layer_config['config']['color'] = color
        layer_config["config"]["visConfig"]["radius"] = radius

        self.user_config["config"]["visState"]["layers"].append(layer_config)
        if time_field is not None and time_field in data.columns:
            self.user_config['config']['visState']['filters'].append(self.get_time_filter(layer_id=layer_id,
                                                                                          time_field=time_field,
                                                                                          s_time=s_time, e_time=e_time,
                                                                                          speed=speed))
        if set_avg_zoom:
            cen_x, cen_y = data[lng_field].mean(), data[lat_field].mean()
            self.user_config["config"]["mapState"]["longitude"] = cen_x
            self.user_config["config"]["mapState"]["latitude"] = cen_y
        self.data_dict[layer_id] = data

    def add_polygon_layer(self, layer_id: str = 'polygon', color: list = None, stroke_color: list = None,
                          width: float = 0.3, data: gpd.GeoDataFrame = None, time_field: str = None,
                          s_time: float = None, e_time: float = None,
                          speed: float = 0.3):
        layer_config = self.get_base_layer()
        layer_config['id'] = layer_id
        layer_config['type'] = 'geojson'
        layer_config['config']['dataId'] = layer_id
        layer_config['config']['label'] = layer_id
        layer_config["config"]["columns"]["geojson"] = 'geometry'
        if color is None:
            layer_config['config']['color'] = [100, 100, 100]
        if stroke_color is None:
            layer_config['config']["visConfig"]['strokeColor'] = [100, 100, 100]
        layer_config['config']["visConfig"]['thickness'] = width
        self.user_config["config"]["visState"]["layers"].append(layer_config)
        if time_field is not None and time_field in data.columns:
            self.user_config['config']['visState']['filters'].append(self.get_time_filter(layer_id=layer_id,
                                                                                          time_field=time_field,
                                                                                          s_time=s_time, e_time=e_time,
                                                                                          speed=speed))
        self.data_dict[layer_id] = data

    def add_trip_layer(self, layer_id: str = 'trip', thickness: float = 8.0, opacity: float = 0.8, color: list = None,
                       trail_length: float = 60.0, data: dict = None):

        layer_config = self.get_base_layer()
        layer_config['id'] = layer_id
        layer_config['type'] = 'trip'
        layer_config['config']['dataId'] = layer_id
        layer_config['config']['label'] = layer_id
        layer_config['config']['color'] = color
        layer_config["config"]["columns"]["geojson"] = '_geojson'
        layer_config['config']["visConfig"]['thickness'] = thickness
        layer_config['config']["visConfig"]['opacity'] = opacity
        layer_config['config']["visConfig"]['trailLength'] = int(trail_length)
        self.user_config["config"]["visState"]["layers"].append(layer_config)
        self.data_dict[layer_id] = data

    def export_html(self, height: float = 600, out_fldr: str = r'./', file_name: str = 'map'):
        try:
            user_map = KeplerGl(height=height, data=self.data_dict)  # data以图层名为键，对应的矢量数据为值
        except:
            user_map = KeplerGl(height=height)  # data以图层名为键，对应的矢量数据为值
            for key in self.data_dict.keys():
                user_map.add_data(self.data_dict[key], name=key)
        user_map.config = self.user_config
        user_map.save_to_html(file_name=os.path.join(out_fldr, file_name + '.html'))  # 导出到本地可编辑html文件
        return user_map

    @staticmethod
    def get_base_layer():
        return kepler_config.get_base_layer_config()

    @staticmethod
    def get_time_filter(layer_id: str = 'layer', time_field: str = 'time', s_time: float = None,
                        e_time: float = None, speed: float = 0.3) -> None:
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


if __name__ == '__main__':
    a = datetime.datetime.now()
    print(a.timestamp())
