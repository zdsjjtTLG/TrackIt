# -- coding: utf-8 --
# @Time    : 2024/1/13 17:21
# @Author  : TangKai
# @Team    : ZheChengData

import os
import datetime
import pandas as pd
import multiprocessing
import geopandas as gpd
from keplergl import KeplerGl
from .tools.group import cut_group
from .GlobalVal import KeplerConfig
from .model.Markov import HiddenMarkov
from .GlobalVal import GpsField, NetField
from .WrapsFunc import function_time_cost
from .tools.coord_trans import LngLatTransfer


con = LngLatTransfer()
gps_field = GpsField()
net_field = NetField()
kepler_config = KeplerConfig()


@function_time_cost
def export_visualization(hmm_obj_list: list[HiddenMarkov], export_all_agents: bool = False, use_gps_source: bool = True,
                         out_fldr: str = None, gps_radius: float = 6.0,
                         export_geo: bool = True, export_html: bool = True, flag_name: str = 'test',
                         multi_core_save: bool = False):
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
    :return:
    """
    out_fldr = './' if out_fldr is None else out_fldr

    if not multi_core_save or len(hmm_obj_list) <= 10 or os.cpu_count() <= 1:
        # print('single export')
        export_vs(hmm_obj_list=hmm_obj_list, use_gps_source=use_gps_source, gps_radius=gps_radius,
                  out_fldr=out_fldr, export_geo=export_geo, export_html=export_html, flag_name=flag_name)
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
                                         flag_name))
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
              flag_name: str = 'test') -> list[HiddenMarkov]:
    _ = [export_v(hmm_obj=hmm_obj, use_gps_source=use_gps_source, gps_radius=gps_radius,
                  out_fldr=out_fldr, export_geo=export_geo, export_html=export_html, flag_name=flag_name) for hmm_obj in hmm_obj_list]
    return _

def export_v(hmm_obj: HiddenMarkov, use_gps_source: bool = True, gps_radius: float = 8.0, out_fldr: str = None,
             export_geo: bool = True, export_html: bool = True, flag_name: str = 'test') -> HiddenMarkov:
    vc = VisualizationCombination(use_gps_source=use_gps_source, hmm_obj=hmm_obj)
    file_name = flag_name + '-' + str(hmm_obj.gps_points.agent_id)
    if export_html:
        try:
            vc.visualization(file_name=file_name, out_fldr=out_fldr, zoom=15,
                             gps_radius=gps_radius)
        except Exception as e:
            print(repr(e))
            print(rf'输出HTML可视化文件, 出现某些错误, 输出失败...')

    if export_geo:
        hmm_obj.acquire_geo_res(out_fldr=out_fldr,
                                flag_name=file_name)
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
                      match_link_width: float = 5.0, gps_radius: float = 3.0) -> None:
        out_fldr = r'./' if out_fldr is None else out_fldr
        base_link_gdf = gpd.GeoDataFrame()
        base_node_gdf = gpd.GeoDataFrame()
        gps_link_gdf = gpd.GeoDataFrame()
        may_error_gdf = gpd.GeoDataFrame()
        for hmm in self.__hmm_obj_list:
            _gps_link_gdf, _base_link_gdf, _base_node_gdf, _error_gdf = hmm.acquire_visualization_res(
                use_gps_source=self.use_gps_source, link_width=link_width, gps_radius=gps_radius,
                match_link_width=match_link_width, node_radius=node_radius)
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
    if gps_field.HEADING_FIELD in mix_gdf.columns:
        mix_gdf.drop(columns=gps_field.HEADING_FIELD, axis=1, inplace=True)
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
        error_item = generate_polygon_layer(color=[245, 97, 129], layer_id=kepler_config.ERROR_XFER)
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

    user_map = KeplerGl(height=600, data=data_item)  # data以图层名为键，对应的矢量数据为值
    user_map.config = user_config
    user_map.save_to_html(file_name=os.path.join(out_fldr, file_name + '.html'))  # 导出到本地可编辑html文件


def generate_polygon_layer(color: list = None, layer_id: str = None) -> dict:
    polygon_item = kepler_config.get_polygon_config()
    polygon_item['id'] = layer_id
    polygon_item['config']['dataId'] = layer_id
    polygon_item['config']['label'] = layer_id
    polygon_item['config']['color'] = color
    return polygon_item


if __name__ == '__main__':
    a = datetime.datetime.now()
    print(a.timestamp())
