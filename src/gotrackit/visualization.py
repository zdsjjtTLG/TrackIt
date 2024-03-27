# -- coding: utf-8 --
# @Time    : 2024/1/13 17:21
# @Author  : TangKai
# @Team    : ZheChengData

import os
import datetime
import pandas as pd
import geopandas as gpd
from keplergl import KeplerGl
# from gotrackit.GlobalVal import KeplerConfig
# from gotrackit.model.Markov import HiddenMarkov
# from gotrackit.GlobalVal import GpsField, NetField
# from gotrackit.tools.coord_trans import LngLatTransfer
from .GlobalVal import KeplerConfig
from .model.Markov import HiddenMarkov
from .GlobalVal import GpsField, NetField
from .tools.coord_trans import LngLatTransfer


con = LngLatTransfer()
gps_field = GpsField()
net_field = NetField()
kepler_config = KeplerConfig()


class VisualizationCombination(object):
    def __init__(self, hmm_obj: HiddenMarkov = None, use_gps_source: bool = False):
        if hmm_obj is None:
            self.__hmm_obj_list = []
        else:
            self.__hmm_obj_list = [hmm_obj]
        self.use_gps_source = use_gps_source

    def collect_hmm(self, hmm_obj: HiddenMarkov = None):
        self.__hmm_obj_list.append(hmm_obj)

    def visualization(self, zoom: int = 15, out_fldr: str = None, file_name: str = None) -> None:
        out_fldr = r'./' if out_fldr is None else out_fldr
        base_link_gdf = gpd.GeoDataFrame()
        base_node_gdf = gpd.GeoDataFrame()
        gps_link_gdf = gpd.GeoDataFrame()
        for hmm in self.__hmm_obj_list:
            _gps_link_gdf, _base_link_gdf, _base_node_gdf = hmm.acquire_visualization_res(
                use_gps_source=self.use_gps_source)
            base_link_gdf = pd.concat([base_link_gdf, _base_link_gdf])
            base_node_gdf = pd.concat([base_node_gdf, _base_node_gdf])
            gps_link_gdf = pd.concat([gps_link_gdf, _gps_link_gdf])

        gps_link_gdf.reset_index(inplace=True, drop=True)
        base_link_gdf.drop_duplicates(subset=[net_field.LINK_ID_FIELD], keep='first', inplace=True)
        base_node_gdf.drop_duplicates(subset=[net_field.NODE_ID_FIELD], keep='first', inplace=True)
        try:
            del base_link_gdf[net_field.LINK_VEC_FIELD]
        except KeyError:
            pass
        base_link_gdf.reset_index(inplace=True, drop=True)
        base_node_gdf.reset_index(inplace=True, drop=True)

        generate_html(mix_gdf=gps_link_gdf, link_gdf=base_link_gdf, node_gdf=base_node_gdf, zoom=zoom,
                      out_fldr=out_fldr,
                      file_name=file_name)


def generate_html(mix_gdf: gpd.GeoDataFrame = None, out_fldr: str = None, file_name: str = None,
                  link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None, zoom: int = 15) -> None:
    """

    :param mix_gdf:
    :param out_fldr:
    :param file_name:
    :param link_gdf:
    :param node_gdf:
    :param zoom:
    :return:
    """
    # 生成KeplerGl对象
    if gps_field.HEADING_FIELD in mix_gdf.columns:
        mix_gdf.drop(columns=gps_field.HEADING_FIELD, axis=1, inplace=True)
    data_item = {kepler_config.MIX_NAME: mix_gdf}

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

    if node_gdf is not None:
        node_item = generate_polygon_layer(color=[100, 100, 100], layer_id=kepler_config.BASE_NODE_NAME)
        user_config["config"]["visState"]["layers"].append(node_item)
        data_item[kepler_config.BASE_NODE_NAME] = node_gdf

    if link_gdf is not None:
        link_item = generate_polygon_layer(color=[65, 72, 88], layer_id=kepler_config.BASE_LINK_NAME)
        user_config["config"]["visState"]["layers"].append(link_item)
        data_item[kepler_config.BASE_LINK_NAME] = link_gdf

    user_map = KeplerGl(height=600, data=data_item)  # data以图层名为键，对应的矢量数据为值
    user_map.config = user_config
    user_map.save_to_html(file_name=os.path.join(out_fldr, file_name + '.html'))  # 导出到本地可编辑html文件
    # print(data_item)
    # print(user_map.config)


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
