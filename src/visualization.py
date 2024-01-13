# -- coding: utf-8 --
# @Time    : 2024/1/13 17:21
# @Author  : TangKai
# @Team    : ZheChengData
import datetime
import os
import json
import geopandas as gpd
from keplergl import KeplerGl
from src.tools.coord_trans import LngLatTransfer
con = LngLatTransfer()
from src.GlobalVal import GpsField

gps_field = GpsField()


def generate_html(mix_gdf: gpd.GeoDataFrame = None, out_fldr: str = None, file_name: str = None,
                  link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None, zoom: int = 15):
    """

    :param mix_gdf:
    :param out_fldr:
    :param file_name:
    :param link_gdf:
    :param node_gdf:
    :param zoom:
    :return:
    """
    # 生成KeplerGl对象s
    data_item = {'mix': mix_gdf}
    # 起点经纬度
    cen_geo = mix_gdf.at[0, 'geometry'].centroid
    cen_x, cen_y = cen_geo.x, cen_geo.y
    s_time, e_time = mix_gdf[gps_field.TIME_FIELD].min().timestamp(), mix_gdf[gps_field.TIME_FIELD].max().timestamp()
    mix_gdf[gps_field.TIME_FIELD] = mix_gdf[gps_field.TIME_FIELD].astype(str)
    with open(r'./config.json') as f:
        user_config = json.load(f)
    user_config["config"]["visState"]["filters"][0]["value"] = [s_time, e_time]
    user_config["config"]["mapState"]["latitude"] = cen_y
    user_config["config"]["mapState"]["longitude"] = cen_x
    user_config["config"]["mapState"]["zoom"] = int(zoom)

    if node_gdf is not None:
        node_item = generate_polygon_layer(color=[100, 100, 100], layer_id='base_node')
        user_config["config"]["visState"]["layers"].append(node_item)
        data_item['base_node'] = node_gdf

    if link_gdf is not None:
        link_item = generate_polygon_layer(color=[65, 72, 88], layer_id='base_link')
        user_config["config"]["visState"]["layers"].append(link_item)
        data_item['base_link'] = link_gdf

    user_map = KeplerGl(height=600, data=data_item)  # data以图层名为键，对应的矢量数据为值
    user_map.config = user_config
    user_map.save_to_html(file_name=os.path.join(out_fldr, file_name + '.html'))  # 导出到本地可编辑html文件

    print(user_map.config)


def generate_polygon_layer(color: list = None, layer_id: str = None):
    polygon_item = {
        "id": layer_id,
        "type": "geojson",
        "config": {
            "dataId": layer_id,
            "label": layer_id,
            "color": color,
            "highlightColor": [
                252,
                242,
                26,
                255
            ],
            "columns": {
                "geojson": "geometry"
            },
            "isVisible": True,
            "visConfig": {
                "opacity": 0.8,
                "strokeOpacity": 0.8,
                "thickness": 0.1,
                "strokeColor": [
                    221,
                    178,
                    124
                ],
                "radius": 10,
                "sizeRange": [
                    0,
                    10
                ],
                "radiusRange": [
                    0,
                    50
                ],
                "heightRange": [
                    0,
                    500
                ],
                "elevationScale": 5,
                "enableElevationZoomFactor": True,
                "stroked": False,
                "filled": True,
                "enable3d": False,
                "wireframe": False
            },
            "hidden": False,
            "textLabel": [
                {
                    "field": None,
                    "color": [
                        255,
                        255,
                        255
                    ],
                    "size": 18,
                    "offset": [
                        0,
                        0
                    ],
                    "anchor": "start",
                    "alignment": "center"
                }
            ]
        },
        "visualChannels": {
            "colorField": {
                "name": "type",
                "type": "string"
            },
            "colorScale": "ordinal",
            "strokeColorField": None,
            "strokeColorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear",
            "heightField": None,
            "heightScale": "linear",
            "radiusField": None,
            "radiusScale": "linear"
        }
    }
    return polygon_item


if __name__ == '__main__':
    a = datetime.datetime.now()
    print(a.timestamp())
