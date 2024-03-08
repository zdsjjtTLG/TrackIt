# -- coding: utf-8 --
# @Time    : 2024/3/8 9:56
# @Author  : TangKai
# @Team    : ZheChengData


import pyproj
import shapely
import pandas as pd
import geopandas as gpd
from shapely.ops import transform
import xml.etree.cElementTree as ET
from shapely.geometry import LineString, Point, Polygon

EDGE_ID_KEY = 'id'
EDGE_FROM_KEY = 'from'
EDGE_TO_KEY = 'to'
EDGE_SPEED_KEY = 'speed'
EDGE_LANES_KEY = 'numLanes'
EDGE_PRIORITY_KEY = 'priority'
EDGE_SHAPE_KEY = 'shape'
EDGE_FUNCTION_KEY = 'function'

LANE_ID_KEY = 'id'
LANE_SPEED_KEY = 'speed'
LANE_LENGTH_KEY = 'length'
LANE_WIDTH_KEY = 'width'
LANE_INDEX_KEY = 'index'
LANE_SHAPE_KEY = 'shape'

NODE_ID_KEY = 'id'
NODE_X_KEY = 'x'
NODE_Y_KEY = 'y'
NODE_TYPE_KEY = 'type'
NODE_COV_BOUND_KEY = 'convBoundary'

JUNCTION_ID_KEY = 'id'
JUNCTION_TYPE_KEY = 'type'
JUNCTION_X_KEY = 'x'
JUNCTION_Y_KEY = 'y'
JUNCTION_SHAPE_KEY = 'shape'


class SumoConvert(object):
    def __init__(self):
        pass

    # 2.解析net.xml文件
    def get_net_shp(self, net_path: str = None, crs: str = 'EPSG:32650') -> tuple[
        gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        :param net_path:
        :param crs:
        :return:
        """
        net_tree = ET.parse(net_path)
        net_root = net_tree.getroot()

        edge_df, junction_df = pd.DataFrame(), pd.DataFrame()

        for child in net_root:
            if child.tag == 'edge':
                _edge_df = self.parse_net_edge(net_edge_obj=child)
                edge_df = pd.concat([edge_df, _edge_df])
            elif child.tag == 'junction':
                _junction_df = self.parse_net_junction(junction_obj=child)
                junction_df = pd.concat([junction_df, _junction_df])
        junction_df.reset_index(inplace=True, drop=True)
        edge_df.reset_index(inplace=True, drop=True)

        edge_df['geometry'] = edge_df[EDGE_SHAPE_KEY].apply(lambda shape: LineString(shape))
        junction_df['geometry'] = junction_df[JUNCTION_SHAPE_KEY].apply(lambda shape: Polygon(shape))

        edge_df.drop(columns=[EDGE_SHAPE_KEY], axis=1, inplace=True)
        junction_df.drop(columns=[JUNCTION_SHAPE_KEY], axis=1, inplace=True)

        junction_gdf = gpd.GeoDataFrame(junction_df, geometry='geometry', crs=crs)
        edge_gdf = gpd.GeoDataFrame(edge_df, geometry='geometry', crs=crs)
        del edge_df, junction_df
        lane_polygon_gdf = edge_gdf[edge_gdf['function'] != 'internal'].copy()
        lane_polygon_gdf['geometry'] = lane_polygon_gdf.apply(lambda item:
                                                              get_off_polygon(l=item['geometry'],
                                                                              off_line_l=(item[
                                                                                              LANE_WIDTH_KEY] - 0.0005) / 2),
                                                              axis=1)
        return edge_gdf, junction_gdf, lane_polygon_gdf

    @staticmethod
    def parse_net_edge(net_edge_obj: ET.Element = None) -> pd.DataFrame:
        """

        :param net_edge_obj:
        :return:
        """
        lane_item_list = []
        edge_id, edge_function = net_edge_obj.get(EDGE_ID_KEY), net_edge_obj.get(EDGE_FUNCTION_KEY)
        for lane_obj in net_edge_obj:
            lane_id, lane_index, lane_speed, lane_length, lane_shape, lane_width = lane_obj.get(LANE_ID_KEY), int(
                lane_obj.get(
                    LANE_INDEX_KEY)), \
                float(lane_obj.get(LANE_SPEED_KEY)), float(lane_obj.get(LANE_LENGTH_KEY)), lane_obj.get(
                LANE_SHAPE_KEY), float(lane_obj.get(LANE_WIDTH_KEY))
            lane_shape = [list(map(float, xy.split(','))) for xy in lane_shape.split(' ')]

            lane_item_list.append([lane_id, lane_index, lane_speed, lane_length, lane_width, lane_shape])

        lane_df = pd.DataFrame(lane_item_list,
                               columns=[LANE_ID_KEY, LANE_INDEX_KEY, LANE_SPEED_KEY, LANE_LENGTH_KEY, LANE_WIDTH_KEY,
                                        LANE_SHAPE_KEY])
        lane_df[EDGE_ID_KEY] = edge_id
        lane_df[EDGE_FUNCTION_KEY] = edge_function
        return lane_df

    @staticmethod
    def parse_net_junction(junction_obj: ET.Element = None) -> pd.DataFrame:
        """

        :param junction_obj:
        :return:
        """
        junction_item_list = []
        junction_id, junction_type, junction_x, junction_y, junction_shape = \
            junction_obj.get(JUNCTION_ID_KEY), junction_obj.get(JUNCTION_TYPE_KEY), junction_obj.get(JUNCTION_X_KEY), \
                junction_obj.get(JUNCTION_Y_KEY), junction_obj.get(JUNCTION_SHAPE_KEY)
        if junction_type == 'internal':
            junction_shape = list(Point(junction_x, junction_y).buffer(1.5).exterior.coords)
        else:
            junction_shape = [list(map(float, xy.split(','))) for xy in junction_shape.split(' ')]

        junction_item_list.append([junction_id, junction_type, junction_x, junction_y, junction_shape])

        junction_df = pd.DataFrame(junction_item_list,
                                   columns=[JUNCTION_ID_KEY, JUNCTION_TYPE_KEY, JUNCTION_X_KEY, JUNCTION_Y_KEY,
                                            JUNCTION_SHAPE_KEY])
        return junction_df


def get_off_polygon(l: LineString = None, off_line_l: float = 1.8):
    a = l.offset_curve(off_line_l)
    b = l.offset_curve(-off_line_l)
    a_list, b_list = list(a.coords), list(b.coords)
    return Polygon(a_list + b_list[::-1])


def prj_xfer(from_crs='EPSG:4326', to_crs='EPSG:32650', origin_p: shapely.geometry = None) -> shapely.geometry:
    before = pyproj.CRS(from_crs)
    after = pyproj.CRS(to_crs)
    project = pyproj.Transformer.from_crs(before, after, always_xy=True).transform
    utm_geo = transform(project, origin_p)
    return utm_geo


if __name__ == '__main__':
    # 4.解析net.xml
    sc = SumoConvert()
    edge_shp, junction_shp, lane_p_shp = sc.get_net_shp(net_path='./data/input/0308/Town01.net.xml', crs='EPSG:32650')

    edge_shp.to_file(r'./data/output/0308/edge.shp', encoding='gbk')
    junction_shp.to_file(r'./data/output/0308/junction.shp', encoding='gbk')
    lane_p_shp.to_file(r'./data/output/0308/lane.shp', encoding='gbk')
