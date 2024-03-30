# -- coding: utf-8 --
# @Time    : 2024/2/10 15:27
# @Author  : TangKai
# @Team    : ZheChengData

import pyproj
import numpy as np
import pandas as pd
import geopandas as gpd
import xml.etree.cElementTree as ET
from shapely.geometry import LineString, Point, Polygon

"""SUMO路网转换的相关方法"""


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
DEFAULT_LANE_WIDTH = '3.2'
DEFAULT_LANE_SPEED = '9.0'

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

    # 1.从解耦的node和edge文件生产shp层
    def get_plain_shp(self, plain_edge_path: str = None, plain_node_path: str = None, crs: str = None) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        从解耦的node和edge文件生产shp层
        :param plain_edge_path:
        :param plain_node_path:
        :param crs:
        :return:
        """
        # 先解析节点
        node_tree = ET.parse(plain_node_path)
        node_root = node_tree.getroot()
        node_df = self.parse_node_palin(plain_node_root=node_root)

        node_loc_dict = {node_id: (x, y) for node_id, x, y in
                         zip(node_df[NODE_ID_KEY], node_df[NODE_X_KEY], node_df[NODE_Y_KEY])}

        node_df['geometry'] = node_df.apply(lambda xy: Point((xy[NODE_X_KEY], xy[NODE_Y_KEY])), axis=1)
        node_df.drop(columns=[NODE_X_KEY, NODE_Y_KEY], axis=1, inplace=True)
        node_gdf = gpd.GeoDataFrame(node_df, geometry='geometry', crs=crs)
        del node_df

        # 再解析edge
        edge_tree = ET.parse(plain_edge_path)
        edge_root = edge_tree.getroot()
        edge_df = self.parse_edge_palin(plain_edge_root=edge_root, node_loc_dict=node_loc_dict)

        # 生产几何列
        edge_df['geometry'] = edge_df[EDGE_SHAPE_KEY].apply(lambda shape: LineString(shape))
        edge_df.drop(columns=[EDGE_SHAPE_KEY], axis=1, inplace=True)

        edge_gdf = gpd.GeoDataFrame(edge_df, geometry='geometry', crs=crs)
        del edge_df

        return edge_gdf, node_gdf

    @staticmethod
    def parse_node_palin(plain_node_root: ET.Element = None) -> pd.DataFrame:
        """
        解析node文件
        :param plain_node_root:
        :return:
        """
        item_list: list[list[str, float, float, str]] = list()
        for child in plain_node_root:
            if child.tag == 'node':
                node_id, node_x, node_y, node_type = child.get(NODE_ID_KEY), float(child.get(NODE_X_KEY)), float(
                    child.get(
                        NODE_Y_KEY)), child.get(NODE_TYPE_KEY)
                item_list.append([node_id, node_x, node_y, node_type])
        return pd.DataFrame(item_list, columns=[NODE_ID_KEY, NODE_X_KEY, NODE_Y_KEY, NODE_TYPE_KEY])

    @staticmethod
    def parse_edge_palin(plain_edge_root: ET.Element = None, node_loc_dict: dict = None) -> pd.DataFrame:
        """
        解析edge文件
        :param plain_edge_root:
        :param node_loc_dict:
        :return:
        """
        edge_item_list = []
        for child in plain_edge_root:
            if child.tag == 'edge':
                edge_id, from_node, to_node, priority, lanes, speed, shape = child.get(EDGE_ID_KEY), child.get(
                    EDGE_FROM_KEY), child.get(
                    EDGE_TO_KEY), int(child.get(EDGE_PRIORITY_KEY)), int(child.get(EDGE_LANES_KEY)), \
                    float(child.get(EDGE_SPEED_KEY)), child.get(EDGE_SHAPE_KEY)
                if shape is None:
                    shape = [node_loc_dict[from_node], node_loc_dict[to_node]]
                else:
                    shape = [list(map(float, xy.split(','))) for xy in shape.split(' ')]
                edge_item_list.append([edge_id, from_node, to_node, priority, lanes, speed, shape])

        edge_df = pd.DataFrame(edge_item_list,
                               columns=[EDGE_ID_KEY, EDGE_FROM_KEY, EDGE_TO_KEY, EDGE_PRIORITY_KEY, EDGE_LANES_KEY,
                                        EDGE_SPEED_KEY, EDGE_SHAPE_KEY])
        return edge_df

    # 2.解析net.xml文件
    def get_net_shp(self, net_path: str = None, crs: str = None, prj4_str: str = None) -> tuple[
        gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        必须时平面投影
        从net.xml解析微观车道级路网
        :param net_path:
        :param crs:
        :return:
        """
        net_tree = ET.parse(net_path)
        net_root = net_tree.getroot()
        if crs is None:
            assert prj4_str is not None
            crs = 'EPSG:' + prj4_2_crs(prj4_str=prj4_str)
        edge_df, junction_df, avg_center_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for child in net_root:
            if child.tag == 'edge':
                _edge_df, _avg_center_df = self.parse_net_edge(net_edge_obj=child)
                edge_df = pd.concat([edge_df, _edge_df])
                avg_center_df = pd.concat([avg_center_df, _avg_center_df])
            elif child.tag == 'junction':
                _junction_df = self.parse_net_junction(junction_obj=child)
                junction_df = pd.concat([junction_df, _junction_df])
        junction_df.reset_index(inplace=True, drop=True)

        edge_df.reset_index(inplace=True, drop=True)
        avg_center_df.reset_index(inplace=True, drop=True)

        edge_df.rename(columns={EDGE_SHAPE_KEY: 'geometry'}, inplace=True)
        avg_center_df.rename(columns={EDGE_SHAPE_KEY: 'geometry'}, inplace=True)
        junction_df.rename(columns={JUNCTION_SHAPE_KEY: 'geometry'}, inplace=True)

        print(junction_df)
        junction_gdf = gpd.GeoDataFrame(junction_df, geometry='geometry', crs=crs)
        edge_gdf = gpd.GeoDataFrame(edge_df, geometry='geometry', crs=crs)
        avg_center_gdf = gpd.GeoDataFrame(avg_center_df, geometry='geometry', crs=crs)

        del edge_df, junction_df, avg_center_df

        lane_polygon_gdf = edge_gdf[edge_gdf['function'] != 'internal'].copy()
        lane_polygon_gdf['geometry'] = lane_polygon_gdf.apply(lambda item:
                                                              get_off_polygon(l=item['geometry'],
                                                                              off_line_l=(item[
                                                                                              LANE_WIDTH_KEY] - 0.05) / 2),
                                                              axis=1)
        return edge_gdf, junction_gdf, lane_polygon_gdf, avg_center_gdf

    @staticmethod
    def parse_net_edge(net_edge_obj: ET.Element = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """

        :param net_edge_obj:
        :return:
        """
        lane_item_list = []
        edge_id = net_edge_obj.get(EDGE_ID_KEY)
        edge_from, edge_to = net_edge_obj.get(EDGE_FROM_KEY), net_edge_obj.get(EDGE_TO_KEY)
        edge_function = net_edge_obj.get(EDGE_FUNCTION_KEY)
        edge_function = 'normal' if edge_function is None else edge_function

        avg_line_shape_list = []
        for lane_obj in net_edge_obj:

            lane_id, lane_index, lane_shape, lane_length, = lane_obj.get(LANE_ID_KEY), \
                lane_obj.get(LANE_INDEX_KEY), lane_obj.get(
                LANE_SHAPE_KEY), lane_obj.get(LANE_LENGTH_KEY)
            if lane_id is None or lane_index is None:
                print(net_edge_obj.get(EDGE_ID_KEY))
                continue

            lane_speed, lane_width = float(try_get_v(lane_obj, LANE_SPEED_KEY, DEFAULT_LANE_SPEED)), \
                float(try_get_v(lane_obj, LANE_WIDTH_KEY, DEFAULT_LANE_WIDTH))

            lane_shape = [list(map(float, xy.split(','))) for xy in lane_shape.split(' ')]
            avg_line_shape_list.append(lane_shape)

            lane_item_list.append([lane_id, lane_index, lane_speed, lane_length, lane_width, LineString(lane_shape)])

        lane_df = pd.DataFrame(lane_item_list,
                               columns=[LANE_ID_KEY, LANE_INDEX_KEY, LANE_SPEED_KEY, LANE_LENGTH_KEY, LANE_WIDTH_KEY,
                                        LANE_SHAPE_KEY])
        lane_df[EDGE_ID_KEY] = edge_id
        lane_df[EDGE_FUNCTION_KEY] = edge_function

        lane_df[EDGE_FROM_KEY], lane_df[EDGE_TO_KEY], lane_df[EDGE_FUNCTION_KEY] = edge_from, edge_to, edge_function

        avg_center_link_df = pd.DataFrame()

        if edge_function == 'normal':
            try:
                avg_center_line = LineString(
                    np.array(avg_line_shape_list).mean(axis=0))
            except ValueError:
                _l = len(avg_line_shape_list)
                select_line = [avg_line_shape_list[int(_l / 2)]]
                avg_center_line = LineString(
                    np.array(select_line).mean(axis=0))
            avg_center_link_df = pd.DataFrame({EDGE_ID_KEY: [edge_id],
                                               EDGE_FROM_KEY: [edge_from],
                                               EDGE_TO_KEY: [edge_to],
                                               'geometry': [avg_center_line]})

        return lane_df, avg_center_link_df

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
            junction_shape = Polygon(list(Point(junction_x, junction_y).buffer(1.5).exterior.coords))
        else:
            junction_shape = [list(map(float, xy.split(','))) for xy in junction_shape.split(' ')]
            _l = len(junction_shape)
            if _l >= 3:
                junction_shape = Polygon(junction_shape)
            elif _l == 2:
                junction_shape = LineString(junction_shape).buffer(1.0)
            else:
                junction_shape = LineString(list(Point(junction_x, junction_y).buffer(1.5).exterior.coords))

        junction_item_list.append([junction_id, junction_type, junction_x, junction_y, junction_shape])

        junction_df = pd.DataFrame(junction_item_list,
                                   columns=[JUNCTION_ID_KEY, JUNCTION_TYPE_KEY, JUNCTION_X_KEY, JUNCTION_Y_KEY,
                                            JUNCTION_SHAPE_KEY])
        return junction_df

    # 转换plain edge和plain node文件的坐标系
    def convert_plain_crs(self, plain_edge_path: str = None, plain_node_path: str = None, from_crs='EPSG:4326',
                          to_crs='EPSG:32650') -> tuple[ET.ElementTree, ET.ElementTree]:
        node_tree = ET.parse(plain_node_path)
        node_root = node_tree.getroot()

        edge_tree = ET.parse(plain_edge_path)
        edge_root = edge_tree.getroot()

        self.convert_plain_node_crs(node_root, from_crs=from_crs, to_crs=to_crs)
        self.convert_plain_edge_crs(edge_root, from_crs=from_crs, to_crs=to_crs)

        return node_tree, edge_tree

    # 转换plain_edge
    @staticmethod
    def convert_plain_edge_crs(plain_edge_root: ET.Element = None,
                               from_crs='EPSG:4326', to_crs='EPSG:32650'):

        for child in plain_edge_root:
            if child.tag == 'edge':
                shape = child.get(EDGE_SHAPE_KEY)
                if shape is None:
                    pass
                else:
                    shape = [list(map(float, xy.split(','))) for xy in shape.split(' ')]
                    origin_p = [Point(loc) for loc in shape]
                    xfer_p = [prj_xfer(origin_p=p, from_crs=from_crs, to_crs=to_crs) for p in origin_p]
                    xfer_shape = [[_xfer_p.x, _xfer_p.y] for _xfer_p in xfer_p]
                    child.set(EDGE_SHAPE_KEY, ' '.join([','.join(list(map(str, loc))) for loc in xfer_shape]))

    @staticmethod
    def convert_plain_node_crs(plain_node_root: ET.Element = None, from_crs='EPSG:4326',
                               to_crs='EPSG:32650') -> None:
        for child in plain_node_root:
            if child.tag == 'node':
                node_x, node_y = float(child.get(NODE_X_KEY)), float(child.get(NODE_Y_KEY))
                xfer_p = prj_xfer(from_crs=from_crs, to_crs=to_crs, origin_p=Point(node_x, node_y))
                child.set(NODE_X_KEY, str(xfer_p.x))
                child.set(NODE_Y_KEY, str(xfer_p.y))
            elif child.tag == 'location':
                # "120.29231071,31.60290186,120.29565375,31.60509861"
                conv_boundary = child.get(NODE_COV_BOUND_KEY).split(',')
                p1, p2 = Point(conv_boundary[0], conv_boundary[1]), Point(conv_boundary[2], conv_boundary[3])
                xfer_p1, xfer_p2 = prj_xfer(from_crs=from_crs, to_crs=to_crs, origin_p=p1), prj_xfer(
                    from_crs=from_crs, to_crs=to_crs, origin_p=p2)
                child.set(NODE_COV_BOUND_KEY, ','.join(list(map(str, [xfer_p1.x, xfer_p1.y, xfer_p2.x, xfer_p2.y]))))


def try_get_v(item_obj: ET.Element = None, k: str = None, default: str = None):
    res = item_obj.get(k)
    if res is None:
        return default
    else:
        return res


def get_off_polygon(l: LineString = None, off_line_l: float = 1.8):
    a = l.offset_curve(off_line_l)
    b = l.offset_curve(-off_line_l)
    a_list, b_list = list(a.coords), list(b.coords)
    return Polygon(a_list + b_list[::-1])


def prj4_2_crs(prj4_str: str = None) -> str:
    crs = pyproj.CRS(prj4_str)
    x = crs.to_epsg()
    return str(x)


if __name__ == '__main__':
    pass
