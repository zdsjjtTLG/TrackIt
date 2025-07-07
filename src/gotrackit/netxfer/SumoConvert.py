# -- coding: utf-8 --
# @Time    : 2024/2/10 15:27
# @Author  : TangKai
# @Team    : ZheChengData

"""SUMO路网转换的相关方法"""

import os
import pyproj
import numpy as np
import pandas as pd
import multiprocessing
from pyproj import CRS
import geopandas as gpd
from shapely.ops import linemerge
from shapely.ops import transform
import xml.etree.cElementTree as ET
from ..tools.group import cut_group
from ..GlobalVal import NetField, GpsField
from ..WrapsFunc import function_time_cost
from shapely.geometry import LineString, Point, Polygon
from ..tools.common import avoid_duplicate_cols
from ..tools.coord_trans import LngLatTransfer
from ..MatchResAna import del_dup_links

net_field = NetField()
gps_field = GpsField()
agent_id_field = gps_field.AGENT_ID_FIELD
time_field = gps_field.TIME_FIELD
link_id_field = net_field.LINK_ID_FIELD
dir_field = net_field.DIRECTION_FIELD
from_node_field = net_field.FROM_NODE_FIELD
to_node_field = net_field.TO_NODE_FIELD
length_field = net_field.LENGTH_FIELD
geometry_field = net_field.GEOMETRY_FIELD
node_id_field = net_field.NODE_ID_FIELD

LW_INFO_FIELD = 'lane_width_info'
LT_INFO_FIELD = 'lane_type_info'
LM_INFO_FIELD = 'lane_mode_info'
LINK_ID_FIELD = 'link_id'
FROM_NODE_FIELD = 'from_node'
TO_NODE_FIELD = 'to_node'
LANE_NUM_FIELD = 'lane_num'
SPEED_FIELD = 'speed'
SPREAD_TYPE = 'spread_type'
WIDTH_FIELD = 'lane_width'
GEOMETRY_FIELD = 'geometry'
EDGE_SPREAD_TYPE = 'spreadType'

DEFAULT_LANE_NUM = 2
DEFAULT_SPEED = 9.0

EDGE_ID_KEY = 'id'
EDGE_FROM_KEY = 'from'
EDGE_TO_KEY = 'to'
EDGE_SPEED_KEY = 'speed'
EDGE_LANES_KEY = 'numLanes'
EDGE_PRIORITY_KEY = 'priority'
EDGE_SHAPE_KEY = 'shape'
EDGE_FUNCTION_KEY = 'function'
ALLOW_MODE_KEY = 'allow'

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

CONN_FROM_EDGE_KEY = 'from'
CONN_TO_EDGE_KEY = 'to'
CONN_FROM_LANE_KEY = 'fromLane'
CONN_TO_LANE_KEY = 'toLane'
CONN_VIA_KEY = 'via'

class SumoConvert(object):
    def __init__(self):
        pass

    def get_plain_shp(self, plain_edge_path: str = None, plain_node_path: str = None, crs: str = None) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """SumoConvert类方法 - get_plain_shp

        - 从解耦的node和edge文件生产shp层

        Args:
            plain_edge_path:
            plain_node_path:
            crs:

        Returns:

        """
        # 先解析节点
        node_tree = ET.parse(plain_node_path)
        node_root = node_tree.getroot()
        node_df = self.parse_node_plain(plain_node_root=node_root)

        node_loc_dict = {node_id: (x, y) for node_id, x, y in
                         zip(node_df[NODE_ID_KEY], node_df[NODE_X_KEY], node_df[NODE_Y_KEY])}

        node_df[geometry_field] = node_df.apply(lambda xy: Point((xy[NODE_X_KEY], xy[NODE_Y_KEY])), axis=1)
        node_df.drop(columns=[NODE_X_KEY, NODE_Y_KEY], axis=1, inplace=True)
        node_gdf = gpd.GeoDataFrame(node_df, geometry=geometry_field, crs=crs)
        del node_df

        # 再解析edge
        edge_tree = ET.parse(plain_edge_path)
        edge_root = edge_tree.getroot()
        edge_df = self.parse_edge_plain(plain_edge_root=edge_root, node_loc_dict=node_loc_dict)

        # 生产几何列
        edge_df[geometry_field] = edge_df[EDGE_SHAPE_KEY].apply(lambda shape: LineString(shape))
        edge_df.drop(columns=[EDGE_SHAPE_KEY], axis=1, inplace=True)

        edge_gdf = gpd.GeoDataFrame(edge_df, geometry=geometry_field, crs=crs)
        del edge_df

        return edge_gdf, node_gdf

    @staticmethod
    def parse_node_plain(plain_node_root: ET.Element = None) -> pd.DataFrame:
        """SumoConvert类方法 - parse_node_plain

        - 解析node文件

        Args:
            plain_node_root:

        Returns:

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
    def parse_edge_plain(plain_edge_root: ET.Element = None, node_loc_dict: dict = None) -> pd.DataFrame:
        """SumoConvert类方法 - parse_edge_plain

        - 解析edge文件

        Args:
            plain_edge_root:
            node_loc_dict:

        Returns:

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

    @function_time_cost
    def get_net_shp(self, net_path: str, core_num: int = 1, l_threshold: float = 1.0) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """SumoConvert类静态方法 - get_net_shp

        - 解析net.xml路网获取GeoDataFrame

        Args:
            net_path: net.xml路网文件路径
            core_num:
            l_threshold: 线型简化阈值(米)

        Returns:
            车道骨架gdf、路口面域gdf、车道面域gdf、路段中心线gdf、路口连接器gdf
        """
        core_num = os.cpu_count() if core_num > os.cpu_count() else core_num
        net_tree = ET.parse(net_path)

        net_root = net_tree.getroot()
        location_ele = net_root.findall('location')[0]
        try:
            prj4_str = location_ele.get('projParameter')
        except:
            raise ValueError('There is no projParameter in net.xml file')


        code = prj4_2_crs(prj4_str=prj4_str)
        if code is None:
            crs, no_crs_flag = None, True
        else:
            crs, no_crs_flag = rf'EPSG:{code}', False

        try:
            x_offset, y_offset = list(map(float, location_ele.get('netOffset').split(',')))
        except:
            x_offset, y_offset = 0, 0

        all_edge_ele = list(net_root.findall('edge'))
        all_junction_ele = list(net_root.findall('junction'))
        all_conn_ele = list(net_root.findall('connection'))

        if core_num > 1:
            # 分组
            edge_ele_group = cut_group(obj_list=all_edge_ele, n=core_num)
            junction_ele_group = cut_group(obj_list=all_junction_ele, n=core_num)
            conn_ele_group = cut_group(obj_list=all_conn_ele, n=core_num)

            edge_ele_group_len, junction_ele_group_len, conn_ele_group_len = \
                len(edge_ele_group), len(junction_ele_group), len(conn_ele_group)

            max_len = max([edge_ele_group_len, junction_ele_group_len, conn_ele_group_len])

            junction_ele_group.extend([] * (max_len - junction_ele_group_len))
            edge_ele_group.extend([] * (max_len - edge_ele_group_len))
            conn_ele_group.extend([] * (max_len - conn_ele_group_len))

            del all_edge_ele, all_junction_ele, all_conn_ele

            pool = multiprocessing.Pool(processes=core_num)
            result_list = []
            for i in range(len(edge_ele_group)):
                result = pool.apply_async(self.parse_elements,
                                          args=(edge_ele_group[i], junction_ele_group[i], conn_ele_group[i], x_offset, y_offset, prj4_str, no_crs_flag))
                result_list.append(result)
            pool.close()
            pool.join()

            # 车道线, edge中心线, 交叉口面域
            lane_gdf, avg_edge_gdf, junction_gdf, conn_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            for res in result_list:
                _lane_df, _avg_edge_df, _junction_df, _conn_df = res.get()
                lane_gdf = pd.concat([lane_gdf, _lane_df])
                avg_edge_gdf = pd.concat([avg_edge_gdf, _avg_edge_df])
                junction_gdf = pd.concat([junction_gdf, _junction_df])
                conn_df = pd.concat([_conn_df, conn_df])
            lane_gdf.reset_index(inplace=True, drop=True)
            avg_edge_gdf.reset_index(inplace=True, drop=True)
            junction_gdf.reset_index(inplace=True, drop=True)
            conn_df.reset_index(inplace=True)
            del result_list
        else:
            lane_gdf, avg_edge_gdf, junction_gdf, conn_df = self.parse_elements(edge_ele_list=all_edge_ele,
                                                                                junction_ele_list=all_junction_ele,
                                                                                conn_ele_list=all_conn_ele,
                                                                                x_offset=x_offset, y_offset=y_offset,
                                                                                proj=prj4_str, no_crs_flag=no_crs_flag)

        avg_edge_gdf.drop(index=avg_edge_gdf[avg_edge_gdf['function'] == 'internal'].index, axis=0, inplace=True)

        # crs convert
        if no_crs_flag:
            crs = 'EPSG:3857'
            if not lane_gdf.empty:
                lane_gdf = gpd.GeoDataFrame(lane_gdf, geometry='geometry', crs='EPSG:4326')
            if not avg_edge_gdf.empty:
                avg_edge_gdf = gpd.GeoDataFrame(avg_edge_gdf, geometry='geometry', crs='EPSG:4326')
            if not junction_gdf.empty:
                junction_gdf = gpd.GeoDataFrame(junction_gdf, geometry='geometry', crs='EPSG:4326')
            lane_gdf = lane_gdf.to_crs(crs)
            avg_edge_gdf = avg_edge_gdf.to_crs(crs)
            junction_gdf = junction_gdf.to_crs(crs)

        try:
            lane_gdf[geometry_field] = gpd.GeoSeries(lane_gdf[geometry_field]).remove_repeated_points(l_threshold)
            avg_edge_gdf[geometry_field] = gpd.GeoSeries(avg_edge_gdf[geometry_field]).remove_repeated_points(
                l_threshold)
        except:
            lane_gdf[geometry_field] = gpd.GeoSeries(lane_gdf[geometry_field]).simplify(l_threshold / 5)
            avg_edge_gdf[geometry_field] = gpd.GeoSeries(avg_edge_gdf[geometry_field]).simplify(l_threshold / 5)

        lane_polygon_gdf = lane_gdf[lane_gdf['function'] != 'internal'].copy()
        lane_polygon_gdf[geometry_field] = \
            lane_polygon_gdf.apply(lambda item:
                                   get_off_polygon(l=item[geometry_field],
                                                   off_line_l=(item[
                                                                   LANE_WIDTH_KEY] - 1e-7) / 2),
                                   axis=1)
        conn_df = self.process_conn(pre_conn_df=conn_df)

        if not no_crs_flag:
            junction_gdf = gpd.GeoDataFrame(junction_gdf, geometry=geometry_field, crs=crs)
            lane_gdf = gpd.GeoDataFrame(lane_gdf, geometry=geometry_field, crs=crs)
            avg_edge_gdf = gpd.GeoDataFrame(avg_edge_gdf, geometry=geometry_field, crs=crs)
            lane_polygon_gdf = gpd.GeoDataFrame(lane_polygon_gdf, geometry=geometry_field, crs=crs)

        if not conn_df.empty:
            conn_gdf = self.tess_lane(conn_df=conn_df, lane_gdf=lane_gdf)
            conn_gdf = conn_gdf.explode(ignore_index=True)
            avg_conn = conn_gdf.drop_duplicates(subset=['from_edge', 'to_edge'], keep='first')[['from_edge', 'to_edge']]
            avg_edge_geo_map = {edge: geo for edge, geo in zip(avg_edge_gdf['edge_id'], avg_edge_gdf[geometry_field])}
            avg_conn[geometry_field] = avg_conn.apply(
                lambda row: LineString([list(avg_edge_geo_map[row['from_edge']].coords)[-1],
                                        list(avg_edge_geo_map[row['to_edge']].coords)[0]]), axis=1,
                result_type='expand')
            del avg_conn['from_edge'], avg_conn['to_edge']
            avg_conn['function'] = 'conn'
            avg_conn = gpd.GeoDataFrame(avg_conn, geometry=geometry_field, crs=avg_edge_gdf.crs)
            avg_edge_gdf = pd.concat([avg_edge_gdf, avg_conn])
            avg_edge_gdf.reset_index(inplace=True, drop=True)
        else:
            conn_gdf = gpd.GeoDataFrame()

        return lane_gdf, junction_gdf, lane_polygon_gdf, avg_edge_gdf, conn_gdf

    @staticmethod
    def tess_lane(conn_df: pd.DataFrame = None, lane_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
        # normal_lane = lane_gdf[~lane_gdf['edge_id'].str.startswith(':')].copy()
        conn_lane = lane_gdf[lane_gdf['edge_id'].str.startswith(':')].copy()
        # print(normal_lane)
        # print(conn_lane)
        conn_df = conn_df.explode(column=['via'], ignore_index=True)
        conn_lane['lane_id'] = conn_lane['edge_id'].astype(str) + '_' + conn_lane['index'].astype(str)

        conn_df = pd.merge(conn_df, conn_lane[['lane_id', geometry_field]], left_on='via', right_on=['lane_id'])
        conn_df = conn_df.groupby(['from_edge', 'from_lane_index',
                                   'to_edge', 'to_lane_index']).agg({geometry_field: list}).reset_index(drop=False)
        conn_df[geometry_field] = conn_df[geometry_field].apply(lambda x: linemerge(x))
        conn_gdf = gpd.GeoDataFrame(conn_df, geometry=geometry_field, crs=lane_gdf.crs)
        return conn_gdf

    def parse_elements(self, edge_ele_list: list[ET.Element] = None, junction_ele_list: list[ET.Element] = None,
                       conn_ele_list: list[ET.Element] = None, x_offset: float = 0.0, y_offset: float = 0.0,
                       proj: str = None, no_crs_flag: bool = False) \
            -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """

        Args:
            edge_ele_list:
            junction_ele_list:
            conn_ele_list:
            x_offset:
            y_offset:
            proj:
            no_crs_flag:

        Returns:

        """
        lane_item_list, avg_edge_item_list = list(), list()
        for edge_ele in edge_ele_list:
            _lane_item_list, avg_edge_item = self.parse_net_edge(net_edge_obj=edge_ele, x_offset=x_offset,
                                                                 y_offset=y_offset, proj=proj, no_crs_flag=no_crs_flag)
            lane_item_list.extend(_lane_item_list)
            avg_edge_item_list.append(avg_edge_item)
        lane_df = pd.DataFrame(lane_item_list,
                               columns=['edge_' + EDGE_ID_KEY, EDGE_FROM_KEY, EDGE_TO_KEY, EDGE_FUNCTION_KEY,
                                        LANE_INDEX_KEY, LANE_SPEED_KEY, LANE_LENGTH_KEY, LANE_WIDTH_KEY,
                                        LANE_SHAPE_KEY, 'allow'])
        avg_edge_df = pd.DataFrame(avg_edge_item_list,
                                   columns=['edge_' + EDGE_ID_KEY, EDGE_FROM_KEY, EDGE_TO_KEY, EDGE_FUNCTION_KEY,
                                            EDGE_SHAPE_KEY, EDGE_SPEED_KEY, LANE_NUM_FIELD])
        lane_df.rename(columns={EDGE_SHAPE_KEY: geometry_field}, inplace=True)
        avg_edge_df.rename(columns={EDGE_SHAPE_KEY: geometry_field}, inplace=True)

        junction_item_list = list()
        for junction_ele in junction_ele_list:
            junction_item = self.parse_net_junction(junction_obj=junction_ele, x_offset=x_offset, y_offset=y_offset,
                                                    proj=proj, no_crs_flag=no_crs_flag)
            junction_item_list.append(junction_item)

        junction_df = pd.DataFrame(junction_item_list, columns=[JUNCTION_ID_KEY, JUNCTION_TYPE_KEY,
                                                                JUNCTION_X_KEY, JUNCTION_Y_KEY, JUNCTION_SHAPE_KEY])
        junction_df.rename(columns={JUNCTION_SHAPE_KEY: geometry_field}, inplace=True)

        conn_item_list = list()
        for conn_ele in conn_ele_list:
            conn_item_list.append(self.parse_connection(conn_ele))
        conn_df = pd.DataFrame(conn_item_list, columns=[CONN_FROM_EDGE_KEY, CONN_TO_EDGE_KEY, CONN_FROM_LANE_KEY,
                                                        CONN_TO_LANE_KEY, CONN_VIA_KEY])
        return lane_df, avg_edge_df, junction_df, conn_df

    @staticmethod
    def parse_net_edge(net_edge_obj: ET.Element = None, x_offset: float = 0.0, y_offset: float = 0.0,
                       proj: str = None, no_crs_flag: bool = False) -> \
            tuple[list[list[str, str, str, str, int, float, float, float, LineString, str]],
                  list[str, str, str, str, LineString, float]]:
        """

        Args:
            net_edge_obj:
            x_offset:
            y_offset:
            proj:
            no_crs_flag:

        Returns:

        """
        edge_id = net_edge_obj.get(EDGE_ID_KEY)
        edge_from, edge_to = net_edge_obj.get(EDGE_FROM_KEY), net_edge_obj.get(EDGE_TO_KEY)
        lane_item_list: list[list[str, str, str, str, int, float, float, float, LineString]] = []

        edge_function = net_edge_obj.get(EDGE_FUNCTION_KEY)
        edge_function = 'normal' if edge_function is None else edge_function
        lane_speed_list = list()
        avg_line_shape_list = []
        for lane_obj in net_edge_obj:

            lane_id, lane_index, lane_shape, lane_length, = lane_obj.get(LANE_ID_KEY), \
                lane_obj.get(LANE_INDEX_KEY), lane_obj.get(
                LANE_SHAPE_KEY), lane_obj.get(LANE_LENGTH_KEY)

            lane_allow = lane_obj.get('allow')
            if lane_allow is None:
                lane_allow = 'passenger'

            if lane_id is None or lane_index is None:
                # print(net_edge_obj.get(EDGE_ID_KEY))
                continue

            lane_speed, lane_width = float(try_get_v(lane_obj, LANE_SPEED_KEY, DEFAULT_LANE_SPEED)), \
                float(try_get_v(lane_obj, LANE_WIDTH_KEY, DEFAULT_LANE_WIDTH))
            lane_speed_list.append(lane_speed)
            lane_shape = [list(map(float, xy.split(','))) for xy in lane_shape.split(' ')]
            lane_shape = np.array(lane_shape) - np.array([x_offset, y_offset])

            # 需要先转化为lng和lat
            if no_crs_flag:
                c = LngLatTransfer()
                lane_shape_x, lane_shape_y = c.loc_convert(lane_shape[:, 0], lane_shape[:, 1], con_type='p-g', proj=proj)
                lane_shape = [[x, y] for x, y in zip(lane_shape_x, lane_shape_y)]
            avg_line_shape_list.append(lane_shape)
            lane_shape = LineString(lane_shape)

            if lane_length is None:
                lane_length = 0.0

            lane_item_list.append(
                [edge_id, edge_from, edge_to, edge_function, int(lane_index), float(lane_speed),
                 float(lane_length), float(lane_width), lane_shape, lane_allow])

        avg_center_line = LineString()
        avg_speed = DEFAULT_LANE_SPEED
        lane_num = 0
        if edge_function == 'normal':
            try:
                avg_center_line = LineString(
                    np.array(avg_line_shape_list).mean(axis=0))
            except Exception as e:
                _l = len(avg_line_shape_list)
                select_line = [avg_line_shape_list[int(_l / 2)]]
                avg_center_line = LineString(
                    np.array(select_line).mean(axis=0))

            avg_speed = np.array(lane_speed_list).mean()
            lane_num = len(avg_line_shape_list)

        return lane_item_list, [edge_id, edge_from, edge_to, edge_function, avg_center_line, avg_speed, lane_num]

    @staticmethod
    def parse_net_junction(junction_obj: ET.Element = None, x_offset: float = 0.0, y_offset: float = 0.0,
                           proj: str = None, no_crs_flag: bool = False) -> list[str, str, float, float, Polygon]:
        """

        Args:
            junction_obj:
            x_offset:
            y_offset:
            proj:
            no_crs_flag:

        Returns:

        """
        junction_id, junction_type, junction_x, junction_y, junction_shape = \
            junction_obj.get(JUNCTION_ID_KEY), junction_obj.get(JUNCTION_TYPE_KEY), junction_obj.get(JUNCTION_X_KEY), \
            junction_obj.get(JUNCTION_Y_KEY), junction_obj.get(JUNCTION_SHAPE_KEY)
        try:
            junction_x, junction_y = float(junction_x) - x_offset, float(junction_y) - y_offset
        except TypeError:
            pass

        buffer = 1.0
        if no_crs_flag:
            c = LngLatTransfer()
            junction_x, junction_y = c.loc_convert(junction_x, junction_y, con_type='p-g', proj=proj)
            buffer = 1e-5

        if junction_type == 'internal':
            junction_shape = Polygon(list(Point(junction_x, junction_y).buffer(buffer).exterior.coords))
        else:
            junction_shape = np.array(
                [np.array(list(map(float, xy.split(',')))) for xy in junction_shape.split(' ')]) - np.array(
                [x_offset, y_offset])
            if no_crs_flag:
                c = LngLatTransfer()
                junction_shape_x, junction_shape_y = c.loc_convert(junction_shape[:, 0], junction_shape[:, 1],
                                                                   con_type='p-g', proj=proj)
                junction_shape = [[x, y] for x, y in zip(junction_shape_x, junction_shape_y)]
            _l = len(junction_shape)
            if _l >= 3:
                junction_shape = Polygon(junction_shape)
            elif _l == 2:
                junction_shape = LineString(junction_shape).buffer(buffer)
            else:
                junction_shape = LineString(list(Point(junction_x, junction_y).buffer(buffer).exterior.coords))

        return [junction_id, junction_type, junction_x, junction_y, junction_shape]

    @staticmethod
    def parse_connection(conn_obj: ET.Element = None) -> tuple[str, str, str, str, str]:
        from_edge, to_edge, from_lane, to_lane, via_lane = \
            conn_obj.get(CONN_FROM_EDGE_KEY), conn_obj.get(CONN_TO_EDGE_KEY), conn_obj.get(CONN_FROM_LANE_KEY), \
            conn_obj.get(CONN_TO_LANE_KEY), None
        try:
            via_lane = conn_obj.get(CONN_VIA_KEY)
        except Exception:
            pass
        return from_edge, to_edge, from_lane, to_lane, via_lane

    @staticmethod
    def process_conn(pre_conn_df: pd.DataFrame = None):
        pre_conn_df['from_lane_id'] = pre_conn_df[CONN_FROM_EDGE_KEY] + '_' + pre_conn_df[CONN_FROM_LANE_KEY]
        pre_conn_df['to_lane_id'] = pre_conn_df[CONN_TO_EDGE_KEY] + '_' + pre_conn_df[CONN_TO_LANE_KEY]

        cm = pre_conn_df['from_lane_id'].str.startswith(':')
        a_pre_conn_df = pre_conn_df[~cm].copy()
        b_pre_conn_df = pre_conn_df[cm].copy()
        entrance_exit_list = []
        turn_list = []
        a_pre_conn_df.reset_index(inplace=True, drop=True)
        a_pre_conn_df['from_edge'] = a_pre_conn_df[EDGE_FROM_KEY]
        for row in a_pre_conn_df.itertuples():
            from_lane_id, to_lane_id, via_lane_id = getattr(row, 'from_lane_id'), getattr(row, 'to_lane_id'), \
                getattr(row, 'via')
            from_edge, from_lane_index, to_edge, to_lane_index = getattr(row, 'from_edge'), \
                getattr(row, CONN_FROM_LANE_KEY), getattr(row, CONN_TO_EDGE_KEY), getattr(row, CONN_TO_LANE_KEY)
            entrance_exit_list.append([from_edge, from_lane_index, to_edge, to_lane_index])
            _turn = []
            _x = b_pre_conn_df[b_pre_conn_df['from_lane_id'] == via_lane_id]
            while not _x.empty:
                _turn.append(via_lane_id)
                via_lane_id = _x.at[_x.index[0], 'via']
                _x = b_pre_conn_df[b_pre_conn_df['from_lane_id'] == via_lane_id]
            turn_list.append(_turn)
        res = pd.DataFrame(entrance_exit_list, columns=['from_edge', 'from_lane_index', 'to_edge', 'to_lane_index'])
        res['via'] = turn_list
        return res

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

    @staticmethod
    def generate_net_from_plain(sumo_home_fldr: str = None, plain_edge_path: str = r'./',
                                plain_node_path: str = r'./', plain_conn_path: str = None, x_offset: float = 0,
                                y_offset: float = 0, out_fldr: str = r'./', out_file_name: str = 'net',
                                join_dist: float = 10.0):
        net_convert_fldr = r'bin/netconvert'
        out_file_name = out_file_name + '.net.xml'
        node_config, edge_config = rf'--node-files={plain_node_path}', rf'--edge-files={plain_edge_path}'
        offset_config = rf'--offset.x {x_offset}  --offset.y {y_offset}'
        output_config = rf'--output-file={os.path.join(out_fldr, out_file_name)}'
        if plain_conn_path is None:
            conn_config = r''
        else:
            conn_config = rf'--connection-files={plain_node_path}'

        os.system(
            rf'{os.path.join(sumo_home_fldr, net_convert_fldr)} {node_config} {edge_config} {conn_config} {offset_config} {output_config} --junctions.join-dist {join_dist}')

    @staticmethod
    def generate_plain_edge(link_gdf: gpd.GeoDataFrame = None, use_lane_ele: bool = False,
                            lane_info_reverse: bool = True):
        """

        Args:
            link_gdf:
            use_lane_ele:
            lane_info_reverse:

        Returns:

        """

        # 创建根节点
        edges_root = ET.Element('edges')

        # 以根节点创建文档树
        tree = ET.ElementTree(edges_root)

        # 设置根节点的属性
        edges_root.set('version', "1.16")
        edges_root.set('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
        edges_root.set('xsi:noNamespaceSchemaLocation', "http://sumo.dlr.de/xsd/edges_file.xsd")
        # new child node
        for _, row in link_gdf.iterrows():
            edge_ele = ET.Element('edge')
            edge_ele.set(EDGE_ID_KEY, str(row[link_id_field]))
            edge_ele.set(EDGE_FROM_KEY, str(row[from_node_field]))
            edge_ele.set(EDGE_TO_KEY, str(row[to_node_field]))
            edge_ele.set(EDGE_PRIORITY_KEY, '-1')
            edge_ele.set(EDGE_SPREAD_TYPE, row[SPREAD_TYPE])
            try:
                edge_ele.set(EDGE_LANES_KEY, str(row[LANE_NUM_FIELD]))
            except KeyError:
                edge_ele.set(EDGE_LANES_KEY, rf'{DEFAULT_LANE_NUM}')

            try:
                edge_ele.set(EDGE_SPEED_KEY, str(row[SPEED_FIELD]))
            except KeyError:
                edge_ele.set(EDGE_SPEED_KEY, rf'{DEFAULT_SPEED}')

            edge_ele.set(EDGE_SHAPE_KEY,
                         ' '.join([','.join(list(map(str, item))) for item in list(row[GEOMETRY_FIELD].coords)]))

            if not use_lane_ele:
                try:
                    edge_ele.set(LANE_WIDTH_KEY, str(row[WIDTH_FIELD]))
                except KeyError:
                    edge_ele.set(LANE_WIDTH_KEY, '3.6')

            if not use_lane_ele:
                try:
                    edge_ele.set('allow', str(row[ALLOW_MODE_KEY]))
                except KeyError:
                    pass

            if use_lane_ele:
                lane_width_info = row[LW_INFO_FIELD].split(',')
                lane_mode_info = row[LM_INFO_FIELD].split(',')
                if lane_info_reverse:
                    lane_width_info = lane_width_info[::-1]
                    lane_mode_info = lane_mode_info[::-1]
                for i, lane_width in enumerate(lane_width_info):
                    lane_ele = ET.Element('lane')
                    lane_ele.set('index', str(i))
                    lane_ele.set(LANE_WIDTH_KEY, str(lane_width_info[i]))
                    # c: passenger, n: bicycle, g: green belt
                    try:
                        now_mode = lane_mode_info[i]
                    except:
                        now_mode = 'c'
                    if now_mode == 'c':
                        pass
                    elif now_mode == 'n':
                        lane_ele.set('allow', 'bicycle')
                    elif now_mode == 'g':
                        lane_ele.set('disallow', 'all')
                    edge_ele.append(lane_ele)
                    stop_ele = ET.Element('stopOffset')
                    stop_ele.set('value', '2.0')
                    lane_ele.append(stop_ele)
            # 将子节点添加到根节点下
            edges_root.append(edge_ele)
        return tree

    @staticmethod
    def generate_plain_node(node_gdf: gpd.GeoDataFrame = None, junction_gdf: gpd.GeoDataFrame = None,
                            x_offset: float = 0.0, y_offset: float = 0.0):
        """

        Args:
            node_gdf:
            junction_gdf:
            x_offset:
            y_offset:

        Returns:

        """
        use_junction_shape = False
        if junction_gdf is not None and not junction_gdf.empty:
            junction_gdf.set_index('node_id', inplace=True)
            use_junction_shape = True
        node_gdf['_x_'], node_gdf['_y_'] = node_gdf['geometry'].x, node_gdf['geometry'].y
        max_x, max_y, min_x, min_y = node_gdf['_x_'].max(), node_gdf['_y_'].max(), node_gdf['_x_'].min(), node_gdf[
            '_y_'].min()

        # create root node
        nodes_root = ET.Element('nodes')
        location_ele = ET.Element('location')
        location_ele.set("netOffset", rf"{x_offset},{y_offset}")
        location_ele.set("convBoundary", rf"{min_x},{min_y},{max_x},{max_y}")
        location_ele.set("origBoundary", "-10000000000.00,-10000000000.00,10000000000.00,10000000000.00")
        location_ele.set('projParameter', str(CRS.from_user_input(node_gdf.crs).to_proj4()))
        print(str(CRS.from_user_input(node_gdf.crs).to_proj4()))
        # 以根节点创建文档树
        tree = ET.ElementTree(nodes_root)

        # 设置根节点的属性
        nodes_root.set('version', "1.16")
        nodes_root.set('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
        nodes_root.set('xsi:noNamespaceSchemaLocation', "http://sumo.dlr.de/xsd/nodes_file.xsd")

        nodes_root.append(location_ele)
        for _, row in node_gdf.iterrows():
            node_ele = ET.Element('node')
            node_id = str(row['node_id'])
            node_ele.set(NODE_ID_KEY, node_id)
            node_ele.set(NODE_X_KEY, str(row['_x_']))
            node_ele.set(NODE_Y_KEY, str(row['_y_']))
            node_ele.set('type', "priority")
            if use_junction_shape:
                try:
                    shape_str = list(junction_gdf.at[node_id, 'geometry'].exterior.coords)
                    shape_str = ' '.join([','.join(list(map(str, loc))) for loc in shape_str])
                    node_ele.set('shape', shape_str)
                except:
                    pass
            nodes_root.append(node_ele)
        del node_gdf['_x_'], node_gdf['_y_']
        return tree

    def generate_hd_map(self, sumo_home_fldr: str, link_gdf: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame,
                        junction_gdf: gpd.GeoDataFrame = None,
                        use_lane_ele: bool = False, lane_info_reverse: bool = True, join_dist: float = 10.0,
                        x_offset: float = 0.0, y_offset: float = 0.0, out_fldr: str = r'./',
                        flag_name: str = 'prj', plain_crs: str = 'EPSG:3857', is_single_link: bool = False):
        """SumoConvert类方法 - generate_hd_map(v0.3.18提供)

        - 基于宏观路网生产.net.xml高精地图文件

        Args:
            sumo_home_fldr: sumo安装目录
            link_gdf: 路网线层
            node_gdf: 路网点层
            is_single_link: 路网线层是否是单向表示
            junction_gdf: 节点面域
            use_lane_ele: 是否启用车道信息自定义
            lane_info_reverse: 是否反转车道索引
            join_dist: 路口合并阈值, 米
            out_fldr: .net.xml路网输出目录
            flag_name: .net.xml路网的名称
            plain_crs: 平面投影坐标系

        Returns:

        """
        if not is_single_link:
            if rf'{SPREAD_TYPE}_ab' not in link_gdf.columns and rf'{SPREAD_TYPE}_ba' not in link_gdf.columns:
                link_gdf.loc[link_gdf[net_field.DIRECTION_FIELD] == 0, SPREAD_TYPE] = 'right'
                link_gdf.loc[link_gdf[net_field.DIRECTION_FIELD] == 1, SPREAD_TYPE] = 'center'
            single_link_gdf = dual2single(link_gdf=link_gdf)
            single_link_gdf[link_id_field] = [i + 1 for i in range(len(single_link_gdf))]
        else:
            single_link_gdf = link_gdf

        if SPREAD_TYPE not in single_link_gdf.columns:
            single_link_gdf[SPREAD_TYPE] = 'right'

        single_link_gdf = single_link_gdf.to_crs(plain_crs)
        node_gdf = node_gdf.to_crs(plain_crs)
        if x_offset == 0:
            x_offset = -node_gdf[geometry_field].x.min()
        if y_offset == 0:
            y_offset = -node_gdf[geometry_field].y.min()
        single_link_gdf[net_field.LINK_ID_FIELD] = [i for i in range(1, len(single_link_gdf) + 1)]
        edge_tree = self.generate_plain_edge(link_gdf=single_link_gdf, use_lane_ele=use_lane_ele,
                                             lane_info_reverse=lane_info_reverse)
        node_tree = self.generate_plain_node(node_gdf=node_gdf, x_offset=x_offset, y_offset=y_offset,
                                             junction_gdf=junction_gdf)
        edge_tree.write(os.path.join(out_fldr, rf'{flag_name}.edg.xml'))
        node_tree.write(os.path.join(out_fldr, rf'{flag_name}.nod.xml'))
        self.generate_net_from_plain(sumo_home_fldr=sumo_home_fldr,
                                     plain_edge_path=os.path.join(out_fldr, rf'{flag_name}.edg.xml'),
                                     plain_node_path=os.path.join(out_fldr, rf'{flag_name}.nod.xml'), x_offset=x_offset,
                                     y_offset=y_offset, join_dist=join_dist, out_fldr=out_fldr,
                                     out_file_name=flag_name)

    @staticmethod
    def match2rou(match_res_df: pd.DataFrame | gpd.GeoDataFrame, edge_id_field: str = 'edge_id',
                  time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's',
                  out_fldr: str = r'./', file_name: str = 'flow') -> ET.ElementTree:
        """SumoConvert类函数 - match2rou(v0.3.15提供)

        - 将路径匹配结果表转化为SUMO的车流路径文件

        Args:
            match_res_df: 匹配结果表
            time_format: 时间列字符串模板
            time_unit: 时间列单位
            edge_id_field: edge_id列字段名称
            out_fldr: rou文件输出目录
            file_name: rou文件名称

        Returns:
            DocTree
        """

        assert edge_id_field in match_res_df.columns, rf'edge_id_field: {edge_id_field} is not found in columns'

        match_res_df.dropna(subset=[edge_id_field], inplace=True, how='any')
        match_res_df.reset_index(inplace=True, drop=True)
        try:
            match_res_df[edge_id_field] = match_res_df[edge_id_field].astype(int)
        except:
            pass
        match_res_df = del_dup_links(match_res_df=match_res_df, use_time=True, time_unit=time_unit,
                                     time_format=time_format)
        match_res_df['dt'] = (match_res_df[time_field] - match_res_df[time_field].min()).dt.total_seconds()
        routes_root = ET.Element('routes')
        tree = ET.ElementTree(routes_root)
        iter_agents(routes_root, match_res_df, edge_id_field)
        if not os.path.exists(out_fldr):
            os.makedirs(out_fldr)
        tree.write(os.path.join(out_fldr, rf'{file_name}.rou.xml'))
        return tree

    def generate_sim(self):
        pass

    @staticmethod
    def get_prj_info(net_path: str = None, crs: str = None, ) -> tuple[float, float, str]:
        """SumoConvert类静态方法 - get_prj_info

        - 获取net.xml的坐标系信息

        Args:
            net_path: .net.xml路网的路径
            crs:

        Returns:
            x_offset, y_offset, crs
        """

        net_tree = ET.parse(net_path)

        net_root = net_tree.getroot()
        location_ele = net_root.findall('location')[0]
        try:
            prj4_str = location_ele.get('projParameter')
        except:
            prj4_str = None

        if crs is None:
            assert prj4_str is not None
            crs = 'EPSG:' + prj4_2_crs(prj4_str=prj4_str)
        try:
            x_offset, y_offset = list(map(float, location_ele.get('netOffset').split(',')))
        except:
            x_offset, y_offset = 0, 0
        return x_offset, y_offset, crs

    @staticmethod
    def generate_sumocfg(net_file_path: str = r'net.net.xml',
                         rou_file_path: str = r'rou.rou.xml', start_time: int = 0,
                         end_time: int = 1200, out_fldr: str = r'./', file_name: str = r'sim'):
        """SumoConvert类静态方法 - generate_sumocfg(v0.3.15提供)

        - 生成sumo仿真配置文件

        Args:
            net_file_path: .net.xml路网的绝对路径
            rou_file_path: .rou.xml车流文件的绝对路径
            start_time: 仿真开始时间戳, 秒
            end_time: 仿真结束时间戳, 秒
            out_fldr: sumo仿真配置文件的存储目录
            file_name: sumo仿真配置文件的名称

        Returns:
            None
        """
        config_root = ET.Element('configuration')
        tree = ET.ElementTree(config_root)
        input_ele = element('input', None, None)
        config_root.append(input_ele)
        net_file_ele, route_file_ele = \
            element('net-file', 'value', net_file_path), \
            element('route-files', 'value', rou_file_path)
        ign_error_ele = element('ignore-route-errors', 'value', "True")
        input_ele.append(net_file_ele)
        input_ele.append(route_file_ele)
        input_ele.append(ign_error_ele)
        sim_time_ele = element('time', None, None)
        begin_time_ele, end_time_ele = \
            element('begin', 'value', rf'{start_time}'), element('end', 'value', rf'{end_time}')
        sim_time_ele.append(begin_time_ele)
        sim_time_ele.append(end_time_ele)
        input_ele.append(sim_time_ele)
        if not os.path.exists(out_fldr):
            os.makedirs(out_fldr)
        tree.write(os.path.join(out_fldr, file_name + '.sumocfg'))

def element(name: str = None, k: str = None, v: str = None):
    ele = ET.Element(name)
    if k is not None:
        ele.set(k, v)
    return ele

def prj4_2_crs(prj4_str: str = None) -> str | None:
    """

    Args:
        prj4_str:

    Returns:

    """
    try:
        _code = pyproj.CRS(prj4_str).to_epsg()
        if _code is not None:
            return str(_code)
    except:
        _code = None
    return _code

def iter_agents(routes_root, match_res_df, edge_id_field):
    """

    Args:
        routes_root:
        match_res_df:
        edge_id_field:

    Returns:

    """
    i = 0
    for agent, df in match_res_df.groupby(agent_id_field):
        i += 1
        route_ele = ET.Element('route')
        route_ele.set('edges', ' '.join(df[edge_id_field].astype(str).to_list()))
        route_ele.set('id', rf'route-{i}')

        vehicle_ele = ET.Element('vehicle')
        vehicle_ele.set('id', str(agent))
        vehicle_ele.set('route', rf'route-{i}')

        depart = df[['dt']].iloc[0, 0]
        vehicle_ele.set('depart', rf'{depart}')

        routes_root.append(route_ele)
        routes_root.append(vehicle_ele)


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


def prj_xfer(from_crs: str = 'EPSG:32650', to_crs: str = 'EPSG:4326', origin_p: Point = None) -> Point:
    f = pyproj.CRS(from_crs)
    t = pyproj.CRS(to_crs)
    project = pyproj.Transformer.from_crs(f, t, always_xy=True).transform
    xfer_point = transform(project, origin_p)
    return xfer_point

def dual2single(link_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """dual2single函数

    - 将具有方向字段的路网格式转化为单向的路网格式(没有方向字段, 仅靠from_node, to_node即可判别方向)

    Args:
        link_gdf: 线层路网数据

    Returns:
        线层路网数据
    """
    # 避免重名, 这里使用了内置字段__temp__
    built_in_col = '__temp__'
    rename_dict = avoid_duplicate_cols(built_in_col_list=[built_in_col], df=link_gdf)
    cols_field_name_list = [LANE_NUM_FIELD, SPEED_FIELD, WIDTH_FIELD, ALLOW_MODE_KEY, SPREAD_TYPE]
    # 找出双向字段, 双向字段都应该以_ab或者_ba结尾
    two_way_field_list = list()
    for cols_name in cols_field_name_list:
        if (cols_name + '_ab' in link_gdf.columns) or (cols_name + '_ba' in link_gdf.columns):
            two_way_field_list.append(cols_name)
    two_way_field_list = list(set(two_way_field_list))
    ab_field_del = [x + '_ab' for x in two_way_field_list]
    ba_field_del = [x + '_ba' for x in two_way_field_list]

    for col in (ab_field_del + ba_field_del):
        assert col in link_gdf.columns, f'缺少字段{col}!'

    ab_rename_dict = {x: y for x, y in zip(ab_field_del, two_way_field_list)}
    ba_rename_dict = {x: y for x, y in zip(ba_field_del, two_way_field_list)}

    # 方向为拓扑反向的
    net_negs = link_gdf[link_gdf[dir_field] == -1].copy()
    net_negs.drop(ab_field_del, axis=1, inplace=True)
    net_negs.rename(columns=ba_rename_dict, inplace=True)
    if not net_negs.empty:
        net_negs[[from_node_field, to_node_field]] = \
            net_negs[[to_node_field, from_node_field]]
        # net_negs[geometry_field] = net_negs[geometry_field].apply(lambda l: LineString(list(l.coords)[::-1]))
        net_negs[geometry_field] = net_negs[geometry_field].reverse()

    # 方向为拓扑正向的
    net_poss = link_gdf[link_gdf[dir_field] == 1].copy()
    net_poss.drop(ba_field_del, axis=1, inplace=True)
    net_poss.rename(columns=ab_rename_dict, inplace=True)

    # 方向为拓扑双向的, 改为拓扑正向
    net_zero_poss = link_gdf[link_gdf[dir_field] == 0].copy()
    net_zero_poss[dir_field] = 1
    net_zero_poss.drop(ba_field_del, axis=1, inplace=True)
    net_zero_poss.rename(columns=ab_rename_dict, inplace=True)

    # 方向为拓扑双向的, 改为拓扑反向
    net_zero_negs = link_gdf[link_gdf[dir_field] == 0].copy()
    net_zero_negs.drop(ab_field_del, axis=1, inplace=True)
    net_zero_negs.rename(columns=ba_rename_dict, inplace=True)
    if not net_zero_negs.empty:
        net_zero_negs[dir_field] = 1
        net_zero_negs[[from_node_field, to_node_field]] = \
            net_zero_negs[[to_node_field, from_node_field]]
        # net_zero_negs[geometry_field] = net_zero_negs[geometry_field].apply(lambda l: LineString(list(l.coords)[::-1]))
        net_zero_negs[geometry_field] = net_zero_negs[geometry_field].reverse()

    net = pd.concat([net_poss, net_zero_poss, net_negs, net_zero_negs]).reset_index(drop=True)

    # 恢复冲突字段
    rename_dict_reverse = dict((v, k) for k, v in rename_dict.items())
    if rename_dict:
        net.rename(columns=rename_dict_reverse, inplace=True)

    cols_list = list(net.columns)
    for col in [from_node_field, to_node_field, length_field]:
        cols_list.remove(col)
    return net[[from_node_field, to_node_field, length_field] + cols_list]

