# -- coding: utf-8 --
# @Time    : 2023/12/10 20:02
# @Author  : TangKai
# @Team    : ZheChengData

"""
路网的信息存储与相关方法
"""

import pandas as pd
import networkx as nx
import geopandas as gpd
from src.GlobalVal import NetField
from shapely.geometry import LineString


net_field = NetField()


class Net(object):

    def __init__(self, link_path=None, node_path=None, weight_field='length',
                 geo_crs: str = 'EPSG:4326', plane_crs: str = 'EPSG:32650'):
        """

        :param link_path:
        :param node_path:
        :param weight_field:
        :param geo_crs:
        :param plane_crs:
        """
        self.geo_crs = geo_crs
        self.plane_crs = plane_crs
        self.__link = Link(link_gdf=gpd.read_file(link_path), weight_field=weight_field)
        self.__node = Node(node_gdf=gpd.read_file(node_path))
        self.__link.create_graph(weight_field=weight_field)

    def get_shortest_path_length(self, o_node=1, d_node=2):
        return self.__link.get_shortest_path_length(o_node=o_node, d_node=d_node)

    def get_shortest_path(self, o_node=1, d_node=2, weight_field: str = None):
        return self.__link.get_shortest_path(o_node=o_node, d_node=d_node, weight_field=weight_field)

    def get_node_loc(self, node_id: int = None):
        return self.__node.get_node_loc(node_id)

    def get_link_data(self):
        return self.__link.get_link_data()

    def get_node_data(self):
        return self.__node.get_node_data()

    def get_link_geo(self, link_id: int = None):
        return self.__link.get_link_geo(link_id=link_id)

    def get_link_from_to(self, link_id: int = None):
        return self.__link.get_link_from_to(link_id=link_id)

    def get_line_geo_by_ft(self, from_node: int = None, to_node: int = None) -> LineString:
        return self.__link.get_geo_by_ft(from_node, to_node)

    def get_link_attr_by_ft(self, attr_name: str = None, from_node: int = None, to_node: int = None):
        return self.__link.get_link_attr_by_ft(from_node=from_node, to_node=to_node, attr_name=attr_name)

    def to_plane_prj(self):
        if self.__link.crs == self.plane_crs:
            pass
        else:
            self.__link.to_plane_prj()
            self.__node.to_plane_prj()

    def to_geo_prj(self):
        if self.__link.crs == self.geo_crs:
            pass
        else:
            self.__link.to_geo_prj()
            self.__node.to_geo_prj()

    @property
    def crs(self):
        return self.__link.crs

    @property
    def bilateral_unidirectional_mapping(self) -> dict[int, tuple[int, int]]:
        return self.__link.bilateral_unidirectional_mapping


class Link(object):
    def __init__(self, link_gdf: gpd.GeoDataFrame = None, geo_crs: str = 'EPSG:4326', plane_crs: str = 'EPSG:32650',
                 weight_field: str = None):

        self.geo_crs = geo_crs
        self.plane_crs = plane_crs
        self.__single_link_gdf = gpd.GeoDataFrame()
        self.__double_single_mapping: dict[int, tuple[int, int]] = dict()
        self.__ft_link_mapping:dict[tuple[int, int], int] = dict()
        self.__graph = nx.DiGraph()
        self.weight_field = weight_field
        self.create_single_link(link_gdf=link_gdf)
        self.__single_link_gdf.set_index(net_field.SINGLE_LINK_ID_FIELD, inplace=True)
        self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD] = list(self.__single_link_gdf.index)

    def create_single_link(self, link_gdf:gpd.GeoDataFrame):
        """
        基于原来路网创建单向路网, 并且建立映射表
        :return:
        """
        link_gdf[net_field.DIRECTION_FIELD] = link_gdf[net_field.DIRECTION_FIELD].astype(int)
        neg_link = link_gdf[link_gdf[net_field.DIRECTION_FIELD] == 0].copy()
        print(len(neg_link))
        if neg_link.empty:
            self.__single_link_gdf = link_gdf.copy()
        else:
            neg_link[[net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD]] = neg_link[
                [net_field.TO_NODE_FIELD, net_field.FROM_NODE_FIELD]]
            neg_link[net_field.GEOMETRY_FIELD] = neg_link[net_field.GEOMETRY_FIELD].apply(
                lambda line_geo: LineString(list(line_geo.coords)[::-1]))
            self.__single_link_gdf = pd.concat([link_gdf, neg_link])
            self.__single_link_gdf.reset_index(inplace=True, drop=True)
        self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD] = [i for i in range(1, len(self.__single_link_gdf) + 1)]
        self.__double_single_mapping = {single_link_id: (link_id, int(direction)) for
                                        single_link_id, link_id, direction in
                                        zip(self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD],
                                            self.__single_link_gdf[net_field.LINK_ID_FIELD],
                                            self.__single_link_gdf[net_field.DIRECTION_FIELD])}
        self.__ft_link_mapping = {(f, t): single_link for f, t, single_link in
                                  zip(self.__single_link_gdf[net_field.FROM_NODE_FIELD],
                                      self.__single_link_gdf[net_field.TO_NODE_FIELD],
                                      self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD])}

    def create_graph(self, weight_field: str = None):
        """
        创建有向图
        :return:
        """
        edge_list = [(f, t, {weight_field: l}) for f, t, l in
                     zip(self.__single_link_gdf[net_field.FROM_NODE_FIELD],
                         self.__single_link_gdf[net_field.TO_NODE_FIELD],
                         self.__single_link_gdf[weight_field])]
        self.__graph.add_edges_from(edge_list)

    def get_shortest_path_length(self, o_node=None, d_node=None):
        try:
            l = nx.dijkstra_path_length(self.__graph, o_node, d_node, weight=self.weight_field)
            return l
        except nx.NetworkXNoPath as e:
            return 300

    def get_shortest_path(self, o_node=None, d_node=None, weight_field: str = None):
        used_weight = weight_field
        if used_weight is None:
            used_weight = self.weight_field
        try:
            node_seq = nx.dijkstra_path(self.__graph, o_node, d_node, weight=used_weight)
            return node_seq
        except nx.NetworkXNoPath as e:
            raise nx.NetworkXNoPath

    def get_link_data(self):
        return self.__single_link_gdf

    def get_link_attr_by_ft(self, attr_name: str = None, from_node: int = None, to_node: int = None):
        return self.__single_link_gdf.at[self.__ft_link_mapping[(from_node, to_node)], attr_name]

    @staticmethod
    def check_link():
        pass

    def get_link_geo(self, link_id: int = None) -> LineString:
        return self.__single_link_gdf.at[link_id, 'geometry']

    def get_link_from_to(self, link_id: int = None) -> tuple[int, int]:
        return self.__single_link_gdf.at[link_id, net_field.FROM_NODE_FIELD], self.__single_link_gdf.at[
            link_id, net_field.TO_NODE_FIELD]

    def get_geo_by_ft(self, from_node: int = None, to_node: int = None) -> LineString:
        return self.__single_link_gdf.at[self.__ft_link_mapping[(from_node, to_node)], net_field.GEOMETRY_FIELD]

    @property
    def crs(self):
        return self.__single_link_gdf.crs

    def to_plane_prj(self):
        if self.__single_link_gdf.crs == self.plane_crs:
            pass
        else:
            self.__single_link_gdf = self.__single_link_gdf.to_crs(self.plane_crs)

    def to_geo_prj(self):
        if self.__single_link_gdf.crs == self.geo_crs:
            pass
        else:
            self.__single_link_gdf = self.__single_link_gdf.to_crs(self.geo_crs)

    @property
    def bilateral_unidirectional_mapping(self) -> dict[int, tuple[int, int]]:
        return self.__double_single_mapping


class Node(object):
    def __init__(self, node_gdf: gpd.GeoDataFrame = None, geo_crs: str = 'EPSG:4326', plane_crs: str = 'EPSG:32650'):
        self.geo_crs = geo_crs
        self.plane_crs = plane_crs
        self.__node_gdf = node_gdf
        self.__node_gdf.set_index(net_field.NODE_ID_FIELD, inplace=True)

    def get_node_geo(self, node_id: int = None):
        return self.__node_gdf.at[node_id, net_field.GEOMETRY_FIELD]

    def get_node_loc(self, node_id: int = None):
        geo = self.get_node_geo(node_id)
        return geo.x, geo.y

    def get_node_data(self):
        return self.__node_gdf

    @property
    def crs(self):
        return self.__node_gdf.crs

    def to_plane_prj(self) -> None:
        if self.__node_gdf.crs == self.plane_crs:
            pass
        else:
            self.__node_gdf = self.__node_gdf.to_crs(self.plane_crs)

    def to_geo_prj(self) -> None:
        if self.__node_gdf.crs == self.geo_crs:
            pass
        else:
            self.__node_gdf = self.__node_gdf.to_crs(self.geo_crs)
