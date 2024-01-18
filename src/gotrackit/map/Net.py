# -- coding: utf-8 --
# @Time    : 2023/12/10 20:02
# @Author  : TangKai
# @Team    : ZheChengData

"""
路网的信息存储与相关方法
"""

import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import LineString
from src.gotrackit.GlobalVal import NetField
from src.gotrackit.WrapsFunc import function_time_cost

NOT_CONN_COST = 200.0
net_field = NetField()


class Net(object):

    @function_time_cost
    def __init__(self, link_path=None, node_path=None, link_gdf: gpd.GeoDataFrame = None,
                 node_gdf: gpd.GeoDataFrame = None, weight_field: str = 'length',
                 geo_crs: str = 'EPSG:4326', plane_crs: str = 'EPSG:32650', init_from_existing: bool = False,
                 is_check: bool = True):
        """
        创建Net类
        :param link_path: link层的路网文件路径, 若指定了该参数, 则直接从磁盘IO创建Net线层
        :param node_path: node层的路网文件路径, 若指定了该参数, 则直接从磁盘IO创建Net点层
        :param link_gdf: 若指定了该参数, 则直接从内存中的gdf创建Net线层
        :param node_gdf: 若指定了该参数, 则直接从内存中的gdf创建Net点层
        :param weight_field: 搜路权重字段
        :param geo_crs:  地理坐标系
        :param plane_crs: 平面投影坐标系
        :param init_from_existing: 是否直接从内存中的gdf创建single_link_gdf, 该参数用于类内部创建子net, 用户不用关心该参数, 使用默认值即可
        """
        self.geo_crs = geo_crs
        self.plane_crs = plane_crs
        self.weight_field = weight_field
        self.all_pair_path_df = pd.DataFrame()
        if link_gdf is None:
            self.__link = Link(link_gdf=gpd.read_file(link_path), weight_field=self.weight_field, geo_crs=self.geo_crs,
                               plane_crs=self.plane_crs, is_check=is_check)
        else:
            self.__link = Link(link_gdf=link_gdf, weight_field=self.weight_field, geo_crs=self.geo_crs,
                               plane_crs=self.plane_crs, is_check=is_check)
        if not init_from_existing:
            self.__link.init_link()
        else:
            self.__link.init_link_from_existing_single_link(single_link_gdf=link_gdf)

        if node_gdf is None:
            self.__node = Node(node_gdf=gpd.read_file(node_path),is_check=is_check)
        else:
            self.__node = Node(node_gdf=node_gdf, is_check=is_check)
        if not init_from_existing:
            self.__node.init_node()
        else:
            pass

        if is_check:
            self.check()
        self.to_plane_prj()

    def check(self) -> None:
        """检查点层线层的关联一致性"""
        node_set = set(self.__node.get_node_data().index)
        link_node_set = set(self.__link.get_link_data()[net_field.FROM_NODE_FIELD]) | \
                        set(self.__link.get_link_data()[net_field.TO_NODE_FIELD])
        assert link_node_set.issubset(node_set), 'Link层中部分节点在Node层中没有记录...'

    @function_time_cost
    def init_net(self) -> None:
        self.__link.create_graph(weight_field=self.weight_field)

    def get_shortest_path_length(self, o_node=1, d_node=2) -> tuple[list, float]:
        return self.__link.get_shortest_path_length(o_node=o_node, d_node=d_node)
    @function_time_cost
    def calc_all_pairs_shortest_path(self) -> None:
        all_pair_path = nx.all_pairs_shortest_path(self.__link.get_graph())
        z = dict(all_pair_path)
        all_pair_path_df = pd.DataFrame(z)
        all_pair_path_df = pd.DataFrame(all_pair_path_df.stack())
        all_pair_path_df.rename(columns={0: net_field.NODE_PATH_FIELD}, inplace=True)
        self.all_pair_path_df = all_pair_path_df

    def get_od_cost(self, o_node: int = None, d_node: int = None) -> tuple[list, float]:
        try:
            node_path = self.all_pair_path_df.at[(d_node, o_node), net_field.NODE_PATH_FIELD]
        except KeyError:
            return [], NOT_CONN_COST

        path_cost = [self.get_link_attr_by_ft(attr_name=self.weight_field, from_node=node_path[i],
                                              to_node=node_path[i + 1]) for i in range(len(node_path) - 1)]
        return node_path, sum(path_cost)

    def search(self, o_node: int = None, d_node: int = None, search_method: str = None) -> tuple[list, float]:
        if search_method == 'all_pairs':
            return self.get_od_cost(o_node=o_node, d_node=d_node)
        elif search_method == 'single':
            return self.get_shortest_path_length(o_node=o_node, d_node=d_node)

    def get_rnd_shortest_path(self) -> list[int]:
        return self.__link.get_rnd_shortest_path()

    def get_node_loc(self, node_id: int = None) -> tuple:
        return self.__node.get_node_loc(node_id)

    def get_one_out_degree_nodes(self) -> list[int]:
        return self.__link.one_out_degree_nodes()

    def get_link_data(self) -> gpd.GeoDataFrame:
        return self.__link.get_link_data()

    def get_node_data(self) -> gpd.GeoDataFrame:
        return self.__node.get_node_data()

    def get_link_geo(self, link_id: int = None) -> LineString:
        return self.__link.get_link_geo(link_id=link_id)

    def get_link_from_to(self, link_id: int = None) -> tuple[int, int]:
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

    def is_geo_crs(self) -> bool:
        if self.crs == self.geo_crs:
            return True
        else:
            return False

    @property
    def bilateral_unidirectional_mapping(self) -> dict[int, tuple[int, int, int, int]]:
        return self.__link.bilateral_unidirectional_mapping
    def get_ft_node_link_mapping(self):
        return self.__link.get_ft_link_mapping()

    @function_time_cost
    def create_computational_net(self, gps_array_buffer:Polygon = None):
        """

        :param gps_array_buffer:
        :return:
        """
        gps_array_buffer_gdf = gpd.GeoDataFrame({'geometry': [gps_array_buffer]}, geometry='geometry',
                                                crs=self.plane_crs)
        sub_single_link_gdf = gpd.sjoin(self.get_link_data(), gps_array_buffer_gdf)
        sub_single_link_gdf.drop(columns=['index_right'], axis=1, inplace=True)
        sub_single_link_gdf.drop_duplicates(subset=[net_field.SINGLE_LINK_ID_FIELD], inplace=True)
        sub_node_list = list(set(sub_single_link_gdf[net_field.FROM_NODE_FIELD]) | \
                             set(sub_single_link_gdf[net_field.TO_NODE_FIELD]))
        sub_node_gdf = self.__node.get_node_data().loc[sub_node_list, :].copy()
        sub_net = Net(link_gdf=sub_single_link_gdf,
                      node_gdf=sub_node_gdf,
                      weight_field=self.weight_field,
                      geo_crs=self.geo_crs, plane_crs=self.plane_crs, init_from_existing=True, is_check=False)
        sub_net.init_net()
        return sub_net

class Link(object):
    def __init__(self, link_gdf: gpd.GeoDataFrame = None, geo_crs: str = 'EPSG:4326', plane_crs: str = 'EPSG:32650',
                 weight_field: str = None, is_check: bool = True):

        self.geo_crs = geo_crs
        self.plane_crs = plane_crs
        self.link_gdf = link_gdf
        self.weight_field = weight_field
        if is_check:
            self.check()
        self.__single_link_gdf = gpd.GeoDataFrame()
        self.__double_single_mapping: dict[int, tuple[int, int, int, int]] = dict()
        self.__ft_link_mapping:dict[tuple[int, int], int] = dict()
        self.__graph = nx.DiGraph()
        self.__one_out_degree_nodes = None

    def check(self):
        gap_set = {net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                   net_field.TO_NODE_FIELD, net_field.DIRECTION_FIELD, self.weight_field,
                   net_field.GEOMETRY_FIELD} - set(self.link_gdf.columns)
        assert len(gap_set) == 0, rf'线层Link缺少以下字段:{gap_set}'
        assert len(self.link_gdf[net_field.LINK_ID_FIELD]) == len(self.link_gdf[net_field.LINK_ID_FIELD].unique()), \
            rf'字段{net_field.LINK_ID_FIELD}不唯一...'
        assert set(self.link_gdf[net_field.DIRECTION_FIELD]).issubset({0, 1}), \
            rf'{net_field.DIRECTION_FIELD}字段值只能为0或者1'
        for col in [net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                    net_field.TO_NODE_FIELD, net_field.DIRECTION_FIELD]:
            assert len(self.link_gdf[self.link_gdf[col].isna()]) == 0, rf'线层Link字段{col}有空值...'
            self.link_gdf[col] = self.link_gdf[col].astype(int)
        assert self.link_gdf.crs == self.geo_crs, rf'源文件Link:地理坐标系指定有误:实际:{self.link_gdf.crs}, 指定: {self.geo_crs}'

    def init_link(self):
        """
        初始化Link, 这里会创建一个single层的link, 并将single_link设置为索引
        :return:
        """
        self.create_single_link(link_gdf=self.link_gdf)
        self.__single_link_gdf.set_index(net_field.SINGLE_LINK_ID_FIELD, inplace=True)
        self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD] = list(self.__single_link_gdf.index)

    def init_link_from_existing_single_link(self, single_link_gdf: gpd.GeoDataFrame = None):
        """通过给定的single_link_gdf初始化link, 用在子net的初始化上"""
        self.__single_link_gdf = single_link_gdf.copy()
        self.__double_single_mapping = {single_link_id: (link_id, int(direction), f, t) for
                                        single_link_id, link_id, direction, f, t in
                                        zip(self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD],
                                            self.__single_link_gdf[net_field.LINK_ID_FIELD],
                                            self.__single_link_gdf[net_field.DIRECTION_FIELD],
                                            self.__single_link_gdf[net_field.FROM_NODE_FIELD],
                                            self.__single_link_gdf[net_field.TO_NODE_FIELD])}
        self.__ft_link_mapping = {(f, t): single_link for f, t, single_link in
                                  zip(self.__single_link_gdf[net_field.FROM_NODE_FIELD],
                                      self.__single_link_gdf[net_field.TO_NODE_FIELD],
                                      self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD])}

    def create_single_link(self, link_gdf:gpd.GeoDataFrame):
        """
        基于原来路网创建单向路网, 并且建立映射表
        :return:
        """
        link_gdf[net_field.DIRECTION_FIELD] = link_gdf[net_field.DIRECTION_FIELD].astype(int)
        neg_link = link_gdf[link_gdf[net_field.DIRECTION_FIELD] == 0].copy()
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
        self.__double_single_mapping = {single_link_id: (link_id, int(direction), f, t) for
                                        single_link_id, link_id, direction, f, t in
                                        zip(self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD],
                                            self.__single_link_gdf[net_field.LINK_ID_FIELD],
                                            self.__single_link_gdf[net_field.DIRECTION_FIELD],
                                            self.__single_link_gdf[net_field.FROM_NODE_FIELD],
                                            self.__single_link_gdf[net_field.TO_NODE_FIELD])}
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

    def get_graph(self):
        return self.__graph

    def get_shortest_path_length(self, o_node=None, d_node=None) -> tuple[list, float]:
        """
        获取两个节点之间的最短路径和开销
        :param o_node:
        :param d_node:
        :return: ([12,13, ...], 263.33)
        """
        try:
            node_path = nx.dijkstra_path(self.__graph, o_node, d_node, weight=self.weight_field)
            cost_list = [self.get_link_attr_by_ft(attr_name=self.weight_field, from_node=node_path[i],
                                                  to_node=node_path[i + 1]) for i in range(len(node_path) - 1)]
            return node_path, sum(cost_list)
        except nx.NetworkXNoPath as e:
            return [], NOT_CONN_COST

    def get_shortest_path(self, o_node=None, d_node=None, weight_field: str = None):
        used_weight = weight_field
        if used_weight is None:
            used_weight = self.weight_field
        try:
            node_seq = nx.dijkstra_path(self.__graph, o_node, d_node, weight=used_weight)
            return node_seq
        except nx.NetworkXNoPath as e:
            raise nx.NetworkXNoPath

    @function_time_cost
    def get_rnd_shortest_path(self) -> list[int]:
        rnd_node = list(self.get_link_data()[net_field.FROM_NODE_FIELD].sample(n=1))[0]
        path_dict = nx.single_source_shortest_path(self.__graph, rnd_node)
        targets = list(path_dict.keys())
        return path_dict[targets[np.random.randint(len(targets))]]

    def get_link_data(self):
        return self.__single_link_gdf.copy()

    def get_link_attr_by_ft(self, attr_name: str = None, from_node: int = None, to_node: int = None):
        """
        通过(from_node, to_node)索引link属性
        :param attr_name:
        :param from_node:
        :param to_node:
        :return:
        """
        return self.__single_link_gdf.at[self.__ft_link_mapping[(from_node, to_node)], attr_name]

    @DeprecationWarning
    def one_out_degree_nodes(self) -> list[int]:
        """只有一个出度的节点集合"""
        if self.__one_out_degree_nodes is None:
            in_degree_df = pd.DataFrame(self.__graph.in_degree(), columns=[net_field.NODE_ID_FIELD, 'in_degree'])
            out_degree_df = pd.DataFrame(self.__graph.out_degree(), columns=[net_field.NODE_ID_FIELD, 'out_degree'])
            self.__one_out_degree_nodes = list(
                set(in_degree_df[(in_degree_df['in_degree'] == 0)][net_field.NODE_ID_FIELD]) & \
                set(out_degree_df[(out_degree_df['out_degree'] == 1)][
                        net_field.NODE_ID_FIELD]))
            return self.__one_out_degree_nodes

        return self.__one_out_degree_nodes

    def get_link_geo(self, link_id: int = None) -> LineString:
        return self.__single_link_gdf.at[link_id, 'geometry']

    def get_link_from_to(self, link_id: int = None) -> tuple[int, int]:
        return self.__single_link_gdf.at[link_id, net_field.FROM_NODE_FIELD], self.__single_link_gdf.at[
            link_id, net_field.TO_NODE_FIELD]

    def get_geo_by_ft(self, from_node: int = None, to_node: int = None) -> LineString:
        return self.__single_link_gdf.at[self.__ft_link_mapping[(from_node, to_node)], net_field.GEOMETRY_FIELD]

    def get_ft_link_mapping(self) -> dict[tuple[int, int], int]:
        return self.__ft_link_mapping

    def to_plane_prj(self) -> None:
        if self.__single_link_gdf.crs == self.plane_crs:
            pass
        else:
            self.__single_link_gdf = self.__single_link_gdf.to_crs(self.plane_crs)

    def to_geo_prj(self) -> None:
        if self.__single_link_gdf.crs == self.geo_crs:
            pass
        else:
            self.__single_link_gdf = self.__single_link_gdf.to_crs(self.geo_crs)
    @property
    def crs(self):
        return self.__single_link_gdf.crs

    @property
    def bilateral_unidirectional_mapping(self) -> dict[int, tuple[int, int, int, int]]:
        return self.__double_single_mapping


class Node(object):
    def __init__(self, node_gdf: gpd.GeoDataFrame = None, geo_crs: str = 'EPSG:4326', plane_crs: str = 'EPSG:32650',
                 is_check: bool = True):
        self.geo_crs = geo_crs
        self.plane_crs = plane_crs
        self.__node_gdf = node_gdf
        if is_check:
            self.check()

    def check(self):
        gap_set = {net_field.NODE_ID_FIELD, net_field.GEOMETRY_FIELD} - set(self.__node_gdf.columns)
        assert len(gap_set) == 0, rf'线层Link缺少以下字段:{gap_set}'
        assert len(self.__node_gdf[net_field.NODE_ID_FIELD]) == len(self.__node_gdf[net_field.NODE_ID_FIELD].unique()), \
            rf'字段{net_field.NODE_ID_FIELD}不唯一...'
        for col in [net_field.NODE_ID_FIELD]:
            assert len(self.__node_gdf[self.__node_gdf[col].isna()]) == 0, rf'点层Node字段{col}有空值...'
            self.__node_gdf[col] = self.__node_gdf[col].astype(int)
        assert self.__node_gdf.crs == self.geo_crs, rf'源文件Node:地理坐标系指定有误:实际:{self.__node_gdf.crs}, 指定: {self.geo_crs}'

    def init_node(self):
        self.__node_gdf.set_index(net_field.NODE_ID_FIELD, inplace=True)

    def get_node_geo(self, node_id: int = None):
        return self.__node_gdf.at[node_id, net_field.GEOMETRY_FIELD]

    def get_node_loc(self, node_id: int = None) -> tuple:
        geo = self.get_node_geo(node_id)
        return geo.x, geo.y

    def get_node_data(self):
        return self.__node_gdf.copy()

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


if __name__ == '__main__':
    g = nx.DiGraph()
    g.add_edges_from([(1, 2), (2, 3), (3, 4)])
    print(list(nx.dfs_preorder_nodes(g, 4)))
    print(nx.degree(g))
    print(dict(g.in_degree()))
    print(dict(g.out_degree()))

