# -- coding: utf-8 --
# @Time    : 2024/2/12 21:28
# @Author  : TangKai
# @Team    : ZheChengData

"""
路网线层存储与相关方法
"""

import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
from ..GlobalVal import NetField, PrjConst
from ..WrapsFunc import function_time_cost
from ..netreverse.RoadNet.Tools.process import merge_double_link

net_field = NetField()
prj_const = PrjConst()

geo_crs = prj_const.PRJ_CRS
link_id_field = net_field.LINK_ID_FIELD
dir_field = net_field.DIRECTION_FIELD
from_node_field = net_field.FROM_NODE_FIELD
to_node_field = net_field.TO_NODE_FIELD
length_field = net_field.LENGTH_FIELD
geometry_field = net_field.GEOMETRY_FIELD
link_vec_field = net_field.LINK_VEC_FIELD


class Link(object):
    def __init__(self, link_gdf: gpd.GeoDataFrame = None, planar_crs: str = None,
                 weight_field: str = None, is_check: bool = True, not_conn_cost: float = 999.0,
                 init_available_link: bool = True, delete_circle: bool = True):

        self.not_conn_cost = not_conn_cost
        self.geo_crs = geo_crs
        self.planar_crs = planar_crs
        self.link_gdf = link_gdf.copy()
        self.link_gdf.index = self.link_gdf[link_id_field]
        try:
            self.link_gdf[geometry_field] = self.link_gdf[geometry_field].remove_repeated_points(1e-7)
        except:
            pass
        self.weight_field = weight_field
        self.__available_link_id = []
        self.delete_circle = delete_circle
        if is_check:
            self.check()
        self.max_link_id = 999
        if init_available_link:
            self.init_available_link_id()
        self.__single_link_gdf = gpd.GeoDataFrame()
        self.__double_single_mapping: dict[int, tuple[int, int, int, int]] = dict()
        self.__ft_link_mapping: dict[tuple[int, int], int] = dict()

        self.__graph = nx.DiGraph()
        self.__ud_graph = nx.Graph()
        self.__one_out_degree_nodes = None
        self.__link_ft_mapping: dict[int, tuple[int, int]] = dict()
        self.__link_f_mapping: dict[int, int] = dict()
        self.__link_t_mapping: dict[int, int] = dict()
        self.__link_geo_mapping: dict[int, LineString] = dict()
        self.done_link_vec = False

    def check(self):
        gap_set = {net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                   net_field.TO_NODE_FIELD, net_field.DIRECTION_FIELD, self.weight_field,
                   net_field.GEOMETRY_FIELD} - set(self.link_gdf.columns)
        assert len(gap_set) == 0, rf'the Link layer lacks the following fields: {gap_set}'
        assert len(self.link_gdf[net_field.LINK_ID_FIELD]) == len(self.link_gdf[net_field.LINK_ID_FIELD].unique()), \
            rf'{net_field.LINK_ID_FIELD} field value is not unique'
        assert set(self.link_gdf[net_field.DIRECTION_FIELD]).issubset({0, 1}), \
            rf'{net_field.DIRECTION_FIELD} field value can only be 0 or 1'
        for col in [net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                    net_field.TO_NODE_FIELD, net_field.DIRECTION_FIELD]:
            assert len(self.link_gdf[self.link_gdf[col].isna()]) == 0, \
                rf'the {col} field of the link layer has an empty value'
            self.link_gdf[col] = self.link_gdf[col].astype(int)

        # 环路检测
        if self.delete_circle:
            circle_idx = self.link_gdf[from_node_field] == self.link_gdf[to_node_field]
            if not self.link_gdf[circle_idx].empty:
                print(rf'a loop was detected in the line layer data, and it was automatically deleted...')
                self.link_gdf.drop(index=self.link_gdf[circle_idx].index, inplace=True, axis=0)

    def init_link(self):
        """
        初始化Link, 这里会创建一个single层的link, 并将single_link设置为索引
        :return:
        """
        self.create_single_link(link_gdf=self.link_gdf)
        self.__single_link_gdf.set_index(net_field.SINGLE_LINK_ID_FIELD, inplace=True)
        self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD] = list(self.__single_link_gdf.index)

    def renew_length(self):
        self.link_gdf[length_field] = self.link_gdf[geometry_field].length

    def init_link_from_existing_single_link(self, single_link_gdf: gpd.GeoDataFrame = None,
                                            ft_link_mapping: dict = None,
                                            double_single_mapping: dict = None, link_ft_mapping: dict = None,
                                            link_t_mapping: dict = None, link_f_mapping: dict = None,
                                            link_geo_mapping: dict = None):
        """通过给定的single_link_gdf初始化link, 用在子net的初始化上"""
        self.__single_link_gdf = single_link_gdf.copy()

        self.__double_single_mapping = double_single_mapping
        self.__ft_link_mapping = ft_link_mapping
        # single_link: (f, t)
        self.__link_ft_mapping = link_ft_mapping
        self.__link_t_mapping = link_t_mapping
        self.__link_f_mapping = link_f_mapping
        self.__link_geo_mapping = link_geo_mapping

    def create_single_link(self, link_gdf: gpd.GeoDataFrame):
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
            # self.__single_link_gdf.reset_index(inplace=True, drop=True)
        self.__single_link_gdf.drop_duplicates(subset=[from_node_field, to_node_field], keep='first', inplace=True)
        self.__single_link_gdf.reset_index(inplace=True, drop=True)
        self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD] = [i for i in range(1, len(self.__single_link_gdf) + 1)]
        self.__single_link_gdf['path'] = self.__single_link_gdf.apply(
            lambda row: [row[net_field.FROM_NODE_FIELD], row[net_field.TO_NODE_FIELD]], axis=1)
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
        self.__link_f_mapping = {v: k[0] for k, v in self.__ft_link_mapping.items()}
        self.__link_t_mapping = {v: k[1] for k, v in self.__ft_link_mapping.items()}
        # self.__link_length = {single_link_id: l for
        #                       single_link_id, l in
        #                       zip(self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD],
        #                           self.__single_link_gdf[net_field.LENGTH_FIELD])}
        # self.__link_geo_mapping = {single_link_id: geo for single_link_id, geo in
        #                            zip(self.__single_link_gdf[net_field.SINGLE_LINK_ID_FIELD],
        #                                self.__single_link_gdf[net_field.GEOMETRY_FIELD])}

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
        # self.__ud_graph = self.__graph.to_undirected()

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
        except Exception as e:
            return [], self.not_conn_cost

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

    def get_bilateral_link_data(self) -> gpd.GeoDataFrame:
        """获取双向路网数据"""
        return self.link_gdf.copy()

    def delete_links(self, link_id_list: list[int] = None) -> gpd.GeoDataFrame:
        del_link_gdf = self.link_gdf.loc[link_id_list, :].copy()
        self.link_gdf.drop(index=link_id_list, inplace=True, axis=0)
        return del_link_gdf

    def append_links(self, link_id: list[int], from_node: list[int], to_node: list[int], dir_val: list[int],
                     geo: list[LineString], **kwargs) -> None:
        assert set(dir_val).issubset({0, 1})
        length_list = [l.length for l in geo]
        attr_dict = {link_id_field: link_id, from_node_field: from_node, to_node_field: to_node,
                     length_field: length_list,
                     dir_field: dir_val, geometry_field: geo}
        attr_dict.update(kwargs)
        self.link_gdf = pd.concat(
            [self.link_gdf, gpd.GeoDataFrame(attr_dict, geometry=geometry_field, crs=self.link_gdf.crs)])
        self.link_gdf.index = self.link_gdf[link_id_field]

    def append_link_gdf(self, link_gdf: gpd.GeoDataFrame = None) -> None:
        assert set(link_gdf[link_id_field]) & set(self.link_gdf[link_id_field]) == set()
        self.link_gdf = pd.concat(
            [self.link_gdf, link_gdf])
        self.link_gdf.index = self.link_gdf[link_id_field]

    def renew_single_link(self):
        pass

    def del_short_links(self, l_threshold: float = 0.5) -> None:
        self.link_gdf.drop(index=self.link_gdf[self.link_gdf[length_field] <= l_threshold].index, inplace=True, axis=0)

    def drop_dup_ft_road(self):
        self.link_gdf.sort_values(by=[from_node_field, to_node_field, dir_field], ascending=[True, True, True],
                                  inplace=True)
        self.link_gdf.drop_duplicates(subset=[from_node_field, to_node_field], inplace=True, keep='first')

    def renew_head_of_geo(self, target_link: list = None, loc_dict: dict[tuple] or dict[list] = None):
        """
        modify the head of the link geo according to loc_dict
        :param target_link:
        :param loc_dict:
        :return:
        """
        self.link_gdf.loc[target_link, geometry_field] = self.link_gdf.loc[target_link, :].apply(
            lambda row: LineString([loc_dict[row[link_id_field]]] + list(row[geometry_field].coords)[1:]), axis=1)

    def calc_link_vec(self) -> None:
        if self.done_link_vec:
            return None

        def get_ht_vec(line: LineString = None) -> np.ndarray:
            cor = list(line.coords)
            return np.array(cor[-1]) - np.array(cor[0])

        self.__single_link_gdf[link_vec_field] = self.__single_link_gdf.apply(
            lambda row: get_ht_vec(line=row[geometry_field]), axis=1)
        self.done_link_vec = True

    def renew_tail_of_geo(self, target_link: list = None, loc_dict: dict[tuple] or dict[list] = None):
        """
        modify the tail of the link geo according to loc_dict
        :param target_link:
        :param loc_dict:
        :return:
        """
        self.link_gdf.loc[target_link, geometry_field] = self.link_gdf.loc[target_link, :].apply(
            lambda row: LineString(list(row[geometry_field].coords)[:-1] + [loc_dict[row[link_id_field]]]), axis=1)

    def renew_geo_of_ht(self, target_link: list = None, head_loc_dict: dict[tuple] or dict[list] = None,
                        tail_loc_dict: dict[tuple] or dict[list] = None):

        self.link_gdf.loc[target_link, geometry_field] = self.link_gdf.loc[target_link, :].apply(
            lambda row: LineString(
                [head_loc_dict[row[link_id_field]]] + list(row[geometry_field].coords)[1:-1] + [tail_loc_dict[
                                                                                                    row[
                                                                                                        link_id_field]]]),
            axis=1)

    def get_double_link_data(self):
        pass

    def get_link_attr_by_ft(self, attr_name: str = None, from_node: int = None, to_node: int = None):
        """
        通过(from_node, to_node)索引link属性
        :param attr_name:
        :param from_node:
        :param to_node:
        :return:
        """
        return self.__single_link_gdf.at[self.__ft_link_mapping[(from_node, to_node)], attr_name]

    def modify_link_gdf(self, link_id_list: list[int], attr_field_list: list[str], val_list: list[list] = None):
        self.link_gdf.loc[link_id_list, attr_field_list] = val_list

    def merge_double_link(self):
        self.link_gdf = merge_double_link(link_gdf=self.link_gdf)
        self.link_gdf.index = self.link_gdf[link_id_field]

    def get_link_geo(self, link_id: int = None, _type: str = 'single') -> LineString:
        if _type == 'single':
            return self.__single_link_gdf.at[link_id, geometry_field]
        else:
            return self.link_gdf.at[link_id, geometry_field]

    def get_link_from_to(self, link_id: int = None, _type='single') -> tuple[int, int]:
        if _type == 'single':
            return self.__single_link_gdf.at[link_id, net_field.FROM_NODE_FIELD], self.__single_link_gdf.at[
                link_id, net_field.TO_NODE_FIELD]
        else:
            return self.link_gdf.at[link_id, net_field.FROM_NODE_FIELD], self.link_gdf.at[
                link_id, net_field.TO_NODE_FIELD]

    def get_geo_by_ft(self, from_node: int = None, to_node: int = None) -> LineString:
        return self.__single_link_gdf.at[self.__ft_link_mapping[(from_node, to_node)], net_field.GEOMETRY_FIELD]

    def get_ft_link_mapping(self) -> dict[tuple[int, int], int]:
        return self.__ft_link_mapping

    def to_plane_prj(self) -> None:
        if self.__single_link_gdf is None or self.__single_link_gdf.empty:
            pass
        else:
            self.__single_link_gdf = self.__single_link_gdf.to_crs(self.planar_crs)
        self.link_gdf = self.link_gdf.to_crs(self.planar_crs)

    def to_geo_prj(self) -> None:
        if self.__single_link_gdf is None or self.__single_link_gdf.empty:
            pass
        else:
            self.__single_link_gdf = self.__single_link_gdf.to_crs(self.geo_crs)
        self.link_gdf = self.link_gdf.to_crs(self.geo_crs)

    @property
    def crs(self):
        return self.link_gdf.crs.srs

    @property
    def bilateral_unidirectional_mapping(self) -> dict[int, tuple[int, int, int, int]]:
        return self.__double_single_mapping.copy()

    def init_available_link_id(self) -> None:
        max_link = self.link_gdf[link_id_field].max()
        self.max_link_id = max_link
        if self.max_link_id >= 100000:
            return None
        self.__available_link_id = list({i for i in range(1, max_link + 1)} - set(self.link_gdf[link_id_field]))

    @property
    def available_link_id(self) -> int:
        if self.__available_link_id:
            now_link_id = self.__available_link_id.pop()
            return now_link_id
        else:
            now_link_id = self.max_link_id
            self.max_link_id += 1
            return now_link_id + 1

    def get_av_link_id(self) -> list:
        return self.__available_link_id

    def get_link_startswith_nodes(self, node_list: list[int], _type: str = 'single') -> list[int, int]:
        if _type == 'single':
            return self.__single_link_gdf[self.__single_link_gdf[from_node_field].isin(node_list)][
                link_id_field].to_list()
        else:
            return self.link_gdf[self.link_gdf[from_node_field].isin(node_list)][link_id_field].to_list()

    def get_link_endswith_nodes(self, node_list: list[int], _type: str = 'single') -> list[int, int]:
        if _type == 'single':
            return self.__single_link_gdf[self.__single_link_gdf[to_node_field].isin(node_list)][
                link_id_field].to_list()
        else:
            return self.link_gdf[self.link_gdf[to_node_field].isin(node_list)][link_id_field].to_list()

    def link_series(self, link_id: int = None) -> gpd.GeoSeries:
        return self.link_gdf.loc[link_id, :].copy()

    @property
    def link_ft_map(self) -> dict[int, tuple[int, int]]:
        return self.__link_ft_mapping

    @property
    def link_f_map(self) -> dict[int, int]:
        return self.__link_f_mapping

    @property
    def link_t_map(self) -> dict[int, int]:
        return self.__link_t_mapping

    @property
    def link_geo_map(self) -> dict[int, LineString]:
        return self.__link_geo_mapping

    def vertex_degree(self, node: int = None) -> int:
        """无向图的节点度"""
        return self.__ud_graph.degree[node]

    def used_node(self) -> set[int]:
        return set(self.link_gdf[from_node_field]) | set(self.link_gdf[to_node_field])
