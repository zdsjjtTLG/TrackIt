# -- coding: utf-8 --
# @Time    : 2023/12/10 20:02
# @Author  : TangKai
# @Team    : ZheChengData

"""
路网信息存储与相关方法
"""

import pandas as pd
from .Link import Link
from .Node import Node
import networkx as nx
import geopandas as gpd
from ..GlobalVal import NetField
from shapely.geometry import LineString
from ..tools.geo_process import prj_inf
from ..tools.save_file import save_file
from ..WrapsFunc import function_time_cost
from shapely.geometry import Polygon, Point


NOT_CONN_COST = 200.0
net_field = NetField()

link_id_field = net_field.LINK_ID_FIELD
dir_field = net_field.DIRECTION_FIELD
from_node_field = net_field.FROM_NODE_FIELD
to_node_field = net_field.TO_NODE_FIELD
length_field = net_field.LENGTH_FIELD
geometry_field = net_field.GEOMETRY_FIELD


class Net(object):

    @function_time_cost
    def __init__(self, link_path: str = None, node_path: str = None, link_gdf: gpd.GeoDataFrame = None,
                 node_gdf: gpd.GeoDataFrame = None, weight_field: str = 'length',
                 geo_crs: str = 'EPSG:4326', plane_crs: str = 'EPSG:32650', init_from_existing: bool = False,
                 is_check: bool = True, create_single: bool = True, search_method: str = 'dijkstra'):
        """
        创建Net类
        :param link_path: link层的路网文件路径, 若指定了该参数, 则直接从磁盘IO创建Net线层
        :param node_path: node层的路网文件路径, 若指定了该参数, 则直接从磁盘IO创建Net点层
        :param link_gdf: 若指定了该参数, 则直接从内存中的gdf创建Net线层
        :param node_gdf: 若指定了该参数, 则直接从内存中的gdf创建Net点层
        :param weight_field: 搜路权重字段
        :param geo_crs:  地理坐标系
        :param plane_crs: 平面投影坐标系
        :param create_single: 是否在初始化的时候创建single层
        :param search_method: 路径搜索方法, 'dijkstra' or 'bellman-ford'
        :param init_from_existing: 是否直接从内存中的gdf创建single_link_gdf, 该参数用于类内部创建子net, 用户不用关心该参数, 使用默认值即可

        """
        self.geo_crs = geo_crs
        self.search_method = search_method
        self.plane_crs = plane_crs
        self.weight_field = weight_field
        self.all_pair_path_df = pd.DataFrame()
        self.__stp_cache = dict()
        self.__done_path_cost = dict()
        if link_gdf is None:
            self.__link = Link(link_gdf=gpd.read_file(link_path), weight_field=self.weight_field, geo_crs=self.geo_crs,
                               plane_crs=self.plane_crs, is_check=is_check)
        else:
            self.__link = Link(link_gdf=link_gdf, weight_field=self.weight_field, geo_crs=self.geo_crs,
                               plane_crs=self.plane_crs, is_check=is_check)
        if not init_from_existing:
            if create_single:
                self.__link.init_link()
        else:
            if create_single:
                self.__link.init_link_from_existing_single_link(single_link_gdf=link_gdf)

        if node_gdf is None:
            self.__node = Node(node_gdf=gpd.read_file(node_path), is_check=is_check, plane_crs=self.plane_crs,
                               geo_crs=self.geo_crs)
        else:
            self.__node = Node(node_gdf=node_gdf, is_check=is_check, plane_crs=self.plane_crs, geo_crs=self.geo_crs)
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
        link_node_set = set(self.__link.get_bilateral_link_data()[net_field.FROM_NODE_FIELD]) | \
                        set(self.__link.get_bilateral_link_data()[net_field.TO_NODE_FIELD])
        assert link_node_set.issubset(node_set), 'Link层中部分节点在Node层中没有记录'

    @function_time_cost
    def init_net(self) -> None:
        self.__link.create_graph(weight_field=self.weight_field)

    def search(self, o_node: int = None, d_node: int = None) -> tuple[list, float]:
        """

        :param o_node:
        :param d_node:
        :return:
        """
        return self.get_od_cost(o_node=o_node, d_node=d_node)

    def get_od_cost(self, o_node: int = None, d_node: int = None) -> tuple[list, float]:
        """

        :param o_node:
        :param d_node:
        :return:
        """

        if o_node in self.__stp_cache.keys():
            try:
                node_path = self.__stp_cache[o_node][d_node]
                cost = self.__done_path_cost[o_node][d_node]
            except KeyError:
                return [], NOT_CONN_COST
        else:
            self.calc_shortest_path(source=o_node, method=self.search_method)
            try:
                node_path = self.__stp_cache[o_node][d_node]
                cost = self.__done_path_cost[o_node][d_node]
            except KeyError:
                return [], NOT_CONN_COST

        return node_path, cost

    def get_shortest_path_length(self, o_node=1, d_node=2) -> tuple[list, float]:
        """

        :param o_node:
        :param d_node:
        :return:
        """
        return self.__link.get_shortest_path_length(o_node=o_node, d_node=d_node)

    def calc_shortest_path(self, source: int = None, method: str = 'dijkstra') -> None:
        if source in self.__stp_cache.keys():
            pass
        else:
            try:
                self.__done_path_cost[source], self.__stp_cache[source] = self._single_source_path(
                    self.__link.get_graph(), source=source,
                    method=method, weight_field=self.weight_field)
            except nx.NetworkXNoPath:
                pass

    @staticmethod
    def _single_source_path(g: nx.DiGraph = None, source: int = None, method: str = 'dijkstra',
                            weight_field: str = None) -> tuple[dict[int, int], dict[int, list]]:
        if method == 'dijkstra':
            return nx.single_source_dijkstra(g, source, weight=weight_field)
        else:
            return nx.single_source_bellman_ford(g, source, weight=weight_field)

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

    def get_node_geo(self, node_id: int = None) -> Point:
        return self.__node.get_node_geo(node_id)

    def get_bilateral_link_data(self) -> gpd.GeoDataFrame:
        return self.__link.get_bilateral_link_data()

    def get_link_geo(self, link_id: int = None, _type: str = 'single') -> LineString:
        return self.__link.get_link_geo(link_id=link_id, _type=_type)

    def get_link_from_to(self, link_id: int = None, _type='bilateral') -> tuple[int, int]:
        return self.__link.get_link_from_to(link_id=link_id, _type=_type)

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

    def del_nodes(self, node_list: list[int] = None) -> None:
        self.__node.delete_nodes(node_list)

    @property
    def bilateral_unidirectional_mapping(self) -> dict[int, tuple[int, int, int, int]]:
        return self.__link.bilateral_unidirectional_mapping

    @property
    def link_ft_map(self) -> dict[int, tuple[int, int]]:
        return self.__link.link_ft_map

    @property
    def available_link_id(self) -> int:
        return self.__link.available_link_id

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
                      geo_crs=self.geo_crs, plane_crs=self.plane_crs, init_from_existing=True, is_check=False,
                      search_method=self.search_method)
        sub_net.init_net()
        return sub_net

    @property
    def graph(self) -> nx.DiGraph:
        return self.__link.get_graph()

    @property
    def node_degree(self, node: int = None) -> int:
        return self.__link.vertex_degree(node)

    def renew_link_head_geo(self, link_list: list[int] = None):
        self.__link.renew_head_of_geo(target_link=link_list,
                                      loc_dict={
                                          link: self.get_node_geo(self.get_link_from_to(link, _type='bilateral')[0]) for
                                          link in link_list})

    def renew_link_tail_geo(self, link_list: list[int] = None):
        self.__link.renew_tail_of_geo(target_link=link_list,
                                      loc_dict={
                                          link: self.get_node_geo(self.get_link_from_to(link, _type='bilateral')[1]) for
                                          link in link_list})

    def renew_link_ht_geo(self, link_list: list[int] = None):
        self.__link.renew_geo_of_ht(target_link=link_list, head_loc_dict={
            link: self.get_node_geo(self.get_link_from_to(link, _type='bilateral')[0]) for
            link in link_list}, tail_loc_dict={
            link: self.get_node_geo(self.get_link_from_to(link, _type='bilateral')[1]) for
            link in link_list})

    def modify_link_gdf(self, link_id_list: list[int], attr_field_list: list[str], val_list: list[list] = None):
        self.__link.modify_link_gdf(link_id_list=link_id_list, attr_field_list=attr_field_list, val_list=val_list)

    def modify_node_gdf(self, node_id_list: list[int], attr_field_list: list[str], val_list: list[list] = None):
        self.__node.modify_node_gdf(node_id_list=node_id_list, attr_field_list=attr_field_list, val_list=val_list)

    def get_link_startswith_nodes(self, node_list: list[int], _type: str = 'single') -> list[int, int]:
        return self.__link.get_link_startswith_nodes(node_list=node_list, _type=_type)

    def get_link_endswith_nodes(self, node_list: list[int], _type: str = 'single') -> list[int, int]:
        return self.__link.get_link_endswith_nodes(node_list=node_list, _type=_type)

    def export_net(self, export_crs: str = 'EPSG:4326', out_fldr: str = None, flag_name: str = None,
                   file_type: str = 'geojson') -> None:
        link_file_name = '_'.join([flag_name, 'link']) if flag_name is not None or flag_name != '' else 'link'
        node_file_name = '_'.join([flag_name, 'node']) if flag_name is not None or flag_name != '' else 'node'

        export_link_gdf = self.__link.link_gdf.to_crs(export_crs)
        export_node_gdf = self.__node.get_node_data().to_crs(export_crs)

        export_link_gdf.reset_index(inplace=True, drop=True)
        export_node_gdf.reset_index(inplace=True, drop=True)

        save_file(data_item=export_link_gdf, file_type=file_type, file_name=link_file_name, out_fldr=out_fldr)
        save_file(data_item=export_node_gdf, file_type=file_type, file_name=node_file_name, out_fldr=out_fldr)

    def split_link(self, p: Point or tuple or list = None, target_link: int = None,
                   omitted_length_threshold: float = 0.8) -> tuple[bool, Point, list[int], str]:
        """
        using one point to split a link: create a new link
        :param p:
        :param target_link:
        :param omitted_length_threshold:
        :return:
        """
        if isinstance(p, Point):
            pass
        else:
            p = Point(p)

        # 获取打断点到target link的投影点信息
        target_link_geo = self.get_link_geo(target_link, _type='bilateral')
        prj_p, p_prj_l, prj_route_l, target_l, split_link_geo_list = prj_inf(p=p, line=target_link_geo)

        if (target_l - prj_route_l) <= omitted_length_threshold:
            # this means that we should merge the to_node and p
            return False, Point(), [], 'tail_beyond'
        elif prj_route_l <= omitted_length_threshold:
            # this means that we should merge the from_node and p
            return False, Point(), [], 'head_beyond'
        else:
            link_info_before_modify = self.__link.link_series(target_link)
            self.__link.modify_link_gdf([target_link], [to_node_field, length_field, geometry_field],
                                        [[-1, split_link_geo_list[0].length, split_link_geo_list[0]]])

            # new link
            new_link_id = self.__link.available_link_id

            # other attr copy
            other_attr = {col: [link_info_before_modify[col]] for col in link_info_before_modify.index if
                          col not in [link_id_field, from_node_field, to_node_field, dir_field, length_field,
                                      geometry_field]}

            self.__link.append_links([new_link_id], [-1], [link_info_before_modify[to_node_field]],
                                     [int(link_info_before_modify[dir_field])], [split_link_geo_list[1]], **other_attr)

            return True, prj_p, [target_link, new_link_id], 'split'

    def drop_dup_ft_road(self):
        self.__link.drop_dup_ft_road()

    def merger_double_link(self):
        self.__link.merge_double_link()

    def del_short_links(self, l_threshold: float = 0.5) -> None:
        self.__link.del_short_links(l_threshold=l_threshold)
        self.del_zero_degree_nodes()

    def del_zero_degree_nodes(self) -> None:
        self.__node.delete_nodes(node_list=list(self.__node.node_id_set() - self.__link.used_node()))
