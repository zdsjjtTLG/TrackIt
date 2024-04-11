# -- coding: utf-8 --
# @Time    : 2023/12/10 20:02
# @Author  : TangKai
# @Team    : ZheChengData

"""
路网信息存储与相关方法
"""

import numpy as np
import pandas as pd
from .Link import Link
from .Node import Node
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
from ..tools.geo_process import prj_inf
from ..tools.save_file import save_file
from ..GlobalVal import NetField, PrjConst
from ..WrapsFunc import function_time_cost
from shapely.geometry import Polygon, Point
from ..tools.geo_process import divide_line_by_l

net_field = NetField()
prj_const = PrjConst()

geo_crs = prj_const.PRJ_CRS
link_id_field = net_field.LINK_ID_FIELD
dir_field = net_field.DIRECTION_FIELD
from_node_field = net_field.FROM_NODE_FIELD
to_node_field = net_field.TO_NODE_FIELD
length_field = net_field.LENGTH_FIELD
geometry_field = net_field.GEOMETRY_FIELD
node_id_field = net_field.NODE_ID_FIELD


class Net(object):

    @function_time_cost
    def __init__(self, link_path: str = None, node_path: str = None, link_gdf: gpd.GeoDataFrame = None,
                 node_gdf: gpd.GeoDataFrame = None, weight_field: str = 'length', init_from_existing: bool = False,
                 is_check: bool = True, create_single: bool = True, search_method: str = 'dijkstra',
                 not_conn_cost: float = 999.0, cache_path: bool = True, cache_id: bool = True):
        """
        创建Net类
        :param link_path: link层的路网文件路径, 若指定了该参数, 则直接从磁盘IO创建Net线层
        :param node_path: node层的路网文件路径, 若指定了该参数, 则直接从磁盘IO创建Net点层
        :param link_gdf: 若指定了该参数, 则直接从内存中的gdf创建Net线层
        :param node_gdf: 若指定了该参数, 则直接从内存中的gdf创建Net点层
        :param weight_field: 搜路权重字段
        :param create_single: 是否在初始化的时候创建single层
        :param search_method: 路径搜索方法, 'dijkstra' or 'bellman-ford'
        :param init_from_existing: 是否直接从内存中的gdf创建single_link_gdf, 该参数用于类内部创建子net, 用户不用关心该参数, 使用默认值即可
        :param not_conn_cost: 不连通路径的阻抗(m)
        """
        self.not_conn_cost = not_conn_cost
        self.geo_crs = geo_crs
        self.search_method = search_method
        self.weight_field = weight_field
        self.all_pair_path_df = pd.DataFrame()
        self.__stp_cache = dict()
        self.__done_path_cost = dict()
        self.cache_path = cache_path
        self.cache_id = cache_id

        if node_gdf is None:
            self.__node = Node(node_gdf=gpd.read_file(node_path), is_check=is_check, init_available_node=self.cache_id)
        else:
            self.__node = Node(node_gdf=node_gdf, is_check=is_check, init_available_node=self.cache_id)

        if link_gdf is None:
            self.__link = Link(link_gdf=gpd.read_file(link_path), weight_field=self.weight_field, is_check=is_check,
                               planar_crs=self.__node.planar_crs, init_available_link=self.cache_id,
                               not_conn_cost=self.not_conn_cost)
        else:
            self.__link = Link(link_gdf=link_gdf, weight_field=self.weight_field, is_check=is_check,
                               planar_crs=self.__node.planar_crs, init_available_link=self.cache_id,
                               not_conn_cost=self.not_conn_cost)
        self.__planar_crs = self.__node.planar_crs
        self.to_plane_prj()
        self.__link.renew_length()
        if not init_from_existing:
            if create_single:
                self.__link.init_link()
        else:
            if create_single:
                self.__link.init_link_from_existing_single_link(single_link_gdf=link_gdf)
        if not init_from_existing:
            self.__node.init_node()
        else:
            pass
        if is_check:
            self.check()
    @property
    def planar_crs(self):
        return self.__planar_crs

    @planar_crs.setter
    def planar_crs(self, val):
        self.__planar_crs = val
        self.__link.planar_crs = val
        self.__node.planar_crs = val

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
                return [], self.not_conn_cost
        else:
            self.calc_shortest_path(source=o_node, method=self.search_method)
            try:
                node_path = self.__stp_cache[o_node][d_node]
                cost = self.__done_path_cost[o_node][d_node]
                if not self.cache_path:
                    del self.__stp_cache[o_node]
                    del self.__done_path_cost[o_node]
            except KeyError:
                return [], self.not_conn_cost

        return node_path, cost

    def get_shortest_path_length(self, o_node=1, d_node=2) -> tuple[list, float]:
        """

        :param o_node:
        :param d_node:
        :return:
        """
        return self.__link.get_shortest_path_length(o_node=o_node, d_node=d_node)

    def calc_shortest_path(self, source: int = None, method: str = 'dijkstra') -> None:

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

    def calc_link_vec(self):
        self.__link.calc_link_vec()

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
        if self.__link.crs == self.planar_crs:
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
                                                crs=self.planar_crs)
        sub_single_link_gdf = gpd.sjoin(self.get_link_data(), gps_array_buffer_gdf)
        if sub_single_link_gdf.empty:
            raise ValueError(rf'GPS数据在指定的buffer范围内关联不到任何路网数据...')
        sub_single_link_gdf.drop(columns=['index_right'], axis=1, inplace=True)
        sub_single_link_gdf.drop_duplicates(subset=[net_field.SINGLE_LINK_ID_FIELD], inplace=True)
        sub_node_list = list(set(sub_single_link_gdf[net_field.FROM_NODE_FIELD]) | \
                             set(sub_single_link_gdf[net_field.TO_NODE_FIELD]))
        sub_node_gdf = self.__node.get_node_data().loc[sub_node_list, :].copy()
        sub_net = Net(link_gdf=sub_single_link_gdf,
                      node_gdf=sub_node_gdf,
                      weight_field=self.weight_field,
                      init_from_existing=True, is_check=False,
                      search_method=self.search_method, cache_path=self.cache_path, cache_id=self.cache_id,
                      not_conn_cost=self.not_conn_cost)
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

    def renew_link_ht_geo(self, link_list: list[int] = None, renew_single: bool = True):
        self.__link.renew_geo_of_ht(target_link=link_list, head_loc_dict={
            link: self.get_node_geo(self.get_link_from_to(link, _type='bilateral')[0]) for
            link in link_list}, tail_loc_dict={
            link: self.get_node_geo(self.get_link_from_to(link, _type='bilateral')[1]) for
            link in link_list})
        if renew_single:
            self.__link.init_link()

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
        prj_p, p_prj_l, prj_route_l, target_l, split_link_geo_list, _ = prj_inf(p=p, line=target_link_geo)

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

    def divide_links(self, divide_l: float = 70.0, min_l: float = 1.0, is_init_link: bool = True,
                     method: str = 'alpha') -> None:
        if method == 'alpha':
            self.divide_links_alpha(divide_l=divide_l, min_l=min_l, is_init_link=is_init_link)
        else:
            self.divide_links_beta(divide_l=divide_l, min_l=min_l, is_init_link=is_init_link)

    def divide_links_alpha(self, divide_l: float = 70.0, min_l: float = 1.0, is_init_link: bool = True) -> None:
        self.to_plane_prj()
        link_gdf = self.__link.get_bilateral_link_data()
        target_index = link_gdf[length_field] > divide_l
        process_link_gdf = link_gdf[target_index].copy()
        del_links = list(process_link_gdf.index)
        max_node = self.__node.max_node_id
        if process_link_gdf.empty:
            return None
        else:
            process_link_gdf[['__divide_l__', '__divide_p__', '__l__']] = process_link_gdf.apply(
                lambda row: divide_line_by_l(line_geo=row[geometry_field], divide_l=divide_l, l_min=min_l), axis=1,
                result_type='expand')
            process_link_gdf.reset_index(inplace=True, drop=True)

            process_link_gdf['__l__'] = process_link_gdf['__l__'] - 1
            process_link_gdf['__increment__'] = np.cumsum(process_link_gdf['__l__'])
            process_link_gdf['__increment__'] = process_link_gdf['__increment__'].shift(1).fillna(0).astype(int)
            process_link_gdf[['__new_ft__', 'new_p']] = process_link_gdf.apply(
                lambda row: self.generate_new_ft(origin_f=row[from_node_field], origin_t=row[to_node_field],
                                                 divide_num=row['__l__'],
                                                 start_node=row['__increment__'] + max_node + 1), axis=1,
                result_type='expand')
            new_node_gdf = process_link_gdf[['__divide_p__', 'new_p']].copy()
            new_node_gdf['p_l'] = new_node_gdf.apply(lambda row: len(row['__divide_p__']), axis=1)
            new_node_gdf.drop(index=new_node_gdf[new_node_gdf['p_l'] == 0].index, axis=0, inplace=True)
            new_node_gdf = pd.DataFrame(new_node_gdf)
            new_node_gdf.drop(columns=['p_l'], axis=1, inplace=True)
            new_node_gdf = new_node_gdf.explode(column=['__divide_p__', 'new_p'], ignore_index=True)
            new_node_gdf.rename(columns={'__divide_p__': geometry_field, 'new_p': node_id_field}, inplace=True)
            new_node_gdf = gpd.GeoDataFrame(new_node_gdf, geometry=geometry_field, crs=self.crs)

            process_link_gdf.drop(columns=['__divide_p__', 'new_p'], axis=1, inplace=True)
            process_link_gdf = pd.DataFrame(process_link_gdf)
            new_link_gdf = process_link_gdf.explode(column=['__divide_l__', '__new_ft__'], ignore_index=True)
            del process_link_gdf
            new_link_gdf[[from_node_field, to_node_field]] = new_link_gdf.apply(lambda row: row['__new_ft__'], axis=1,
                                                                                result_type='expand')
            new_link_gdf.drop(columns=['__increment__', geometry_field, '__new_ft__', '__l__'], axis=1, inplace=True)
            new_link_gdf.rename(columns={'__divide_l__': geometry_field}, inplace=True)
            max_link_id = self.__link.max_link_id
            new_link_gdf['_parent_link'] = new_link_gdf[link_id_field]
            new_link_gdf[length_field] = new_link_gdf.apply(lambda row: row[geometry_field].length, axis=1)
            new_link_gdf[link_id_field] = [i + max_link_id for i in range(1, len(new_link_gdf) + 1)]
            new_link_gdf = gpd.GeoDataFrame(new_link_gdf, geometry=geometry_field, crs=self.crs)

            self.__link.delete_links(link_id_list=del_links)
            self.__link.append_link_gdf(new_link_gdf)
            self.__node.append_node_gdf(new_node_gdf)
            self.__link.init_available_link_id()
            self.__node.init_available_node_id()
            if is_init_link:
                self.check()
                self.__link.init_link()
                self.__node.init_node()

    @staticmethod
    def generate_new_ft(origin_f: int = None, origin_t: int = None,
                        divide_num: int = 2, start_node: int = None) -> tuple[list, list]:
        _ = [origin_f] + [start_node + i for i in range(divide_num)] + [origin_t]
        return [[_[i], _[i + 1]] for i in range(len(_) - 1)], _[1:-1]

    def divide_links_beta(self, divide_l: float = 70.0, min_l: float = 1.0, is_init_link: bool = True):
        flag = True
        done_divide_set = set()
        while flag:
            candidate_link_set = self.get_greater_than_threshold(l_threshold=divide_l)
            target_link_set = candidate_link_set - done_divide_set
            if not target_link_set:
                break
            else:
                target_link = target_link_set.pop()
                done_divide_set.add(target_link)
                split_ok, prj_p, modified_link, res_type = self.split_link(
                    self.__link.get_link_geo(target_link, _type='bilateral').interpolate(divide_l),
                    target_link, omitted_length_threshold=min_l)
                if split_ok:
                    new_node_id = self.__node.available_node_id
                    self.__node.append_nodes(node_id=[new_node_id], geo=[prj_p])
                    self.modify_link_gdf(link_id_list=[modified_link[0]], attr_field_list=[to_node_field],
                                         val_list=[[new_node_id]])
                    self.modify_link_gdf(link_id_list=[modified_link[1]], attr_field_list=[from_node_field],
                                         val_list=[[new_node_id]])
                    self.renew_link_tail_geo(link_list=[modified_link[0]])
                    self.renew_link_head_geo(link_list=[modified_link[1]])

        if is_init_link:
            self.check()
            self.__link.init_link()
            self.__node.init_node()

    def get_greater_than_threshold(self, l_threshold: float = None) -> set[int]:
        link_gdf = self.__link.link_gdf
        return set(link_gdf[link_gdf[length_field] > l_threshold][link_id_field])