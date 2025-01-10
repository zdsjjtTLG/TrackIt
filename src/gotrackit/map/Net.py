# -- coding: utf-8 --
# @Time    : 2023/12/10 20:02
# @Author  : TangKai
# @Team    : ZheChengData

"""
路网信息存储与相关方法
"""
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing
from .Link import Link
from .Node import Node
import geopandas as gpd
from itertools import islice
from ..tools.group import cut_group
from shapely.geometry import LineString
from ..tools.geo_process import prj_inf
from ..tools.grid import get_grid_data
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
path_field = net_field.NODE_PATH_FIELD
cost_field = net_field.COST_FIELD
grid_id_field = net_field.GRID_ID
o_node_field, d_node_field = net_field.S_NODE, net_field.T_NODE


class Net(object):
    """

    """
    @function_time_cost
    def __init__(self, link_path: str = None, node_path: str = None, link_gdf: gpd.GeoDataFrame = None,
                 node_gdf: gpd.GeoDataFrame = None, weight_field: str = 'length', init_from_existing: bool = False,
                 is_check: bool = True, create_single: bool = True, search_method: str = 'dijkstra',
                 ft_link_mapping: dict = None, double_single_mapping: dict = None, link_ft_mapping: dict = None,
                 link_t_mapping: dict = None, link_f_mapping: dict = None, link_geo_mapping: dict = None,
                 not_conn_cost: float = 1000.0, cache_path: bool = True, cache_id: bool = True,
                 is_sub_net: bool = False, fmm_cache: bool = False, cache_cn: int = 1, cache_slice: int = None,
                 fmm_cache_fldr: str = r'./', prj_cache: bool = False, grid_len: float = 2000.0, is_hierarchical: bool = False,
                 cache_name: str = 'cache', recalc_cache: bool = True,
                 cut_off: float = 1200.0, delete_circle: bool = True, plane_crs: str = None):
        """Net类初始化
        
        Args:
            link_gdf: 线层数据, 必需参数
            node_gdf: 点层数据, 必需参数
            link_path: link层的路网文件路径, 若指定了该参数, 则直接从磁盘读取线层, 可选参数
            node_path: node层的路网文件路径, 若指定了该参数, 则直接从磁盘读取点层, 可选参数
            cut_off: 路径搜索截断长度, 米
            not_conn_cost: 不连通路径的阻抗(m)
            fmm_cache: 路径预计算参数1：是否启用路径预计算
            fmm_cache_fldr: 路径预计算参数2：存储路径预计算结果的文件目录
            recalc_cache: 路径预计算参数3：是否重新进行路径预计算, 取值False时, 程序会去fmm_cache_fldr下读取缓存, 若读取失败则会重新进行路径预计算
            cache_cn: 路径预计算参数4：使用几个核进行路径预计算
            cache_slice: 路径预计算参数5：大于0的整数, 表示将路径预计算结果切分为cache_slice份(大规模路网启用预计算时, 增大该值可以防止内存溢出)
            prj_cache: 是否启用投影缓存
            is_hierarchical: 是否启用空间分层
            cache_path: 是否缓存最短路径, 该参数只在use_sub_net=False且fmm_cache=False时生效, 即：匹配过程中计算好的单源最短路信息会被缓存用于减少后续的最短路计算量, 该选项启用后有可能会造成内存溢出问题
            grid_len: 启用空间分层时, 该参数起效, 意为将路网区域划分为grid_len(m)的栅格
            cache_name: 路径预存储的标志名称, 默认cache
            plane_crs: 要使用的平面投影坐标系, 用户若不指定, 程序会依据路网的经纬度范围自动进行6度投影带的选择, 推荐使用程序自动
            weight_field: 搜路权重字段, 目前只能为length
            create_single: 是否在初始化的时候创建单向路网(若该Net传入地图匹配模块中, 无需指定使用默认值即可)
            search_method: 路径搜索方法, 目前只能为dijkstra
            is_sub_net: 用户不可指定
            init_from_existing: 用户不可指定
            double_single_mapping: 用户不可指定
            link_ft_mapping: 用户不可指定
            link_f_mapping: 用户不可指定
            link_t_mapping: 用户不可指定
            link_geo_mapping: 用户不可指定
            ft_link_mapping: 用户不可指定
        """
        self.not_conn_cost = not_conn_cost
        self.geo_crs = geo_crs
        self.search_method = search_method
        self.weight_field = weight_field
        self.all_pair_path_df = pd.DataFrame()
        self.__stp_cache = dict() or pd.DataFrame
        self.__done_path_cost = dict() or pd.DataFrame
        self.__done_stp_cost_df = pd.DataFrame()
        self.__cache_prj_inf = dict()
        self.cache_path = cache_path
        self.cache_id = cache_id
        self.__is_sub_net = is_sub_net
        self.fmm_cache = fmm_cache
        self.cut_off = cut_off
        if cache_cn > os.cpu_count():
            cache_cn = os.cpu_count()
        self.cache_cn = cache_cn
        self.cache_name = cache_name
        self.fmm_cache_fldr = fmm_cache_fldr
        self.prj_cache = prj_cache
        self.recalc_cache = recalc_cache
        self.cache_slice = cache_slice
        self.delete_circle = delete_circle
        self.grid_len = 2000.0 if grid_len < 2000.0 else grid_len
        self.is_hierarchical = is_hierarchical
        self.region_grid = gpd.GeoDataFrame()
        self.grid_cor_link = pd.DataFrame()
        self.done_sjoin_cache = False
        if self.cache_slice is None:
            self.cache_slice = 2 * self.cache_cn

        if node_gdf is None:
            self.__node = Node(node_gdf=gpd.read_file(node_path), is_check=is_check, init_available_node=self.cache_id,
                               plane_crs=plane_crs)
        else:
            self.__node = Node(node_gdf=node_gdf, is_check=is_check, init_available_node=self.cache_id,
                               plane_crs=plane_crs)

        if not init_from_existing:
            self.__node.init_node()
        else:
            pass

        if link_gdf is None:
            self.__link = Link(link_gdf=gpd.read_file(link_path), weight_field=self.weight_field, is_check=is_check,
                               planar_crs=self.__node.planar_crs, init_available_link=self.cache_id,
                               not_conn_cost=self.not_conn_cost, delete_circle=self.delete_circle)
        else:
            self.__link = Link(link_gdf=link_gdf, weight_field=self.weight_field, is_check=is_check,
                               planar_crs=self.__node.planar_crs, init_available_link=self.cache_id,
                               not_conn_cost=self.not_conn_cost, delete_circle=self.delete_circle)
        self.__planar_crs = self.__node.planar_crs
        self.to_plane_prj()
        if not self.is_sub_net:
            self.del_zero_degree_nodes()
            self.__link.renew_length()
        if not init_from_existing:
            if create_single:
                self.__link.init_link()
        else:
            if create_single:
                # for sub net
                self.__link.init_link_from_existing_single_link(single_link_gdf=link_gdf,
                                                                double_single_mapping=double_single_mapping,
                                                                ft_link_mapping=ft_link_mapping,
                                                                link_ft_mapping=link_ft_mapping,
                                                                link_t_mapping=link_t_mapping,
                                                                link_f_mapping=link_f_mapping,
                                                                link_geo_mapping=link_geo_mapping)
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

    @property
    def is_sub_net(self) -> bool:
        return self.__is_sub_net

    def get_path_cache(self) -> pd.DataFrame:
        return self.__done_stp_cost_df

    def set_path_cache(self, stp_cost_df: pd.DataFrame = None) -> None:
        self.__done_stp_cost_df = stp_cost_df

    def get_prj_cache(self) -> dict:
        return self.__cache_prj_inf

    def set_prj_cache(self, cache_prj_inf: dict = None) -> None:
        self.__cache_prj_inf = cache_prj_inf

    def check(self) -> None:
        """检查点层线层的关联一致性"""
        node_set = set(self.__node.get_node_data().index)
        link_node_set = set(self.__link.get_bilateral_link_data()[net_field.FROM_NODE_FIELD]) | \
                        set(self.__link.get_bilateral_link_data()[net_field.TO_NODE_FIELD])
        assert link_node_set.issubset(node_set), 'some nodes in the link layer are not recorded in the node layer'

    def init_net(self, stp_cost_cache_df: pd.DataFrame = None, cache_prj_inf: dict = None) -> None:
        self.__link.create_graph(weight_field=self.weight_field)
        if not self.is_sub_net and self.is_hierarchical:
            try:
                self.cal_sjoin_cache()
            except Exception as e:
                print(fr'spatial layered association failure: {repr(e)}')
        if self.is_sub_net:
            self.set_prj_cache(cache_prj_inf)
        else:
            if self.prj_cache:
                try:
                    self.cache_prj_info()
                except Exception as e:
                    print(repr(e))

        if self.fmm_cache:
            if self.is_sub_net:
                self.set_path_cache(stp_cost_cache_df)
            else:
                self.fmm_path_cache()

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

    def get_shortest_path(self, o_node=1, d_node=2) -> list:
        """

        :param o_node:
        :param d_node:
        :return:
        """
        return self.__link.get_shortest_path(o_node=o_node, d_node=d_node)

    def get_shortest_length(self,  o_node=1, d_node=2) -> float:

        return self.__link.get_shortest_length(o_node=o_node, d_node=d_node)

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
        self.__link.to_plane_prj()
        self.__node.to_plane_prj()

    def to_geo_prj(self):
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
    def link_t_map(self) -> dict[int, int]:
        return self.__link.link_t_map

    @property
    def link_geo_map(self) -> dict[int, LineString]:
        return self.__link.link_geo_map

    @property
    def link_f_map(self) -> dict[int, int]:
        return self.__link.link_f_map

    @property
    def available_link_id(self) -> int:
        return self.__link.available_link_id

    def get_ft_node_link_mapping(self):
        return self.__link.get_ft_link_mapping()

    @function_time_cost
    def create_computational_net(self, gps_array_buffer: Polygon = None, weight_field: str = 'length',
                                 cache_path: bool = True, cache_id: bool = True, not_conn_cost: float = 999.0,
                                 fmm_cache: bool = False, prj_cache: bool = True, must_contain_link: list[int] = None):
        """

        Args:
            gps_array_buffer:
            weight_field:
            cache_path:
            cache_id:
            not_conn_cost:
            fmm_cache:
            prj_cache:
            must_contain_link:

        Returns:

        """
        if gps_array_buffer is None:
            return None
        gps_array_buffer_gdf = gpd.GeoDataFrame({'geometry': [gps_array_buffer]}, geometry='geometry',
                                                crs=self.planar_crs)
        single_link_gdf = self.get_link_data()
        single_link_gdf.reset_index(inplace=True, drop=True)
        if self.is_hierarchical:
            try:
                pre_filter_link = self.calc_pre_filter(gps_array_buffer_gdf)
                single_link_gdf = single_link_gdf[single_link_gdf[link_id_field].isin(pre_filter_link)]
            except Exception as e:
                print(repr(e), 'spatial layered association failure')
        sub_single_link_gdf = gpd.sjoin(single_link_gdf, gps_array_buffer_gdf)

        if must_contain_link is not None and must_contain_link:
            _gap = set(must_contain_link) - set(sub_single_link_gdf[net_field.SINGLE_LINK_ID_FIELD])
            if _gap:
                sub_single_link_gdf = pd.concat(
                    [sub_single_link_gdf,
                     single_link_gdf[single_link_gdf[net_field.SINGLE_LINK_ID_FIELD].isin(_gap)]])

        if sub_single_link_gdf.empty:
            print(rf'the GPS data cannot be associated with any road network data within the specified buffer range...')
            return None
        sub_single_link_gdf.drop(columns=['index_right'], axis=1, inplace=True)
        sub_single_link_gdf.drop_duplicates(subset=[net_field.SINGLE_LINK_ID_FIELD], inplace=True)
        sub_node_list = list(set(sub_single_link_gdf[net_field.FROM_NODE_FIELD]) | \
                             set(sub_single_link_gdf[net_field.TO_NODE_FIELD]))
        sub_node_gdf = self.__node.get_node_data().loc[sub_node_list, :].copy()
        sub_net = Net(link_gdf=sub_single_link_gdf,
                      node_gdf=sub_node_gdf,
                      weight_field=weight_field,
                      init_from_existing=True, is_check=False, cache_path=cache_path, cache_id=cache_id,
                      not_conn_cost=not_conn_cost, is_sub_net=True, fmm_cache=fmm_cache, prj_cache=prj_cache,
                      ft_link_mapping=self.get_ft_node_link_mapping(),
                      link_ft_mapping=self.link_ft_map, link_f_mapping=self.link_f_map, link_t_mapping=self.link_t_map,
                      double_single_mapping=self.bilateral_unidirectional_mapping, cut_off=self.cut_off,
                      delete_circle=False, is_hierarchical=False)
        sub_net.init_net(stp_cost_cache_df=self.get_path_cache(), cache_prj_inf=self.get_prj_cache())
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
        """

        Args:
            export_crs:
            out_fldr:
            flag_name:
            file_type:

        Returns:

        """
        link_file_name = '_'.join([flag_name, 'link']) if flag_name is not None and flag_name != '' else 'link'
        node_file_name = '_'.join([flag_name, 'node']) if flag_name is not None and flag_name != '' else 'node'

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
        prj_p, p_prj_l, prj_route_l, target_l, split_link_geo_list, _, _ = prj_inf(p=p, line=target_link_geo)

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

    def check_ln_consistency(self):
        l_node = self.__link.used_node()
        n_node = self.__node.node_id_set()
        gap = n_node - l_node
        # print(rf'1: {gap}')
        if gap:
            # 点层中存在线层没有用的点, 直接删掉
            self.__node.delete_nodes(node_list=list(gap))

        # n_node = self.__node.node_id_set()
        # gap = l_node - n_node
        # print(rf'2: {gap}')

    def del_short_links(self, l_threshold: float = 0.5) -> None:
        self.__link.del_short_links(l_threshold=l_threshold)
        self.del_zero_degree_nodes()

    def del_zero_degree_nodes(self) -> None:
        self.__node.delete_nodes(node_list=list(self.__node.node_id_set() - self.__link.used_node()))

    def divide_links(self, divide_l: float = 70.0, min_l: float = 1.0, is_init_link: bool = True,
                     method: str = 'alpha') -> None:
        """

        Args:
            divide_l:
            min_l:
            is_init_link:
            method:

        Returns:

        """
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
            process_link_gdf[['__divide_l__', '__divide_p__', '__l__', '__seq__']] = process_link_gdf.apply(
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
            # new_node_gdf['p_l'] = new_node_gdf.apply(lambda row: len(row['__divide_p__']), axis=1)
            new_node_gdf['p_l'] = new_node_gdf['__divide_p__'].apply(lambda x: len(x))
            new_node_gdf.drop(index=new_node_gdf[new_node_gdf['p_l'] == 0].index, axis=0, inplace=True)
            new_node_gdf = pd.DataFrame(new_node_gdf)
            new_node_gdf.drop(columns=['p_l'], axis=1, inplace=True)
            new_node_gdf = new_node_gdf.explode(column=['__divide_p__', 'new_p'], ignore_index=True)
            new_node_gdf.rename(columns={'__divide_p__': geometry_field, 'new_p': node_id_field}, inplace=True)
            new_node_gdf = gpd.GeoDataFrame(new_node_gdf, geometry=geometry_field, crs=self.crs)

            process_link_gdf.drop(columns=['__divide_p__', 'new_p'], axis=1, inplace=True)
            process_link_gdf = pd.DataFrame(process_link_gdf)
            new_link_gdf = process_link_gdf.explode(column=['__divide_l__', '__new_ft__', '__seq__'], ignore_index=True)
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

    def get_circle_link(self) -> set[int]:
        link_gdf = self.__link.link_gdf
        return set(link_gdf[link_gdf[from_node_field] == link_gdf[to_node_field]][link_id_field])

    def get_same_ft_link(self) -> set[int]:
        link_gdf = self.__link.link_gdf
        link_gdf['__temp__'] = link_gdf.apply(
            lambda row: tuple(sorted((row[net_field.FROM_NODE_FIELD], row[net_field.TO_NODE_FIELD]))), axis=1)
        dup_ft = set(link_gdf[link_gdf.duplicated(subset=['__temp__'])][
                         net_field.LINK_ID_FIELD])
        del link_gdf['__temp__']
        return dup_ft

    def process_circle(self):
        """处理环路和相同f-t node的路"""
        candidate_link_set = self.get_circle_link()
        self.process_target(target_link_list=candidate_link_set)

        candidate_link_set = self.get_same_ft_link()
        self.process_target(target_link_list=candidate_link_set)

        self.check()

    def process_target(self, target_link_list: list[int] or set[int] = None):
        for target_link in target_link_list:
            target_geo = self.__link.get_link_geo(target_link, _type='bilateral')
            split_ok, prj_p, modified_link, res_type = \
                self.split_link(target_geo.interpolate(target_geo.length / 2),
                                target_link, omitted_length_threshold=0.01)
            if split_ok:
                new_node_id = self.__node.available_node_id
                self.__node.append_nodes(node_id=[new_node_id], geo=[prj_p])
                self.modify_link_gdf(link_id_list=[modified_link[0]], attr_field_list=[to_node_field],
                                     val_list=[[new_node_id]])
                self.modify_link_gdf(link_id_list=[modified_link[1]], attr_field_list=[from_node_field],
                                     val_list=[[new_node_id]])
                self.renew_link_tail_geo(link_list=[modified_link[0]])
                self.renew_link_head_geo(link_list=[modified_link[1]])

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

    @function_time_cost
    def fmm_path_cache(self):
        if self.fmm_cache_fldr is None:
            self.fmm_cache_fldr = r'./'
        if not self.recalc_cache:
            try:
                with open(os.path.join(self.fmm_cache_fldr, rf'{self.cache_name}_path_cache'), 'rb') as f:
                    done_stp_cache_df = pickle.load(f)
                self.set_path_cache(done_stp_cache_df)
                print(rf'using local fmm cache...')
                return None
            except Exception as e:
                print(repr(e))

        link = self.__link.get_bilateral_link_data()
        g = self.graph
        node_list = list(set(link[net_field.FROM_NODE_FIELD]) | set(link[net_field.TO_NODE_FIELD]))
        del link
        print(rf'calc fmm cache...')
        if self.cache_cn <= 1:
            done_stp_cost_df = self.single_source_cache(node_list, g, self.cut_off, self.weight_field, self.cache_slice)
        else:
            done_stp_cost_df = pd.DataFrame()
            node_group = cut_group(obj_list=node_list, n=self.cache_cn)
            pool = multiprocessing.Pool(processes=len(node_group))
            result_list = []
            for i in range(0, len(node_group)):
                result = pool.apply_async(self.single_source_cache,
                                          args=(node_group[i], g, self.cut_off, self.weight_field, self.cache_slice))
                result_list.append(result)
            pool.close()
            pool.join()
            for res in result_list:
                done_stp_cost_df = pd.concat([done_stp_cost_df, res.get()])
            done_stp_cost_df.reset_index(inplace=True, drop=True)
        _ = self.__link.get_link_data()[[net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD,
                                         self.weight_field]].rename(
            columns={net_field.FROM_NODE_FIELD: o_node_field, net_field.TO_NODE_FIELD: d_node_field,
                     self.weight_field: cost_field})
        _[path_field] = _.apply(lambda row: [int(row[o_node_field]), int(row[d_node_field])], axis=1)
        done_stp_cost_df = pd.concat([_, done_stp_cost_df])
        del _
        done_stp_cost_df.drop_duplicates(subset=[o_node_field, d_node_field], keep='first', inplace=True)
        done_stp_cost_df.reset_index(inplace=True, drop=True)
        done_stp_cost_df['2nd_node'], done_stp_cost_df['-2nd_node'] = -1, -1
        normal_path_idx = done_stp_cost_df[cost_field] > 0
        try:
            done_stp_cost_df.loc[normal_path_idx, '2nd_node'] = done_stp_cost_df.loc[normal_path_idx, :][
                path_field].apply(
                lambda x: x[1])
            done_stp_cost_df.loc[normal_path_idx, '-2nd_node'] = done_stp_cost_df.loc[normal_path_idx, :][
                path_field].apply(
                lambda x: x[-2])
        except:
            pass
        with open(os.path.join(self.fmm_cache_fldr, rf'{self.cache_name}_path_cache'), 'wb') as f:
            pickle.dump(done_stp_cost_df, f)
        self.set_path_cache(done_stp_cost_df)

    @staticmethod
    def slice_save(done_stp_cache: dict = None, done_cost_cache: dict = None, n: int = 3) -> pd.DataFrame:
        temp_stp_list = [{} for i in range(n)]
        temp_cost_list = [{} for i in range(n)]
        _ = [temp_stp_list[i % n].update({key: done_stp_cache[key]}) for i, key in enumerate(done_stp_cache.keys())]
        _ = [temp_cost_list[i % n].update({key: done_cost_cache[key]}) for i, key in enumerate(done_cost_cache.keys())]

        stp_cost_res = pd.DataFrame()
        del done_stp_cache, done_cost_cache
        for stp_cache, cost_cache in zip(temp_stp_list, temp_cost_list):
            done_stp_cache_df = pd.DataFrame(stp_cache).stack().reset_index(drop=False).rename(
                columns={'level_0': d_node_field, 'level_1': o_node_field, 0: path_field})
            done_stp_cache_df.dropna(subset=[o_node_field, d_node_field], how='any', inplace=True)

            done_cost_cache_df = pd.DataFrame(cost_cache).stack().reset_index(drop=False).rename(
                columns={'level_0': d_node_field, 'level_1': o_node_field, 0: cost_field})
            done_cost_cache_df.dropna(subset=[o_node_field, d_node_field], how='any', inplace=True)
            done_cost_cache_df[cost_field] = np.around(done_cost_cache_df[cost_field], decimals=1)

            stp_cost_cache_df = pd.merge(done_cost_cache_df, done_stp_cache_df, on=[o_node_field, d_node_field])
            del done_stp_cache_df, done_cost_cache_df
            stp_cost_res = pd.concat([stp_cost_res, stp_cost_cache_df])
            del stp_cost_cache_df
        stp_cost_res.reset_index(inplace=True, drop=True)
        return stp_cost_res

    def single_source_cache(self, node_list: list = None, g: nx.DiGraph = None,
                            cut_off: float = 500.0, weight_field: str = 'length', n: int = 2) -> pd.DataFrame:
        done_cost_cache, done_stp_cache = {}, {}
        for node in node_list:
            try:
                done_cost_cache[node], done_stp_cache[node] = nx.single_source_dijkstra(g, node, weight=weight_field,
                                                                                        cutoff=cut_off)
            except Exception as e:
                pass

        done_stp_cost_df = self.slice_save(done_stp_cache=done_stp_cache,
                                           done_cost_cache=done_cost_cache,
                                           n=n)
        return done_stp_cost_df

    def single_link_ft_cost(self) -> pd.DataFrame:
        _ = self.__link.get_link_data()[
            [net_field.SINGLE_LINK_ID_FIELD, net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD, self.weight_field,
             'path']].copy()
        _.rename(columns={self.weight_field: 'cost'}, inplace=True)
        return _

    @function_time_cost
    def cache_prj_info(self):
        single_link_gdf = self.__link.get_link_data()
        single_link_gdf = single_link_gdf[
            [net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD, net_field.GEOMETRY_FIELD, net_field.LENGTH_FIELD]].copy()

        cache_prj_gdf = self.split_segment(single_link_gdf)
        cache_prj_gdf[net_field.SEG_COUNT] = \
            cache_prj_gdf.groupby([net_field.FROM_NODE_FIELD,
                                   net_field.TO_NODE_FIELD])[net_field.X_DIFF].transform('count')
        dup = cache_prj_gdf[(cache_prj_gdf[net_field.VEC_LEN] == 0) &
                            (cache_prj_gdf[net_field.SEG_COUNT] > 1)]
        if not dup.empty:
            cache_prj_gdf.drop(index=dup.index, inplace=True, axis=0)
            cache_prj_gdf.reset_index(inplace=True, drop=True)
            cache_prj_gdf[net_field.SEG_COUNT] = \
                cache_prj_gdf.groupby([net_field.FROM_NODE_FIELD,
                                       net_field.TO_NODE_FIELD])[net_field.X_DIFF].transform('count')
        cache_prj_inf = {1: cache_prj_gdf[cache_prj_gdf[net_field.SEG_COUNT] == 1].copy().reset_index(drop=True),
                         2: cache_prj_gdf[cache_prj_gdf[net_field.SEG_COUNT] > 1].copy().reset_index(drop=True)}
        del cache_prj_gdf
        self.set_prj_cache(cache_prj_inf)

    def calc_pre_filter(self, gps_rou_buffer_gdf: gpd.GeoDataFrame = None) -> set[int]:
        # sjoin between gps & grid
        gps_cor_grid_df = gpd.sjoin(gps_rou_buffer_gdf, self.region_grid)
        if gps_cor_grid_df.empty:
            raise ValueError(r'the GPS data cannot be associated with any road network data within the specified buffer range...')
        gps_cor_grid = set(gps_cor_grid_df[grid_id_field])
        pre_link = set(self.grid_cor_link[self.grid_cor_link[grid_id_field].isin(gps_cor_grid)][link_id_field])
        return pre_link

    @function_time_cost
    def cal_sjoin_cache(self):
        if not self.done_sjoin_cache:
            min_x, min_y, max_x, max_y = self.get_bounds()
            self.region_grid = \
                self.generate_region_grid(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                                          grid_len=self.grid_len, crs=self.crs)
            link_gdf = self.__link.link_gdf[[link_id_field, geometry_field]]
            link_gdf.reset_index(inplace=True, drop=True)
            self.grid_cor_link = gpd.sjoin(self.region_grid[[grid_id_field, geometry_field]], link_gdf)
            self.region_grid = self.region_grid[
                self.region_grid[grid_id_field].isin(set(self.grid_cor_link[grid_id_field]))]
            # self.region_grid.to_file(r'grid.shp')
            del self.grid_cor_link['index_right']
            del self.grid_cor_link[geometry_field]
            self.done_sjoin_cache = True

    @staticmethod
    def generate_region_grid(grid_len: float = 5000, max_y: float = None, min_y: float = None,
                             max_x: float = None, min_x: float = None, crs: str = 'EPSG:32650') -> gpd.GeoDataFrame:
        grid_df = get_grid_data(polygon_gdf=gpd.GeoDataFrame(geometry=[Polygon([(min_x, min_y), (max_x, min_y),
                                                                                (max_x, max_y),
                                                                                (min_x, max_y)])],
                                                             crs=crs), meter_step=grid_len, is_geo_coord=False,
                                generate_index=False)
        return grid_df

    def get_bounds(self) -> tuple[float, float, float, float]:
        bound_link = self.__link.link_gdf.bounds
        min_lng, min_lat, max_lng, max_lat = \
            bound_link['minx'].min(), bound_link['miny'].min(), bound_link['maxx'].max(), bound_link['maxy'].max()
        return min_lng, min_lat, max_lng, max_lat

    @staticmethod
    def split_segment(path_gdf: gpd.GeoDataFrame = None, del_loc: bool = True) -> gpd.GeoDataFrame or pd.DataFrame:
        """
        拆解轨迹坐标, 并且粗去重(按照路段的起终点坐标)
        :param path_gdf: gpd.GeoDataFrame(), 必需参数, 必须字段: [geometry], crs要求EPSG:4326
        :param del_loc:
        :return: gpd.GeoDataFrame()
        """
        path_gdf['point_list'] = path_gdf[net_field.GEOMETRY_FIELD].apply(lambda x: list(x.coords))
        path_gdf['line_list'] = path_gdf['point_list'].apply(
            lambda x: [(x[i], x[i + 1]) for i in range(0, len(x) - 1)])
        path_gdf['topo_seq'] = path_gdf['line_list'].apply(lambda x: [i for i in range(len(x))])
        path_gdf.drop(columns=[net_field.GEOMETRY_FIELD, 'point_list'], axis=1, inplace=True)
        path_gdf = pd.DataFrame(path_gdf)
        path_gdf = path_gdf.explode(column=['line_list', 'topo_seq'], ignore_index=True)
        path_gdf.rename(columns={'line_list': 'ft-loc'}, inplace=True)
        # path_gdf = gpd.GeoDataFrame(path_gdf, crs=origin_crs, geometry=net_field.GEOMETRY_FIELD)
        path_gdf['f_x'] = path_gdf['ft-loc'].apply(lambda x: x[0][0])
        path_gdf['f_y'] = path_gdf['ft-loc'].apply(lambda x: x[0][1])
        path_gdf['t_x'] = path_gdf['ft-loc'].apply(lambda x: x[1][0])
        path_gdf['t_y'] = path_gdf['ft-loc'].apply(lambda x: x[1][1])
        del path_gdf['ft-loc'], path_gdf[net_field.LENGTH_FIELD]
        path_gdf[net_field.X_DIFF] = path_gdf['t_x'] - path_gdf['f_x']
        path_gdf[net_field.Y_DIFF] = path_gdf['t_y'] - path_gdf['f_y']
        if del_loc:
            del path_gdf['f_x'], path_gdf['f_y'], path_gdf['t_x'], path_gdf['t_y']
        path_gdf[net_field.VEC_LEN] = np.sqrt(path_gdf[net_field.X_DIFF] ** 2 + path_gdf[net_field.Y_DIFF] ** 2)
        # path_gdf['__l__'] = path_gdf[net_field.GEOMETRY_FIELD].length
        path_gdf[net_field.SEG_ACCU_LENGTH] = \
            path_gdf.groupby([net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD])[[net_field.VEC_LEN]].cumsum()
        path_gdf['topo_seq'] = path_gdf['topo_seq'].astype(int)
        # del path_gdf['__l__']
        return path_gdf

    def shortest_k_paths(self, o: int = None, d: int = None, k: int = 2) -> list:
        """Net类方法 - shortest_k_paths：

        - 计算两个节点之间的K条最短路

        Args:
            o: 起点节点
            d: 终点节点
            k: 路径数目

        Returns:
            list
        """
        g = self.__link.get_graph()
        return list(islice(nx.shortest_simple_paths(g, o, d, weight=self.weight_field), k))

    def get_single_link(self):
        return self.__link.get_link_data()

    def divide_links_gamma(self, break_pts: gpd.GeoDataFrame, min_l: float = 1.0, is_init_link: bool = True):
        break_pts[node_id_field] = '-1'
        for index, stop_row in break_pts.iterrows():
            stop_point = stop_row.geometry
            # 找到最近的 road 线段
            road_link_o = self.__link.link_gdf
            nearest_road = road_link_o.iloc[road_link_o.geometry.distance(stop_point).argmin()]
            nearest_road_line = nearest_road[geometry_field]

            # 计算线段上距离 stop 点最近的点
            # nearest_pt = nearest_points(stop_point, nearest_road_line)[1]  # nearest_points 返回两个点，这里取第二个（road 上的点）
            split_ok, prj_p, modified_link, res_type = self.split_link(stop_point, nearest_road[link_id_field],
                                                                       omitted_length_threshold=2)
            if split_ok:
                new_node_id = self.__node.available_node_id
                break_pts.at[index, node_id_field] = new_node_id
                self.__node.append_nodes(node_id=[new_node_id], geo=[prj_p])
                self.modify_link_gdf(link_id_list=[modified_link[0]], attr_field_list=[to_node_field],
                                     val_list=[[new_node_id]])
                self.modify_link_gdf(link_id_list=[modified_link[1]], attr_field_list=[from_node_field],
                                     val_list=[[new_node_id]])
                self.renew_link_tail_geo(link_list=[modified_link[0]])
                self.renew_link_head_geo(link_list=[modified_link[1]])
            else:
                road_node_o = self.__node.get_node_data()
                nearest_node = road_node_o.iloc[road_node_o.geometry.distance(stop_point).argmin()]
                break_pts.at[index, node_id_field] = nearest_node[node_id_field]

        if is_init_link:
            self.check()
            self.__link.init_link()
            self.__node.init_node()

        return break_pts
