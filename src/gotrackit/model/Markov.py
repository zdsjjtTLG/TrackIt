# -- coding: utf-8 --
# @Time    : 2023/12/9 8:29
# @Author  : TangKai
# @Team    : ZheChengData

"""Markov Model Class"""


import time
import os.path
import datetime
import warnings
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from ..map.Net import Net
from datetime import timedelta
from ..map.Net import NOT_CONN_COST
from ..solver.Viterbi import Viterbi
from ..gps.LocGps import GpsPointsGdf
from ..WrapsFunc import function_time_cost
from ..tools.geo_process import n_equal_points
from shapely.geometry import Point, LineString
from ..GlobalVal import NetField, GpsField, MarkovField

gps_field = GpsField()
net_field = NetField()
markov_field = MarkovField()

from_link_f, to_link_f = markov_field.FROM_STATE, markov_field.TO_STATE
from_link_n_f, to_link_n_f = markov_field.FROM_STATE_N, markov_field.TO_STATE_N
from_gps_f, to_gps_f = gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ
MIN_P = 1e-10


class HiddenMarkov(object):
    """隐马尔可夫模型类"""
    def __init__(self, net: Net, gps_points: GpsPointsGdf, beta: float = 30.1, gps_sigma: float = 20.0):
        self.gps_points = gps_points
        self.net = net
        # (gps_seq, single_link_id): (prj_p, prj_dis, route_dis)
        self.__done_prj_dict: dict[tuple[int, int]: tuple[Point, float, float, float]] = dict()
        self.__adj_seq_path_dict: dict[tuple[int, int], list[int, int]] = dict()
        self.__ft_transition_dict = dict()
        self.__ft_mapping_dict = dict()
        self.beta = beta
        self.gps_sigma = gps_sigma
        self.__emission_mat_dict = dict()
        self.__solver = None
        self.index_state_list = None
        self.gps_match_res_gdf = None
        # {(from_seq, to_seq): pd.DataFrame()}
        self.__s2s_route_l = dict()
        self.__plot_mix_gdf, self.__base_link_gdf, self.__base_node_gdf = None, None, None
        self.path_cost_df = pd.DataFrame()
        self.is_warn = False

    def generate_markov_para(self):

        # self.__generate_markov_para()
        self.__generate_transition_mat()
        self.__generate_emission_mat()

    @function_time_cost
    def __generate_transition_mat(self):

        # 依据一辆车的时序gps点和和底层路网生成转移概率矩阵和生成概率矩阵
        # seq, geometry, single_link_id, from_node, to_node, dir, length
        gps_candidate_link, _gap = self.gps_points.generate_candidate_link(net=self.net)

        if _gap:
            warnings.warn(rf'seq为: {_gap}的GPS点没有关联到任何候选路段..., 不会用于路径匹配计算...')

            # 删除关联不到任何路段的gps点
            self.gps_points.delete_target_gps(target_seq_list=list(_gap))

        # 一定要排序
        seq_list = sorted(list(gps_candidate_link[gps_field.POINT_SEQ_FIELD].unique()))

        if len(seq_list) <= 1:
            raise ValueError(r'GPS数据样本点不足2个, 请检查...')

        self.gps_points.calc_gps_point_dis()

        # 计算状态转移概率矩阵
        for i in range(0, len(seq_list) - 1):
            from_link = gps_candidate_link[gps_candidate_link[gps_field.POINT_SEQ_FIELD] == seq_list[i]][
                net_field.SINGLE_LINK_ID_FIELD].to_list()
            to_link = gps_candidate_link[gps_candidate_link[gps_field.POINT_SEQ_FIELD] == seq_list[i + 1]][
                net_field.SINGLE_LINK_ID_FIELD].to_list()

            transition_df = pd.DataFrame([[f, t] for f in from_link for t in to_link], columns=[markov_field.FROM_STATE,
                                                                                                markov_field.TO_STATE])

            transition_df[markov_field.ROUTE_LENGTH] = \
                transition_df.apply(
                    lambda item: self.calc_route_length(from_gps_seq=seq_list[i],
                                                        to_gps_seq=seq_list[i + 1],
                                                        from_link_id=item[markov_field.FROM_STATE],
                                                        to_link_id=item[markov_field.TO_STATE]), axis=1)

            transition_df[markov_field.DIS_GAP] = np.abs(-transition_df[
                markov_field.ROUTE_LENGTH] + self.gps_points.get_gps_point_dis((seq_list[i], seq_list[i + 1])))

            self.__s2s_route_l[(seq_list[i], seq_list[i + 1])] = transition_df[
                [markov_field.FROM_STATE, markov_field.TO_STATE, markov_field.ROUTE_LENGTH]].copy().set_index(
                [markov_field.FROM_STATE, markov_field.TO_STATE])

            # 转成matrix
            transition_mat = transition_df[
                [markov_field.FROM_STATE, markov_field.TO_STATE, markov_field.DIS_GAP]].set_index(
                [markov_field.FROM_STATE, markov_field.TO_STATE]).unstack().values

            # 索引映射
            f_mapping, t_mapping = {i: f for i, f in zip(range(len(from_link)), sorted(from_link))}, \
                {i: t for i, t in zip(range(len(to_link)), sorted(to_link))}
            transition_mat = self.transition_probability(self.beta, transition_mat)

            self.__ft_transition_dict[seq_list[i]] = transition_mat
            self.__ft_mapping_dict[seq_list[i]] = f_mapping
            self.__ft_mapping_dict[seq_list[i + 1]] = t_mapping

    @function_time_cost
    def __generate_emission_mat(self):

        # 计算每个观测点的生成概率
        emission_p_df = pd.DataFrame(self.__done_prj_dict).T.reset_index(drop=False).rename(
            columns={'level_0': gps_field.POINT_SEQ_FIELD, 'level_1': net_field.SINGLE_LINK_ID_FIELD,
                     1: markov_field.PRJ_L})[
            [gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD, markov_field.PRJ_L]]
        emission_p_df.sort_values(by=[gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD],
                                  ascending=[True, True], inplace=True)
        emission_p_df = emission_p_df.groupby([gps_field.POINT_SEQ_FIELD]).agg(
            {markov_field.PRJ_L: np.array}).reset_index(
            drop=False)

        self.__emission_mat_dict = {
            int(row[gps_field.POINT_SEQ_FIELD]): self.emission_probability(dis=row[markov_field.PRJ_L],
                                                                           sigma=self.gps_sigma) for _, row in
            emission_p_df.iterrows()}

    def solve(self, use_lop_p: bool = True):
        """

        :param use_lop_p: 是否使用对数概率, 避免浮点数精度下溢
        :return:
        """

        # 使用viterbi模型求解
        self.__solver = Viterbi(observation_list=self.gps_points.used_observation_seq_list,
                                o_mat_dict=self.__emission_mat_dict,
                                t_mat_dict=self.__ft_transition_dict, use_log_p=use_lop_p)
        self.__solver.init_model()
        self.index_state_list = self.__solver.iter_model()

        print(self.index_state_list)

    def calc_route_length(self, from_gps_seq: int = None, to_gps_seq: int = None, from_link_id: int = None,
                          to_link_id: int = None) -> float:
        """
        :param from_gps_seq: 上一观测时刻的gps点序号
        :param to_gps_seq: 当前观测时刻的gps点序号
        :param from_link_id: 上一观测时刻候选link_id
        :param to_link_id: 当前观测时刻候选link_id
        :return:
        """
        # prj_p, prj_dis, route_dis
        (from_prj_p, from_prj_dis, from_route_dis, from_l_length) = \
            self.cache_emission_data(gps_seq=from_gps_seq, single_link_id=from_link_id)

        (to_prj_p, to_prj_dis, to_route_dis, to_l_length) = \
            self.cache_emission_data(gps_seq=to_gps_seq, single_link_id=to_link_id)

        # if (from_gps_seq, from_link_id) in self.__done_prj_dict.keys():
        #     (from_prj_p, from_prj_dis, from_route_dis, from_l_length) = self.__done_prj_dict[
        #         (from_gps_seq, from_link_id)]
        # else:
        #     (from_prj_p, from_prj_dis, from_route_dis, from_l_length) = self.get_gps_prj_info(
        #         target_link_id=from_link_id,
        #         gps_seq=from_gps_seq)
        #     self.__done_prj_dict.update(
        #         {(from_gps_seq, from_link_id): (from_prj_p, from_prj_dis, from_route_dis, from_l_length)})
        # if (to_gps_seq, to_link_id) in self.__done_prj_dict.keys():
        #     (to_prj_p, to_prj_dis, to_route_dis, to_l_length) = self.__done_prj_dict[(to_gps_seq, to_link_id)]
        # else:
        #     (to_prj_p, to_prj_dis, to_route_dis, to_l_length) = self.get_gps_prj_info(target_link_id=to_link_id,
        #                                                                               gps_seq=to_gps_seq)
        #     self.__done_prj_dict.update({(to_gps_seq, to_link_id): (to_prj_p, to_prj_dis, to_route_dis, to_l_length)})

        # 基于投影信息计算路径长度
        from_link_ft, to_link_ft = self.net.get_link_from_to(from_link_id, _type='single'), \
            self.net.get_link_from_to(to_link_id, _type='single')

        # same link
        if from_link_id == to_link_id:
            route_l = np.absolute(from_route_dis - to_route_dis)
            return route_l

        # one same node
        dup_node_list = list(set(from_link_ft) & set(to_link_ft))
        if len(dup_node_list) == 1:
            dup_node = dup_node_list[0]
            if (dup_node == from_link_ft[1]) and (dup_node == to_link_ft[0]):
                route_l = from_l_length - from_route_dis + to_route_dis
                return np.absolute(route_l)
            else:
                return NOT_CONN_COST
        # 正好相反的f-t
        elif len(dup_node_list) == 2:
            route_l = from_l_length - from_route_dis + to_route_dis
            return np.absolute(route_l)

        route_item = self.net.search(o_node=from_link_ft[0], d_node=to_link_ft[0])
        if len(route_item[0]) > 2:
            self.__adj_seq_path_dict[(from_link_id, to_link_id)] = route_item[0]
        if route_item[0]:
            if route_item[0][1] != from_link_ft[1]:
                # abnormal
                # route_item_alpha = self.net.search(o_node=from_link_ft[1], d_node=to_link_ft[0],
                #                                    search_method=self.search_method)
                # if route_item_alpha[0]:
                #     route_l1 = route_item[1] + from_route_dis
                # else:
                #     return NOT_CONN_COST
                return NOT_CONN_COST
            else:
                route_l1 = route_item[1] - from_route_dis

            if route_item[0][-2] == to_link_ft[1]:
                # abnormal
                return NOT_CONN_COST
            else:
                route_l2 = to_route_dis

            route_l = np.absolute(route_l1 + route_l2)
            return route_l
        else:
            return NOT_CONN_COST

    def cache_emission_data(self, gps_seq: int = None, single_link_id: int = None) -> tuple[Point, float, float, float]:
        """
        :param gps_seq:
        :param single_link_id:
        :return:
        """
        if (gps_seq, single_link_id) in self.__done_prj_dict.keys():
            # already calculated
            (prj_p, prj_dis, route_dis, l_length) = self.__done_prj_dict[
                (gps_seq, single_link_id)]
        else:
            # new calc and cache
            (prj_p, prj_dis, route_dis, l_length) = self.get_gps_prj_info(
                target_link_id=single_link_id,
                gps_seq=gps_seq)
            self.__done_prj_dict.update(
                {(gps_seq, single_link_id): (prj_p, prj_dis, route_dis, l_length)})
        return prj_p, prj_dis, route_dis, l_length


    # @function_time_cost
    # def __generate_markov_para(self, use_swifter: bool = False):
    #     # 依据一辆车的时序gps点和和底层路网生成转移概率矩阵和生成概率矩阵
    #     # seq, geometry, single_link_id, from_node, to_node, dir, length
    #     gps_candidate_link, _gap = self.gps_points.generate_candidate_link(net=self.net)
    #
    #     # {seq1: {'single_link_id': [candidate_link]}, seq2: ...}
    #     gps_candidate_link_dict = gps_candidate_link.groupby([gps_field.POINT_SEQ_FIELD]).agg(
    #         {net_field.SINGLE_LINK_ID_FIELD: list}).to_dict(orient='index')
    #
    #     if _gap:
    #         warnings.warn(rf'seq为: {_gap}的GPS点没有关联到任何候选路段..., 不会用于路径匹配计算...')
    #
    #         # 删除关联不到任何路段的gps点
    #         self.gps_points.delete_target_gps(target_seq_list=list(_gap))
    #
    #     # 一定要排序
    #     seq_list = sorted(list(gps_candidate_link_dict.keys()))
    #
    #     if len(seq_list) <= 1:
    #         raise ValueError(r'GPS数据样本点不足2个, 请检查...')
    #
    #     self.gps_points.calc_gps_point_dis()
    #
    #     ft_gps_candidate = \
    #         {(seq_list[i], seq_list[i + 1]): [
    #             list(itertools.product(gps_candidate_link_dict[seq_list[i]][net_field.SINGLE_LINK_ID_FIELD],
    #                                    gps_candidate_link_dict[seq_list[i + 1]][net_field.SINGLE_LINK_ID_FIELD]))]
    #             for i in range(0, len(seq_list) - 1)}
    #
    #     seq_link_df = gps_candidate_link.groupby([gps_field.POINT_SEQ_FIELD]).agg(
    #         {net_field.SINGLE_LINK_ID_FIELD: list})
    #     self.__ft_mapping_dict = {
    #         seq: {i: link for i, link in enumerate(sorted(seq_link_df.at[seq, net_field.SINGLE_LINK_ID_FIELD]))} for seq
    #         in seq_link_df.index}
    #
    #     # print(self.__ft_mapping_dict)
    #     transit_df = pd.DataFrame(ft_gps_candidate).T.reset_index(drop=False).rename(
    #         columns={'level_0': from_gps_f, 'level_1': to_gps_f, 0: 'ft_link'})
    #
    #     transit_df = transit_df.explode(column='ft_link', ignore_index=True)
    #
    #     all_required_source_list = \
    #         gps_candidate_link[gps_candidate_link[gps_field.POINT_SEQ_FIELD].isin(seq_list[:-1])][
    #             net_field.FROM_NODE_FIELD].unique()
    #
    #     # 提前计算最短路信息
    #     path_cost_df = pd.DataFrame({'source': all_required_source_list})
    #     if not use_swifter:
    #         path_cost_df['path'] = path_cost_df.apply(lambda row:
    #                                                   self.net._single_source_path(g=self.net.graph,
    #                                                                                source=row['source'],
    #                                                                                method=self.net.search_method,
    #                                                                                weight_field=self.net.weight_field),
    #                                                   axis=1)
    #     else:
    #         path_cost_df['path'] = path_cost_df.swifter.apply(lambda row:
    #                                                            self.net._single_source_path(g=self.net.graph,
    #                                                                                         source=row['source'],
    #                                                                                         method=self.net.search_method,
    #                                                                                         weight_field=self.net.weight_field),
    #                                                            axis=1)
    #
    #     # 计算发射概率
    #
    #     print(path_cost_df)
    #     path_cost_df.set_index('source', inplace=True)
    #     self.path_cost_df = path_cost_df
    #
    #
    #     # dis of (gps, prj_gps)
    #     gps_candidate_link[['prj_p', markov_field.PRJ_L, 'route_dis', 'l_length']] = gps_candidate_link.apply(
    #         lambda row: self.get_gps_prj_info(target_link_id=row[net_field.SINGLE_LINK_ID_FIELD],
    #                                           gps_seq=row[gps_field.POINT_SEQ_FIELD]), axis=1, result_type='expand')
    #
    #     # 计算done_prj_dict
    #     self.__done_prj_dict = {(row[gps_field.POINT_SEQ_FIELD], row[net_field.SINGLE_LINK_ID_FIELD]):
    #                                 (row['prj_p'], row[markov_field.PRJ_L], row['route_dis'], row['l_length']) for
    #                             _, row in
    #                             gps_candidate_link.iterrows()}
    #
    #
    #     emission_p_df = gps_candidate_link[
    #         [gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD, markov_field.PRJ_L]].copy()
    #
    #     emission_p_df.sort_values(by=[gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD],
    #                               ascending=[True, True], inplace=True)
    #
    #     emission_p_df = \
    #         emission_p_df.groupby([gps_field.POINT_SEQ_FIELD])[markov_field.PRJ_L].agg(
    #             lambda x: list(x)).reset_index(drop=False)
    #     # emission_p_df[markov_field.PRJ_L] = emission_p_df[markov_field.PRJ_L].swifter.apply(lambda row: np.array(row))
    #     emission_p_df[markov_field.PRJ_L] = emission_p_df[markov_field.PRJ_L].apply(lambda row: np.array(row))
    #
    #     self.__emission_mat_dict = {
    #         int(row[gps_field.POINT_SEQ_FIELD]): self.emission_probability(dis=row[markov_field.PRJ_L],
    #                                                                        sigma=self.gps_sigma) for _, row in
    #         emission_p_df.iterrows()}
    #
    #     def ft_link_path_item(from_link, to_link):
    #         try:
    #             source_item = path_cost_df.at[from_link, 'path']
    #             return source_item[1][to_link], source_item[0][to_link]
    #         except KeyError:
    #             return [], NOT_CONN_COST
    #
    #     # 计算转移概率
    #     transit_df[markov_field.ROUTE_LENGTH] = \
    #         transit_df.apply(
    #             lambda item: self._calc_route_length(from_link_id=item['ft_link'][0],
    #                                                  to_link_id=item['ft_link'][1],
    #                                                  from_route_dis=self.__done_prj_dict[
    #                                                      (item[from_gps_f], item['ft_link'][0])][1],
    #                                                  to_route_dis=self.__done_prj_dict[
    #                                                      (item[to_gps_f], item['ft_link'][1])][1],
    #                                                  from_l_length=self.__done_prj_dict[
    #                                                      (item[from_gps_f], item['ft_link'][0])][3]
    #                                                  ), axis=1)
    #
    #
    #     # transit_df[markov_field.ROUTE_LENGTH] = \
    #     #     transit_df.swifter.apply(
    #     #         lambda item: self._calc_route_length(from_link_id=item['ft_link'][0],
    #     #                                              to_link_id=item['ft_link'][1],
    #     #                                              from_route_dis=self.__done_prj_dict[
    #     #                                                  (item[from_gps_f], item['ft_link'][0])][1],
    #     #                                              to_route_dis=self.__done_prj_dict[
    #     #                                                  (item[to_gps_f], item['ft_link'][1])][1],
    #     #                                              from_l_length=self.__done_prj_dict[
    #     #                                                  (item[from_gps_f], item['ft_link'][0])][3]
    #     #                                              ), axis=1)
    #     print(transit_df)
    #
    # def _calc_route_length(self, from_route_dis: float = None, to_route_dis: float = None,
    #                        from_l_length: float = None,
    #                        from_link_id: int = None, to_link_id: int = None) -> float:
    #     """
    #     :param from_route_dis: 上一观测时刻的gps点序号
    #     :param to_route_dis: 当前观测时刻的gps点序号
    #     :param from_link_id: 上一观测时刻候选link_id
    #     :param to_link_id: 当前观测时刻候选link_id
    #     :return:
    #     """
    #     # 基于投影信息计算路径长度
    #     from_link_ft, to_link_ft = self.net.get_link_from_to(from_link_id), self.net.get_link_from_to(to_link_id)
    #
    #     # same link
    #     if from_link_id == to_link_id:
    #         route_l = np.absolute(from_route_dis - to_route_dis)
    #         return route_l
    #
    #     # one same node
    #     dup_node_list = list(set(from_link_ft) & set(to_link_ft))
    #     if len(dup_node_list) == 1:
    #         dup_node = dup_node_list[0]
    #         if (dup_node == from_link_ft[1]) and (dup_node == to_link_ft[0]):
    #             route_l = from_l_length - from_route_dis + to_route_dis
    #             return np.absolute(route_l)
    #         else:
    #             return NOT_CONN_COST
    #     # 正好相反的f-t
    #     elif len(dup_node_list) == 2:
    #         route_l = from_l_length - from_route_dis + to_route_dis
    #         return np.absolute(route_l)
    #
    #     try:
    #         source_item = self.path_cost_df.at[from_link_ft[0], 'path']
    #         route_item = [source_item[1][to_link_ft[0]], source_item[0][to_link_ft[0]]]
    #     except KeyError:
    #         route_item = [], NOT_CONN_COST
    #
    #     if len(route_item[0]) > 2:
    #         self.__adj_seq_path_dict[(from_link_id, to_link_id)] = route_item[0]
    #     if route_item[0]:
    #         if route_item[0][1] != from_link_ft[1]:
    #             return NOT_CONN_COST
    #         else:
    #             route_l1 = route_item[1] - from_route_dis
    #
    #         if route_item[0][-2] == to_link_ft[1]:
    #             # abnormal
    #             return NOT_CONN_COST
    #         else:
    #             route_l2 = to_route_dis
    #
    #         route_l = np.absolute(route_l1 + route_l2)
    #         return route_l
    #     else:
    #         return NOT_CONN_COST
    #
    #
    def get_gps_prj_info(self, gps_seq: int = None, target_link_id: int = None) -> tuple[Point, float, float, float]:
        return self.gps_points.get_prj_inf(line=self.net.get_link_geo(target_link_id, _type='single'), seq=gps_seq)

    @staticmethod
    def transition_probability(beta: float = 30.2, dis_gap: float or np.ndarray = None):
        """
        dis_gap = straight_l - route_l
        :param beta:
        :param dis_gap:
        :return:
        """
        # p = (1 / beta) * np.e ** (- 0.1 * dis_gap / beta)
        p = np.e ** (- 0.1 * dis_gap / beta)
        return p

    @staticmethod
    def emission_probability(sigma: float = 1.0, dis: float = 6.0) -> float:
        # p = (1 / (sigma * (2 * np.pi) ** 0.5)) * (np.e ** (-0.5 * (0.1 * dis / sigma) ** 2))
        p = np.e ** (-0.5 * (0.1 * dis / sigma) ** 2)
        return p

    def acquire_res(self) -> gpd.GeoDataFrame():
        # 观测序列 -> (观测序列, single_link)
        single_link_state_list = [(observe_val, self.__ft_mapping_dict[observe_val][state_index]) for
                                  observe_val, state_index in
                                  zip(self.gps_points.used_observation_seq_list,
                                      self.index_state_list)]
        print(single_link_state_list)
        # 映射回原路网link_id, 以及dir
        # {[link_id, dir, from_node, to_node], [link_id, dir, from_node, to_node]...}
        bilateral_unidirectional_mapping = self.net.bilateral_unidirectional_mapping
        link_state_list = [item + bilateral_unidirectional_mapping[item[1]] for item in single_link_state_list]

        gps_link_state_df = pd.DataFrame(link_state_list, columns=[gps_field.POINT_SEQ_FIELD,
                                                                   net_field.SINGLE_LINK_ID_FIELD,
                                                                   net_field.LINK_ID_FIELD,
                                                                   net_field.DIRECTION_FIELD,
                                                                   net_field.FROM_NODE_FIELD,
                                                                   net_field.TO_NODE_FIELD])
        gps_link_state_df[gps_field.SUB_SEQ_FIELD] = 0

        gps_link_state_df[markov_field.PRJ_GEO] = \
            gps_link_state_df.apply(lambda item: self.__done_prj_dict[(item[gps_field.POINT_SEQ_FIELD],
                                                                       item[net_field.SINGLE_LINK_ID_FIELD])][0], axis=1)

        gps_link_state_df[['next_single', 'next_seq']] = gps_link_state_df[
            [net_field.SINGLE_LINK_ID_FIELD, gps_field.POINT_SEQ_FIELD]].shift(-1)
        gps_link_state_df.fillna(-99, inplace=True)
        gps_link_state_df[['next_single', 'next_seq']] = gps_link_state_df[['next_single', 'next_seq']].astype(int)
        gps_link_state_df[markov_field.DIS_TO_NEXT] = \
            gps_link_state_df.apply(
                lambda item: self.__s2s_route_l[(item[gps_field.POINT_SEQ_FIELD], item['next_seq'])].at[
                    (item[net_field.SINGLE_LINK_ID_FIELD],
                     item['next_single']), markov_field.ROUTE_LENGTH] if item['next_single'] != -99 else np.nan, axis=1)

        del link_state_list

        # agent_id, seq
        gps_match_res_gdf = self.gps_points.gps_gdf

        # 获取补全的路径
        # gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD, gps_field.SUB_SEQ_FIELD
        # net_field.LINK_ID_FIELD, net_field.DIRECTION_FIELD, net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD
        # gps_field.TIME_FIELD, net_field.GEOMETRY_FIELD
        omitted_gps_state_df = self.acquire_omitted_match_item(gps_link_state_df=gps_link_state_df)
        if not omitted_gps_state_df.empty:
            gps_link_state_df = pd.concat([gps_link_state_df, omitted_gps_state_df[[gps_field.POINT_SEQ_FIELD,
                                                                                    net_field.SINGLE_LINK_ID_FIELD,
                                                                                    net_field.LINK_ID_FIELD,
                                                                                    net_field.DIRECTION_FIELD,
                                                                                    net_field.FROM_NODE_FIELD,
                                                                                    net_field.TO_NODE_FIELD,
                                                                                    gps_field.SUB_SEQ_FIELD]]])
            gps_link_state_df.sort_values(by=[gps_field.POINT_SEQ_FIELD, gps_field.SUB_SEQ_FIELD],
                                          ascending=[True, True], inplace=True)
            gps_link_state_df.reset_index(inplace=True, drop=True)

        # 给每个gps点打上路网link标签, 存在GPS匹配不到路段的情况(比如buffer范围内无候选路段)
        gps_match_res_gdf = pd.merge(gps_match_res_gdf[[gps_field.POINT_SEQ_FIELD, gps_field.AGENT_ID_FIELD,
                                                        gps_field.LNG_FIELD, gps_field.LAT_FIELD, gps_field.TIME_FIELD,
                                                        gps_field.GEOMETRY_FIELD]],
                                     gps_link_state_df, on=gps_field.POINT_SEQ_FIELD, how='right')
        gps_match_res_gdf.loc[gps_match_res_gdf[gps_field.NEXT_LINK_FIELD].isna(), net_field.GEOMETRY_FIELD] = \
            omitted_gps_state_df[net_field.GEOMETRY_FIELD].to_list()
        gps_match_res_gdf.loc[gps_match_res_gdf[gps_field.NEXT_LINK_FIELD].isna(), gps_field.TIME_FIELD] = \
            omitted_gps_state_df[gps_field.TIME_FIELD].to_list()

        gps_match_res_gdf.drop(columns=[gps_field.NEXT_LINK_FIELD], axis=1, inplace=True)
        gps_match_res_gdf = gpd.GeoDataFrame(gps_match_res_gdf, geometry=net_field.GEOMETRY_FIELD,
                                             crs=self.gps_points.crs)
        self.gps_match_res_gdf = gps_match_res_gdf
        return gps_match_res_gdf

    @function_time_cost
    def acquire_omitted_match_item(self, gps_link_state_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        推算补全不完整的路径之间的path以及GPS点
        :param gps_link_state_df: 初步的匹配结果, 可能存在断掉的路径
        :return:
        """
        bilateral_unidirectional_mapping = self.net.bilateral_unidirectional_mapping
        gps_match_res_gdf = self.gps_points.gps_gdf

        # 找出断掉的路径
        gps_link_state_df.sort_values(by=gps_field.POINT_SEQ_FIELD, ascending=True, inplace=True)
        gps_link_state_df[gps_field.NEXT_LINK_FIELD] = gps_link_state_df[net_field.SINGLE_LINK_ID_FIELD].shift(-1)
        gps_link_state_df[gps_field.NEXT_LINK_FIELD] = gps_link_state_df[gps_field.NEXT_LINK_FIELD].fillna(-1)
        gps_link_state_df[gps_field.NEXT_LINK_FIELD] = gps_link_state_df[gps_field.NEXT_LINK_FIELD].astype(int)
        gps_link_state_df.reset_index(inplace=True, drop=True)

        ft_node_link_mapping = self.net.get_ft_node_link_mapping()
        omitted_gps_state_item = []
        omitted_gps_points = []
        omitted_gps_points_time = []
        used_observation_seq_list = self.gps_points.used_observation_seq_list
        for i, used_o in enumerate(used_observation_seq_list[:-1]):
            ft_state = (int(gps_link_state_df.at[i, net_field.SINGLE_LINK_ID_FIELD]),
                        int(gps_link_state_df.at[i, gps_field.NEXT_LINK_FIELD]))

            now_from_node, now_to_node = int(gps_link_state_df.at[i, net_field.FROM_NODE_FIELD]), \
                int(gps_link_state_df.at[i, net_field.TO_NODE_FIELD])

            next_from_node, next_to_node = int(gps_link_state_df.at[i + 1, net_field.FROM_NODE_FIELD]), \
                int(gps_link_state_df.at[i + 1, net_field.TO_NODE_FIELD])

            if ((now_from_node, now_to_node) == (next_from_node, next_to_node)) or now_to_node == next_from_node:
                pass
            else:
                if ft_state in self.__adj_seq_path_dict.keys():
                    pre_seq = int(gps_link_state_df.at[i, gps_field.POINT_SEQ_FIELD])
                    next_seq = int(gps_link_state_df.at[i + 1, gps_field.POINT_SEQ_FIELD])
                    node_seq = self.__adj_seq_path_dict[ft_state]
                    if node_seq[1] != now_to_node:
                        warnings.warn(rf'相邻link状态不连通...ft:{(now_from_node, now_to_node)} -> ft:{(next_from_node, next_to_node)}, 可能是GPS太稀疏或者路网本身不连通')
                        _single_link_list = [ft_node_link_mapping[(node_seq[i], node_seq[i + 1])] for i in
                                             range(0, len(node_seq) - 1)]
                    else:
                        _single_link_list = [ft_node_link_mapping[(node_seq[i], node_seq[i + 1])] for i in
                                             range(1, len(node_seq) - 1)]

                    omitted_gps_state_item += [
                        (pre_seq, _single_link, sub_seq) + bilateral_unidirectional_mapping[_single_link]
                        for _single_link, sub_seq in zip(_single_link_list,
                                                         range(1, len(_single_link_list) + 1))]

                    # 利用前后的GPS点信息来补全缺失的GPS点
                    pre_order_gps, next_order_gps = gps_match_res_gdf.at[pre_seq, net_field.GEOMETRY_FIELD], \
                        gps_match_res_gdf.at[next_seq, net_field.GEOMETRY_FIELD]
                    omitted_gps_points.extend(n_equal_points(len(_single_link_list) + 1, from_point=pre_order_gps,
                                                             to_point=next_order_gps, add_noise=True, noise_frac=0.3))

                    # 一条补出来的路生成一个GPS点
                    pre_seq_time, next_seq_time = gps_match_res_gdf.at[pre_seq, gps_field.TIME_FIELD], \
                        gps_match_res_gdf.at[next_seq, gps_field.TIME_FIELD]
                    dt = (next_seq_time - pre_seq_time).total_seconds() / (len(omitted_gps_points) + 1)
                    omitted_gps_points_time.extend(
                        [pre_seq_time + timedelta(seconds=dt * n) for n in range(1, len(_single_link_list) + 1)])
                else:
                    self.is_warn = True
                    warnings.warn(rf'相邻link状态不连通...ft:{(now_from_node, now_to_node)} -> ft:{(next_from_node, next_to_node)}, 可能是GPS太稀疏或者路网本身不连通')

        omitted_gps_state_df = pd.DataFrame(omitted_gps_state_item, columns=[gps_field.POINT_SEQ_FIELD,
                                                                             net_field.SINGLE_LINK_ID_FIELD,
                                                                             gps_field.SUB_SEQ_FIELD,
                                                                             net_field.LINK_ID_FIELD,
                                                                             net_field.DIRECTION_FIELD,
                                                                             net_field.FROM_NODE_FIELD,
                                                                             net_field.TO_NODE_FIELD])
        del omitted_gps_state_item

        omitted_gps_state_df[gps_field.TIME_FIELD] = omitted_gps_points_time
        omitted_gps_state_df[net_field.GEOMETRY_FIELD] = [Point(loc) for loc in omitted_gps_points]

        return omitted_gps_state_df

    def acquire_visualization_res(self, use_gps_source: bool = False) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """获取可视化结果"""
        if self.__plot_mix_gdf is None:
            single_link_gdf = self.net.get_link_data()
            node_gdf = self.net.get_node_data()
            net_crs = self.net.crs
            plain_crs = self.net.plane_crs
            is_geo_crs = self.net.is_geo_crs()
            double_link_geo = {l: geo for l, geo in zip(single_link_gdf[net_field.LINK_ID_FIELD],
                                                        single_link_gdf[net_field.GEOMETRY_FIELD])}

            if self.gps_match_res_gdf is None:
                self.acquire_res()

            # 匹配路段
            if use_gps_source:
                plot_gps_gdf = self.gps_points.source_gps[
                    [gps_field.POINT_SEQ_FIELD, gps_field.AGENT_ID_FIELD, gps_field.LNG_FIELD,
                     gps_field.LAT_FIELD, gps_field.TIME_FIELD, gps_field.GEOMETRY_FIELD]].copy()
            else:
                plot_gps_gdf = self.gps_match_res_gdf.copy()
                plot_gps_gdf = plot_gps_gdf[[gps_field.POINT_SEQ_FIELD, gps_field.AGENT_ID_FIELD, gps_field.LNG_FIELD,
                                             gps_field.LAT_FIELD, gps_field.TIME_FIELD,
                                             gps_field.GEOMETRY_FIELD]].copy()

            # GPS点转化为circle polygon
            plot_gps_gdf.drop(columns=[gps_field.LNG_FIELD, gps_field.LAT_FIELD], axis=1, inplace=True)
            if plot_gps_gdf.crs != plain_crs:
                plot_gps_gdf = plot_gps_gdf.to_crs(plain_crs)
            plot_gps_gdf[net_field.GEOMETRY_FIELD] = plot_gps_gdf[net_field.GEOMETRY_FIELD].apply(lambda p: p.buffer(3.0))
            plot_gps_gdf[gps_field.TYPE_FIELD] = 'gps'

            # 匹配路段GDF
            plot_match_link_gdf = self.gps_match_res_gdf.copy()
            plot_match_link_gdf.drop(columns=[markov_field.PRJ_GEO], axis=1, inplace=True)
            plot_match_link_gdf[gps_field.TYPE_FIELD] = 'link'
            plot_match_link_gdf[net_field.GEOMETRY_FIELD] = plot_match_link_gdf[net_field.LINK_ID_FIELD].apply(
                lambda x: double_link_geo[x])
            plot_match_link_gdf.crs = net_crs
            plot_match_link_gdf.drop_duplicates(subset=net_field.LINK_ID_FIELD, keep='first', inplace=True)
            plot_match_link_gdf.reset_index(drop=True, inplace=True)
            if is_geo_crs:
                plot_match_link_gdf = plot_match_link_gdf.to_crs(plain_crs)
            plot_match_link_gdf[net_field.GEOMETRY_FIELD] = plot_match_link_gdf[net_field.GEOMETRY_FIELD].apply(
                lambda l: l.buffer(5.0))
            plot_match_link_gdf.drop(columns=[gps_field.LNG_FIELD, gps_field.LAT_FIELD], axis=1, inplace=True)

            plot_gps_gdf = plot_gps_gdf.to_crs(self.net.geo_crs)
            plot_match_link_gdf = plot_match_link_gdf.to_crs(self.net.geo_crs)
            gps_link_gdf = pd.concat([plot_gps_gdf, plot_match_link_gdf])
            gps_link_gdf.reset_index(inplace=True, drop=True)

            # 路网底图
            origin_link_gdf = single_link_gdf.drop_duplicates(subset=[net_field.LINK_ID_FIELD], keep='first').copy()
            if is_geo_crs:
                origin_link_gdf = origin_link_gdf.to_crs(plain_crs)
                node_gdf = node_gdf.to_crs(plain_crs)
            origin_link_gdf[net_field.GEOMETRY_FIELD] = origin_link_gdf[net_field.GEOMETRY_FIELD].apply(
                lambda x: x.buffer(1.5))
            node_gdf[net_field.GEOMETRY_FIELD] = node_gdf[net_field.GEOMETRY_FIELD].apply(
                lambda x: x.buffer(1.5))

            origin_link_gdf = origin_link_gdf.to_crs(self.net.geo_crs)
            node_gdf = node_gdf.to_crs(self.net.geo_crs)

            self.__plot_mix_gdf, self.__base_link_gdf, self.__base_node_gdf = gps_link_gdf, origin_link_gdf, node_gdf
            return gps_link_gdf, origin_link_gdf, node_gdf

        else:
            return self.__plot_mix_gdf, self.__base_link_gdf, self.__base_node_gdf

    def acquire_geo_res(self, out_fldr: str = None, flag_name: str = 'flag'):
        """获取矢量结果文件, 可以在qgis中可视化"""
        if self.gps_match_res_gdf is None:
            self.acquire_res()

        # gps
        gps_layer = self.gps_match_res_gdf[[gps_field.POINT_SEQ_FIELD, gps_field.SUB_SEQ_FIELD,
                                            gps_field.GEOMETRY_FIELD]].copy()
        gps_layer = gps_layer.to_crs(self.net.geo_crs)

        # prj_point
        prj_p_layer = self.gps_match_res_gdf[
            [gps_field.POINT_SEQ_FIELD, gps_field.SUB_SEQ_FIELD, net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
             net_field.TO_NODE_FIELD, markov_field.DIS_TO_NEXT, markov_field.PRJ_GEO, gps_field.GEOMETRY_FIELD]].copy()
        prj_p_layer.dropna(subset=[markov_field.PRJ_GEO], inplace=True)

        prj_p_layer['__geo'] = prj_p_layer.apply(
            lambda item: LineString((item[gps_field.GEOMETRY_FIELD], item[markov_field.PRJ_GEO])), axis=1)

        # prj_line
        prj_l_layer = prj_p_layer[['__geo']].copy()
        prj_l_layer.rename(columns={'__geo': gps_field.GEOMETRY_FIELD}, inplace=True)
        prj_l_layer = gpd.GeoDataFrame(prj_l_layer, geometry=gps_field.GEOMETRY_FIELD, crs=prj_p_layer.crs)

        prj_p_layer.set_geometry(markov_field.PRJ_GEO, inplace=True, crs=prj_p_layer.crs)
        prj_p_layer.drop(columns=['__geo', gps_field.GEOMETRY_FIELD], axis=1, inplace=True)

        prj_l_layer = prj_l_layer.to_crs(self.net.geo_crs)
        prj_p_layer = prj_p_layer.to_crs(self.net.geo_crs)

        # match_link
        match_link_gdf = self.gps_match_res_gdf[[gps_field.POINT_SEQ_FIELD, gps_field.SUB_SEQ_FIELD,
                                                net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                                                net_field.TO_NODE_FIELD]].copy()
        match_link_gdf.drop_duplicates(subset=[net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                                               net_field.TO_NODE_FIELD], keep='first', inplace=True)
        match_link_gdf.reset_index(inplace=True, drop=True)
        match_link_gdf = pd.merge(match_link_gdf,
                                  self.net.get_link_data()[[net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                                                            net_field.TO_NODE_FIELD, net_field.GEOMETRY_FIELD]],
                                  on=[net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                                      net_field.TO_NODE_FIELD],
                                  how='left')
        match_link_gdf = gpd.GeoDataFrame(match_link_gdf, geometry=net_field.GEOMETRY_FIELD, crs=self.net.crs)

        match_link_gdf = match_link_gdf.to_crs(self.net.geo_crs)

        for gdf, name in zip([gps_layer, prj_p_layer, prj_l_layer, match_link_gdf],
                             ['gps', 'prj_p', 'prj_l', 'match_link']):
            gdf.to_file(os.path.join(out_fldr, '-'.join([flag_name, name]) + '.geojson'), driver='GeoJSON')


if __name__ == '__main__':
    pass
    # a = LineString([(0, 0), (0, 1)])
    # z = a.segmentize(1/3 + 0.1 * 1/ 3)
    # print(z)

    # x = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    # print(x)
    # x.loc[x['a'] >= 2, ['a', 'b']] = [[12, 11], [121,344]]
    # print(x)
    #
    # x = datetime.datetime.now()
    # time.sleep(2)
    # x1 = datetime.datetime.now()
    # print((x1 - x).total_seconds())




