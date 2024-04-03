# -- coding: utf-8 --
# @Time    : 2023/12/9 8:29
# @Author  : TangKai
# @Team    : ZheChengData

"""Markov Model Class"""

import time
import os.path
import datetime
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing
import geopandas as gpd
from ..map.Net import Net
from itertools import chain
from datetime import timedelta
from ..solver.Viterbi import Viterbi
from ..gps.LocGps import GpsPointsGdf
from ..tools.geo_process import prj_inf
from ..WrapsFunc import function_time_cost
from ..tools.group import cut_group_for_df, cut_group
from shapely.geometry import Point, LineString
from ..GlobalVal import NetField, GpsField, MarkovField
from ..tools.geo_process import n_equal_points, hmm_vector_angle


gps_field = GpsField()
net_field = NetField()
markov_field = MarkovField()

from_link_f, to_link_f = markov_field.FROM_STATE, markov_field.TO_STATE
from_link_n_f, to_link_n_f = markov_field.FROM_STATE_N, markov_field.TO_STATE_N
from_gps_f, to_gps_f = gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ
MIN_P = 1e-10


class HiddenMarkov(object):
    """隐马尔可夫模型类"""

    def __init__(self, net: Net, gps_points: GpsPointsGdf, beta: float = 30.1, gps_sigma: float = 20.0,
                 not_conn_cost: float = 999.0, use_heading_inf: bool = True, heading_para_array: np.ndarray = None,
                 dis_para: float = 0.1, top_k: int = 25, omitted_l: float = 6.0, multi_core: bool = False,
                 core_num: int = 1):
        self.gps_points = gps_points
        self.net = net
        # (gps_seq, single_link_id): (prj_p, prj_dis, route_dis)
        self.__done_prj_dict: dict[tuple[int, int]: tuple[Point, float, float, float, np.ndarray]] = dict()
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
        self.not_conn_cost = not_conn_cost
        self.use_heading_inf = use_heading_inf
        if heading_para_array is None:
            self.heading_para_array = np.array([1.0, 1.0, 1.0, 0.1, 0.00001, 0.00001, 0.00001, 0.000001, 0.000001])
        self.angle_slice = 180 / len(self.heading_para_array)
        self.dis_para = dis_para
        self.warn_info = list()
        self.top_k = top_k
        self.omitted_l = omitted_l
        self.multi_core = multi_core
        self.core_num = core_num if core_num <= os.cpu_count() else os.cpu_count()

    def generate_markov_para(self):

        # self.__generate_markov_para()
        if self.multi_core and self.core_num >= 1:
            self.__generate_transition_mat_alpha_multi()
        else:
            self.__generate_transition_mat()
        self.__generate_emission_mat()

    def __generate_prj_info(self):
        """计算GPS点到候选路段的投影信息"""
        pass

    @function_time_cost
    def __generate_transition_mat(self):

        # 依据一辆车的时序gps点和和底层路网生成转移概率矩阵和生成概率矩阵
        # seq, geometry, single_link_id, from_node, to_node, dir, length
        gps_candidate_link, _gap = self.gps_points.generate_candidate_link(net=self.net)
        if gps_candidate_link.empty:
            raise ValueError(r'GPS数据样本点无法关联到任何路段...')

        if _gap:
            warnings.warn(rf'seq为: {_gap}的GPS点没有关联到任何候选路段..., 不会用于路径匹配计算...')

            # 删除关联不到任何路段的gps点
            self.gps_points.delete_target_gps(target_seq_list=list(_gap))

        gps_candidate_link = self.filter_k_candidates(preliminary_candidate_link=gps_candidate_link, top_k=self.top_k)

        # 一定要排序
        seq_list = sorted(list(gps_candidate_link[gps_field.POINT_SEQ_FIELD].unique()))
        if len(seq_list) <= 1:
            raise ValueError(r'GPS数据样本点不足2个, 请检查...')

        self.gps_points.calc_gps_point_dis()
        # _ = pd.DataFrame()
        # 计算状态转移概率矩阵
        for i in range(0, len(seq_list) - 1):
            from_link = gps_candidate_link[gps_candidate_link[gps_field.POINT_SEQ_FIELD] == seq_list[i]][
                net_field.SINGLE_LINK_ID_FIELD].to_list()
            to_link = gps_candidate_link[gps_candidate_link[gps_field.POINT_SEQ_FIELD] == seq_list[i + 1]][
                net_field.SINGLE_LINK_ID_FIELD].to_list()

            transition_df = pd.DataFrame([[int(f), int(t)] for f in from_link for t in to_link],
                                         columns=[markov_field.FROM_STATE,
                                                  markov_field.TO_STATE])

            transition_df[markov_field.ROUTE_LENGTH] = \
                transition_df.apply(
                    lambda item: self.calc_route_length(from_gps_seq=seq_list[i],
                                                        to_gps_seq=seq_list[i + 1],
                                                        from_link_id=item[markov_field.FROM_STATE],
                                                        to_link_id=item[markov_field.TO_STATE]), axis=1)

            transition_df[markov_field.DIS_GAP] = np.abs(-transition_df[
                markov_field.ROUTE_LENGTH] + self.gps_points.get_gps_point_dis((seq_list[i], seq_list[i + 1])))

            # _ = pd.concat([_, transition_df])
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
            transition_mat = self.transition_probability(self.beta, transition_mat, self.dis_para)

            self.__ft_transition_dict[seq_list[i]] = transition_mat
            self.__ft_mapping_dict[seq_list[i]] = f_mapping
            self.__ft_mapping_dict[seq_list[i + 1]] = t_mapping

        # print(_)

    def filter_k_candidates(self, preliminary_candidate_link: gpd.GeoDataFrame or pd.DataFrame = None,
                            top_k: int = 10):
        """
        对Buffer范围内的初步候选路段进行二次筛选, 需按照投影距离排名前K位的
        :param preliminary_candidate_link:
        :param top_k
        :return:
        """

        preliminary_candidate_link['prj_info'] = preliminary_candidate_link.apply(
            lambda row: prj_inf(p=row[gps_field.GEOMETRY_FIELD], line=row['single_link_geo']), axis=1)
        preliminary_candidate_link['prj_dis'] = preliminary_candidate_link.apply(lambda row: row['prj_info'][1], axis=1)
        preliminary_candidate_link.sort_values(by=[gps_field.POINT_SEQ_FIELD, 'prj_dis'], ascending=[True, True],
                                               inplace=True)
        candidate_link = preliminary_candidate_link.groupby(gps_field.POINT_SEQ_FIELD).head(top_k)
        candidate_link.reset_index(inplace=True, drop=True)

        self.__done_prj_dict = {
            (gps_seq, single_link_id): (prj_info[0], prj_info[1], prj_info[2], prj_info[3], prj_info[5]) for
            gps_seq, single_link_id, prj_info in
            zip(candidate_link[gps_field.POINT_SEQ_FIELD],
                candidate_link[net_field.SINGLE_LINK_ID_FIELD],
                candidate_link['prj_info'])}
        return candidate_link

    @function_time_cost
    def __generate_emission_mat(self):

        # 计算每个观测点的生成概率, 这是在计算状态转移概率之后, 已经将关联不到的GPS点删除了
        # 这里的向量是候选路段的投影点的方向向量
        emission_p_df = pd.DataFrame(self.__done_prj_dict).T.reset_index(drop=False).rename(
            columns={'level_0': gps_field.POINT_SEQ_FIELD, 'level_1': net_field.SINGLE_LINK_ID_FIELD,
                     1: markov_field.PRJ_L, 4: net_field.LINK_VEC_FIELD})[
            [gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD, markov_field.PRJ_L, net_field.LINK_VEC_FIELD]]

        if self.use_heading_inf:
            self.gps_points.calc_diff_heading()
            emission_p_df = pd.merge(emission_p_df, self.gps_points.gps_gdf[[gps_field.POINT_SEQ_FIELD,
                                                                             gps_field.DIFF_VEC]], how='left',
                                     on=gps_field.POINT_SEQ_FIELD)
            emission_p_df[markov_field.HEADING_GAP] = \
                emission_p_df.apply(
                    lambda row: hmm_vector_angle(gps_diff_vec=row[gps_field.DIFF_VEC],
                                                 link_dir_vec=row[net_field.LINK_VEC_FIELD], omitted_l=self.omitted_l),
                    axis=1)
        else:
            emission_p_df[markov_field.HEADING_GAP] = 0
        emission_p_df[markov_field.HEADING_GAP] = emission_p_df[markov_field.HEADING_GAP].astype(object)
        emission_p_df[markov_field.PRJ_L] = emission_p_df[markov_field.PRJ_L].astype(object)
        emission_p_df.sort_values(by=[gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD],
                                  ascending=[True, True], inplace=True)
        emission_p_df = emission_p_df.groupby([gps_field.POINT_SEQ_FIELD]).agg(
            {markov_field.PRJ_L: np.array, markov_field.HEADING_GAP: np.array}).reset_index(
            drop=False)

        self.__emission_mat_dict = {
            int(row[gps_field.POINT_SEQ_FIELD]): self.emission_probability(dis=row[markov_field.PRJ_L],
                                                                           sigma=self.gps_sigma,
                                                                           heading_gap=row[markov_field.HEADING_GAP])
            for _, row in
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

    @function_time_cost
    def __generate_transition_mat_alpha_multi(self):
        n = self.core_num
        # 依据一辆车的时序gps点和和底层路网生成转移概率矩阵和生成概率矩阵
        # seq, geometry, single_link_id, from_node, to_node, dir, length
        gps_candidate_link, _gap = self.gps_points.generate_candidate_link(net=self.net)

        if gps_candidate_link.empty:
            raise ValueError(r'GPS数据样本点无法关联到任何路段...')

        if _gap:
            warnings.warn(rf'seq为: {_gap}的GPS点没有关联到任何候选路段..., 不会用于路径匹配计算...')

            # 删除关联不到任何路段的gps点
            self.gps_points.delete_target_gps(target_seq_list=list(_gap))

        gps_candidate_link = self.filter_k_candidates(preliminary_candidate_link=gps_candidate_link, top_k=self.top_k)

        # 一定要排序
        seq_list = sorted(list(gps_candidate_link[gps_field.POINT_SEQ_FIELD].unique()))

        if len(seq_list) <= 1:
            raise ValueError(r'GPS数据样本点不足2个, 请检查...')

        # 已经删除了关联不到任何路段的GPS点, 基于新的序列计算相邻GPS点距离
        # gps_field.POINT_SEQ_FIELD, gps_field.NEXT_SEQ, gps_field.ADJ_DIS
        gps_pre_next_dis_df = self.gps_points.calc_pre_next_dis()
        gps_pre_next_dis_df.rename(columns={gps_field.POINT_SEQ_FIELD: gps_field.FROM_GPS_SEQ,
                                            gps_field.NEXT_SEQ: gps_field.TO_GPS_SEQ}, inplace=True)
        # 计算每个seq点对应的candidate_link_list
        seq_candidate = \
            gps_candidate_link.groupby(gps_field.POINT_SEQ_FIELD).agg({net_field.SINGLE_LINK_ID_FIELD: list})
        del gps_candidate_link
        ft_list = [[i, i + 1, seq_list[i], seq_list[i + 1]] for i in range(0, len(seq_list) - 1)]
        ft_group = cut_group(obj_list=ft_list, n=n)
        single_link_gdf = self.net.get_link_data()
        single_link_gdf.reset_index(inplace=True, drop=True)
        single_link_ft_df = single_link_gdf[[net_field.SINGLE_LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                                             net_field.TO_NODE_FIELD]].copy()

        del single_link_gdf
        del ft_list

        pool = multiprocessing.Pool(processes=n)
        result_list = []
        g = self.net.graph
        # self.generate_transition_mat_alpha(ft_group[0], single_link_ft_df, seq_candidate, gps_pre_next_dis_df, g,
        #                                    self.__done_prj_dict,
        #                                    self.net.search_method, self.net.weight_field, self.net.cache_path,
        #                                    self.net.not_conn_cost)
        for i in range(0, len(ft_group)):
            result = pool.apply_async(self.generate_transition_mat_alpha,
                                      args=(ft_group[i], single_link_ft_df, seq_candidate,
                                            gps_pre_next_dis_df, g,
                                            self.__done_prj_dict,
                                            self.net.search_method, self.net.weight_field, self.net.cache_path,
                                            self.net.not_conn_cost))
            result_list.append(result)
        pool.close()
        pool.join()
        adj_seq_path_dict, ft_transition_dict, ft_mapping_dict, s2s_route_l = dict(), dict(), dict(), dict()
        for res in result_list:
            _adj_seq_path_dict, _ft_transition_dict, _ft_mapping_dict, _s2s_route_l = res.get()
            adj_seq_path_dict.update(_adj_seq_path_dict)
            ft_transition_dict.update(_ft_transition_dict)
            ft_mapping_dict.update(_ft_mapping_dict)
            s2s_route_l.update(_s2s_route_l)
        self.__adj_seq_path_dict = adj_seq_path_dict
        self.__ft_mapping_dict = ft_mapping_dict
        self.__ft_transition_dict = ft_transition_dict
        self.__s2s_route_l = s2s_route_l

    def generate_transition_mat_alpha(self, gps_ft_list: list = None, single_link_ft_df: pd.DataFrame = None,
                                      seq_candidate: pd.DataFrame = None,
                                      gps_pre_next_dis_df: pd.DataFrame = None,
                                      g: nx.DiGraph = None,
                                      prj_done_dict: dict = None,
                                      method: str = None, weight_field: str = 'length',
                                      cache_path: bool = True, not_conn_cost: float = 999.0):
        done_stp_cache, done_cost_cache, adj_seq_path_dict, s2s_route_l = dict(), dict(), dict(), dict()
        ft_transition_dict, ft_mapping_dict = dict(), dict()
        all_ft_state_list = list(chain(*[[[idx, next_idx, f_gps_seq, t_gps_seq, from_link, to_link]
                                          for from_link in seq_candidate.at[f_gps_seq, net_field.SINGLE_LINK_ID_FIELD]
                                          for to_link in seq_candidate.at[t_gps_seq, net_field.SINGLE_LINK_ID_FIELD]]
                                         for idx, next_idx, f_gps_seq, t_gps_seq in gps_ft_list]))

        del seq_candidate
        transition_df = pd.DataFrame(all_ft_state_list, columns=['idx', 'next_idx',
                                                                 gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ,
                                                                 markov_field.FROM_STATE, markov_field.TO_STATE])
        del all_ft_state_list
        transition_df = self.diy_merge(left_df=transition_df,
                                       right_df=single_link_ft_df,
                                       left_key=markov_field.FROM_STATE, right_key=net_field.SINGLE_LINK_ID_FIELD,
                                       label='from')
        transition_df = self.diy_merge(left_df=transition_df,
                                       right_df=single_link_ft_df,
                                       left_key=markov_field.TO_STATE, right_key=net_field.SINGLE_LINK_ID_FIELD,
                                       label='to')
        del single_link_ft_df

        transition_df[markov_field.ROUTE_LENGTH] = \
            transition_df.apply(
                lambda item: self.calc_route_length_alpha(from_gps_seq=item[gps_field.FROM_GPS_SEQ],
                                                          to_gps_seq=item[gps_field.TO_GPS_SEQ],
                                                          from_link_id=item[markov_field.FROM_STATE],
                                                          to_link_id=item[markov_field.TO_STATE],
                                                          from_link_ft=(item['from_link_f'], item['from_link_t']),
                                                          to_link_ft=(item['to_link_f'], item['to_link_t']),
                                                          di_g=g,
                                                          weight_field=weight_field,
                                                          method=method,
                                                          prj_done_dict=prj_done_dict,
                                                          done_stp_cache=done_stp_cache,
                                                          done_cost_cache=done_cost_cache,
                                                          cache_path=cache_path,
                                                          not_conn_cost=not_conn_cost,
                                                          adj_seq_path_dict=adj_seq_path_dict), axis=1)
        del done_cost_cache, done_stp_cache, g
        transition_df = pd.merge(transition_df, gps_pre_next_dis_df, on=[gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ],
                                 how='left')

        transition_df[markov_field.DIS_GAP] = np.abs(
            -transition_df[markov_field.ROUTE_LENGTH] + transition_df[gps_field.ADJ_DIS])

        for (idx, next_idx, f_gps_seq, t_gps_seq), df in transition_df.groupby(
                ['idx', 'next_idx', gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ]):
            s2s_route_l[(f_gps_seq, t_gps_seq)] = df[
                [markov_field.FROM_STATE, markov_field.TO_STATE, markov_field.ROUTE_LENGTH]].copy().set_index(
                [markov_field.FROM_STATE, markov_field.TO_STATE])

            # 转成matrix
            transition_mat = df[
                [markov_field.FROM_STATE, markov_field.TO_STATE, markov_field.DIS_GAP]].set_index(
                [markov_field.FROM_STATE, markov_field.TO_STATE]).unstack().values
            from_link, to_link = list(set(df[markov_field.FROM_STATE])), list(set(df[markov_field.TO_STATE]))
            # 索引映射
            f_mapping, t_mapping = {i: f for i, f in zip(range(len(from_link)), sorted(from_link))}, \
                {i: t for i, t in zip(range(len(to_link)), sorted(to_link))}
            transition_mat = self.transition_probability(self.beta, transition_mat, self.dis_para)

            ft_transition_dict[f_gps_seq] = transition_mat
            ft_mapping_dict[f_gps_seq] = f_mapping
            ft_mapping_dict[t_gps_seq] = t_mapping

        return adj_seq_path_dict, ft_transition_dict, ft_mapping_dict, s2s_route_l

    @staticmethod
    def diy_merge(left_df: pd.DataFrame, right_df: pd.DataFrame or gpd.GeoDataFrame = None, left_key: str = None,
                  right_key: str = None, label: str = 'from'):
        df = pd.merge(left_df, right_df, left_on=left_key, right_on=right_key, how='left')

        df.rename(columns={net_field.FROM_NODE_FIELD: label + '_link_f',
                           net_field.TO_NODE_FIELD: label + '_link_t'}, inplace=True)

        df.drop(columns=[right_key], axis=1, inplace=True)
        return df

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
        (from_prj_p, from_prj_dis, from_route_dis, from_l_length, from_p_vec) = \
            self.cache_emission_data(gps_seq=from_gps_seq, single_link_id=from_link_id)

        (to_prj_p, to_prj_dis, to_route_dis, to_l_length, to_p_vec) = \
            self.cache_emission_data(gps_seq=to_gps_seq, single_link_id=to_link_id)

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
                return self.not_conn_cost
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
                return self.not_conn_cost
            else:
                route_l1 = route_item[1] - from_route_dis

            if route_item[0][-2] == to_link_ft[1]:
                # abnormal
                return self.not_conn_cost
            else:
                route_l2 = to_route_dis

            route_l = np.absolute(route_l1 + route_l2)
            return route_l
        else:
            return self.not_conn_cost

    def calc_route_length_alpha(self, from_gps_seq: int = None, to_gps_seq: int = None,
                                from_link_id: int = None, to_link_id: int = None,
                                from_link_ft=None, to_link_ft=None,
                                done_stp_cache: dict = None, done_cost_cache: dict = None,
                                prj_done_dict: dict = None,
                                adj_seq_path_dict: dict = None,
                                di_g: nx.DiGraph = None,
                                method: str = None, weight_field: str = 'length',
                                cache_path: bool = True, not_conn_cost: float = 999.0) -> float:
        """"""
        from_prj_p, from_prj_dis, from_route_dis, from_l_length, from_p_vec = prj_done_dict[
            (from_gps_seq, from_link_id, )]
        to_prj_p, to_prj_dis, to_route_dis, to_l_length, to_p_vec = prj_done_dict[
            (to_gps_seq, to_link_id,)]

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
                return self.not_conn_cost
        # 正好相反的f-t
        elif len(dup_node_list) == 2:
            route_l = from_l_length - from_route_dis + to_route_dis
            return np.absolute(route_l)
        o_node,  d_node = from_link_ft[0], to_link_ft[0]
        route_item = self.get_od_cost_alpha(g=di_g, o_node=o_node, d_node=d_node,
                                            done_stp_cache=done_stp_cache, done_cost_cache=done_cost_cache,
                                            method=method,
                                            weight_field=weight_field, cache_path=cache_path,
                                            not_conn_cost=not_conn_cost)
        if len(route_item[0]) > 2:
            adj_seq_path_dict[(from_link_id, to_link_id)] = route_item[0]
        if route_item[0]:
            if route_item[0][1] != from_link_ft[1]:
                return not_conn_cost
            else:
                route_l1 = route_item[1] - from_route_dis

            if route_item[0][-2] == to_link_ft[1]:
                # abnormal
                return not_conn_cost
            else:
                route_l2 = to_route_dis

            route_l = np.absolute(route_l1 + route_l2)
            return route_l
        else:
            return not_conn_cost

    def get_od_cost_alpha(self, g: nx.DiGraph = None, o_node: int = None, d_node: int = None,
                          cache_path: bool = True, done_stp_cache: dict = None,
                          done_cost_cache: dict = None, method: str = None,
                          weight_field: str = 'length', not_conn_cost: float = 999.0) -> tuple[list, float]:
        """"""

        if o_node in done_stp_cache.keys():
            try:
                node_path = done_stp_cache[o_node][d_node]
                cost = done_cost_cache[o_node][d_node]
            except KeyError:
                return [], not_conn_cost
        else:
            self.calc_shortest_path_alpha(g=g, source=o_node, method=method, done_cost_cache=done_cost_cache,
                                          done_stp_cache=done_stp_cache, weight_field=weight_field)
            try:
                node_path = done_stp_cache[o_node][d_node]
                cost = done_cost_cache[o_node][d_node]
                if not cache_path:
                    del done_stp_cache[o_node]
                    del done_cost_cache[o_node]
            except KeyError:
                return [], not_conn_cost

        return node_path, cost

    def calc_shortest_path_alpha(self, g: nx.DiGraph = None, source: int = None, done_stp_cache: dict = None,
                                 done_cost_cache: dict = None, method: str = None,
                                 weight_field: str = 'length'):
        """

        :param g:
        :param source:
        :param done_stp_cache:
        :param done_cost_cache:
        :param method:
        :param weight_field:
        :return:
        """
        try:
            done_cost_cache[source], done_stp_cache[source] = self._single_source_path_alpha(
                g, source=source,
                method=method, weight_field=weight_field)
        except nx.NetworkXNoPath:
            pass

    @staticmethod
    def _single_source_path_alpha(g: nx.DiGraph = None, source: int = None, method: str = 'dijkstra',
                                  weight_field: str = None) -> tuple[dict[int, int], dict[int, list]]:
        if method == 'dijkstra':
            return nx.single_source_dijkstra(g, source, weight=weight_field)
        else:
            return nx.single_source_bellman_ford(g, source, weight=weight_field)


    def cache_emission_data_alpha(self, gps_seq: int = None, single_link_id: int = None, done_prj_dict: dict = None,
                                  target_link_geo: LineString = None, gps_geo: Point = None) -> \
            tuple[Point, float, float, float, np.ndarray]:
        """
        :param gps_seq:
        :param single_link_id:
        :param target_link_geo
        :param gps_geo
        :param done_prj_dict
        :return:
        """
        if (gps_seq, single_link_id) in done_prj_dict.keys():
            # already calculated
            (prj_p, prj_dis, route_dis, l_length, p_vec) = done_prj_dict[
                (gps_seq, single_link_id)]
        else:
            # new calc and cache
            (prj_p, prj_dis, route_dis, l_length, p_vec) = self.get_gps_prj_info_alpha(target_link_geo=target_link_geo,
                                                                                       gps_geo=gps_geo)
            done_prj_dict.update(
                {(gps_seq, single_link_id): (prj_p, prj_dis, route_dis, l_length, p_vec)})
        return prj_p, prj_dis, route_dis, l_length, p_vec

    def cache_emission_data(self, gps_seq: int = None, single_link_id: int = None) -> \
            tuple[Point, float, float, float, np.ndarray]:
        """
        :param gps_seq:
        :param single_link_id:
        :return:
        """
        if (gps_seq, single_link_id) in self.__done_prj_dict.keys():
            # already calculated
            (prj_p, prj_dis, route_dis, l_length, p_vec) = self.__done_prj_dict[
                (gps_seq, single_link_id)]
        else:
            # new calc and cache
            print('# new calc and cache')
            (prj_p, prj_dis, route_dis, l_length, p_vec) = self.get_gps_prj_info(
                target_link_id=single_link_id,
                gps_seq=gps_seq)
            self.__done_prj_dict.update(
                {(gps_seq, single_link_id): (prj_p, prj_dis, route_dis, l_length, p_vec)})
        return prj_p, prj_dis, route_dis, l_length, p_vec

    def get_gps_prj_info(self, gps_seq: int = None, target_link_id: int = None) -> \
            tuple[Point, float, float, float, np.ndarray]:
        return self.gps_points.get_prj_inf(line=self.net.get_link_geo(target_link_id, _type='single'), seq=gps_seq)

    def get_gps_prj_info_alpha(self, gps_geo: Point = None, target_link_geo: LineString = None) -> \
            tuple[Point, float, float, float, np.ndarray]:
        (prj_p, prj_dis, route_dis, l_length, _, p_vec) = prj_inf(gps_geo, target_link_geo)
        return prj_p, prj_dis, route_dis, l_length, p_vec

    @staticmethod
    def transition_probability(beta: float = 30.2, dis_gap: float or np.ndarray = None, dis_para: float = 0.1):
        """
        dis_gap = straight_l - route_l
        :param beta:
        :param dis_gap:
        :param dis_para
        :return:
        """
        # p = (1 / beta) * np.e ** (- 0.1 * dis_gap / beta)
        p = np.e ** (- dis_para * dis_gap / beta)
        return p

    def emission_probability(self, sigma: float = 1.0, dis: np.ndarray = 6.0, heading_gap: np.ndarray = None) -> float:
        # p = (1 / (sigma * (2 * np.pi) ** 0.5)) * (np.e ** (-0.5 * (0.1 * dis / sigma) ** 2))
        # print(heading_gap)
        heading_gap = self.heading_para_array[(heading_gap / self.angle_slice).astype(int)]
        # print(heading_gap)
        p = heading_gap * np.e ** (-0.5 * (self.dis_para * dis / sigma) ** 2)
        return p

    def acquire_res(self) -> gpd.GeoDataFrame():
        # 观测序列 -> (观测序列, single_link)
        single_link_state_list = [(observe_val, self.__ft_mapping_dict[observe_val][state_index]) for
                                  observe_val, state_index in
                                  zip(self.gps_points.used_observation_seq_list,
                                      self.index_state_list)]
        # print(single_link_state_list)
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
                                                                       item[net_field.SINGLE_LINK_ID_FIELD])][0],
                                    axis=1)

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
        self.gps_match_res_gdf = gps_match_res_gdf.to_crs(self.gps_points.geo_crs)
        return self.gps_match_res_gdf

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
                        self.warn_info.append([(now_from_node, now_to_node), (next_from_node, next_to_node)])
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
                    self.warn_info.append([(now_from_node, now_to_node), (next_from_node, next_to_node)])

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
            plain_crs = self.net.planar_crs
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
        out_fldr = r'./' if out_fldr is None else out_fldr
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
        prj_p_layer.set_geometry(markov_field.PRJ_GEO, inplace=True, crs=self.gps_points.plane_crs)
        prj_p_layer = prj_p_layer.to_crs(self.gps_points.geo_crs)

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




