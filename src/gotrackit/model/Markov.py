# -- coding: utf-8 --
# @Time    : 2023/12/9 8:29
# @Author  : TangKai
# @Team    : ZheChengData

"""Markov Model Class"""

import os.path
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from ..map.Net import Net
from datetime import timedelta
from ..solver.Viterbi import Viterbi
from ..gps.LocGps import GpsPointsGdf
from ..tools.geo_process import prj_inf
from ..WrapsFunc import function_time_cost
from shapely.geometry import Point, LineString
from ..GlobalVal import NetField, GpsField, MarkovField
from ..tools.geo_process import n_equal_points, hmm_vector_angle


gps_field = GpsField()
net_field = NetField()
markov_field = MarkovField()

from_link_f, to_link_f = markov_field.FROM_STATE, markov_field.TO_STATE
from_link_n_f, to_link_n_f = markov_field.FROM_STATE_N, markov_field.TO_STATE_N
from_gps_f, to_gps_f = gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ
path_field = net_field.NODE_PATH_FIELD
cost_field = net_field.COST_FIELD
o_node_field, d_node_field = net_field.S_NODE, net_field.T_NODE
MIN_P = 1e-10


class HiddenMarkov(object):
    """隐马尔可夫模型类"""

    def __init__(self, net: Net, gps_points: GpsPointsGdf, beta: float = 30.1, gps_sigma: float = 20.0,
                 not_conn_cost: float = 999.0, use_heading_inf: bool = True, heading_para_array: np.ndarray = None,
                 dis_para: float = 0.1, top_k: int = 25, omitted_l: float = 6.0):
        self.gps_points = gps_points
        self.net = net
        # (gps_seq, single_link_id): (prj_p, prj_dis, route_dis)
        self.__done_prj_dict: dict[tuple[int, int]: tuple[Point, float, float, float, np.ndarray]] = dict()
        self.__done_prj_df: gpd.GeoDataFrame or pd.DataFrame = None
        self.__adj_seq_path_dict: dict[tuple[int, int], list[int, int]] = dict()
        self.__ft_transition_dict = dict()
        self.__ft_mapping_dict = dict()
        self.__ft_idx_map = pd.DataFrame()
        self.beta = beta
        self.gps_sigma = gps_sigma
        self.__emission_mat_dict = dict()
        self.__solver = None
        self.index_state_list = None
        self.gps_match_res_gdf = None
        # {(from_seq, to_seq): pd.DataFrame()}
        self.__s2s_route_l = pd.DataFrame()
        self.__plot_mix_gdf, self.__base_link_gdf, self.__base_node_gdf, self.__may_error = None, None, None, None
        self.path_cost_df = pd.DataFrame()
        self.is_warn = False
        self.not_conn_cost = not_conn_cost
        self.use_heading_inf = use_heading_inf
        if heading_para_array is None:
            self.heading_para_array = np.array([1.0, 1.0, 1.0, 0.1, 0.0001, 0.0001, 0.00001, 0.000001, 0.000001])
        else:
            self.heading_para_array = heading_para_array
        self.angle_slice = 180 / len(self.heading_para_array)
        self.dis_para = dis_para
        self.warn_info = {'from_ft': [], 'to_ft': []}
        self.format_warn_info = pd.DataFrame()
        self.top_k = top_k
        self.omitted_l = omitted_l
        self.gps_candidate_link = None

    @function_time_cost
    def generate_markov_para(self, add_single_ft: list[bool] = None) -> bool:

        is_success = self.__generate_transition_mat_alpha_multi(add_single_ft)
        if is_success:
            self.__generate_emission_mat()
        return is_success

    def __generate_emission_mat(self):

        # 计算每个观测点的生成概率, 这是在计算状态转移概率之后, 已经将关联不到的GPS点删除了
        # 这里的向量是候选路段的投影点的方向向量
        emission_p_df = self.__done_prj_df.copy()
        emission_p_df.rename(columns={'p_vec': net_field.LINK_VEC_FIELD}, inplace=True)
        if self.use_heading_inf:
            self.gps_points.calc_diff_heading()
            emission_p_df = pd.merge(emission_p_df, self.gps_points.gps_gdf[[gps_field.POINT_SEQ_FIELD,
                                                                             gps_field.DIFF_VEC]], how='left',
                                     on=gps_field.POINT_SEQ_FIELD)
            emission_p_df[markov_field.HEADING_GAP] = \
                [hmm_vector_angle(gps_diff_vec=diff_vec,
                                  link_dir_vec=link_vec,
                                  omitted_l=self.omitted_l) for diff_vec, link_vec in
                 zip(emission_p_df[gps_field.DIFF_VEC], emission_p_df[net_field.LINK_VEC_FIELD])]
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

    def __generate_candidates(self) -> list[int]:
        # 初步依据gps点buffer得到候选路段, 关联不到的GPS点删除掉
        # seq, geometry, single_link_id, from_node, to_node, dir, length
        gps_candidate_link, _gap = self.gps_points.generate_candidate_link(net=self.net)

        if gps_candidate_link.empty:
            print(r'GPS数据样本点无法关联到任何路段, 请检查路网完整性或者增加gps_buffer参数')
            return list()

        if _gap:
            warnings.warn(rf'seq为: {_gap}的GPS点没有关联到任何候选路段..., 不会用于路径匹配计算...')
            # 删除关联不到任何路段的gps点
            self.gps_points.delete_target_gps(target_seq_list=list(_gap))

        # 一定要排序
        seq_list = sorted(list(gps_candidate_link[gps_field.POINT_SEQ_FIELD].unique()))
        if len(seq_list) <= 1:
            print(r'经过路段关联, 删除掉无法关联到路段的GPS点后, GPS数据样本点不足2个, 请检查路网完整性或者增加gps_buffer参数')
            return list()
        self.gps_candidate_link = gps_candidate_link
        return seq_list

    def filter_k_candidates(self, preliminary_candidate_link: gpd.GeoDataFrame or pd.DataFrame = None,
                            top_k: int = 10, using_cache: bool = True,
                            cache_prj_inf: dict = None) -> gpd.GeoDataFrame or pd.DataFrame:
        """
        对Buffer范围内的初步候选路段进行二次筛选, 需按照投影距离排名前K位, 并且得到计算发射概率需要的数据
        :param preliminary_candidate_link:
        :param top_k
        :param using_cache
        :param cache_prj_inf
        :return:
        """
        preliminary_candidate_link.reset_index(inplace=True, drop=True)
        preliminary_candidate_link['quick_stl'] = preliminary_candidate_link[gps_field.GEOMETRY_FIELD].shortest_line(
            preliminary_candidate_link['single_link_geo']).length
        preliminary_candidate_link.sort_values(by=[gps_field.POINT_SEQ_FIELD, 'quick_stl'], ascending=[True, True],
                                               inplace=True)
        preliminary_candidate_link = preliminary_candidate_link.groupby(gps_field.POINT_SEQ_FIELD).head(
            top_k).reset_index(drop=True)
        if not using_cache:
            prj_info_list = [prj_inf(p=geo, line=single_link_geo) for geo, single_link_geo in
                             zip(preliminary_candidate_link[gps_field.GEOMETRY_FIELD],
                                 preliminary_candidate_link['single_link_geo'])]
            prj_df = pd.DataFrame(prj_info_list,
                                  columns=['prj_p', 'prj_dis', 'route_dis', 'l_length', 'split_line', 'p_vec'])
            del prj_df['split_line']
            del preliminary_candidate_link['quick_stl']
            preliminary_candidate_link = pd.merge(preliminary_candidate_link, prj_df, left_index=True, right_index=True)
            k_candidate_link = preliminary_candidate_link
            del prj_df
            return k_candidate_link
        else:
            return self.filter_k_candidates_by_cache(preliminary_candidate_link=preliminary_candidate_link,
                                                     cache_prj_inf=cache_prj_inf)

    @staticmethod
    def filter_k_candidates_by_cache(preliminary_candidate_link: gpd.GeoDataFrame or pd.DataFrame = None,
                                     cache_prj_inf: dict = None) -> pd.DataFrame:
        del preliminary_candidate_link[net_field.LENGTH_FIELD]
        preliminary_candidate_link['route_dis'] = preliminary_candidate_link['single_link_geo'].project(
            preliminary_candidate_link[gps_field.GEOMETRY_FIELD])
        preliminary_candidate_link['prj_p'] = preliminary_candidate_link['single_link_geo'].interpolate(
            preliminary_candidate_link['route_dis'])
        if 1 in cache_prj_inf.keys():
            cache_prj_gdf_a = cache_prj_inf[1]
            preliminary_candidate_link = pd.merge(preliminary_candidate_link, cache_prj_gdf_a, how='left',
                                                  on=[net_field.FROM_NODE_FIELD,
                                                      net_field.TO_NODE_FIELD])
            a_info = preliminary_candidate_link[~preliminary_candidate_link[net_field.SEG_ACCU_LENGTH].isna()].copy()
            preliminary_candidate_link = preliminary_candidate_link[
                preliminary_candidate_link[net_field.SEG_ACCU_LENGTH].isna()].copy()
            preliminary_candidate_link.drop(columns=[net_field.SEG_ACCU_LENGTH, 'topo_seq', net_field.LENGTH_FIELD,
                                                     'dir_vec', net_field.SEG_COUNT], axis=1, inplace=True)
            del a_info[net_field.SEG_ACCU_LENGTH], a_info[net_field.SEG_COUNT], a_info['topo_seq']
            if preliminary_candidate_link.empty:
                a_info.rename(columns={'quick_stl': 'prj_dis', 'dir_vec': 'p_vec'}, inplace=True)
                return a_info
        else:
            a_info = pd.DataFrame()

        if 2 in cache_prj_inf.keys():
            cache_prj_gdf_b = cache_prj_inf[2]
            b_info = pd.merge(preliminary_candidate_link, cache_prj_gdf_b, how='inner', on=[net_field.FROM_NODE_FIELD,
                                                                                            net_field.TO_NODE_FIELD])
            b_info['ratio'] = b_info['route_dis'] / b_info[net_field.SEG_ACCU_LENGTH]

            c_info = b_info[b_info['ratio'] >= 1].copy()
            if not c_info.empty:
                c_info.sort_values(by=[gps_field.POINT_SEQ_FIELD, net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD,
                                       'topo_seq'], ascending=True)
                c_info = c_info.groupby(
                    [gps_field.POINT_SEQ_FIELD, net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD]).tail(n=1)
                del c_info['topo_seq'], c_info['ratio'], c_info[net_field.SEG_ACCU_LENGTH]
            d_info = b_info[b_info['ratio'] < 1].copy()
            if not d_info.empty:
                d_info['ck'] = \
                    d_info.groupby([gps_field.POINT_SEQ_FIELD, net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD])[
                    [net_field.FROM_NODE_FIELD]].transform('count')
                d_info = d_info[d_info['ck'] == d_info[net_field.SEG_COUNT]].copy()

                d_info = d_info.groupby(
                    [gps_field.POINT_SEQ_FIELD, net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD]).head(n=1)
                del d_info['ck'], d_info['topo_seq'], d_info['ratio'], d_info[net_field.SEG_ACCU_LENGTH]

            preliminary_candidate_link = pd.concat([a_info, c_info, d_info])
            preliminary_candidate_link.reset_index(inplace=True, drop=True)
            preliminary_candidate_link.rename(columns={'quick_stl': 'prj_dis', 'dir_vec': 'p_vec'}, inplace=True)
            return preliminary_candidate_link
        else:
            return a_info

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

    def __generate_transition_mat_alpha_multi(self, add_single_ft: list[bool] = None) -> bool:
        # 计算 初步候选, 经过这一步, 实际匹配用到的GPS点已经完全确定
        # 得到self.gps_candidate_link
        seq_list = self.__generate_candidates()
        if not seq_list:
            return False

        # 已经删除了关联不到任何路段的GPS点, 基于新的序列计算相邻GPS点距离
        # gps_field.POINT_SEQ_FIELD, gps_field.NEXT_SEQ, gps_field.ADJ_DIS
        gps_pre_next_dis_df = self.gps_points.calc_pre_next_dis()
        gps_pre_next_dis_df.rename(columns={gps_field.POINT_SEQ_FIELD: gps_field.FROM_GPS_SEQ,
                                            gps_field.NEXT_SEQ: gps_field.TO_GPS_SEQ}, inplace=True)

        # 抽取单向信息
        single_link_ft_df = self.net.single_link_ft_cost()
        single_link_ft_df.reset_index(inplace=True, drop=True)
        g = self.net.graph
        is_sub_net = self.net.is_sub_net
        fmm_cache = self.net.fmm_cache
        cut_off, max_cut_off = self.net.cut_off, self.net.max_cut_off
        # print(cut_off)
        if is_sub_net:
            if fmm_cache:
                done_stp_cost_df = self.net.get_path_cache()
                cache_prj_info = self.net.get_prj_cache()
            else:
                done_stp_cost_df = pd.DataFrame()
                cache_prj_info = dict()
        else:
            done_stp_cost_df = self.net.get_path_cache()
            cache_prj_info = self.net.get_prj_cache()

        adj_seq_path_dict, ft_transition_dict, ft_idx_map, s2s_route_l, prj_done_df, \
            done_stp_cost_df = \
            self.generate_transition_mat_beta(single_link_ft_df, self.gps_candidate_link,
                                              gps_pre_next_dis_df, g, self.net.search_method,
                                              self.net.weight_field, self.net.cache_path, self.net.not_conn_cost,
                                              done_stp_cost_df, is_sub_net, fmm_cache,
                                              cut_off, max_cut_off, cache_prj_info, add_single_ft)
        # print(len(done_stp_cost_df))
        ft_idx_map.reset_index(inplace=True, drop=True)
        s2s_route_l.reset_index(inplace=True, drop=True)
        self.__adj_seq_path_dict = adj_seq_path_dict
        self.__ft_idx_map = ft_idx_map
        self.__ft_transition_dict = ft_transition_dict
        self.__s2s_route_l = s2s_route_l
        self.__done_prj_df = prj_done_df

        if not is_sub_net and not fmm_cache:
            self.net.set_path_cache(stp_cost_df=done_stp_cost_df)
        return True

    # @function_time_cost
    def generate_transition_mat_beta(self, single_link_ft_df: pd.DataFrame = None,
                                     pre_seq_candidate: pd.DataFrame = None,
                                     gps_pre_next_dis_df: pd.DataFrame = None,
                                     g: nx.DiGraph = None,
                                     method: str = None, weight_field: str = 'length',
                                     cache_path: bool = True, not_conn_cost: float = 999.0,
                                     done_stp_cost_df: pd.DataFrame = None,
                                     is_sub_net: bool = True, fmm_cache: bool = False, cut_off: float = 600.0,
                                     max_cut_off: float = 2000.0, cache_prj_inf: dict = None,
                                     add_single_ft: list[bool] = None) -> \
            tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # K候选
        seq_k_candidate_info = \
            self.filter_k_candidates(preliminary_candidate_link=pre_seq_candidate, using_cache=fmm_cache,
                                     top_k=self.top_k, cache_prj_inf=cache_prj_inf)
        # print(rf'{len(seq_k_candidate_info)}个候选路段...')
        seq_k_candidate_info['idx'] = seq_k_candidate_info.groupby(gps_field.POINT_SEQ_FIELD)[
                                          net_field.SINGLE_LINK_ID_FIELD].rank(method='min').astype(int) - 1

        ft_idx_map = seq_k_candidate_info[[gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD, 'idx']].copy()
        ft_idx_map.sort_values(by=[gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD], inplace=True)
        del seq_k_candidate_info['idx']
        del pre_seq_candidate
        seq_k_candidate = seq_k_candidate_info.groupby(gps_field.POINT_SEQ_FIELD).agg(
            {net_field.SINGLE_LINK_ID_FIELD: list, gps_field.POINT_SEQ_FIELD: list}).rename(
            columns={gps_field.POINT_SEQ_FIELD: 'g_s'})

        gps_match_link_route_dis = seq_k_candidate_info[
            [gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD, 'route_dis']].copy()
        seq_k_candidate.rename(columns={net_field.SINGLE_LINK_ID_FIELD: markov_field.FROM_STATE,
                                        'g_s': gps_field.FROM_GPS_SEQ}, inplace=True)

        seq_k_candidate[markov_field.TO_STATE] = seq_k_candidate[markov_field.FROM_STATE].shift(-1)
        seq_k_candidate[gps_field.TO_GPS_SEQ] = seq_k_candidate[gps_field.FROM_GPS_SEQ].shift(-1)

        seq_k_candidate.dropna(subset=[markov_field.TO_STATE], inplace=True)

        from_state = seq_k_candidate[[markov_field.FROM_STATE, gps_field.FROM_GPS_SEQ]].reset_index(drop=False).rename(
            columns={gps_field.POINT_SEQ_FIELD: 'g'}).explode(
            column=[markov_field.FROM_STATE, gps_field.FROM_GPS_SEQ], ignore_index=True)

        to_state = seq_k_candidate[
            [markov_field.TO_STATE, gps_field.TO_GPS_SEQ]].reset_index(drop=False).rename(
            columns={gps_field.POINT_SEQ_FIELD: 'g'}).explode(column=[markov_field.TO_STATE, gps_field.TO_GPS_SEQ],
                                                              ignore_index=True)

        transition_df = pd.merge(from_state, to_state, on='g', how='outer')
        transition_df.reset_index(inplace=True, drop=True)
        # print(rf'{len(transition_df)}次状态转移...')
        transition_df = self.diy_merge(left_df=transition_df,
                                       right_df=single_link_ft_df[
                                           [net_field.SINGLE_LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                                            net_field.TO_NODE_FIELD]],
                                       left_key=markov_field.FROM_STATE, right_key=net_field.SINGLE_LINK_ID_FIELD,
                                       label='from')
        transition_df = self.diy_merge(left_df=transition_df,
                                       right_df=single_link_ft_df[
                                           [net_field.SINGLE_LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
                                            net_field.TO_NODE_FIELD]],
                                       left_key=markov_field.TO_STATE, right_key=net_field.SINGLE_LINK_ID_FIELD,
                                       label='to')
        transition_df = self.diy_merge(left_df=transition_df,
                                       right_df=gps_match_link_route_dis,
                                       left_key=[gps_field.FROM_GPS_SEQ, markov_field.FROM_STATE],
                                       right_key=[gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD],
                                       label='from', merge_type='prj')
        transition_df = self.diy_merge(left_df=transition_df,
                                       right_df=gps_match_link_route_dis,
                                       left_key=[gps_field.TO_GPS_SEQ, markov_field.TO_STATE],
                                       right_key=[gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD],
                                       label='to', merge_type='prj')
        del gps_match_link_route_dis

        now_source_node = set(transition_df['from_link_f'])
        if not fmm_cache:
            # 先计算所有要计算的path
            if o_node_field in done_stp_cost_df.columns:
                already_cache_node = set(done_stp_cost_df[o_node_field])
            else:
                already_cache_node = set()
            gap = now_source_node - already_cache_node
            del single_link_ft_df[net_field.SINGLE_LINK_ID_FIELD]
            if gap:
                if not cache_path:
                    add_single_ft[0] = True
                done_stp_cost_df = self.add_path_cache(done_stp_cost_df=done_stp_cost_df,
                                                       source_node_list=gap, cut_off=cut_off,
                                                       single_link_ft_path_df=single_link_ft_df,
                                                       weight_field=weight_field, method=method, g=g,
                                                       add_single_ft=add_single_ft)
        del single_link_ft_df, g

        transition_df = pd.merge(transition_df, done_stp_cost_df, left_on=['from_link_f', 'to_link_f'],
                                 right_on=[o_node_field, d_node_field], how='left')
        del transition_df[o_node_field], transition_df[d_node_field]
        # sub_net do not share path within different agents
        if is_sub_net or fmm_cache or not cache_path:
            del done_stp_cost_df
            done_stp_cost_df = pd.DataFrame()

        transition_df[cost_field] = transition_df[cost_field].fillna(0)
        transition_df.reset_index(inplace=True, drop=True)
        transition_df[markov_field.ROUTE_LENGTH] = not_conn_cost * 1.0

        _ = transition_df[transition_df[cost_field] > 0]
        adj_seq_path_dict = {(int(f_state), int(t_state)): node_path for f_state, t_state, node_path, c in
                             zip(_[markov_field.FROM_STATE],
                                 _[markov_field.TO_STATE],
                                 _[path_field],
                                 _[cost_field]) if c > 0}
        same_link_idx = transition_df[markov_field.FROM_STATE] == transition_df[markov_field.TO_STATE]
        transition_df.loc[same_link_idx, markov_field.ROUTE_LENGTH] = \
            np.abs(transition_df.loc[same_link_idx, :]['from_route_dis'] -
                   transition_df.loc[same_link_idx, :]['to_route_dis'])
        transition_df['2nd_node'] = -1
        transition_df['-2nd_node'] = -1
        normal_path_idx_a = transition_df[cost_field] > 0
        # normal_path_idx_a可能全是False
        try:
            transition_df.loc[normal_path_idx_a, '2nd_node'] = transition_df.loc[normal_path_idx_a, :][
                path_field].apply(
                lambda x: x[1])
            transition_df.loc[normal_path_idx_a, '-2nd_node'] = transition_df.loc[normal_path_idx_a, :][
                path_field].apply(
                lambda x: x[-2])
        except:
            pass
        normal_path_idx_b = (normal_path_idx_a & \
                             (transition_df['2nd_node'] == transition_df['from_link_t']) & \
                             (transition_df['-2nd_node'] != transition_df['to_link_t'])) | \
                            ((transition_df['from_link_f'] == transition_df['to_link_t']) &
                             (transition_df['from_link_t'] == transition_df['to_link_f']))

        transition_df.loc[normal_path_idx_b, markov_field.ROUTE_LENGTH] = \
            np.abs(transition_df.loc[normal_path_idx_b, :][cost_field] -
                   transition_df.loc[normal_path_idx_b, :]['from_route_dis'] +
                   transition_df.loc[normal_path_idx_b, :]['to_route_dis'])

        transition_df = pd.merge(transition_df, gps_pre_next_dis_df, on=[gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ],
                                 how='left')
        transition_df[markov_field.DIS_GAP] = np.abs(
            -transition_df[markov_field.ROUTE_LENGTH] + transition_df[gps_field.ADJ_DIS])

        s2s_route_l = transition_df[[gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ, markov_field.ROUTE_LENGTH,
                                     markov_field.FROM_STATE, markov_field.TO_STATE]].copy()
        transition_df['trans_values'] = \
            self.transition_probability(self.beta, transition_df[markov_field.DIS_GAP].values, self.dis_para)
        ft_transition_dict = {f_gps_seq: df[
            [markov_field.FROM_STATE, markov_field.TO_STATE, 'trans_values']].set_index(
            [markov_field.FROM_STATE, markov_field.TO_STATE]).unstack().values for (f_gps_seq, t_gps_seq), df in
                              transition_df.groupby([gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ])}
        return adj_seq_path_dict, ft_transition_dict, ft_idx_map, s2s_route_l, seq_k_candidate_info, done_stp_cost_df

    def add_path_cache(self, done_stp_cost_df: pd.DataFrame = None, source_node_list: list[int] or set[int] = None,
                       g: nx.DiGraph = None, method: str = 'dijkstra', single_link_ft_path_df: pd.DataFrame = None,
                       weight_field: str = None, cut_off: float = 600.0, add_single_ft: list[bool] = True) \
            -> pd.DataFrame:
        _done_stp_cost_df = self.single_source_path_cache(node_list=source_node_list, method=method, cut_off=cut_off,
                                                          weight_field=weight_field, g=g)
        done_stp_cost_df = pd.concat([done_stp_cost_df, _done_stp_cost_df])
        del _done_stp_cost_df
        if add_single_ft[0]:
            single_link_ft_path_df = single_link_ft_path_df[single_link_ft_path_df[self.net.weight_field] > cut_off]
            if not single_link_ft_path_df.empty:
                done_stp_cost_df = pd.concat(
                    [done_stp_cost_df, single_link_ft_path_df.rename(columns={net_field.FROM_NODE_FIELD: o_node_field,
                                                                              net_field.TO_NODE_FIELD: d_node_field})])
                done_stp_cost_df.drop_duplicates(subset=[o_node_field, d_node_field], keep='first', inplace=True)
            print('add single ft')
            add_single_ft[0] = False
        done_stp_cost_df.reset_index(inplace=True, drop=True)
        return done_stp_cost_df

    def single_source_path_cache(self, node_list: list[int] or set[int] = None, g: nx.DiGraph = None,
                                 method: str = 'dijkstra',
                                 weight_field: str = None, cut_off: float = 600.0) -> pd.DataFrame:
        cost, stp = dict(), dict()
        stp_cost_df = pd.DataFrame()
        for n in node_list:
            try:
                cost[n], stp[n] = \
                    self._single_source_path_alpha(g, source=n,
                                                   method=method,
                                                   weight_field=weight_field,
                                                   cut_off=cut_off)
            except nx.NetworkXNoPath:
                pass
        if stp:
            stp_df = pd.DataFrame(stp).stack().reset_index(drop=False).rename(
                columns={'level_0': d_node_field, 'level_1': o_node_field, 0: path_field})
            cost_df = pd.DataFrame(cost).stack().reset_index(drop=False).rename(
                columns={'level_0': d_node_field, 'level_1': o_node_field, 0: cost_field})
            cost_df[cost_field] = np.around(cost_df[cost_field], decimals=1)
            stp_cost_df = pd.merge(stp_df, cost_df, on=[o_node_field, d_node_field])
            del stp_df, cost_df
            stp_cost_df.reset_index(inplace=True, drop=True)
        return stp_cost_df

    @staticmethod
    def diy_merge(left_df: pd.DataFrame, right_df: pd.DataFrame or gpd.GeoDataFrame = None,
                  left_key: str or list[str] = None,
                  right_key: str or list[str] = None, label: str = 'from', merge_type: str = 'ft'):
        df = pd.merge(left_df, right_df, left_on=left_key, right_on=right_key, how='left')

        if merge_type == 'ft':
            df.rename(columns={net_field.FROM_NODE_FIELD: label + '_link_f',
                               net_field.TO_NODE_FIELD: label + '_link_t'}, inplace=True)

            df.drop(columns=[right_key], axis=1, inplace=True)
        else:
            df.rename(columns={'route_dis': label + '_route_dis'}, inplace=True)
            df.drop(columns=right_key, axis=1, inplace=True)
        return df


    @staticmethod
    def _single_source_path_alpha(g: nx.DiGraph = None, source: int = None, method: str = 'dijkstra',
                                  weight_field: str = None, cut_off: float = 600.0) -> \
            tuple[dict[int, int], dict[int, list]]:
        if method == 'dijkstra':
            return nx.single_source_dijkstra(g, source, weight=weight_field, cutoff=cut_off)
        else:
            return nx.single_source_bellman_ford(g, source, weight=weight_field)

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
        state_idx_df = pd.DataFrame(
            {'idx': self.index_state_list, gps_field.POINT_SEQ_FIELD: self.gps_points.used_observation_seq_list})

        gps_link_state_df = pd.merge(state_idx_df, self.__ft_idx_map, on=[gps_field.POINT_SEQ_FIELD, 'idx'])

        single_link = self.net.get_link_data()[[net_field.SINGLE_LINK_ID_FIELD,
                                                net_field.LINK_ID_FIELD,
                                                net_field.DIRECTION_FIELD,
                                                net_field.FROM_NODE_FIELD,
                                                net_field.TO_NODE_FIELD]].copy()
        single_link.reset_index(inplace=True, drop=True)
        gps_link_state_df = pd.merge(gps_link_state_df, single_link, on=net_field.SINGLE_LINK_ID_FIELD, how='left')

        gps_link_state_df[gps_field.SUB_SEQ_FIELD] = 0
        gps_link_state_df = pd.merge(gps_link_state_df, self.__done_prj_df[[gps_field.POINT_SEQ_FIELD,
                                                                            net_field.SINGLE_LINK_ID_FIELD,
                                                                            markov_field.PRJ_GEO]],
                                     on=[gps_field.POINT_SEQ_FIELD,
                                         net_field.SINGLE_LINK_ID_FIELD], how='left')
        gps_link_state_df[['next_single', 'next_seq']] = gps_link_state_df[
            [net_field.SINGLE_LINK_ID_FIELD, gps_field.POINT_SEQ_FIELD]].shift(-1)
        gps_link_state_df.fillna(-99, inplace=True)
        gps_link_state_df[['next_single', 'next_seq']] = gps_link_state_df[['next_single', 'next_seq']].astype(int)

        self.__s2s_route_l.rename(columns={gps_field.FROM_GPS_SEQ: gps_field.POINT_SEQ_FIELD,
                                           gps_field.TO_GPS_SEQ: 'next_seq',
                                           markov_field.FROM_STATE: net_field.SINGLE_LINK_ID_FIELD,
                                           markov_field.TO_STATE: 'next_single',
                                           markov_field.ROUTE_LENGTH: markov_field.DIS_TO_NEXT}, inplace=True)

        gps_link_state_df = pd.merge(gps_link_state_df, self.__s2s_route_l, how='left', on=[gps_field.POINT_SEQ_FIELD,
                                                                                            'next_seq',
                                                                                            net_field.SINGLE_LINK_ID_FIELD,
                                                                                            'next_single'])
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
        del gps_link_state_df
        gps_match_res_gdf.loc[gps_match_res_gdf[gps_field.NEXT_LINK_FIELD].isna(), net_field.GEOMETRY_FIELD] = \
            omitted_gps_state_df[net_field.GEOMETRY_FIELD].to_list()
        gps_match_res_gdf.loc[gps_match_res_gdf[gps_field.NEXT_LINK_FIELD].isna(), gps_field.TIME_FIELD] = \
            omitted_gps_state_df[gps_field.TIME_FIELD].to_list()

        gps_match_res_gdf.drop(columns=[gps_field.NEXT_LINK_FIELD], axis=1, inplace=True)
        gps_match_res_gdf = gpd.GeoDataFrame(gps_match_res_gdf, geometry=gps_field.GEOMETRY_FIELD,
                                             crs=self.gps_points.crs)
        gps_match_res_gdf = gps_match_res_gdf.to_crs(self.gps_points.geo_crs)
        gps_match_res_gdf[[gps_field.LNG_FIELD, gps_field.LAT_FIELD]] = gps_match_res_gdf.apply(
            lambda row: (row[gps_field.GEOMETRY_FIELD].x, row[gps_field.GEOMETRY_FIELD].y), axis=1,
            result_type='expand')
        prj_gdf = gps_match_res_gdf[[markov_field.PRJ_GEO]].copy()
        del gps_match_res_gdf[markov_field.PRJ_GEO]
        prj_gdf.dropna(subset=[markov_field.PRJ_GEO], inplace=True)
        prj_gdf = gpd.GeoDataFrame(prj_gdf, geometry=markov_field.PRJ_GEO, crs=self.gps_points.plane_crs)
        prj_gdf = prj_gdf.to_crs(self.gps_points.geo_crs)
        prj_gdf[['prj_lng', 'ptj_lat']] = prj_gdf.apply(
            lambda row: (row[markov_field.PRJ_GEO].x, row[markov_field.PRJ_GEO].y), axis=1,
            result_type='expand')
        gps_match_res_gdf = pd.merge(gps_match_res_gdf, prj_gdf, how='left', left_index=True, right_index=True)
        del prj_gdf
        self.gps_match_res_gdf = gps_match_res_gdf
        return self.gps_match_res_gdf

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
                pre_seq = int(gps_link_state_df.at[i, gps_field.POINT_SEQ_FIELD])
                next_seq = int(gps_link_state_df.at[i + 1, gps_field.POINT_SEQ_FIELD])
                if ft_state in self.__adj_seq_path_dict.keys():
                    # pre_seq = int(gps_link_state_df.at[i, gps_field.POINT_SEQ_FIELD])
                    # next_seq = int(gps_link_state_df.at[i + 1, gps_field.POINT_SEQ_FIELD])
                    node_seq = self.__adj_seq_path_dict[ft_state]
                    if node_seq[1] != now_to_node:
                        warnings.warn(
                            rf'gps seq: {pre_seq} -> {next_seq} 状态转移出现问题, from_link:{(now_from_node, now_to_node)} -> to_link:{(next_from_node, next_to_node)}')
                        # self.warn_info.append([(now_from_node, now_to_node), (next_from_node, next_to_node)])
                        self.warn_info['from_ft'].append(
                            (ft_state[0], now_from_node, now_to_node, rf'seq:{pre_seq}-{next_seq}'))
                        self.warn_info['to_ft'].append(
                            (ft_state[1], next_from_node, next_to_node, rf'seq:{pre_seq}-{next_seq}'))
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
                    warnings.warn(
                        rf'gps seq: {pre_seq} -> {next_seq} 状态转移出现问题, from_link:{(now_from_node, now_to_node)} -> to_link:{(next_from_node, next_to_node)}')
                    # self.warn_info.append([(now_from_node, now_to_node), (next_from_node, next_to_node)])
                    self.warn_info['from_ft'].append(
                        (ft_state[0], now_from_node, now_to_node, rf'seq:{pre_seq}-{next_seq}'))
                    self.warn_info['to_ft'].append(
                        (ft_state[1], next_from_node, next_to_node, rf'seq:{pre_seq}-{next_seq}'))

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

    def acquire_visualization_res(self, use_gps_source: bool = False,
                                  link_width: float = 1.5, node_radius: float = 1.5,
                                  match_link_width: float = 5.0, gps_radius: float = 3.0) -> tuple[gpd.GeoDataFrame,
                                  gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """获取可视化结果"""
        if self.__plot_mix_gdf is None:
            # print('初次计算')
            single_link_gdf = self.net.get_link_data()
            single_link_gdf.reset_index(inplace=True, drop=True)
            node_gdf = self.net.get_node_data()
            net_crs = self.net.crs
            plain_crs = self.net.planar_crs
            is_geo_crs = self.net.is_geo_crs()

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
            plot_gps_gdf[net_field.GEOMETRY_FIELD] = plot_gps_gdf[net_field.GEOMETRY_FIELD].buffer(gps_radius)
            plot_gps_gdf[gps_field.TYPE_FIELD] = 'gps'

            # 匹配路段GDF
            plot_match_link_gdf = self.gps_match_res_gdf.copy()
            plot_match_link_gdf.drop(columns=[markov_field.PRJ_GEO], axis=1, inplace=True)
            plot_match_link_gdf[gps_field.TYPE_FIELD] = 'link'
            del plot_match_link_gdf[net_field.GEOMETRY_FIELD]
            plot_match_link_gdf = pd.merge(plot_match_link_gdf, single_link_gdf[[net_field.SINGLE_LINK_ID_FIELD,
                                                                                 net_field.GEOMETRY_FIELD]],
                                           on=net_field.SINGLE_LINK_ID_FIELD,
                                           how='inner')
            plot_match_link_gdf = gpd.GeoDataFrame(plot_match_link_gdf, geometry=net_field.GEOMETRY_FIELD, crs=net_crs)
            plot_match_link_gdf.drop_duplicates(subset=net_field.LINK_ID_FIELD, keep='first', inplace=True)
            plot_match_link_gdf.reset_index(drop=True, inplace=True)
            if is_geo_crs:
                plot_match_link_gdf = plot_match_link_gdf.to_crs(plain_crs)
            plot_match_link_gdf[net_field.GEOMETRY_FIELD] = \
                plot_match_link_gdf[net_field.GEOMETRY_FIELD].buffer(match_link_width)
            plot_match_link_gdf.drop(columns=[gps_field.LNG_FIELD, gps_field.LAT_FIELD], axis=1, inplace=True)

            plot_gps_gdf = plot_gps_gdf.to_crs(self.net.geo_crs)
            plot_match_link_gdf = plot_match_link_gdf.to_crs(self.net.geo_crs)
            gps_link_gdf = pd.concat([plot_gps_gdf, plot_match_link_gdf])
            gps_link_gdf.reset_index(inplace=True, drop=True)

            # 错误信息
            may_error_gdf = gpd.GeoDataFrame()
            if self.format_warn_info.empty:
                pass
            else:
                self.format_warn_info[['from_single', 'to_single', 'ft_gps']] = self.format_warn_info.apply(
                    lambda row: (row['from_ft'][0], row['to_ft'][0], row['from_ft'][3]),
                    axis=1, result_type='expand')
                format_warn_df = pd.concat([self.format_warn_info[['from_single', 'ft_gps']].rename(
                    columns={'from_single': net_field.SINGLE_LINK_ID_FIELD}),
                                            self.format_warn_info[['to_single', 'ft_gps']].rename(
                                                columns={'to_single': net_field.SINGLE_LINK_ID_FIELD})])
                format_warn_df.reset_index(inplace=True, drop=True)
                may_error_gdf = pd.merge(format_warn_df,
                                         single_link_gdf, on=net_field.SINGLE_LINK_ID_FIELD, how='left')
                may_error_gdf = gpd.GeoDataFrame(may_error_gdf, geometry=net_field.GEOMETRY_FIELD,
                                                 crs=single_link_gdf.crs.srs)

            # 路网底图
            origin_link_gdf = single_link_gdf.drop_duplicates(subset=[net_field.LINK_ID_FIELD], keep='first').copy()
            del single_link_gdf
            if is_geo_crs:
                origin_link_gdf = origin_link_gdf.to_crs(plain_crs)
                node_gdf = node_gdf.to_crs(plain_crs)
                may_error_gdf = may_error_gdf.to_crs(plain_crs)
            origin_link_gdf[net_field.GEOMETRY_FIELD] = origin_link_gdf[net_field.GEOMETRY_FIELD].buffer(link_width)
            node_gdf[net_field.GEOMETRY_FIELD] = node_gdf[net_field.GEOMETRY_FIELD].buffer(node_radius)

            if not may_error_gdf.empty:
                may_error_gdf[net_field.GEOMETRY_FIELD] = may_error_gdf[net_field.GEOMETRY_FIELD].buffer(link_width)

            origin_link_gdf = origin_link_gdf.to_crs(self.net.geo_crs)
            if not may_error_gdf.empty:
                may_error_gdf = may_error_gdf.to_crs(self.net.geo_crs)
            node_gdf = node_gdf.to_crs(self.net.geo_crs)

            self.__plot_mix_gdf, self.__base_link_gdf, self.__base_node_gdf, self.__may_error = \
                gps_link_gdf, origin_link_gdf, node_gdf, may_error_gdf
            return gps_link_gdf, origin_link_gdf, node_gdf, may_error_gdf
        else:
            # print('利用重复值')
            return self.__plot_mix_gdf, self.__base_link_gdf, self.__base_node_gdf, self.__may_error

    def acquire_geo_res(self, out_fldr: str = None, flag_name: str = 'flag'):
        """获取矢量结果文件, 可以在qgis中可视化"""
        out_fldr = r'./' if out_fldr is None else out_fldr
        if self.gps_match_res_gdf is None:
            self.acquire_res()

        # gps
        gps_layer = self.gps_match_res_gdf[[gps_field.POINT_SEQ_FIELD, gps_field.SUB_SEQ_FIELD,
                                            gps_field.GEOMETRY_FIELD]].copy()

        # prj_point
        prj_p_layer = self.gps_match_res_gdf[
            [gps_field.POINT_SEQ_FIELD, gps_field.SUB_SEQ_FIELD, net_field.LINK_ID_FIELD, net_field.FROM_NODE_FIELD,
             net_field.TO_NODE_FIELD, markov_field.DIS_TO_NEXT, markov_field.PRJ_GEO, gps_field.GEOMETRY_FIELD]].copy()
        prj_p_layer.dropna(subset=[markov_field.PRJ_GEO], inplace=True)
        prj_p_layer.set_geometry(markov_field.PRJ_GEO, inplace=True, crs=self.gps_points.geo_crs)

        prj_p_layer['__geo'] = prj_p_layer.apply(
            lambda item: LineString((item[gps_field.GEOMETRY_FIELD], item[markov_field.PRJ_GEO])), axis=1)

        # prj_line
        prj_l_layer = prj_p_layer[['__geo']].copy()
        prj_l_layer.rename(columns={'__geo': gps_field.GEOMETRY_FIELD}, inplace=True)
        prj_l_layer = gpd.GeoDataFrame(prj_l_layer, geometry=gps_field.GEOMETRY_FIELD, crs=prj_p_layer.crs)

        prj_p_layer.set_geometry(markov_field.PRJ_GEO, inplace=True, crs=prj_p_layer.crs)

        prj_p_layer.drop(columns=['__geo', gps_field.GEOMETRY_FIELD], axis=1, inplace=True)

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
        match_link_gdf = gpd.GeoDataFrame(match_link_gdf, geometry=net_field.GEOMETRY_FIELD, crs=self.net.planar_crs)
        match_link_gdf = match_link_gdf.to_crs(self.net.geo_crs)

        for gdf, name in zip([gps_layer, prj_p_layer, prj_l_layer, match_link_gdf],
                             ['gps', 'prj_p', 'prj_l', 'match_link']):
            gdf.to_file(os.path.join(out_fldr, '-'.join([flag_name, name]) + '.geojson'), driver='GeoJSON')

    def format_war_info(self):
        if self.warn_info['from_ft']:
            self.format_warn_info = pd.DataFrame(self.warn_info)
        del self.warn_info


    # @function_time_cost
    # def generate_transition_mat_beta1(self, gps_ft_list: list = None, single_link_ft_df: pd.DataFrame = None,
    #                                   pre_seq_candidate: pd.DataFrame = None,
    #                                   gps_pre_next_dis_df: pd.DataFrame = None,
    #                                   g: nx.DiGraph = None,
    #                                   method: str = None, weight_field: str = 'length',
    #                                   cache_path: bool = True, not_conn_cost: float = 999.0,
    #                                   done_stp_cost_df: pd.DataFrame = None,
    #                                   is_sub_net: bool = True, fmm_cache: bool = False, cut_off: float = 600.0,
    #                                   max_cut_off: float = 2000.0, cache_prj_inf: dict = None) -> \
    #         tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     gps_ft_df = pd.DataFrame(gps_ft_list, columns=['f_gps_seq', 't_gps_seq'])
    #     used_gps_seq = list(set(gps_ft_df['f_gps_seq']) | set(gps_ft_df['t_gps_seq']))
    #
    #     # 确定最终的top_k候选
    #     pre_seq_candidate = pre_seq_candidate[
    #         pre_seq_candidate[gps_field.POINT_SEQ_FIELD].isin(used_gps_seq)].copy()
    #     seq_k_candidate_info = self.filter_k_candidates(preliminary_candidate_link=pre_seq_candidate,
    #                                                     top_k=self.top_k, cache_prj_inf=cache_prj_inf)
    #
    #     # 存储状态的索引和状态序列的映射
    #     seq_k_candidate_info['idx'] = seq_k_candidate_info.groupby(gps_field.POINT_SEQ_FIELD)[
    #                                       net_field.SINGLE_LINK_ID_FIELD].rank(method='min').astype(int) - 1
    #     ft_idx_map = seq_k_candidate_info[[gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD, 'idx']].copy()
    #     ft_idx_map.sort_values(by=[gps_field.POINT_SEQ_FIELD, net_field.SINGLE_LINK_ID_FIELD], inplace=True)
    #     del seq_k_candidate_info['idx']
    #
    #     del pre_seq_candidate
    #     del used_gps_seq
    #
    #     s = time.time()
    #     # 构造状态转移表
    #     seq_k_candidate = seq_k_candidate_info.groupby(gps_field.POINT_SEQ_FIELD).agg(
    #         {net_field.SINGLE_LINK_ID_FIELD: list, gps_field.POINT_SEQ_FIELD: list,
    #          net_field.FROM_NODE_FIELD: list, net_field.TO_NODE_FIELD: list, 'route_dis': list}).rename(
    #         columns={gps_field.POINT_SEQ_FIELD: gps_field.FROM_GPS_SEQ,
    #                  net_field.SINGLE_LINK_ID_FIELD: markov_field.FROM_STATE,
    #                  net_field.FROM_NODE_FIELD: 'from_link_f', net_field.TO_NODE_FIELD: 'from_link_t',
    #                  'route_dis': 'from_route_dis'})
    #
    #     seq_k_candidate[[gps_field.TO_GPS_SEQ, markov_field.TO_STATE, 'to_link_f', 'to_link_t', 'to_route_dis']] = \
    #         seq_k_candidate[[gps_field.FROM_GPS_SEQ, markov_field.FROM_STATE, 'from_link_f', 'from_link_t',
    #                          'from_route_dis']].shift(-1)
    #     seq_k_candidate.dropna(subset=[markov_field.TO_STATE], inplace=True)
    #
    #     from_state = seq_k_candidate[
    #         [markov_field.FROM_STATE, gps_field.FROM_GPS_SEQ,
    #          'from_link_f', 'from_link_t', 'from_route_dis']].reset_index(drop=False).explode(
    #         column=[markov_field.FROM_STATE, gps_field.FROM_GPS_SEQ, 'from_link_f', 'from_link_t',
    #                 'from_route_dis'], ignore_index=True)
    #
    #     to_state = seq_k_candidate[
    #         [markov_field.TO_STATE, gps_field.TO_GPS_SEQ, 'to_link_f', 'to_link_t', 'to_route_dis']].reset_index(
    #         drop=False).explode(
    #         column=[markov_field.TO_STATE, gps_field.TO_GPS_SEQ, 'to_link_f', 'to_link_t', 'to_route_dis'],
    #         ignore_index=True)
    #     del seq_k_candidate
    #     transition_df = pd.merge(from_state, to_state, on=gps_field.POINT_SEQ_FIELD, how='outer')
    #     del transition_df[gps_field.POINT_SEQ_FIELD]
    #     del from_state, to_state
    #     transition_df.reset_index(inplace=True, drop=True)
    #     print(rf'构造: {time.time() - s}')
    #     t = time.time()
    #     del single_link_ft_df
    #
    #     now_source_node = set(transition_df['from_link_f'])
    #     if not fmm_cache:
    #         # 先计算所有要计算的path
    #         already_cache_node = set(done_stp_cost_df['o_node'])
    #         gap = now_source_node - already_cache_node
    #         if not gap:
    #             done_stp_cost_df = self.add_path_cache(done_stp_cost_df=done_stp_cost_df,
    #                                                    source_node_list=gap, cut_off=cut_off,
    #                                                    weight_field=weight_field, method=method, g=g)
    #             now_source_node = set(done_stp_cost_df['o_node'])
    #
    #     # 如果某些点除开自身-自身没有其他点, 则加一个5km搜索
    #     done_stp_cost_df.drop(index=done_stp_cost_df[done_stp_cost_df['o_node'] == done_stp_cost_df['d_node']].index,
    #                           inplace=True)
    #     no_path_node = now_source_node - set(done_stp_cost_df['o_node'])
    #     if no_path_node:
    #         done_stp_cost_df = self.add_path_cache(done_stp_cost_df=done_stp_cost_df,
    #                                                source_node_list=no_path_node, cut_off=max_cut_off,
    #                                                weight_field=weight_field, method=method, g=g)
    #     del g
    #
    #     # transition_df['from_route_dis'] = transition_df['from_route_dis'].astype(float)
    #     # transition_df['to_route_dis'] = transition_df['to_route_dis'].astype(float)
    #
    #     transition_df = pd.merge(transition_df, done_stp_cost_df, left_on=['from_link_f', 'to_link_f'],
    #                              right_on=['o_node', 'd_node'], how='left')
    #     del transition_df['o_node'], transition_df['d_node']
    #     # sub_net do not share path within different agents
    #     if is_sub_net or fmm_cache or not cache_path:
    #         del done_stp_cost_df
    #         done_stp_cost_df = pd.DataFrame()
    #
    #     transition_df['cost'] = transition_df['cost'].fillna(0)
    #     transition_df.reset_index(inplace=True, drop=True)
    #     transition_df[markov_field.ROUTE_LENGTH] = not_conn_cost * 1.0
    #
    #     _ = transition_df[transition_df['cost'] > 0]
    #     adj_seq_path_dict = {(int(f_state), int(t_state)): node_path for f_state, t_state, node_path, c in
    #                          zip(_[markov_field.FROM_STATE],
    #                              _[markov_field.TO_STATE],
    #                              _['path'],
    #                              _['cost']) if c > 0}
    #     z = time.time()
    #     same_link_idx = transition_df[markov_field.FROM_STATE] == transition_df[markov_field.TO_STATE]
    #     transition_df.loc[same_link_idx, markov_field.ROUTE_LENGTH] = \
    #         np.abs(transition_df.loc[same_link_idx, :]['from_route_dis'].values.astype(float) -
    #                transition_df.loc[same_link_idx, :]['to_route_dis'].values.astype(float))
    #     transition_df['2nd_node'] = -1
    #     transition_df['-2nd_node'] = -1
    #     normal_path_idx_a = transition_df['cost'] > 0
    #     t2 = time.time()
    #     transition_df.loc[normal_path_idx_a, '2nd_node'] = transition_df.loc[normal_path_idx_a, :]['path'].apply(
    #         lambda x: x[1])
    #     transition_df.loc[normal_path_idx_a, '-2nd_node'] = transition_df.loc[normal_path_idx_a, :]['path'].apply(
    #         lambda x: x[-2])
    #     t3 = time.time()
    #     normal_path_idx_b = (normal_path_idx_a & \
    #                          (transition_df['2nd_node'] == transition_df['from_link_t']) & \
    #                          (transition_df['-2nd_node'] != transition_df['to_link_t'])) | \
    #                         ((transition_df['from_link_f'] == transition_df['to_link_t']) &
    #                          (transition_df['from_link_t'] == transition_df['to_link_f']))
    #
    #     transition_df.loc[normal_path_idx_b, markov_field.ROUTE_LENGTH] = \
    #         np.abs(transition_df.loc[normal_path_idx_b, :]['cost'].values -
    #                transition_df.loc[normal_path_idx_b, :]['from_route_dis'].values.astype(float) +
    #                transition_df.loc[normal_path_idx_b, :]['to_route_dis'].values.astype(float))
    #     print(rf'cache_len: {len(done_stp_cost_df)}')
    #
    #     transition_df = pd.merge(transition_df, gps_pre_next_dis_df, on=[gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ],
    #                              how='left')
    #
    #     transition_df[markov_field.DIS_GAP] = np.abs(
    #         -transition_df[markov_field.ROUTE_LENGTH] + transition_df[gps_field.ADJ_DIS])
    #     t4 = time.time()
    #     # print(transition_df[markov_field.DIS_GAP])
    #     print(rf'向量化总时间: {t4 - z}')
    #     print(rf'apply: {t3 - t2}')
    #     print(rf'总计算: {t4 - t}')
    #
    #     s2s_route_l = transition_df[[gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ, markov_field.ROUTE_LENGTH,
    #                                  markov_field.FROM_STATE, markov_field.TO_STATE]].copy()
    #
    #     transition_df['trans_values'] = \
    #         self.transition_probability(self.beta, transition_df[markov_field.DIS_GAP].values, self.dis_para)
    #
    #     ft_transition_dict = {f_gps_seq: df[
    #         [markov_field.FROM_STATE, markov_field.TO_STATE, 'trans_values']].set_index(
    #         [markov_field.FROM_STATE, markov_field.TO_STATE]).unstack().values for (f_gps_seq, t_gps_seq), df in
    #                           transition_df.groupby([gps_field.FROM_GPS_SEQ, gps_field.TO_GPS_SEQ])}
    #     print(rf'归并mat: {time.time() - t4}')
    #
    #     return adj_seq_path_dict, ft_transition_dict, ft_idx_map, s2s_route_l, seq_k_candidate_info, done_stp_cost_df


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




