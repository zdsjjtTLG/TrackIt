# -- coding: utf-8 --
# @Time    : 2023/12/9 8:29
# @Author  : TangKai
# @Team    : ZheChengData
import time

import numpy as np
import pandas as pd
from src.map.Net import Net
from shapely.geometry import Point
from src.solver.Viterbi import Viterbi
from src.gps.LocGps import GpsPointsGdf
from src.WrapsFunc import function_time_cost
from src.GlobalVal import NetField, GpsField, MarkovField
from src.map.Net import NOT_CONN_COST

gps_field = GpsField()
net_field = NetField()
markov_field = MarkovField()


class HiddenMarkov(object):

    def __init__(self, net: Net, gps_points: GpsPointsGdf, beta: float = 30.1, gps_sigma: float = 20.0,
                 search_method: str = 'all_pairs'):
        self.gps_points = gps_points
        self.net = net
        assert search_method in ['all_pairs', 'single'], 'search_method must in [\'all_pairs\', \'single\'] '
        self.search_method = search_method
        # (from_gps_seq, from_link_id): (from_prj_p, from_prj_dis, from_route_dis)
        self.__done_prj_dict: dict[tuple[int, int]: tuple[Point, float, float]] = dict()
        self.__ft_transition_dict = dict()
        self.__ft_mapping_dict = dict()
        self.beta = beta
        self.gps_sigma = gps_sigma
        self.__emission_mat_dict = dict()
        self.__solver = None
        self.index_state_list = None

    def generate_markov_para(self):
        self.__generate_transition_mat()
        self.__generate_emission_mat()

    def solve(self):
        # 使用viterbi模型求解
        self.__solver = Viterbi(observation_num=self.gps_points.gps_list_length,
                                o_mat_dict=self.__emission_mat_dict,
                                t_mat_dict=self.__ft_transition_dict)
        self.__solver.init_model()
        self.index_state_list = self.__solver.iter_model()

        print(self.index_state_list)

    def acquire_res(self):
        single_link_state_list = [self.__ft_mapping_dict[observe_seq][state_index] for observe_seq, state_index in
                                  zip(range(len(self.index_state_list)),
                                      self.index_state_list)]
        # print(single_link_state_list)
        link_state_list = [self.net.bilateral_unidirectional_mapping[link] for link in single_link_state_list]
        # print(link_state_list)
        # print(len(link_state_list))

        df = pd.DataFrame(link_state_list, columns=['link_id', 'dir'])
        map_res = {i:link for i, link in zip(range(len(df)), df['link_id'])}
        gps_gdf = self.gps_points.gps_gdf
        # print(gps_gdf)
        print(map_res)

        single_link_gdf = self.net.get_link_data()
        match_link_res = single_link_gdf[single_link_gdf['link_id'].isin(list(map_res.values()))].copy()
        match_link_res.drop_duplicates(subset=['link_id'], inplace=True)
        match_link_res.reset_index(drop=True, inplace=True)
        match_link_res.to_file(r'./data/output/match/res_link.shp')

        gps_gdf['match_link'] = gps_gdf[gps_field.POINT_SEQ_FIELD].apply(lambda x: map_res[x])
        gps_gdf.drop(columns=[gps_field.TIME_FIELD], axis=1, inplace=True)
        gps_gdf.to_file(r'./data/output/match/res.shp', encoding='gbk')
        return link_state_list

    @function_time_cost
    def __generate_transition_mat(self):
        # 依据一辆车的时序gps点和和底层路网生成转移概率矩阵和生成概率矩阵
        # seq, geometry, single_link_id, from_node, to_node, dir, length
        gps_candidate_link = self.gps_points.generate_candidate_link(net=self.net)

        seq_list = sorted(list(gps_candidate_link[gps_field.POINT_SEQ_FIELD].unique()))

        if self.search_method == 'all_pairs':
            self.net.calc_all_pairs_shortest_path()

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
        if (from_gps_seq, from_link_id) in self.__done_prj_dict.keys():
            (from_prj_p, from_prj_dis, from_route_dis, from_l_length) = self.__done_prj_dict[
                (from_gps_seq, from_link_id)]
        else:
            (from_prj_p, from_prj_dis, from_route_dis, from_l_length) = self.get_gps_prj_info(
                target_link_id=from_link_id,
                gps_seq=from_gps_seq)
            self.__done_prj_dict.update(
                {(from_gps_seq, from_link_id): (from_prj_p, from_prj_dis, from_route_dis, from_l_length)})

        if (to_gps_seq, to_link_id) in self.__done_prj_dict.keys():
            (to_prj_p, to_prj_dis, to_route_dis, to_l_length) = self.__done_prj_dict[(to_gps_seq, to_link_id)]
        else:
            (to_prj_p, to_prj_dis, to_route_dis, to_l_length) = self.get_gps_prj_info(target_link_id=to_link_id,
                                                                                      gps_seq=to_gps_seq)
            self.__done_prj_dict.update({(to_gps_seq, to_link_id): (to_prj_p, to_prj_dis, to_route_dis, to_l_length)})

        # 基于投影信息计算路径长度
        from_link_ft, to_link_ft = self.net.get_link_from_to(from_link_id), self.net.get_link_from_to(to_link_id)

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

        route_item = self.net.search(o_node=from_link_ft[0], d_node=to_link_ft[0], search_method=self.search_method)

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

    def get_gps_prj_info(self, gps_seq: int = None, target_link_id: int = None) -> tuple[Point, float, float, float]:
        return self.gps_points.get_prj_inf(line=self.net.get_link_geo(target_link_id), seq=gps_seq)

    @staticmethod
    def transition_probability(beta: float = 30.2, dis_gap: float or np.ndarray = None):
        """
        dis_gap = straight_l - route_l
        :param beta:
        :param dis_gap:
        :return:
        """
        # p = (1 / beta) * np.e ** (- dis_gap / beta)
        dis_gap = dis_gap / 10
        p = np.e ** (- dis_gap / beta)
        return p

    @staticmethod
    def emission_probability(sigma: float = 1.0, dis: float = 10.0) -> float:
        # p = (1 / (sigma * (2 * np.pi) ** 0.5)) * (np.e ** (-0.5 * (dis / sigma) ** 2))
        dis = dis / 10
        p = np.e ** (-0.5 * (dis / sigma) ** 2)
        return p


if __name__ == '__main__':
    # df = pd.DataFrame({'a': [12, 2, 31, 4], 'b': [34, 12, 11, 1241], 'c': [123, 222, 444, 555]})
    # print(df.set_index(['a', 'b']).unstack())
    # print(df.set_index(['a', 'b']).unstack().values)
    #
    # a = np.array([[1, 2, 3], [3, 1, 0]])
    #
    # print(a)
    # print(HiddenMarkov.transition_probability(0.2, a))

    # a = {(1,2) :(3,45,12), (3,1):(234,12,11)}
    # pd.DataFrame(a).T.reset_index(drop=False)

    # print(HiddenMarkov.emission_probability(sigma=20.0, dis=0))
    pass

