# -- coding: utf-8 --
# @Time    : 2023/12/9 8:02
# @Author  : TangKai
# @Team    : ZheChengData

"""viterbi算法求解动态规划"""

import numpy as np


class Viterbi(object):
    def __init__(self, observation_list: list[int], t_mat_dict: dict[int, np.ndarray] = None,
                 o_mat_dict: dict[int, np.ndarray] = None, use_log_p: bool = True,
                 initial_ep: dict[int, np.ndarray] = None):
        """
        :param t_mat_dict: 存储每个观测点到下一观测点(相邻观测点)之间的状态转移概率矩阵
        :param o_mat_dict: 存储每个观测点的观测概率矩阵
        :param use_log_p: 是否使用对数概率
        """
        self.use_log_p = use_log_p
        self.o_seq_list = observation_list  # 观测序列值(GPS数据的seq值, 值不一定连续, 但是一定递增)
        self.T = len(observation_list)  # 获取观测点数目
        assert self.T > 1, '至少2个观测点'
        self.psi_array_dict = dict()  # [1, observation_num]
        self.zeta_array_dict = dict()  # [0, observation_num - 1]

        # 确保每个状态的观测概率矩阵是一个1 * 状态数的矩阵
        o_mat_dict = {observe_seq: o_mat_dict[observe_seq].reshape(1, o_mat_dict[observe_seq].shape[0]) if len(
            o_mat_dict[observe_seq].shape) == 1 else o_mat_dict[observe_seq] for observe_seq in o_mat_dict.keys()}
        # 转移概率矩阵哈希表, key值为 t∈[0 ~ T - 2], 对应的矩阵为 M * N 矩阵
        # t值对应的矩阵, 其ij值表示由 t时刻状态i 转移至 t + 1时刻状态j的概率
        self.AMat: dict[int, np.ndarray] = t_mat_dict

        # 观测概率矩阵哈希表, key值为 t∈[0 ~ T - 1], 迭代元素是一个列 1 * N 的矩阵, 表示由当前时刻各状态生产当前观测的概率
        self.BMat: dict[int, np.ndarray] = o_mat_dict

        # 如果只有一个元素, 则认为所有观测时刻的转移概率矩阵和观测概率矩阵一样
        if len(self.AMat) == 1:
            assert self.o_seq_list[0] in self.AMat.keys()
            self.AMat = {k: self.AMat[self.o_seq_list[0]] for k in self.o_seq_list[:-1]}
        assert len(self.AMat) == self.T - 1
        if len(self.BMat) == 1:
            assert self.o_seq_list[0] in self.BMat.keys()
            self.BMat = {k: self.BMat[self.o_seq_list[0]] for k in self.o_seq_list}
        assert len(self.BMat) == self.T
        self.initial_ep = initial_ep

    def init_model(self) -> None:
        """初始化模型"""
        if self.initial_ep is None:
            # 初始化, 获取初始观测态的可选状态数量
            init_n = self.AMat[self.o_seq_list[0]].shape[0]
            # 初始观测概率矩阵
            init_b = self.BMat[self.o_seq_list[0]]
            self.initial_ep = (1 / init_n) * init_b.astype(float)
            if self.use_log_p:
                self.zeta_array_dict[self.o_seq_list[0]] = np.log(self.initial_ep)
            else:
                self.zeta_array_dict[self.o_seq_list[0]] = self.initial_ep
        else:
            self.zeta_array_dict[self.o_seq_list[0]] = self.initial_ep

        # print(rf'初始化后:{self.zeta_array_dict[0]}')

    def iter_model(self) -> list[int]:
        """动态规划迭代求解"""

        # 1.迭代状态转移(i状态向 i + 1状态 转移)
        for i, o in enumerate(self.o_seq_list[:-1]):

            # 第i个观测点的各状态的概率 * (i -> i + 1)状态转移概率 * 当前状态的概率
            # self.zeta_array_dict[self.o_seq_list[i]].T * self.AMat[self.o_seq_list[i]] 是一个m * n的矩阵
            # self.BMat[self.o_seq_list[i + 1]]是一个n * 1的矩阵
            # m是i观测点的可选状态数, n是i+1观测点的可选状态数

            # t = self.zeta_array_dict[self.o_seq_list[i]].T * self.AMat[self.o_seq_list[i]] * self.BMat[self.o_seq_list[i + 1]]
            # check_t = np.log(t.astype(np.float32))
            # temp_t = np.log(self.zeta_array_dict[self.o_seq_list[i]].T.astype(np.float32)) + \
            #          np.log(self.AMat[self.o_seq_list[i]].astype(np.float32)) + \
            #          np.log(self.BMat[self.o_seq_list[i + 1]].astype(np.float32))
            # print(temp_t)

            t = self.calc_zeta_p(zeta_now_array=self.zeta_array_dict[self.o_seq_list[i]].T,
                                 a_now_array=self.AMat[self.o_seq_list[i]],
                                 b_next_array=self.BMat[self.o_seq_list[i + 1]], use_log=self.use_log_p)

            # 找出当前每种状态的最大值, last_state_index是一个 n * 1的矩阵
            last_state_index = np.argmax(t, axis=0)

            # 记录
            self.psi_array_dict[self.o_seq_list[i + 1]] = last_state_index
            self.zeta_array_dict[self.o_seq_list[i + 1]] = \
                np.array([t[last_state_index, [i for i in range(0, len(last_state_index))]]])

        # 2.回溯
        state_list = []
        last_max_state = -1
        for i in range(self.T - 1, 0, -1):
            if i == self.T - 1:
                # 回溯起点, 找出当前最大概率的状态
                start_max_state = np.argmax(self.zeta_array_dict[self.o_seq_list[i]])
                state_list.append(start_max_state)
                last_max_state = self.psi_array_dict[self.o_seq_list[i]][start_max_state]
                state_list.append(last_max_state)
            else:
                start_max_state = last_max_state
                last_max_state = self.psi_array_dict[self.o_seq_list[i]][start_max_state]
                state_list.append(last_max_state)
        state_list = state_list[::-1]
        return state_list

    @staticmethod
    def calc_zeta_p(zeta_now_array: np.ndarray = None,
                    a_now_array: np.ndarray = None,
                    b_next_array: np.ndarray = None, use_log: bool = True) -> np.ndarray:
        if use_log:
            return zeta_now_array.astype(np.float32) + np.log(a_now_array.astype(np.float32)) + \
                np.log(b_next_array.astype(np.float32))
        else:
            return zeta_now_array * a_now_array * b_next_array


if __name__ == '__main__':

    # 0: 没分享
    # 1: 分享了
    state_map = {0: '海润C', 1: '新羽胜', 2: '亿利达'}

    # tests = Viterbi(observation_list=[0, 1, 2, 3],
    #                t_mat_dict={0: np.array([[1, 1.2, 1.3],
    #                                         [1.2, 1.3, 1.4],
    #                                         [1.5, 1.4, 1.3]]),
    #                            1: np.array([[0.25, 0.4],
    #                                         [0.36, 0.3],
    #                                         [0.2, 0.26]]),
    #                            2: np.array([[0.50, 0.4, 0.30],
    #                                         [0.40, 0.8, 0.4]])
    #                            },
    #
    #                o_mat_dict={0: np.array([0.5, 0.2, 0.3]),
    #                            1: np.array([0.5, 0.5, 0.1]),
    #                            2: np.array([0.4, 0.2]),
    #                            3: np.array([0.1, 0.5, 0.1])})

    # tests = Viterbi(observation_list=[0, 2, 4, 7],
    #                t_mat_dict={0: np.array([[1, 1.2, 1.3],
    #                                         [1.2, 1.3, 1.4],
    #                                         [1.5, 1.4, 1.3]]),
    #                            2: np.array([[0.25, 0.4],
    #                                         [0.36, 0.3],
    #                                         [0.2, 0.26]]),
    #                            4: np.array([[0.50, 0.4, 0.30],
    #                                         [0.40, 0.8, 0.4]])
    #                            },
    #
    #                o_mat_dict={0: np.array([0.5, 0.2, 0.3]),
    #                            2: np.array([0.5, 0.5, 0.1]),
    #                            4: np.array([0.4, 0.2]),
    #                            7: np.array([0.1, 0.5, 0.1])})

    test = Viterbi(observation_list=[0, 2],
                   t_mat_dict={0: np.array([[1, 1.2, 1.3],
                                            [1.2, 1.3, 1.4],
                                            [1.5, 1.4, 1.3]])},

                   o_mat_dict={0: np.array([0.5, 0.2, 0.3]),
                               2: np.array([0.5, 0.5, 0.1]),})
    test.init_model()
    state_l = test.iter_model()
    print(state_l)
    print([state_map[i] for i in state_l])

    # a = np.array([[1, 2, 3]])
    # b = np.array([[1,3,1], [2,12,14], [11,9,6]])
    # c = np.array([[4, 5, 6]])
    # #
    # print(a)
    # print(b)
    # x = a.T*b
    # print(x)
    # z = x * c
    # print(z)
    #
    # print(np.argmax(z, axis=0))

    # x = np.array([1,2,3])
    # print(x.shape[0])
    #
    # print(x.reshape(1, 3))
    # o_mat_dict = {1: np.array([[1,2,3]]), 2: np.array([[1,56,3]]), }
    # o_mat_dict = {observe_seq: o_mat_dict[observe_seq].reshape(1, o_mat_dict[observe_seq].shape[0]) if len(
    #     o_mat_dict[observe_seq].shape) == 1 else o_mat_dict[observe_seq] for observe_seq in o_mat_dict.keys()}
    # for i in o_mat_dict.keys():
    #     print(i)
    #     print(o_mat_dict[i])
    #     print(o_mat_dict[i].shape)
    #     print('********')
