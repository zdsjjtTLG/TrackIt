# -- coding: utf-8 --
# @Time    : 2023/12/9 8:02
# @Author  : TangKai
# @Team    : ZheChengData

"""viterbi算法求解动态规划"""

import numpy as np
from src.WrapsFunc import function_time_cost


class Viterbi(object):
    def __init__(self, observation_num: int = 2, t_mat_dict: dict[int, np.ndarray] = None,
                 o_mat_dict: dict[int, np.ndarray] = None):
        """

        :param observation_num: 观测点数目
        :param t_mat_dict: 存储每个观测点到下一观测点(相邻观测点)之间的状态转移概率矩阵
        :param o_mat_dict: 存储每个观测点的观测概率矩阵
        """
        self.T = observation_num  # 获取观测点数目
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
            assert 0 in self.AMat.keys()
            self.AMat = {k: self.AMat[0] for k in range(0, self.T - 1)}
        assert len(self.AMat) == self.T - 1
        if len(self.BMat) == 1:
            assert 0 in self.BMat.keys()
            self.BMat = {k: self.BMat[0] for k in range(0, self.T)}
        assert len(self.BMat) == self.T

    def init_model(self) -> None:
        """初始化模型"""
        # 初始化, 获取初始观测态的可选状态数量
        init_n = self.AMat[0].shape[0]

        # 初始观测概率矩阵
        init_b = self.BMat[0]

        self.zeta_array_dict[0] = (1 / init_n) * init_b
        print(rf'初始化后:{self.zeta_array_dict[0]}')

    @function_time_cost
    def iter_model(self) -> list[int]:
        """动态规划迭代求解"""

        # 1.迭代状态转移(i状态向 i + 1状态 转移)
        for i in range(0, self.T - 1):
            # print('**********')

            # print(self.zeta_array_dict[i].T)
            # print(self.AMat[i])
            # print(self.zeta_array_dict[i].T * self.AMat[i])
            # print(self.BMat[i + 1])
            # print(self.zeta_array_dict[i].T * self.AMat[i] * self.BMat[i + 1])

            # 第i个观测点的各状态的概率 * (i -> i + 1)状态转移概率 * 当前状态的概率
            # self.zeta_array_dict[i].T * self.AMat[i] 是一个m * n的矩阵, self.BMat[i + 1]是一个n * 1的矩阵
            # m是i观测点的可选状态数, n是i+1观测点的可选状态数
            x = self.zeta_array_dict[i].T * self.AMat[i] * self.BMat[i + 1]
            print(x)
            # 找出当前每种状态的最大值, last_state_index是一个 n * 1的矩阵
            last_state_index = np.argmax(x, axis=0)

            # 记录
            self.psi_array_dict[i + 1] = last_state_index
            self.zeta_array_dict[i + 1] = np.array([x[last_state_index, [i for i in range(0, len(last_state_index))]]])

            # print(self.psi_array_dict[i + 1])
            # print(self.zeta_array_dict[i + 1])

        # 2.回溯
        print('回溯......')
        state_list = []
        last_max_state = -1
        if self.T == 1:
            return list(np.argmax(self.zeta_array_dict[0], axis=1))

        for i in range(self.T - 1, 0, -1):
            # print(i)
            if i == self.T - 1:
                # 回溯起点, 找出当前最大概率的状态
                start_max_state = np.argmax(self.zeta_array_dict[i])
                state_list.append(start_max_state)
                last_max_state = self.psi_array_dict[i][start_max_state]
                state_list.append(last_max_state)
            else:
                start_max_state = last_max_state
                last_max_state = self.psi_array_dict[i][start_max_state]
                state_list.append(last_max_state)
        state_list = state_list[::-1]
        return state_list


if __name__ == '__main__':

    # 0: 没分享
    # 1: 分享了
    state_map = {0: '海润C', 1: '新羽胜', 2: '亿利达'}

    test = Viterbi(observation_num=4,
                   t_mat_dict={0: np.array([[1, 1.2, 1.3],
                                            [1.2, 1.3, 1.4],
                                            [1.5, 1.4, 1.3]]),
                               1: np.array([[0.25, 0.4],
                                            [0.36, 0.3],
                                            [0.2, 0.26]]),
                               2: np.array([[0.50, 0.4, 0.30],
                                            [0.40, 0.8, 0.4]])
                               },

                   o_mat_dict={0: np.array([0.5, 0.2, 0.3]),
                               1: np.array([0.5, 0.5, 0.1]),
                               2: np.array([0.4, 0.2]),
                               3: np.array([0.1, 0.5, 0.1])})

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
