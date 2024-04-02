# -- coding: utf-8 --
# @Time    : 2024/3/25 14:13
# @Author  : TangKai
# @Team    : ZheChengData


import numpy as np
import pandas as pd
from .map.Net import Net
from .gps.LocGps import GpsPointsGdf
from .model.Markov import HiddenMarkov
from .GlobalVal import NetField, GpsField
from .visualization import VisualizationCombination

gps_field = GpsField()
net_field = NetField()
agent_if_field = gps_field.AGENT_ID_FIELD
node_id_field = net_field.NODE_ID_FIELD


class MapMatch(object):
    def __init__(self, flag_name: str = 'test', net: Net = None, use_sub_net: bool = True, gps_df: pd.DataFrame = None,
                 time_format: str = "%Y-%m-%d %H:%M:%S", time_unit: str = 's',
                 gps_buffer: float = 90, gps_route_buffer_gap: float = 25.0,
                 max_increment_times: int = 2, increment_buffer: float = 20.0,
                 beta: float = 20.0, gps_sigma: float = 20.0, dis_para: float = 0.1,
                 is_lower_f: bool = False, lower_n: int = 2,
                 use_heading_inf: bool = False, heading_para_array: np.ndarray = None,
                 dense_gps: bool = True, dense_interval: float = 25.0,
                 is_rolling_average: bool = False, window: int = 2,
                 export_html: bool = False, use_gps_source: bool = False, html_fldr: str = None,
                 export_geo_res: bool = False, geo_res_fldr: str = None,
                 node_num_threshold: int = 2000, top_k: int = 20):
        """

        :param flag_name: 标记字符名称, 会用于标记输出的可视化文件, 默认"test"
        :param net: gotrackit路网对象, 必须指定
        :param use_sub_net: 是否在子网络上进行计算, 默认True
        :param gps_df: GPS数据, 必须指定
        :param time_format: GPS数据中时间列的格式, 默认"%Y-%m-%d %H:%M:%S"
        :param time_unit: GPS数据中时间列的单位, 如果时间列是数值(秒或者毫秒), 默认's'
        :param gps_buffer: GPS的搜索半径, 单位米, 意为只选取每个gps_buffer点附近100米范围内的路段作为候选路段, 默认90.0
        :param gps_route_buffer_gap: 半径增量, gps_buffer + gps_route_buffer_gap 的半径范围用于计算子网络, 默认25.0
        :param max_increment_times: 增量搜索次数, 默认2
        :param increment_buffer: 增量半径, 默认20.0
        :param beta: 该值越大, 状态转移概率对于距离越不敏感, 默认20m
        :param gps_sigma: 该值越大, 发射概率对距离越不敏感, 默认20m
        :param dis_para: 距离的折减系数, 默认0.1
        :param is_lower_f: 是否对GPS数据进行数据降频率, 适用于: 高频-高定位误差 GPS数据, 默认False
        :param lower_n: 频率倍率, 默认2
        :param use_heading_inf: 是否利用GPS的差分方向向量修正发射概率, 适用于: 低定位误差 GPS数据 或者低频定位数据(配合加密参数), 默认False
        :param heading_para_array: 差分方向修正参数, 默认np.array([1.0, 1.0, 1.0, 0.1, 0.00001, 0.000001, 0.00001, 0.000001, 0.000001])
        :param dense_gps: 是否对GPS数据进行加密, 默认False
        :param dense_interval: 当前后GPS点的直线距离l超过dense_interval即进行加密, 进行 int(l / dense_interval) + 1 等分加密, 默认25.0
        :param is_rolling_average: 是否启用滑动窗口平均对GPS数据进行降噪, 默认False
        :param window: 滑动窗口大小, 默认2
        :param export_html: 是否输出网页可视化结果html文件, 默认True
        :param use_gps_source: 是否在可视化结果中使用GPS源数据进行展示, 默认False
        :param html_fldr: 保存网页可视化结果的文件目录, 默认当前目录
        :param export_geo_res: 是否输出匹配结果的几何可视化文件, 默认False
        :param geo_res_fldr: 存储几何可视化文件的目录, 默认当前目录
        :param node_num_threshold: 默认2000
        """
        # 坐标系投影
        self.plain_crs = net.planar_crs
        self.geo_crs = net.geo_crs

        # 用于自动确定是否使用全局路网的指标
        self.node_num_threshold = node_num_threshold

        # gps参数
        self.gps_df = gps_df
        self.time_format = time_format  # 时间列格式
        self.time_unit = time_unit  # 时间列单位
        self.is_lower_f = is_lower_f  # 是否降频率
        self.lower_n = lower_n  # 降频倍数
        self.is_rolling_average = is_rolling_average  # 是否启用滑动窗口平均
        self.rolling_window = window  # 窗口大小
        self.dense_gps = dense_gps  # 是否加密GPS
        self.dense_interval = dense_interval  # 加密阈值(前后GPS点直线距离超过该值就会启用线性加密)
        self.gps_buffer = gps_buffer  # gps的关联范围(m)
        self.use_sub_net = use_sub_net  # 是否启用子网络
        self.max_increment_times = max_increment_times
        self.increment_buffer = increment_buffer
        self.gps_route_buffer_gap = gps_route_buffer_gap
        self.use_heading_inf = use_heading_inf
        self.heading_para_array = heading_para_array
        self.top_k = top_k

        self.beta = beta  # 状态转移概率参数, 概率与之成正比
        self.gps_sigma = gps_sigma  # 发射概率参数, 概率与之成正比
        self.flag_name = flag_name
        self.dis_para = dis_para

        self.export_html = export_html
        self.export_geo_res = export_geo_res
        self.geo_res_fldr = geo_res_fldr
        self.html_fldr = html_fldr

        self.may_error_list = dict()

        self.my_net = net
        self.not_conn_cost = self.my_net.not_conn_cost
        self.use_gps_source = use_gps_source

    def execute(self):

        match_res_df = pd.DataFrame()
        if len(self.my_net.get_node_data()[node_id_field].unique()) <= self.node_num_threshold:
            self.use_sub_net = False

        # 对每辆车的轨迹进行匹配
        for agent_id, _gps_df in self.gps_df.groupby(gps_field.AGENT_ID_FIELD):
            file_name = '-'.join([self.flag_name, str(agent_id)])
            print(rf'agent: {agent_id}')
            _gps_df.reset_index(inplace=True, drop=True)
            gps_obj = GpsPointsGdf(gps_points_df=_gps_df, time_format=self.time_format,
                                   buffer=self.gps_buffer, time_unit=self.time_unit,
                                   plane_crs=self.plain_crs,
                                   max_increment_times=self.max_increment_times, increment_buffer=self.increment_buffer,
                                   dense_gps=self.dense_gps, dense_interval=self.dense_interval)
            del _gps_df
            # 降频处理
            if self.is_lower_f:
                print(rf'lower {self.lower_n} - frequency')
                gps_obj.lower_frequency(n=self.lower_n)

            if self.is_rolling_average:
                print(rf'rolling average by window size - {self.rolling_window}')
                gps_obj.rolling_average(window=self.rolling_window)

            if self.dense_gps:
                print(rf'dense gps by interval - {self.dense_interval}m')
                gps_obj.dense()

            # 依据当前的GPS数据(源数据)做一个子网络
            if self.use_sub_net:
                print(rf'using sub net')
                sub_net = self.my_net.create_computational_net(
                    gps_array_buffer=gps_obj.get_gps_array_buffer(buffer=self.gps_buffer + self.gps_route_buffer_gap))
                # 初始化一个隐马尔可夫模型
                hmm_obj = HiddenMarkov(net=sub_net, gps_points=gps_obj, beta=self.beta, gps_sigma=self.gps_sigma,
                                       not_conn_cost=self.not_conn_cost, use_heading_inf=self.use_heading_inf,
                                       heading_para_array=self.heading_para_array, dis_para=self.dis_para,
                                       top_k=self.top_k)
            else:
                print(rf'using whole net')
                hmm_obj = HiddenMarkov(net=self.my_net, gps_points=gps_obj, beta=self.beta, gps_sigma=self.gps_sigma,
                                       not_conn_cost=self.not_conn_cost, use_heading_inf=self.use_heading_inf,
                                       heading_para_array=self.heading_para_array, dis_para=self.dis_para,
                                       top_k=self.top_k)

            # 求解参数
            hmm_obj.generate_markov_para()
            hmm_obj.solve()
            _match_res_df = hmm_obj.acquire_res()
            if hmm_obj.is_warn:
                self.may_error_list[agent_id] = hmm_obj.warn_info

            if self.export_geo_res:
                hmm_obj.acquire_geo_res(out_fldr=self.geo_res_fldr,
                                        flag_name=file_name)

            match_res_df = pd.concat([match_res_df, _match_res_df])

            if self.export_html:
                # 4.初始化一个匹配结果管理器
                vc = VisualizationCombination(use_gps_source=self.use_gps_source)
                vc.collect_hmm(hmm_obj)
                vc.visualization(zoom=15, out_fldr=self.html_fldr,
                                 file_name=file_name)
                del vc
        match_res_df.reset_index(inplace=True, drop=True)
        return match_res_df, self.may_error_list
