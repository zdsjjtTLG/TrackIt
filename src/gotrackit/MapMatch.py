# -- coding: utf-8 --
# @Time    : 2024/3/25 14:13
# @Author  : TangKai
# @Team    : ZheChengData


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
    def __init__(self, net: Net = None,
                 node_num_threshold: int = 2000, gps_df: pd.DataFrame = None, time_format: str = "%Y-%m-%d %H:%M:%S",
                 time_unit: str = 's', max_increment_times: int = 2, increment_buffer: bool = 20.0,
                 is_lower_f: bool = False, lower_n: int = 2, is_rolling_average: bool = False, window: int = 2,
                 gps_buffer: float = 90, use_sub_net: bool = True, gps_route_buffer_gap: float = 25.0,
                 beta: float = 20.2, gps_sigma: float = 20.0, flag_name: str = 'test',
                 export_html: bool = False, export_geo_res: bool = False, geo_res_fldr: str = None,
                 html_fldr: str = None, dense_gps: bool = True, dense_interval: float = 25.0,
                 use_gps_source: bool = False, use_heading_inf: bool = True):

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

        self.beta = beta  # 状态转移概率参数, 概率与之成正比
        self.gps_sigma = gps_sigma  # 发射概率参数, 概率与之成正比
        self.flag_name = flag_name

        self.export_html = export_html
        self.export_geo_res = export_geo_res
        self.geo_res_fldr = geo_res_fldr
        self.html_fldr = html_fldr

        self.may_error_list = []

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
                                       not_conn_cost=self.not_conn_cost, use_heading_inf=self.use_heading_inf)
            else:
                print(rf'using whole net')
                hmm_obj = HiddenMarkov(net=self.my_net, gps_points=gps_obj, beta=self.beta, gps_sigma=self.gps_sigma,
                                       not_conn_cost=self.not_conn_cost, use_heading_inf=self.use_heading_inf)

            # 求解参数
            hmm_obj.generate_markov_para()
            hmm_obj.solve()
            _match_res_df = hmm_obj.acquire_res()
            if hmm_obj.is_warn:
                self.may_error_list.append(agent_id)

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
