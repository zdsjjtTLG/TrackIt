# -- coding: utf-8 --
# @Time    : 2024/3/25 14:13
# @Author  : TangKai
# @Team    : ZheChengData

import os
import numpy as np
import pandas as pd
import multiprocessing
import geopandas as gpd
from .map.Net import Net
from .model.Para import ParaGrid
from .tools.group import cut_group
from .gps.LocGps import GpsPointsGdf
from .model.Markov import HiddenMarkov
from .GlobalVal import NetField, GpsField
from .visualization import export_visualization


gps_field = GpsField()
net_field = NetField()
agent_id_field = gps_field.AGENT_ID_FIELD
node_id_field = net_field.NODE_ID_FIELD


class MapMatch(object):
    def __init__(self, flag_name: str = 'test', net: Net = None, use_sub_net: bool = True,
                 time_format: str = "%Y-%m-%d %H:%M:%S", time_unit: str = 's',
                 gps_buffer: float = 200.0, use_node_restrict: bool = False, gps_route_buffer_gap: float = 15.0,
                 beta: float = 6.0, gps_sigma: float = 30.0, dis_para: float = 0.1,
                 is_lower_f: bool = False, lower_n: int = 2,
                 use_heading_inf: bool = False, heading_para_array: np.ndarray = None,
                 dense_gps: bool = True, dense_interval: float = 100.0,
                 dwell_l_length: float = 5.0, dwell_n: int = 2, del_dwell: bool = True,
                 dup_threshold: float = 10.0,
                 is_rolling_average: bool = False, window: int = 2,
                 export_html: bool = False, use_gps_source: bool = False, out_fldr: str = None,
                 export_geo_res: bool = False, top_k: int = 20, omitted_l: float = 6.0,
                 link_width: float = 1.5, node_radius: float = 1.5,
                 match_link_width: float = 5.0, gps_radius: float = 6.0, export_all_agents: bool = False,
                 visualization_cache_times: int = 10, multi_core_save: bool = False, instant_output: bool = False,
                 use_para_grid: bool = False, para_grid: ParaGrid = None, user_field_list: list[str] = None,
                 heading_vec_len: float = 15.0):
        """
        :param flag_name: 标记字符名称, 会用于标记输出的可视化文件, 默认"test"
        :param net: gotrackit路网对象, 必须指定
        :param use_sub_net: 是否在子网络上进行计算, 默认True
        :param time_format: GPS数据中时间列的格式, 默认"%Y-%m-%d %H:%M:%S"
        :param time_unit: GPS数据中时间列的单位, 如果时间列是数值(秒或者毫秒), 默认's'
        :param gps_buffer: GPS的搜索半径, 单位米, 意为只选取每个gps_buffer点附近100米范围内的路段作为候选路段, 默认100.0
        :param gps_route_buffer_gap: 半径增量, gps_buffer + gps_route_buffer_gap 的半径范围用于计算子网络, 默认25.0
        :param beta: 该值越大, 状态转移概率对于距离越不敏感, 默认20m
        :param gps_sigma: 该值越大, 发射概率对距离越不敏感, 默认20m
        :param dis_para: 距离的折减系数, 默认0.1
        :param is_lower_f: 是否对GPS数据进行数据降频率, 适用于: 高频-高定位误差 GPS数据, 默认False
        :param lower_n: 频率倍率, 默认2
        :param use_heading_inf: 是否利用GPS的差分方向向量修正发射概率, 适用于: 低定位误差 GPS数据 或者低频定位数据(配合加密参数), 默认False
        :param heading_para_array: 差分方向修正参数, 默认np.array([1.0, 1.0, 1.0, 0.1, 0.00001, 0.000001, 0.00001, 0.000001, 0.000001])
        :param dense_gps: 是否对GPS数据进行加密, 默认False
        :param dense_interval: 当前后GPS点的直线距离l超过dense_interval即进行加密, 进行 int(l / dense_interval) + 1 等分加密, 默认25.0
        :param dup_threshold: 利用GPS轨迹计算sub_net时, 先对GPS点原始轨迹做简化, 重复点检测阈值, 默认5m
        :param is_rolling_average: 是否启用滑动窗口平均对GPS数据进行降噪, 默认False
        :param window: 滑动窗口大小, 默认2
        :param export_html: 是否输出网页可视化结果html文件, 默认True
        :param use_gps_source: 是否在可视化结果中使用GPS源数据进行展示, 默认False
        :param out_fldr: 保存匹配结果的文件目录, 默认当前目录
        :param export_geo_res: 是否输出匹配结果的几何可视化文件, 默认False
        :param omitted_l: 当某GPS点与前后GPS点的平均距离小于该距离(m)时, 该GPS点的方向限制作用被取消
        :param gps_radius: HTML可视化中GPS点的半径大小，单位米，默认8米
        :param export_all_agents: 是否将所有agent的可视化存储于一个html文件中
        :param visualization_cache_times: 每匹配完几辆车再进行(html or geojson文件)结果的统一存储, 默认50
        :param multi_core_save: 是否启用多进程进行结果存储
        :param instant_output: 是否每匹配完一条轨迹就存储csv匹配结果
        :param use_para_grid: 是否启用网格参数搜索
        :param para_grid: 网格参数对象
        :param heading_vec_len: 匹配航向向量的长度(控制geojson中的可视化)
        :param user_field_list: gps数据中用户想要输出的额外字段
        """
        # 坐标系投影
        self.plain_crs = net.planar_crs
        self.geo_crs = net.geo_crs

        # gps参数
        self.time_format = time_format  # 时间列格式
        self.time_unit = time_unit  # 时间列单位
        self.is_lower_f = is_lower_f  # 是否降频率
        self.lower_n = lower_n  # 降频倍数
        self.del_dwell = del_dwell
        self.dwell_n = dwell_n
        self.dwell_l_length = dwell_l_length
        self.is_rolling_average = is_rolling_average  # 是否启用滑动窗口平均
        self.rolling_window = window  # 窗口大小
        self.dense_gps = dense_gps  # 是否加密GPS
        self.dense_interval = dense_interval  # 加密阈值(前后GPS点直线距离超过该值就会启用线性加密)
        self.gps_buffer = gps_buffer  # gps的关联范围(m)
        self.use_node_restrict = use_node_restrict
        self.use_sub_net = use_sub_net  # 是否启用子网络
        self.gps_route_buffer_gap = gps_route_buffer_gap
        self.use_heading_inf = use_heading_inf
        self.heading_para_array = heading_para_array
        if not omitted_l < dense_interval / 2:
            omitted_l = dense_interval / 2 - 0.1
        self.omitted_l = omitted_l
        self.top_k = top_k
        if not dup_threshold < (self.gps_buffer + self.gps_route_buffer_gap) / 3:
            dup_threshold = (self.gps_buffer + self.gps_route_buffer_gap) / 3 - 0.1
        self.dup_threshold = dup_threshold

        self.beta = beta  # 状态转移概率参数, 概率与之成正比
        self.gps_sigma = gps_sigma  # 发射概率参数, 概率与之成正比
        self.flag_name = flag_name
        self.dis_para = dis_para

        self.export_html = export_html
        self.export_geo_res = export_geo_res
        self.out_fldr = r'./' if out_fldr is None else out_fldr

        self.my_net = net
        self.not_conn_cost = self.my_net.not_conn_cost
        self.use_gps_source = use_gps_source

        # 网页可视化参数
        self.link_width = link_width
        self.node_radius = node_radius
        self.match_link_width = match_link_width
        self.gps_radius = gps_radius

        self.export_all_agents = export_all_agents
        self.visualization_cache_times = visualization_cache_times
        self.multi_core_save = multi_core_save
        self.sub_net_buffer = self.gps_buffer + self.gps_route_buffer_gap
        self.instant_output = instant_output

        self.use_para_grid = use_para_grid
        self.para_grid = para_grid
        self.heading_vec_len = heading_vec_len

        self.user_field_list = user_field_list

    def execute(self, gps_df: pd.DataFrame or gpd.GeoDataFrame = None) -> tuple[pd.DataFrame, dict, list]:
        """
        :param gps_df:
        :return:
        """
        # check and format
        self.node_restrict_format(gps_df=gps_df)
        user_field_list = GpsPointsGdf.check(gps_points_df=gps_df, user_field_list=self.user_field_list)

        match_res_df = pd.DataFrame()
        may_error_list = dict()
        error_list = list()
        hmm_res_list = []  # save hmm_res
        gps_df.dropna(subset=[agent_id_field], inplace=True)
        agent_num = len(gps_df[gps_field.AGENT_ID_FIELD].unique())
        if agent_num == 0:
            print('after removing the rows with empty values in the agent_id column, the gps data is empty...')
            return match_res_df, may_error_list, error_list

        # 对每辆车的轨迹进行匹配
        agent_count = 0
        add_single_ft = [True]

        for agent_id, _gps_df in gps_df.groupby(gps_field.AGENT_ID_FIELD):
            agent_count += 1
            print(rf'- gotrackit ------> No.{agent_count}: agent: {agent_id} ')
            try:
                gps_obj = GpsPointsGdf(gps_points_df=_gps_df, time_format=self.time_format,
                                       buffer=self.gps_buffer, time_unit=self.time_unit,
                                       plane_crs=self.plain_crs, user_filed_list=user_field_list)
            except Exception as e:
                print(f'error constructing GPS object: {repr(e)}')
                error_list.append(agent_id)
                continue

            del _gps_df

            # gps-process
            try:
                self.gps_reprocess(gps_obj=gps_obj,
                                   is_lower_f=self.is_lower_f, lower_n=self.lower_n,
                                   is_rolling_average=self.is_rolling_average, rolling_window=self.rolling_window,
                                   dense_gps=self.dense_gps, dense_interval=self.dense_interval,
                                   del_dwell=self.del_dwell, dwell_l_length=self.dwell_l_length, dwell_n=self.dwell_n)
            except Exception as e:
                print(rf'GPS data preprocessing error: {repr(e)}')
                error_list.append(agent_id)
                continue

            if len(gps_obj.gps_gdf) <= 1:
                print(r'after data preprocessing, there are less than 2 GPS observation points.')
                error_list.append(agent_id)
                continue

            # 依据当前的GPS数据(源数据)做一个子网络
            if self.use_sub_net:
                print(rf'using sub net')
                add_single_ft[0] = True
                used_net = self.my_net.create_computational_net(
                    gps_array_buffer=gps_obj.get_gps_array_buffer(buffer=self.sub_net_buffer,
                                                                  dup_threshold=self.dup_threshold),
                    fmm_cache=self.my_net.fmm_cache, weight_field=self.my_net.weight_field,
                    cache_path=self.my_net.cache_path, cache_id=self.my_net.cache_id,
                    not_conn_cost=self.my_net.not_conn_cost)
                if used_net is None:
                    error_list.append(agent_id)
                    continue
            else:
                used_net = self.my_net
                print(rf'using whole net')

            # 初始化一个隐马尔可夫模型
            hmm_obj = HiddenMarkov(net=used_net, gps_points=gps_obj, beta=self.beta, gps_sigma=self.gps_sigma,
                                   not_conn_cost=self.not_conn_cost, use_heading_inf=self.use_heading_inf,
                                   heading_para_array=self.heading_para_array, dis_para=self.dis_para,
                                   top_k=self.top_k, use_node_restrict=self.use_node_restrict,
                                   omitted_l=self.omitted_l, para_grid=self.para_grid,
                                   heading_vec_len=self.heading_vec_len)
            if not self.use_para_grid:
                is_success, _match_res_df = hmm_obj.hmm_execute(add_single_ft=add_single_ft)
            else:
                is_success, _match_res_df = hmm_obj.hmm_para_grid_execute(add_single_ft=add_single_ft,
                                                                          agent_id=agent_id)
                self.para_grid.init_para_grid()

            if not is_success:
                error_list.append(agent_id)
                continue
            hmm_obj.del_ft_trans()

            if hmm_obj.is_warn:
                may_error_list[agent_id] = hmm_obj.format_warn_info

            if self.instant_output:
                _match_res_df.to_csv(os.path.join(self.out_fldr, rf'{agent_id}_match_res.csv'),
                                     encoding='utf_8_sig', index=False)
            else:
                match_res_df = pd.concat([match_res_df, _match_res_df])

            # if export files
            if self.export_html or self.export_geo_res:
                hmm_res_list.append(hmm_obj)
                if len(hmm_res_list) >= self.visualization_cache_times or agent_count == agent_num:
                    export_visualization(hmm_obj_list=hmm_res_list, use_gps_source=self.use_gps_source,
                                         export_geo=self.export_geo_res, export_html=self.export_html,
                                         gps_radius=self.gps_radius, export_all_agents=self.export_all_agents,
                                         out_fldr=self.out_fldr, flag_name=self.flag_name,
                                         multi_core_save=self.multi_core_save, sub_net_buffer=self.sub_net_buffer,
                                         dup_threshold=self.dup_threshold)
                    del hmm_res_list
                    hmm_res_list = []
        if hmm_res_list:
            export_visualization(hmm_obj_list=hmm_res_list, use_gps_source=self.use_gps_source,
                                 export_geo=self.export_geo_res, export_html=self.export_html,
                                 gps_radius=self.gps_radius, export_all_agents=self.export_all_agents,
                                 out_fldr=self.out_fldr, flag_name=self.flag_name,
                                 multi_core_save=self.multi_core_save, sub_net_buffer=self.sub_net_buffer,
                                 dup_threshold=self.dup_threshold)
        return match_res_df, may_error_list, error_list

    def multi_core_execute(self, gps_df: pd.DataFrame or gpd.GeoDataFrame = None, core_num: int = 2) -> \
            tuple[pd.DataFrame, dict, list]:
        assert gps_field.AGENT_ID_FIELD in set(gps_df.columns), 'gps data is missing agent_id field'
        agent_id_list = list(gps_df[gps_field.AGENT_ID_FIELD].unique())
        core_num = os.cpu_count() if core_num > os.cpu_count() else core_num
        agent_group = cut_group(agent_id_list, n=core_num)
        n = len(agent_group)
        print(f'using multiprocessing - {n} cores')
        pool = multiprocessing.Pool(processes=n)
        result_list = []
        for i in range(0, n):
            core_out_fldr = os.path.join(self.out_fldr, rf'core{i}')
            if os.path.exists(core_out_fldr):
                pass
            else:
                os.makedirs(core_out_fldr)

            agent_id_list = agent_group[i]
            _gps_df = gps_df[gps_df[gps_field.AGENT_ID_FIELD].isin(agent_id_list)]
            mmp = MapMatch(net=self.my_net, use_sub_net=self.use_sub_net, time_format=self.time_format,
                           time_unit=self.time_unit, gps_buffer=self.gps_buffer,
                           gps_route_buffer_gap=self.gps_route_buffer_gap,
                           beta=self.beta,
                           gps_sigma=self.gps_sigma, dis_para=self.dis_para, is_lower_f=self.is_lower_f,
                           lower_n=self.lower_n,
                           use_heading_inf=self.use_heading_inf, heading_para_array=self.heading_para_array,
                           dense_gps=self.dense_gps,
                           dense_interval=self.dense_interval, dwell_l_length=self.dwell_l_length, dwell_n=self.dwell_n,
                           del_dwell=self.del_dwell,
                           dup_threshold=self.dup_threshold, is_rolling_average=self.is_rolling_average,
                           window=self.rolling_window,
                           export_html=self.export_html,
                           use_gps_source=self.use_gps_source, out_fldr=core_out_fldr,
                           export_geo_res=self.export_geo_res,
                           top_k=self.top_k, omitted_l=self.omitted_l, link_width=self.link_width,
                           node_radius=self.node_radius,
                           match_link_width=self.match_link_width, gps_radius=self.gps_radius,
                           export_all_agents=self.export_all_agents,
                           visualization_cache_times=self.visualization_cache_times,
                           multi_core_save=False, instant_output=self.instant_output, use_para_grid=self.use_para_grid,
                           para_grid=self.para_grid, user_field_list=self.user_field_list,
                           heading_vec_len=self.heading_vec_len)
            result = pool.apply_async(mmp.execute,
                                      args=(_gps_df, ))
            result_list.append(result)
        pool.close()
        pool.join()

        match_res, may_error, error = pd.DataFrame(), dict(), list()
        for res in result_list:
            _match_res, _may_error, _error = res.get()
            match_res = pd.concat([match_res, _match_res])
            may_error.update(_may_error)
            error.extend(_error)
        match_res.reset_index(inplace=True, drop=True)
        return match_res, may_error, error

    def gps_reprocess(self, gps_obj: GpsPointsGdf = None,
                      del_dwell: bool = True, dwell_l_length: float = 5.0, dwell_n: int = 2,
                      is_lower_f: bool = False, lower_n: int = 2,
                      is_rolling_average: bool = False, rolling_window: int = 2,
                      dense_gps: bool = True, dense_interval: float = 120.0) -> None:
        # gps-process
        if del_dwell:
            print(rf'gps-preprocessing: del dwell')
            gps_obj.del_dwell_points(dwell_l_length=dwell_l_length, dwell_n=dwell_n)

        # 降频处理
        if is_lower_f:
            print(rf'gps-preprocessing: lower frequency, size: {self.lower_n}')
            gps_obj.lower_frequency(lower_n=lower_n, multi_agents=False)

        if is_rolling_average:
            print(rf'gps-preprocessing: rolling average, window size: {self.rolling_window}')
            gps_obj.rolling_average(rolling_window=rolling_window, multi_agents=False)

        if dense_gps:
            print(rf'gps-preprocessing: dense gps by interval: {self.dense_interval}m')
            gps_obj.dense(dense_interval=dense_interval)

    def node_restrict_format(self, gps_df: pd.DataFrame = None):
        if self.use_node_restrict:
            assert node_id_field in gps_df.columns, 'turn on use_node_restrict, but no node_id field in data'
            if self.user_field_list is None:
                self.user_field_list = [node_id_field]
            if node_id_field not in self.user_field_list:
                self.user_field_list.append(node_id_field)

class OnLineMapMatch(MapMatch):
    def __init__(self, flag_name: str = 'test', net: Net = None, use_sub_net: bool = False,
                 time_format: str = "%Y-%m-%d %H:%M:%S", time_unit: str = 's',
                 gps_buffer: float = 200.0, gps_route_buffer_gap: float = 15.0,
                 beta: float = 6.0, gps_sigma: float = 30.0, dis_para: float = 0.1,
                 is_lower_f: bool = False, lower_n: int = 2,
                 use_heading_inf: bool = False, heading_para_array: np.ndarray = None,
                 dense_gps: bool = True, dense_interval: float = 100.0,
                 dwell_l_length: float = 5.0, dwell_n: int = 2, del_dwell: bool = True,
                 dup_threshold: float = 10.0,
                 is_rolling_average: bool = False, window: int = 2,
                 export_html: bool = False, use_gps_source: bool = False, out_fldr: str = None,
                 export_geo_res: bool = False, top_k: int = 20, omitted_l: float = 6.0,
                 link_width: float = 1.5, node_radius: float = 1.5,
                 match_link_width: float = 5.0, gps_radius: float = 6.0, export_all_agents: bool = False,
                 visualization_cache_times: int = 5, multi_core_save: bool = False, instant_output: bool = False,
                 user_field_list: list[str] = None, heading_vec_len: float = 15.0):
        MapMatch.__init__(self, flag_name=flag_name, net=net, use_sub_net=use_sub_net, time_format=time_format,
                          time_unit=time_unit, gps_buffer=gps_buffer, gps_route_buffer_gap=gps_route_buffer_gap,
                          beta=beta, gps_sigma=gps_sigma, dis_para=dis_para, is_lower_f=is_lower_f,
                          lower_n=lower_n, use_heading_inf=use_heading_inf, heading_para_array=heading_para_array,
                          dense_gps=dense_gps, dense_interval=dense_interval,
                          dwell_l_length=dwell_l_length, dwell_n=dwell_n, del_dwell=del_dwell,
                          dup_threshold=dup_threshold, is_rolling_average=is_rolling_average, window=window,
                          export_html=export_html, use_gps_source=use_gps_source, out_fldr=out_fldr,
                          export_geo_res=export_geo_res, top_k=top_k, omitted_l=omitted_l, gps_radius=gps_radius,
                          link_width=link_width, node_radius=node_radius, match_link_width=match_link_width,
                          export_all_agents=export_all_agents, visualization_cache_times=visualization_cache_times,
                          multi_core_save=multi_core_save, instant_output=instant_output,
                          user_field_list=user_field_list, heading_vec_len=heading_vec_len)
        self.his_hmm_dict = dict()
        self.his_gps = dict()
        self.add_single_ft = [True]

    def execute(self, gps_df: pd.DataFrame or gpd.GeoDataFrame = None,
                gps_already_plain: bool = False, time_gap_threshold: float = 1800.0,
                dis_gap_threshold: float = 600.0,
                overlapping_window: int = 3) -> tuple[pd.DataFrame, dict, list]:
        """

        :param gps_df:
        :param gps_already_plain: gps数据的lng, lat列是否已经是平面投影坐标系
        :param time_gap_threshold: 时间间隔阈值
        :param dis_gap_threshold: 距离阈值
        :param overlapping_window: 重叠窗口长度
        :return:
        """
        # check and format
        self.is_rolling_average = False
        # self.del_dwell = False
        user_field_list = GpsPointsGdf.check(gps_points_df=gps_df, user_field_list=self.user_field_list)
        may_error_list = dict()
        error_list = list()
        match_res_df = pd.DataFrame()
        hmm_res_list = []  # save hmm_res
        gps_df.dropna(subset=[agent_id_field], inplace=True)
        agent_num = len(gps_df[gps_field.AGENT_ID_FIELD].unique())
        if agent_num == 0:
            print('after removing the rows with empty values in the agent_id column, the gps data is empty...')
            return match_res_df, may_error_list, error_list

        # 对每辆车的轨迹进行匹配
        agent_count = 0
        # self.add_single_ft = [True]

        for agent_id, _gps_df in gps_df.groupby(gps_field.AGENT_ID_FIELD):
            cor_with_his = False
            agent_count += 1
            if agent_id in self.his_hmm_dict.keys():
                self.gps_buffer = self.his_hmm_dict[agent_id].gps_points.buffer
            else:
                if len(_gps_df) <= 1 and agent_id not in self.his_gps.keys():
                    self.his_gps[agent_id] = _gps_df.copy()
                    continue
                else:
                    if agent_id in self.his_gps.keys():
                        _gps_df = pd.concat([self.his_gps[agent_id], _gps_df])
                        del self.his_gps[agent_id]
                    _gps_df.reset_index(inplace=True, drop=True)

            print(rf'- gotrackit ------> No.{agent_count}: agent: {agent_id} ')
            try:
                gps_obj = GpsPointsGdf(gps_points_df=_gps_df, time_format=self.time_format,
                                       buffer=self.gps_buffer, time_unit=self.time_unit,
                                       plane_crs=self.plain_crs,
                                       user_filed_list=user_field_list, already_plain=gps_already_plain)
            except Exception as e:
                print(f'error constructing GPS object: {repr(e)}')
                error_list.append(agent_id)
                continue
            del _gps_df
            # judge if the same chain with his agent
            last_seq_list = list()
            if agent_id in self.his_hmm_dict.keys():
                cor_with_his, last_seq_list = gps_obj.merge_gps(
                    gps_obj=self.his_hmm_dict[agent_id].gps_points,
                    depth=overlapping_window,
                    dis_gap_threshold=dis_gap_threshold,
                    time_gap_threshold=time_gap_threshold)

            # gps-process
            try:
                self.gps_reprocess(gps_obj=gps_obj,
                                   is_lower_f=self.is_lower_f, lower_n=self.lower_n,
                                   is_rolling_average=self.is_rolling_average, rolling_window=self.rolling_window,
                                   dense_gps=self.dense_gps, dense_interval=self.dense_interval,
                                   del_dwell=self.del_dwell, dwell_l_length=self.dwell_l_length, dwell_n=self.dwell_n)
            except Exception as e:
                print(rf'GPS data preprocessing error: {repr(e)}')
                error_list.append(agent_id)
                continue

            if len(gps_obj.gps_gdf) <= 1:
                print(r'after data preprocessing, there are less than 2 GPS observation points.')
                error_list.append(agent_id)
                continue

            # 依据当前的GPS数据(源数据)做一个子网络
            if self.use_sub_net:
                print(rf'using sub net')
                self.add_single_ft[0] = True
                used_net = self.my_net.create_computational_net(
                    gps_array_buffer=gps_obj.get_gps_array_buffer(buffer=self.sub_net_buffer,
                                                                  dup_threshold=self.dup_threshold),
                    fmm_cache=self.my_net.fmm_cache, weight_field=self.my_net.weight_field,
                    cache_path=self.my_net.cache_path, cache_id=self.my_net.cache_id,
                    not_conn_cost=self.my_net.not_conn_cost)
                if used_net is None:
                    error_list.append(agent_id)
                    continue
            else:
                used_net = self.my_net
                print(rf'using whole net')

            hmm_obj = HiddenMarkov(net=used_net, gps_points=gps_obj, beta=self.beta,
                                   gps_sigma=self.gps_sigma,
                                   not_conn_cost=self.not_conn_cost,
                                   use_heading_inf=self.use_heading_inf,
                                   heading_para_array=self.heading_para_array,
                                   dis_para=self.dis_para,
                                   top_k=self.top_k, omitted_l=self.omitted_l,
                                   para_grid=self.para_grid,
                                   heading_vec_len=self.heading_vec_len, flag_name=self.flag_name,
                                   out_fldr=self.out_fldr)
            his_emission = dict()
            his_ft_idx_map = pd.DataFrame()
            if cor_with_his:
                his_emission = self.his_hmm_dict[agent_id].extract_emission(seq_list=last_seq_list)
                his_ft_idx_map = self.his_hmm_dict[agent_id].get_ft_idx_map

            is_success, _match_res_df = \
                hmm_obj.hmm_rt_execute(add_single_ft=self.add_single_ft, last_em_para=his_emission,
                                       last_seq_list=last_seq_list, his_ft_idx_map=his_ft_idx_map)
            if not is_success:
                error_list.append(agent_id)
                continue
            hmm_obj.del_ft_trans()

            self.his_hmm_dict[agent_id] = hmm_obj

            if hmm_obj.is_warn:
                may_error_list[agent_id] = hmm_obj.format_warn_info

            if self.instant_output:
                _match_res_df.to_csv(os.path.join(self.out_fldr, rf'{agent_id}_match_res.csv'),
                                     encoding='utf_8_sig', index=False)
            else:
                match_res_df = pd.concat([match_res_df, _match_res_df])

            # if export files
            if self.export_html or self.export_geo_res:
                hmm_res_list.append(hmm_obj)
                if len(hmm_res_list) >= self.visualization_cache_times or agent_count == agent_num:
                    export_visualization(hmm_obj_list=hmm_res_list, use_gps_source=self.use_gps_source,
                                         export_geo=self.export_geo_res, export_html=self.export_html,
                                         gps_radius=self.gps_radius, export_all_agents=self.export_all_agents,
                                         out_fldr=self.out_fldr, flag_name=self.flag_name,
                                         multi_core_save=self.multi_core_save, sub_net_buffer=self.sub_net_buffer,
                                         dup_threshold=self.dup_threshold)
                    del hmm_res_list
                    hmm_res_list = []

        if hmm_res_list:
            export_visualization(hmm_obj_list=hmm_res_list, use_gps_source=self.use_gps_source,
                                 export_geo=self.export_geo_res, export_html=self.export_html,
                                 gps_radius=self.gps_radius, export_all_agents=self.export_all_agents,
                                 out_fldr=self.out_fldr, flag_name=self.flag_name,
                                 multi_core_save=self.multi_core_save, sub_net_buffer=self.sub_net_buffer,
                                 dup_threshold=self.dup_threshold)
        return match_res_df, may_error_list, error_list
