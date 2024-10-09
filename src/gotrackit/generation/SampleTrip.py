# -- coding: utf-8 --
# @Time    : 2024/5/26 19:56
# @Author  : TangKai
# @Team    : ZheChengData

import datetime
from ..map import Net
from tqdm import tqdm
import geopandas as gpd
from .GpsGen import Route
from .GpsGen import Car, RouteInfoCollector


class TripGeneration(object):
    def __init__(self, net: Net = None, time_step: float = 0.1, speed_miu: float = 12.0,
                 speed_sigma: float = 3.6, save_gap: int = 1, loc_frequency: float = 2.0,
                 loc_error_sigma: float = 40.0, loc_error_miu: float = 0.0):
        """

        :param net: 路网对象
        :param time_step: 仿真步长(s)
        :param speed_miu: 仿真速度均值(m/s)
        :param speed_sigma: 仿真速度标准差(m/s)
        :param save_gap: 每多少仿真步保存一次真实轨迹数据
        :param loc_frequency: 每多少s进行一次GPS定位
        :param loc_error_sigma: 定位误差均值(m)
        :param loc_error_miu: 定位误差标准差(m)
        """
        self.net = net
        self.time_step = time_step
        self.speed_miu = speed_miu
        self.speed_sigma = speed_sigma
        self.save_gap = save_gap
        self.loc_frequency = loc_frequency
        self.loc_error_sigma = loc_error_sigma
        self.loc_error_miu = loc_error_miu

    def generate_rand_trips(self, trip_num: int = 10, instant_output: bool = False,
                            out_fldr: str = r'./', time_format: str = "%Y-%m-%d %H:%M:%S", agent_flag: str = 'agent',
                            start_year: int = 2022, start_month: int = 5, start_day: int = 15, start_hour: int = 10,
                            start_minute: int = 20, start_second: int = 12,
                            file_type: str = 'geojson'):

        return self.__generate_trips(trip_num=trip_num, instant_output=instant_output, out_fldr=out_fldr,
                                     time_format=time_format, agent_flag=agent_flag, start_year=start_year,
                                     start_month=start_month, start_day=start_day, start_hour=start_hour,
                                     start_minute=start_minute, start_second=start_second, file_type=file_type)

    def generate_od_trips(self, od_set: list or tuple = None, instant_output: bool = False,
                          out_fldr: str = r'./', time_format: str = "%Y-%m-%d %H:%M:%S", agent_flag: str = 'agent',
                          start_year: int = 2022, start_month: int = 5, start_day: int = 15, start_hour: int = 10,
                          start_minute: int = 20, start_second: int = 12,
                          file_type: str = 'geojson'):
        if od_set is None or not od_set:
            return None
        return self.__generate_trips(od_set=od_set, instant_output=instant_output, out_fldr=out_fldr,
                                     time_format=time_format, agent_flag=agent_flag, start_year=start_year,
                                     start_month=start_month, start_day=start_day, start_hour=start_hour,
                                     start_minute=start_minute, start_second=start_second, file_type=file_type)

    def generate_destined_trips(self, node_paths: list or tuple = None, instant_output: bool = False,
                                out_fldr: str = r'./', time_format: str = "%Y-%m-%d %H:%M:%S",
                                agent_flag: str = 'agent',
                                start_year: int = 2022, start_month: int = 5, start_day: int = 15, start_hour: int = 10,
                                start_minute: int = 20, start_second: int = 12,
                                file_type: str = 'geojson'):
        if node_paths is None or not node_paths:
            return None
        return self.__generate_trips(node_paths=node_paths, instant_output=instant_output, out_fldr=out_fldr,
                                     time_format=time_format, agent_flag=agent_flag, start_year=start_year,
                                     start_month=start_month, start_day=start_day, start_hour=start_hour,
                                     start_minute=start_minute, start_second=start_second, file_type=file_type)

    def __generate_trips(self, trip_num: int = 10, instant_output: bool = False,
                         out_fldr: str = r'./', time_format: str = "%Y-%m-%d %H:%M:%S", agent_flag: str = 'agent',
                         start_year: int = 2022, start_month: int = 5, start_day: int = 15, start_hour: int = 10,
                         start_minute: int = 20, start_second: int = 12, od_set: list or tuple = None,
                         node_paths: list or tuple = None, file_type: str = 'geojson'):

        trajectory_gdf, gps_gdf, mix_gdf = gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()
        data_col = RouteInfoCollector(from_crs=self.net.planar_crs, to_crs=self.net.geo_crs, convert_prj_sys=True)

        if od_set:
            trip_num = len(od_set)
        if node_paths:
            trip_num = len(node_paths)

        # 开始行车
        for i in tqdm(range(trip_num), desc=rf'TripGeneration', ncols=100):
            # 新建车对象, 分配一个车辆ID, 配备一个Net和一个Route, 并且设置仿真参数
            car_id = rf'{agent_flag}_{i + 1}'
            car = Car(net=self.net, time_step=self.time_step, save_log=False,
                      route=Route(net=self.net),
                      agent_id=car_id, speed_miu=self.speed_miu, speed_sigma=self.speed_sigma,
                      loc_frequency=self.loc_frequency, loc_error_sigma=self.loc_error_sigma,
                      loc_error_miu=self.loc_error_miu,
                      start_time=datetime.datetime(year=start_year, month=start_month, day=start_day,
                                                   hour=start_hour, minute=start_minute, second=start_second),
                      save_gap=self.save_gap)

            if od_set:
                car.route = Route(o_node=od_set[i][0], d_node=od_set[i][1], net=self.net)
            else:
                if node_paths:
                    node_seq = node_paths[i]
                    car.route = Route(net=self.net,
                                      ft_seq=[(node_seq[i], node_seq[i + 1]) for i in range(len(node_seq) - 1)])

            # 开始行车
            try:
                is_success = car.start_drive()
                if not is_success:
                    continue
            except:
                continue

            # 收集数据
            data_col.collect_trajectory(car.get_trajectory_info())
            data_col.collect_gps(car.get_gps_loc_info())
            if instant_output:
                trajectory_gdf = data_col.save_trajectory(file_type=file_type,
                                                          out_fldr=out_fldr,
                                                          file_name=rf'{car_id}-trajectory',
                                                          time_format=time_format)
                gps_gdf = data_col.save_gps_info(file_type=file_type, out_fldr=out_fldr,
                                                 file_name=rf'{car_id}-gps',
                                                 time_format=time_format)
                mix_gdf = data_col.save_mix_info(file_type='geojson', out_fldr=out_fldr,
                                                 file_name=rf'{car_id}-mix',
                                                 time_format=time_format)
                data_col = RouteInfoCollector(from_crs=self.net.planar_crs, to_crs=self.net.geo_crs,
                                              convert_prj_sys=True)
        if not instant_output:
            trajectory_gdf = data_col.save_trajectory(file_type=file_type, out_fldr=out_fldr,
                                                      file_name='-'.join([agent_flag, 'trajectory']),
                                                      time_format=time_format)
            gps_gdf = data_col.save_gps_info(file_type=file_type, out_fldr=out_fldr,
                                             file_name='-'.join([agent_flag, 'gps']),
                                             time_format=time_format)
            mix_gdf = data_col.save_mix_info(file_type=file_type, out_fldr=out_fldr,
                                             file_name='-'.join([agent_flag, 'mix']),
                                             time_format=time_format)
        return trajectory_gdf, gps_gdf, mix_gdf
