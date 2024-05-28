# -- coding: utf-8 --
# @Time    : 2024/5/26 19:56
# @Author  : TangKai
# @Team    : ZheChengData


import datetime
from ..map import Net
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
                            out_fldr: str = r'./', time_format: str = "%Y-%m-%d %H:%M:%S", agent_flag: str = 'agent'):
        trajectory_gdf, gps_gdf, mix_gdf = gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()
        data_col = RouteInfoCollector(from_crs=self.net.planar_crs, to_crs=self.net.geo_crs, convert_prj_sys=True)

        # 开始行车
        for car_id in [rf'{agent_flag}_{i}' for i in range(1, trip_num + 1)]:
            # 新建车对象, 分配一个车辆ID, 配备一个Net和一个Route, 并且设置仿真参数
            car = Car(net=self.net, time_step=self.time_step, save_log=False,
                      route=Route(net=self.net),
                      agent_id=car_id, speed_miu=self.speed_miu, speed_sigma=self.speed_sigma,
                      loc_frequency=self.loc_frequency, loc_error_sigma=self.loc_error_sigma,
                      loc_error_miu=self.loc_error_miu,
                      start_time=datetime.datetime(year=2022, month=5, day=12, hour=16, minute=14, second=0),
                      save_gap=self.save_gap)

            # 开始行车
            car.start_drive()

            # 收集数据
            data_col.collect_trajectory(car.get_trajectory_info())
            data_col.collect_gps(car.get_gps_loc_info())
            if instant_output:
                trajectory_gdf = data_col.save_trajectory(file_type='geojson',
                                                          out_fldr=out_fldr,
                                                          file_name=rf'{car_id}-trajectory',
                                                          time_format=time_format)
                gps_gdf = data_col.save_gps_info(file_type='geojson', out_fldr=out_fldr,
                                                 file_name=rf'{car_id}-gps',
                                                 time_format=time_format)
                mix_gdf = data_col.save_mix_info(file_type='geojson', out_fldr=out_fldr,
                                                 file_name=rf'{car_id}-mix',
                                                 time_format=time_format)
                data_col = RouteInfoCollector(from_crs=self.net.planar_crs, to_crs=self.net.geo_crs,
                                              convert_prj_sys=True)
        if not instant_output:
            trajectory_gdf = data_col.save_trajectory(file_type='geojson', out_fldr=out_fldr,
                                                      file_name='-'.join([agent_flag, 'trajectory']),
                                                      time_format=time_format)
            gps_gdf = data_col.save_gps_info(file_type='geojson', out_fldr=out_fldr,
                                             file_name='-'.join([agent_flag, 'gps']),
                                             time_format=time_format)
            mix_gdf = data_col.save_mix_info(file_type='geojson', out_fldr=out_fldr,
                                             file_name='-'.join([agent_flag, 'mix']),
                                             time_format=time_format)
        return trajectory_gdf, gps_gdf, mix_gdf
