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
        """TripGeneration类初始化

        Args:
            net: 路网对象
            time_step: 仿真步长(s)
            speed_miu: 仿真速度均值(m/s)
            speed_sigma: 仿真速度标准差(m/s)
            save_gap: 每多少仿真步保存一次真实轨迹数据，整数
            loc_frequency: 每多少s进行一次GPS定位，该值必须大于仿真步长
            loc_error_sigma: 定位误差标准差(m)
            loc_error_miu: 定位误差均值(m)
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
        """TripGeneration的类方法：generate_rand_trips函数
        - 生成随机路径的trip

        Args:
            trip_num:
            instant_output: 是否即时输出，即是否每计算完一次行程就存储GPS数据文件、逐帧轨迹文件
            out_fldr: 存储输出文件的目录
            time_format: 输出GPS数据的时间列的格式
            agent_flag: 标志字符
            start_year: 车辆出发的起始年
            start_month: 车辆出发的起始月
            start_day: 车辆出发的起始日
            start_hour: 车辆出发的起始小时
            start_minute: 车辆出发的起始分钟
            start_second: 车辆出发的起始秒
            file_type: 输出文件的类型，shp或者geojson

        Returns:
            真实轨迹数据表、GPS轨迹数据表、混合数据表
        """

        return self.__generate_trips(trip_num=trip_num, instant_output=instant_output, out_fldr=out_fldr,
                                     time_format=time_format, agent_flag=agent_flag, start_year=start_year,
                                     start_month=start_month, start_day=start_day, start_hour=start_hour,
                                     start_minute=start_minute, start_second=start_second, file_type=file_type)

    def generate_od_trips(self, od_set: list or tuple = None, instant_output: bool = False,
                          out_fldr: str = r'./', time_format: str = "%Y-%m-%d %H:%M:%S", agent_flag: str = 'agent',
                          start_year: int = 2022, start_month: int = 5, start_day: int = 15, start_hour: int = 10,
                          start_minute: int = 20, start_second: int = 12,
                          file_type: str = 'geojson'):
        """TripGeneration的类方法：generate_od_trips函数
        - 生成指定OD路径的trip

        Args:
            od_set: 起始节点OD对，按照起始节点进行路径搜索后生成trip，形如[(o_node, d_node), …] 例如[(12, 23), (34, 111)]，表示生成2个trip，分别为节点12到节点23的最短路径、节点34到节点111的最短路径
            instant_output: 是否即时输出，即是否每计算完一次行程就存储GPS数据文件、逐帧轨迹文件
            out_fldr: 存储输出文件的目录
            time_format: 输出GPS数据的时间列的格式
            agent_flag: 标志字符
            start_year: 车辆出发的起始年
            start_month: 车辆出发的起始月
            start_day: 车辆出发的起始日
            start_hour: 车辆出发的起始小时
            start_minute: 车辆出发的起始分钟
            start_second: 车辆出发的起始秒
            file_type: 输出文件的类型，shp或者geojson

        Returns:
            真实轨迹数据表、GPS轨迹数据表、混合数据表
        """
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
        """TripGeneration的类方法：generate_destined_trips函数
        - 生成指定路径的trip

        Args:
            node_paths: 路径节点序列，形如[[node1, node2, node3, …], [node5, node7, node9, …],…] 例如[(12, 90, 9012, 123), (3412, 23, 112, 23), (34, 344, 111)]，表示生成3个trip，每个trip的节点序列已经指定好
            instant_output: 是否即时输出，即是否每计算完一次行程就存储GPS数据文件、逐帧轨迹文件
            out_fldr: 存储输出文件的目录
            time_format: 输出GPS数据的时间列的格式
            agent_flag: 标志字符
            start_year: 车辆出发的起始年
            start_month: 车辆出发的起始月
            start_day: 车辆出发的起始日
            start_hour: 车辆出发的起始小时
            start_minute: 车辆出发的起始分钟
            start_second: 车辆出发的起始秒
            file_type: 输出文件的类型，shp或者geojson

        Returns:
            真实轨迹数据表、GPS轨迹数据表、混合数据表
        """
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
