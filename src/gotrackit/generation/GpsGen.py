# -- coding: utf-8 --
# @Time    : 2023/12/30 12:40
# @Author  : TangKai
# @Team    : ZheChengData

"""模拟车辆在路径上行驶, 并生成GPS定位信息"""

import sys
import logging
import os.path
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from ..map.Net import Net
from datetime import timedelta
from ..WrapsFunc import function_time_cost
from ..GlobalVal import GpsField, NetField
from ..tools.coord_trans import LngLatTransfer
from shapely.geometry import LineString, Point


gps_field = GpsField()
net_field = NetField()


class Route(object):
    """路径类"""
    def __init__(self, net: Net = None, o_node: int = None, d_node: int = None, ft_seq: list[tuple[int, int]] = None):
        """
        若不指定ft_seq, 则使用o_node -> d_node进行搜录获取路径, 若没有指定o_node和d_node则使用随机路径
        :param net:
        :param o_node: 起点节点ID
        :param d_node: 终点节点ID
        :param ft_seq: 路径节点序列, [(12, 24), (24, 121), (121, 90)...]
        """
        self.net = net
        self._o_node = o_node
        self._d_node = d_node
        self._ft_seq = ft_seq

    @property
    def o_node(self):
        return self._o_node

    @o_node.setter
    def o_node(self, value: int = None):
        if value not in list(self.net.get_node_data().index):
            raise ValueError
        self._o_node = value

    @property
    def d_node(self):
        return self._d_node

    @d_node.setter
    def d_node(self, value: int = None):
        if value not in list(self.net.get_node_data().index):
            raise ValueError
        self._d_node = value

    @property
    def ft_seq(self):
        return self._ft_seq

    @ft_seq.setter
    def ft_seq(self, ft_seq: list[int] = None):
        self._ft_seq = ft_seq

    @property
    def random_route(self) -> list[tuple[int, int]]:
        node_route = self.net.get_rnd_shortest_path()
        ft_seq = [(node_route[i], node_route[i + 1]) for i in range(len(node_route) - 1)]
        return ft_seq

    @property
    def od_route(self) -> tuple[list[tuple[int, int]], bool]:
        """通过指定起终点节点获取路径"""
        node_route, route_cost = self.net.get_shortest_path_length(o_node=self.o_node, d_node=self.d_node)
        if node_route:
            ft_seq = [(node_route[i], node_route[i + 1]) for i in range(len(node_route) - 1)]
            return ft_seq, True
        else:
            print(rf'{self.o_node}->{self.d_node}: no path, ignore...')
            return [(-1, -1)], False


class GpsDevice(object):
    """gps设备类"""
    def __init__(self, time_step: float = 0.5, loc_frequency: float = 1.0, agent_id: str = None,
                 loc_error_miu: float = 0.0, loc_error_sigma: float = 15,
                 heading_error_sigma: float = 10.0, heading_error_miu: float = 0.0):
        """
        假定gps的定位误差服从正态分布 ~ N(μ, σ2) ~ N(loc_error_miu, loc_error_sigma^2)
        落在区间(μ-σ, μ+σ)内的概率为0.68，横轴区间(μ-1.96σ, μ+1.96σ)内的概率为0.95，横轴区间(μ-2.58σ,μ+2.58σ)内的概率为0.997
        :param agent_id: 车辆id
        :param time_step: 仿真时间步长, s
        :param loc_frequency: gps定位频率(s), 每多少秒记录一次定位
        :param loc_error_miu: gps定位误差参数, N(μ, σ2)中的μ
        :param loc_error_sigma: gps定位误差参数, N(μ, σ2)中的σ
        """
        self.time_step = time_step
        self.loc_frequency = loc_frequency
        self.agent_id = agent_id

        self.tic_miu = 0
        self.tic_sigma = loc_frequency * 0.06

        self.loc_error_miu = loc_error_miu
        self.loc_error_sigma = loc_error_sigma
        self.heading_error_sigma = heading_error_sigma
        self.heading_error_miu = heading_error_miu

        # 用于记录定位数据的list
        # time, geo, heading, agent_id, link, from_node, to_node
        self.__final_loc_gps: list[tuple[datetime.datetime, Point, float, str, int, int, int]] = []
        self._f = "%Y/%m/%d %H:%M:%S"

    def receive_car_loc(self, now_time: datetime.datetime = None, loc: tuple[float, float, float, int, int, int] = None) -> None:
        """
        接收车辆的定位信息, 依据定位频率判断是否记录定位信息
        :param now_time:
        :param loc:
        :return:
        """
        if not self.__final_loc_gps or (now_time - self.__final_loc_gps[-1][0]).total_seconds() > self.tic_gap:
            loc_error = self.loc_error
            device_loc_x, device_loc_y, device_heading = \
                loc[0] + loc_error[0], loc[1] + loc_error[1], loc[2] + loc_error[2]
            link, f, t = int(loc[3]), int(loc[4]), int(loc[5])
            self.__final_loc_gps.append((now_time, Point(device_loc_x, device_loc_y), round(device_heading, 2),
                                         self.agent_id, link, f, t))
            # logging.info(
            #     rf'LOCATION - Car-{self.agent_id}, LocTime:{now_time.strftime(self._f)}, XY:{device_loc_x, device_loc_y, device_heading}, Error: {round(loc_error[0], 2), round(loc_error[1], 2), round(loc_error[2], 2)}  - LOCATION')

    @property
    def tic_gap(self) -> float:
        return self.loc_frequency + np.random.normal(loc=self.tic_miu, scale=self.tic_sigma)

    @property
    def loc_error(self) -> (float, float, float):
        """x, y, heading的定位误差"""
        return 0.707106 * np.random.normal(loc=self.loc_error_miu, scale=self.loc_error_sigma), \
               0.707106 * np.random.normal(loc=self.loc_error_miu, scale=self.loc_error_sigma), \
            np.random.normal(loc=self.heading_error_miu, scale=self.heading_error_sigma)

    def get_gps_loc_info(self) -> list[tuple[datetime.datetime, Point, float, str, int, int, int]]:
        return self.__final_loc_gps


class Car(object):
    """车类"""

    def __init__(self, agent_id: str = None, speed_miu: float = 10.0, speed_sigma: float = 3,
                 net: Net = None, time_step: float = 0.2, start_time: datetime.datetime = None,
                 loc_frequency: float = 10.0, loc_error_miu: float = 0.0, loc_error_sigma: float = 15,
                 heading_error_sigma: float = 10.0, heading_error_miu: float = 0.0,
                 save_gap: int = 5, route: Route = None, save_log: bool = False):
        """
        使用正态分布对车辆速度进行简单建模, 车辆速度服从正态分布 ~ N(speed_miu, speed_sigma^2)
        :param agent_id: 车辆ID
        :param speed_miu: 速度速度期望值, m/s
        :param speed_sigma: 速度标准差, m/s
        :param net: Net
        :param time_step: 仿真步长, s
        :param start_time: datetime.datetime, 行车开始的时间
        :param loc_frequency: GPS定位频率(s), 每多少秒定位一次
        :param loc_error_miu: 定位误差期望值参数
        :param loc_error_sigma: 定位误差标准差参数
        :param heading_error_sigma: 航向角误差标准差参数
        :param heading_error_miu: 航向角误差期望值
        :param save_gap: 每多少个仿真步存储一次车辆的真实轨迹, 每 save_gap * time_step (s)存储一次车辆的真实坐标
        """

        self.net = net
        self.time_step = time_step
        self.save_log = save_log
        assert loc_frequency / self.time_step >= 2, 'loc_frequency/time_step must be greater than or equal to 2.0'
        self.agent_id = agent_id
        self.speed_miu = speed_miu
        self.speed_sigma = speed_sigma
        self.start_time = start_time
        self.heading_error_sigma = heading_error_sigma
        self.heading_error_miu = heading_error_miu
        self.__time_tic = 0
        self.__step_loc: list[tuple[datetime.datetime, Point, float, str]] = list()  # 用于存储逐帧的轨迹
        self.save_gap = save_gap
        self.route = route

        # 装备GPS设备
        self.gps_loc_frequency = loc_frequency
        self.gps_loc_error_miu = loc_error_miu
        self.gps_loc_error_sigma = loc_error_sigma
        self.gps_device = GpsDevice(time_step=self.time_step,
                                    loc_frequency=self.gps_loc_frequency, agent_id=self.agent_id,
                                    loc_error_miu=self.gps_loc_error_miu, loc_error_sigma=self.gps_loc_error_sigma,
                                    heading_error_miu=self.heading_error_miu,
                                    heading_error_sigma=self.heading_error_sigma)

        # logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
        console_handler.setLevel(logging.INFO)

        if self.save_log:
            file_handler = logging.FileHandler(fr'./car_gps_{self.agent_id}.log', mode='a')
            file_handler.setFormatter(
                logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
            file_handler.setLevel(logging.INFO)
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                                handlers=[file_handler, console_handler])
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                                handlers=[console_handler])
        logging.info(rf'Car {self.agent_id}_logging_info:.....')

    def acquire_route(self) -> tuple[list[tuple[int, int]], bool]:
        if self.route.ft_seq is None:
            if self.route.o_node is None or self.route.d_node is None:
                return self.route.random_route, True
            else:
                return self.route.od_route
        else:
            return self.route.ft_seq, True

    @function_time_cost
    def start_drive(self) -> bool:
        """开始模拟行车"""
        ft_seq, is_success = self.acquire_route()
        if not is_success:
            return False
        for ft in ft_seq:
            f, t = ft[0], ft[1]
            now_drive_link_geo = self.net.get_line_geo_by_ft(from_node=f, to_node=t)
            now_drive_link_name = self.net.get_link_attr_by_ft(attr_name='road_name', from_node=f, to_node=t)
            now_drive_link = self.net.get_link_attr_by_ft(attr_name=net_field.LINK_ID_FIELD, from_node=f, to_node=t)
            link_length = now_drive_link_geo.length
            # logging.info(
            #     rf'### Car-{self.agent_id} driving in Node({ft[0]}) - Node({ft[1]}), {now_drive_link_name}, {link_length} ###')
            done_l = 0  # 用于记录在当前link上的累计行进距离
            his_index = 0  # 用于记录上一仿真时间步所到达的link折点索引

            # 计算link上的速度
            if now_drive_link_name == '路口转向':
                speed = np.abs(self.speed - np.random.randint(0, 5))
                speed = 0.01 if speed < 0.01 else speed
            else:
                speed = np.abs(self.speed)
            while True:
                # 计算当前时间和车辆所处的位置
                now_time = self.start_time + timedelta(seconds=self.__time_tic * self.time_step)
                now_loc_x, now_loc_y, now_heading, his_index = \
                    self.location(line=now_drive_link_geo, distance=done_l, history_index=his_index)

                # 位置传递给gps设备, 由设备决定是否记录数据
                self.gps_device.receive_car_loc(now_time=now_time,
                                                loc=(now_loc_x, now_loc_y, now_heading, now_drive_link, f, t))

                # 判断是否到达轨迹存储条件
                if self.__time_tic % self.save_gap == 0:
                    # logging.info(rf'saving true trajectory ......')
                    self.__step_loc.append((now_time, Point(now_loc_x, now_loc_y), now_heading, self.agent_id))

                # speed加一个微小的扰动, 计算即将开始的仿真步的车辆速度和沿着link移动的路径长度
                used_speed = speed + np.random.normal(loc=0, scale=speed * 0.2)
                # print(used_speed)
                step_l = used_speed * self.time_step

                # logging.info(
                #     rf'Car-{self.agent_id}: TimeStep-{self.__time_tic}, NowTime-{now_time.strftime("%Y/%m/%d %H:%M:%S")}, speed: {round(used_speed, 2)}m/s, accu_l: {round(done_l, 2)}m')

                self.__time_tic += 1
                done_l += step_l

                # 防止误差累积
                if (link_length + 0.5 < done_l < link_length - 0.5) or link_length < done_l:
                    break

        self.__time_tic = 0
        return True

    @property
    def speed(self) -> float:
        """车辆速度, 服从正态分布"""
        return np.random.normal(loc=self.speed_miu, scale=self.speed_sigma)

    @staticmethod
    def location(line: LineString = None, distance: float = None, history_index: int = 0) -> tuple or list:
        """
        计算车辆位置、航向角
        :param line:
        :param distance:
        :param history_index:
        :return: (x, y, heading, his_index)
        """
        coords = list(line.coords)
        if distance <= 0.0:
            loc_xy = coords[0]
            next_xy = coords[1]
            heading_vec = np.array([next_xy[0] - loc_xy[0],
                                    next_xy[1] - loc_xy[1]])
            return loc_xy[0], loc_xy[1], calc_north_angle(dir_vec=heading_vec), 0
        elif distance >= line.length:
            loc_xy = coords[-1]
            pre_xy = coords[-2]
            heading_vec = np.array([loc_xy[0] - pre_xy[0],
                                    loc_xy[1] - pre_xy[1]])
            return loc_xy[0], loc_xy[1], calc_north_angle(dir_vec=heading_vec), len(list(line.coords)) - 1
        else:
            for i in range(history_index, len(coords)):
                p = coords[i]
                # for i, p in enumerate(coords):
                #     if i < history_index:
                #         continue
                xd = line.project(Point(p))
                if xd == distance:
                    loc_xy = p
                    try:
                        pre_xy = coords[i - 1]
                        heading_vec = np.array([loc_xy[0] - pre_xy[0],
                                                loc_xy[1] - pre_xy[1]])
                        return loc_xy[0], loc_xy[1], calc_north_angle(dir_vec=heading_vec), i - 1
                    except IndexError or KeyError:
                        next_xy = coords[i + 1]
                        heading_vec = np.array([next_xy[0] - loc_xy[0],
                                                next_xy[1] - loc_xy[1]])
                        return loc_xy[0], loc_xy[1], calc_north_angle(dir_vec=heading_vec), i - 1
                if xd > distance:
                    cp = line.interpolate(distance)
                    next_xy = coords[i]
                    pre_xy = coords[i - 1]
                    heading_vec = np.array([next_xy[0] - pre_xy[0],
                                            next_xy[1] - pre_xy[1]])
                    return cp.x, cp.y, calc_north_angle(dir_vec=heading_vec), i - 1

    def get_trajectory_info(self):
        return self.__step_loc

    def get_gps_loc_info(self) -> list[tuple[datetime.datetime, Point, float, str, int, int, int]]:
        return self.gps_device.get_gps_loc_info()


class RouteInfoCollector(object):
    """真实路径坐标以及GPS坐标收集器"""
    def __init__(self, convert_prj_sys: bool = True, convert_type: str = 'bd-84',
                 convert_loc: bool = False, from_crs: str = None, to_crs: str = None, crs: str = None) -> None:
        """

        :param convert_prj_sys: 是否转换坐标系(GCJ02、百度、84之间的转化)
        :param convert_type: 'gc-84', 'gc-bd'分别代表GCJ02转化为84, GCJ02转化为百度,
        :param from_crs: str
        :param to_crs: str
        :param crs: str
        :param convert_loc: 是否进行地理坐标系到平面投影系的转化

        """
        self.convert_prj_sys = convert_prj_sys
        self.from_crs = from_crs
        self.to_crs = to_crs
        self.crs = crs
        self.convert_loc = convert_loc
        self.convert_type = convert_type

        self.gps_info_list: list[tuple[datetime.datetime, Point, float, str, int, int, int]] = []
        self.trajectory_info_list: list[tuple[datetime.datetime, Point, float, str]] = []
        self.gps_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame()
        self.trajectory_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame()

    def collect_gps(self, gps_info: list[tuple[datetime.datetime, Point, float, str, int, int, int]] = None):
        self.gps_info_list.extend(gps_info)

    @function_time_cost
    def collect_trajectory(self, trajectory_info: list[tuple[datetime.datetime, Point, float, str]] = None):
        self.trajectory_info_list.extend(trajectory_info)

    @staticmethod
    @function_time_cost
    def format_gdf(convert_prj_sys: bool = False, from_crs: str = None, to_crs: str = None, crs: str = None,
                   convert_loc: bool = False,
                   convert_type: str = 'bd-84',
                   info_list: list = None, attr_name_field_list: list[str] = None,
                   time_format="%Y-%m-%d %H:%M:%S") -> gpd.GeoDataFrame:

        format_df = pd.DataFrame(info_list, columns=attr_name_field_list)

        if to_crs is not None:
            used_crs = from_crs
        else:
            used_crs = crs
            to_crs = crs

        format_gdf = gpd.GeoDataFrame(format_df, geometry=gps_field.GEOMETRY_FIELD, crs=used_crs)
        format_gdf = format_gdf.to_crs(to_crs)
        format_gdf[gps_field.TIME_FIELD] = format_gdf[gps_field.TIME_FIELD].apply(
            lambda t: t.strftime(time_format))

        if convert_loc:
            con = LngLatTransfer()
            format_gdf[gps_field.GEOMETRY_FIELD] = \
                format_gdf[gps_field.GEOMETRY_FIELD].apply(
                    lambda geo: con.obj_convert(geo_obj=geo, con_type=convert_type))
        format_gdf[gps_field.LNG_FIELD] = format_gdf[gps_field.GEOMETRY_FIELD].x
        format_gdf[gps_field.LAT_FIELD] = format_gdf[gps_field.GEOMETRY_FIELD].y

        attr_name_field_list.remove(gps_field.GEOMETRY_FIELD)
        attr_name_field_list.extend([gps_field.LNG_FIELD, gps_field.LAT_FIELD, gps_field.GEOMETRY_FIELD])
        return format_gdf[attr_name_field_list]

    @function_time_cost
    def save_trajectory(self, out_fldr: str = r'./', file_name: str = None, file_type: str = 'csv',
                        time_format="%Y-%m-%d %H:%M:%S") -> gpd.GeoDataFrame:
        """
        存储轨迹信息
        :param out_fldr:
        :param file_name:
        :param file_type:
        :param time_format:
        :return:
        """
        if self.trajectory_gdf.empty and self.trajectory_info_list:
            self.trajectory_gdf = self.format_gdf(from_crs=self.from_crs,
                                                  to_crs=self.to_crs,
                                                  crs=self.crs, convert_loc=self.convert_loc,
                                                  convert_type=self.convert_type,
                                                  info_list=self.trajectory_info_list,
                                                  attr_name_field_list=[gps_field.TIME_FIELD, gps_field.GEOMETRY_FIELD,
                                                                        gps_field.HEADING_FIELD,
                                                                        gps_field.AGENT_ID_FIELD],
                                                  time_format=time_format)
        else:
            return gpd.GeoDataFrame()
        if out_fldr is None:
            pass
        else:
            self.save_file(file_type=file_type, df=self.trajectory_gdf, out_fldr=out_fldr, file_name=file_name)
        return self.trajectory_gdf

    @function_time_cost
    def save_gps_info(self, out_fldr: str = r'./', file_name: str = None, file_type: str = 'csv',
                      time_format="%Y-%m-%d %H:%M:%S") -> gpd.GeoDataFrame:
        """
        存储gps信息
        :param out_fldr:
        :param file_name:
        :param file_type:
        :param time_format:
        :return:
        """
        if self.gps_gdf.empty and self.gps_info_list:
            self.gps_gdf = self.format_gdf(convert_prj_sys=self.convert_prj_sys, from_crs=self.from_crs,
                                           to_crs=self.to_crs,
                                           crs=self.crs, convert_loc=self.convert_loc, convert_type=self.convert_type,
                                           info_list=self.gps_info_list,
                                           attr_name_field_list=[gps_field.TIME_FIELD, gps_field.GEOMETRY_FIELD,
                                                                 gps_field.HEADING_FIELD, gps_field.AGENT_ID_FIELD,
                                                                 net_field.LINK_ID_FIELD,
                                                                 net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD],
                                           time_format=time_format)
        else:
            return gpd.GeoDataFrame()
        if out_fldr is None:
            pass
        else:
            self.save_file(file_type=file_type, df=self.gps_gdf, out_fldr=out_fldr, file_name=file_name)
        return self.gps_gdf

    @function_time_cost
    def save_mix_info(self, out_fldr: str = r'./', file_name: str = None, convert_prj_sys: bool = True,
                      from_crs: str = None, to_crs: str = None, crs: str = None,
                      convert_loc: bool = False, convert_type: str = 'bd-84', file_type: str = 'csv',
                      time_format="%Y-%m-%d %H:%M:%S") -> gpd.GeoDataFrame:
        """

        :param out_fldr:
        :param file_name:
        :param convert_prj_sys:
        :param from_crs:
        :param to_crs:
        :param crs:
        :param convert_loc:
        :param convert_type:
        :param file_type:
        :param time_format
        :return:
        """
        if self.gps_gdf.empty and self.gps_info_list:
            self.gps_gdf = self.format_gdf(convert_prj_sys=convert_prj_sys, from_crs=from_crs, to_crs=to_crs,
                                           crs=crs, convert_loc=convert_loc, convert_type=convert_type,
                                           info_list=self.gps_info_list,
                                           attr_name_field_list=[gps_field.TIME_FIELD, gps_field.GEOMETRY_FIELD,
                                                                 gps_field.HEADING_FIELD, gps_field.AGENT_ID_FIELD,
                                                                 net_field.LINK_ID_FIELD,
                                                                 net_field.FROM_NODE_FIELD, net_field.TO_NODE_FIELD],
                                           time_format=time_format)
        if self.trajectory_gdf.empty and self.trajectory_info_list:
            self.trajectory_gdf = self.format_gdf(convert_prj_sys=convert_prj_sys, from_crs=from_crs, to_crs=to_crs,
                                                  crs=crs, convert_loc=convert_loc, convert_type=convert_type,
                                                  info_list=self.trajectory_info_list,
                                                  attr_name_field_list=[gps_field.TIME_FIELD, gps_field.GEOMETRY_FIELD,
                                                                        gps_field.HEADING_FIELD,
                                                                        gps_field.AGENT_ID_FIELD],
                                                  time_format=time_format)
        if not self.trajectory_gdf.empty:
            self.gps_gdf[gps_field.TYPE_FIELD] = 'gps'
            self.trajectory_gdf[gps_field.TYPE_FIELD] = 'trajectory'
            mix_gdf = pd.concat([self.trajectory_gdf, self.gps_gdf])
            mix_gdf.reset_index(inplace=True, drop=True)
            if out_fldr is None:
                pass
            else:
                self.save_file(file_type=file_type, df=mix_gdf,
                               file_name=file_name, out_fldr=out_fldr)
            return mix_gdf
        else:
            return gpd.GeoDataFrame()

    @staticmethod
    def save_file(file_type: str = 'csv', df: pd.DataFrame or gpd.GeoDataFrame = None,
                  out_fldr: str = r'./', file_name: str = None):
        """

        :param file_type:
        :param df:
        :param out_fldr:
        :param file_name:
        :return:
        """
        if file_type == 'csv':
            df.to_csv(os.path.join(out_fldr, ''.join([file_name, '.csv'])), encoding='utf_8_sig',
                      index=False)
        elif file_type == 'geojson':
            assert isinstance(df, gpd.GeoDataFrame)
            df.to_file(os.path.join(out_fldr, ''.join([file_name, '.geojson'])), encoding='gbk',
                       driver='GeoJSON')
        else:
            assert isinstance(df, gpd.GeoDataFrame)
            df.to_file(os.path.join(out_fldr, ''.join([file_name, '.shp'])), encoding='gbk')


def calc_north_angle(dir_vec: np.ndarray = None) -> float:
    """"""
    north_vec = np.array([0, 1])
    if np.linalg.norm(dir_vec) <= 0:
        raise ValueError('向量的模为0...')
    _heading = 180 * np.arccos(
        np.dot(north_vec, dir_vec) / (np.linalg.norm(dir_vec) * np.linalg.norm(north_vec))) / np.pi
    if dir_vec[0] < 0:
        return 360.0 - _heading
    else:
        return _heading


if __name__ == '__main__':
    print(calc_north_angle(dir_vec=np.array([-1, 100])))


