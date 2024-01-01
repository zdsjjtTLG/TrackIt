# -- coding: utf-8 --
# @Time    : 2023/12/30 12:40
# @Author  : TangKai
# @Team    : ZheChengData

"""模拟车辆在路径上行驶, 并生成GPS定位信息"""

import sys
import pyproj
import logging
import os.path
import datetime
import numpy as np
import pandas as pd
from src.map.Net import Net
from datetime import timedelta
from shapely.ops import transform
from shapely.geometry import LineString, Point

from src.GlobalVal import GpsField

gps_field = GpsField()

class GpsDevice(object):
    """gps设备类"""
    def __init__(self, time_step: float = 0.5, loc_frequency: float = 1.0, agent_id: int = None,
                 loc_error_miu: float = 0.0, loc_error_sigma: float = 15):
        """
        假定gps的定位误差服从正态分布 ~ N(μ, σ2)
        落在区间(μ-σ, μ+σ)内的概率为0.68，横轴区间(μ-1.96σ, μ+1.96σ)内的概率为0.95，横轴区间(μ-2.58σ,μ+2.58σ)内的概率为0.997
        :param agent_id: 车辆id
        :param time_step: 仿真时间步长
        :param loc_frequency: gps定位频率(s), 每多少秒记录一次定位
        :param loc_error_miu: gps定位误差参数, N(μ, σ2)中的μ
        :param loc_error_sigma: gps定位误差参数, N(μ, σ2)中的σ
        """
        self.time_step = time_step
        self.loc_frequency = loc_frequency
        self.agent_id = agent_id

        self.tic_miu = 0
        self.tic_sigma = loc_frequency * 0.1

        self.loc_error_miu = loc_error_miu
        self.loc_error_sigma = loc_error_sigma

        # 用于记录定位数据的list
        self.final_loc_gps: list[tuple[datetime.datetime, Point]] = []
        self._f = "%Y/%m/%d %H:%M:%S"

    def receive_car_loc(self, now_time: datetime.datetime = None, loc: tuple[float, float] = None) -> None:
        """
        接收车辆的定位信息, 依据定位频率判断是否记录定位信息
        :param now_time:
        :param loc:
        :return:
        """
        if not self.final_loc_gps or (now_time - self.final_loc_gps[-1][0]).seconds > self.tic_gap:
            xy_error = self.loc_error
            real_loc_x, real_loc_y = loc[0] + xy_error[0], loc[1] + xy_error[1]
            self.final_loc_gps.append((now_time, Point(real_loc_x, real_loc_y)))
            logging.info(
                rf'LOCATION - Car-{self.agent_id}, LocTime:{now_time.strftime(self._f)}, XY:{real_loc_x, real_loc_y}, Error: {round(xy_error[0], 2), round(xy_error[1], 2)}  - LOCATION')

    def export_data(self, convert_loc: bool = True, from_crs: str = None, to_crs: str = None, out_fldr: str = None,
                    file_name: str = None) -> None:
        """
        行驶结束后, 存储gps数据
        :param convert_loc: 是否需要进行坐标转换
        :param from_crs:
        :param to_crs:
        :param out_fldr:
        :param file_name:
        :return:
        """
        if convert_loc:
            assert from_crs is not None
            assert to_crs is not None
            loc_res = self.convert_loc(from_crs=from_crs, to_crs=to_crs)
        else:
            loc_res = self.final_loc_gps
        gps_df = pd.DataFrame(loc_res, columns=[gps_field.TIME_FIELD, 'geometry'])
        gps_df[gps_field.LNG_FIELD] = gps_df['geometry'].apply(lambda p: p.x)
        gps_df[gps_field.LAT_FIELD] = gps_df['geometry'].apply(lambda p: p.y)
        gps_df[gps_field.POINT_SEQ_FIELD] = [i + 1 for i in range(len(gps_df))]
        gps_df[gps_field.AGENT_ID_FIELD] = self.agent_id
        gps_df[[gps_field.AGENT_ID_FIELD, gps_field.TIME_FIELD, gps_field.LNG_FIELD, gps_field.LAT_FIELD,
                gps_field.POINT_SEQ_FIELD]].to_csv(os.path.join(out_fldr, file_name + '.csv'), encoding='utf_8_sig',
                                                   index=False)

    def convert_loc(self, from_crs: str = None, to_crs: str = None) -> list[tuple[datetime.datetime, Point]]:
        """
        地理坐标和平面坐标之间的转换
        :param from_crs:
        :param to_crs:
        :return:
        """
        before = pyproj.CRS(from_crs)
        after = pyproj.CRS(to_crs)
        project = pyproj.Transformer.from_crs(before, after, always_xy=True).transform
        loc_res = [(loc_item[0], transform(project, loc_item[1])) for loc_item in self.final_loc_gps]
        return loc_res
    @property
    def tic_gap(self) -> float:
        return self.loc_frequency + np.random.normal(loc=self.tic_miu, scale=self.tic_sigma)

    @property
    def loc_error(self) -> (float, float):
        """x,y方向的定位误差"""
        return 0.707106 * np.random.normal(loc=self.loc_error_miu, scale=self.loc_error_sigma), \
            0.707106 * np.random.normal(loc=self.loc_error_miu, scale=self.loc_error_sigma)


class Car(object):
    """车类"""
    def __init__(self, agent_id: int = None, speed_miu: float = 10.0, speed_sigma: float = 3,
                 net: Net = None, time_step: float = 0.2, start_time: datetime.datetime = None,
                 loc_frequency: float = 1.0, loc_error_miu: float = 0.0, loc_error_sigma: float = 15):
        """

        :param agent_id:
        :param speed_miu:
        :param speed_sigma:
        :param net:
        :param time_step:
        :param start_time:
        :param loc_frequency:
        :param loc_error_miu:
        :param loc_error_sigma:
        """
        self.net = net
        self.time_step = time_step
        assert self.time_step <= 0.5, '仿真时间步长应该小于0.5s'
        self.agent_id = agent_id
        self.speed_miu = speed_miu
        self.speed_sigma = speed_sigma
        self.start_time = start_time
        self.__time_tic = 0
        self.ft_seq: list[tuple[int, int]] = []

        # 装备GPS设备
        self.gps_loc_frequency = loc_frequency
        self.gps_loc_error_miu = loc_error_miu
        self.gps_loc_error_sigma = loc_error_sigma
        self.gps_device = GpsDevice(time_step=self.time_step,
                                    loc_frequency=self.gps_loc_frequency, agent_id=self.agent_id,
                                    loc_error_miu=self.gps_loc_error_miu, loc_error_sigma=self.gps_loc_error_sigma)

        # logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
        console_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(fr'./car_gps_{self.agent_id}.log', mode='a')
        file_handler.setFormatter(
            logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
        file_handler.setLevel(logging.INFO)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                            handlers=[file_handler, console_handler])
        logging.info(rf'Car {self.agent_id}_logging_info:.....')

    def acquire_route_by_od(self, o_node: int = None, d_node: int = None) -> list[tuple[int, int]]:
        """通过指定起终点节点获取路径"""
        node_route = self.net.get_shortest_path(o_node=o_node, d_node=d_node)
        self.ft_seq = [(node_route[i], node_route[i + 1]) for i in range(len(node_route) - 1)]
        return self.ft_seq

    def start_drive(self) -> None:
        """开始模拟行车"""
        for ft in self.ft_seq:
            now_drive_link_geo = self.net.get_line_geo_by_ft(from_node=ft[0], to_node=ft[1])
            now_drive_link_name = self.net.get_link_attr_by_ft(attr_name='road_name', from_node=ft[0], to_node=ft[1])
            link_length = now_drive_link_geo.length
            logging.info(
                rf'### Car-{self.agent_id} driving in Node({ft[0]}) - Node({ft[1]}), {now_drive_link_name}, {link_length} ###')
            done_l = 0  # 用于记录在当前link上的累计行进距离
            his_index = 0  # 用于记录上一仿真时间步所到达的link折点索引

            # 计算link上的速度
            if now_drive_link_name == '路口转向':
                speed = np.abs(self.speed - np.random.randint(0, 5))
            else:
                speed = self.speed

            while True:
                # 计算当前时间和车辆所处的位置
                now_time = self.start_time + timedelta(seconds=self.__time_tic * self.time_step)
                now_loc_x, now_loc_y, his_index = \
                    self.location(line=now_drive_link_geo, distance=done_l, history_index=his_index)

                # 位置传递给gps设备, 由设备决定是否记录数据
                self.gps_device.receive_car_loc(now_time=now_time, loc=(now_loc_x, now_loc_y))

                # speed加一个微小的扰动, 计算即将开始的仿真步的车辆速度和沿着link移动的路径长度
                used_speed = speed + np.random.normal(loc=0, scale=speed * 0.08)
                step_l = used_speed * self.time_step

                logging.info(
                    rf'Car-{self.agent_id}: TimeStep-{self.__time_tic}, NowTime-{now_time.strftime("%Y/%m/%d %H:%M:%S")}, speed: {round(used_speed, 2)}m/s, accu_l: {round(done_l, 2)}m')

                self.__time_tic += 1
                done_l += step_l

                # 防止误差累积
                if (link_length + 0.5 < done_l < link_length - 0.5) or link_length < done_l:
                    break

        self.__time_tic = 0

    @property
    def speed(self) -> float:
        """车辆速度, 服从正态分布"""
        return np.random.normal(loc=self.speed_miu, scale=self.speed_sigma)

    @staticmethod
    def location(line: LineString = None, distance: float = None, history_index: int = 0) -> tuple or list:
        """
        计算车辆位置
        :param line:
        :param distance:
        :param history_index:
        :return:
        """

        if distance <= 0.0:
            xy = list(line.coords)[0]
            return xy[0], xy[1], 0
        elif distance >= line.length:
            xy = list(line.coords)[-1]
            return xy[0], xy[1], len(list(line.coords)) - 1
        else:
            coords = list(line.coords)
            for i, p in enumerate(coords):
                if i < history_index:
                    continue
                xd = line.project(Point(p))
                if xd == distance:
                    return coords[i], i - 1
                if xd > distance:
                    cp = line.interpolate(distance)
                    return cp.x, cp.y, i - 1


if __name__ == '__main__':
    pass

