# -- coding: utf-8 --
# @Time    : 2023/12/9 8:29
# @Author  : TangKai
# @Team    : ZheChengData


import datetime
import pandas as pd
from src.map.Net import Net
from src.generation.GpsGen import Car, RouteInfoCollector
from src.generation.GpsGen import Route
import geopandas as gpd
from src.gps.LocGps import GpsPointsGdf
from src.model.Markov import HiddenMarkov
from src.GlobalVal import NetField, GpsField
net_field = NetField()
gps_field = GpsField()


if __name__ == '__main__':
    # 1.新建一个路网对象, 并且使用平面坐标
    my_net = Net(link_path=r'./data/input/net/FinalLink.shp',
                 node_path=r'./data/input/net/FinalNode.shp',
                 weight_field='length', geo_crs='EPSG:4326', plane_crs='EPSG:32650')
    # 初始化
    my_net.init_net()

    _time_step = 0.1  # 仿真步长, s
    agent_id = rf'car_{3}'
    # o_node, d_node = 5953, 8528

    # 新建一个route
    route = Route(net=my_net)
    # route.o_node = 16283
    # route.d_node = 12912
    # 2.新建一个车对象, 配备一个电子地图net, 仿真步长为{_time_step}s
    car = Car(agent_id=agent_id, speed_miu=12.0, speed_sigma=3.6,
              net=my_net, time_step=_time_step,
              save_gap=5,
              loc_frequency=1.0, loc_error_sigma=20.0, loc_error_miu=0.0,
              start_time=datetime.datetime.now(), route=route)

    # # 依据起终结点获得route
    # car.acquire_route_by_od(o_node=o_node, d_node=d_node)
    # 开始行车
    car.start_drive()

    data_col = RouteInfoCollector(from_crs='EPSG:32650', to_crs='EPSG:4326', convert_prj_sys=True, convert_type='gc-84',
                                  convert_loc=False)
    data_col.collect_trajectory(car.get_trajectory_info())
    data_col.collect_gps(car.get_gps_loc_info())

    data_col.save_trajectory(file_type='geojson', out_fldr=r'./data/output/trajectory/', file_name=agent_id)

    data_col.save_gps_info(file_type='geojson', out_fldr=r'./data/output/gps/', file_name=agent_id)
    data_col.save_mix_info(file_type='geojson', out_fldr=r'./data/output/mix/', file_name=agent_id)




