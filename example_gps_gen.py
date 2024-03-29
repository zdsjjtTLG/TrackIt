# -- coding: utf-8 --
# @Time    : 2023/12/9 8:29
# @Author  : TangKai
# @Team    : ZheChengData

"""依据路网生成GPS数据"""

import datetime
# from src.gotrackit.map.Net import Net
# from src.gotrackit.generation.GpsGen import Route
# from src.gotrackit.GlobalVal import NetField, GpsField
# from src.gotrackit.generation.GpsGen import Car, RouteInfoCollector

from gotrackit.map.Net import Net
from gotrackit.generation.GpsGen import Route
from gotrackit.GlobalVal import NetField, GpsField
from gotrackit.generation.GpsGen import Car, RouteInfoCollector


net_field = NetField()
gps_field = GpsField()


if __name__ == '__main__':
    # 1.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_path=r'data/input/net/xian/modifiedConn_link.shp',
                 node_path=r'data/input/net/xian/modifiedConn_node.shp',
                 weight_field='length')
    # 路网对象初始化
    my_net.init_net()

    # 2.新建一个route, 用于车辆car路径导航
    # 必须指定一个net对象
    # 若不指定ft_seq, 则使用o_node -> d_node进行搜路获取路径, 若没有指定o_node和d_node则使用随机路径
    route = Route(net=my_net, o_node=None, d_node=None, ft_seq=None)
    # route.o_node = 176356
    # route.d_node = 228133
    # route.ft_seq = [(137226, 42212), (42212, 21174), (21174, 39617)]

    # 3.新建一个行程信息收集器对象, 对数据进行统一管理
    # 轨迹信息和GPS坐标信息都是平面坐标系, 需要转化为地理坐标系后再进行存储
    data_col = RouteInfoCollector(from_crs=my_net.planar_crs, to_crs=my_net.geo_crs, convert_prj_sys=True)

    # 4.设置仿真参数, 并且初始化一个车辆实体
    _time_step = 0.1  # 仿真步长, s
    speed_miu = 12.0  # 速度期望值
    speed_sigma = 3.6  # 速度标准差
    save_gap = 5  # 每多少仿真步保存一次车辆真实位置数据
    loc_frequency_list = [3, 6, 10, 15]  # 每多少s进行一次GPS定位
    loc_error_sigma = 40.0  # 定位误差标准差(m)
    loc_error_miu = 0.0  # 定位误差标准期望值(m)

    k = 1
    # 开始行车
    for car_id in [rf'xa_car_{i}' for i in range(k, k + 10)]:
        # 新建车对象, 分配一个车辆ID, 配备一个Net和一个Route, 并且设置仿真参数
        car = Car(net=my_net, time_step=_time_step, route=route, save_log=False,
                  agent_id=car_id, speed_miu=speed_miu, speed_sigma=speed_sigma,
                  loc_frequency=loc_frequency_list[k % 4], loc_error_sigma=loc_error_sigma, loc_error_miu=loc_error_miu,
                  start_time=datetime.datetime(year=2022, month=5, day=12, hour=16, minute=14, second=0),
                  save_gap=save_gap)

        # 开始行车
        car.start_drive()

        # 收集数据
        data_col.collect_trajectory(car.get_trajectory_info())
        data_col.collect_gps(car.get_gps_loc_info())

    # 存储数据
    tra_name = '0329sample'
    trajectory_gdf = data_col.save_trajectory(file_type='geojson', out_fldr=r'./data/output/trajectory/sample/',
                                              file_name=tra_name)
    gps_gdf = data_col.save_gps_info(file_type='geojson', out_fldr=r'./data/output/gps/sample/', file_name=tra_name)
    mix_gdf = data_col.save_mix_info(file_type='geojson', out_fldr=r'./data/output/mix/sample/', file_name=tra_name)

    print(trajectory_gdf)
    print(gps_gdf)
    print(mix_gdf)







