# -- coding: utf-8 --
# @Time    : 2023/12/9 8:29
# @Author  : TangKai
# @Team    : ZheChengData

"""依据路网生成GPS数据"""

import datetime
from src.map.Net import Net
from src.generation.GpsGen import Route
from src.GlobalVal import NetField, GpsField
from src.generation.GpsGen import Car, RouteInfoCollector


net_field = NetField()
gps_field = GpsField()


if __name__ == '__main__':
    # 1.新建一个路网对象, 并且指定其地理坐标系(shp源文件的crs)以及要使用的投影坐标系
    # 示例为西安的路网, 使用6度带中的32649
    plain_crs = 'EPSG:32649'
    geo_crs = 'EPSG:4326'
    my_net = Net(link_path=r'data/input/net/xian/link.shp',
                 node_path=r'data/input/net/xian/node.shp',
                 weight_field='length', geo_crs=geo_crs, plane_crs=plain_crs)
    # 路网对象初始化
    my_net.init_net()

    # 2.新建一个route, 用于车辆car路径导航
    # 必须指定一个net对象
    # 若不指定ft_seq, 则使用o_node -> d_node进行搜路获取路径, 若没有指定o_node和d_node则使用随机路径
    route = Route(net=my_net, o_node=None, d_node=None, ft_seq=None)
    # route.o_node = 176356
    # route.d_node = 228133
    # ft_seq = [(137226, 42212), (42212, 21174), (21174, 39617)]

    # 3.新建一个车对象, 分配一个车辆ID, 配备一个电子地图net, 且设置仿真参数
    _time_step = 0.1  # 仿真步长, s
    agent_id = rf'xa_car_{3}'  # 车辆ID
    speed_miu = 12.0  # 速度期望值
    speed_sigma = 3.6  # 速度标准差
    save_gap = 5  # 每多少仿真步保存一次车辆真实位置数据
    loc_frequency = 1.0  # 每多少s进行一次GPS定位
    loc_error_sigma = 20.0  # 定位误差标准差(m)
    loc_error_miu = 0.0  # 定位误差标准期望值(m)

    car = Car(net=my_net, time_step=_time_step, route=route,
              agent_id=agent_id, speed_miu=speed_miu,speed_sigma=speed_sigma,
              loc_frequency=loc_frequency, loc_error_sigma=loc_error_sigma, loc_error_miu=loc_error_miu,
              start_time=datetime.datetime.now(), save_gap=save_gap)

    # 4.开始行车
    car.start_drive()

    # 新建一个行程信息收集器对象
    data_col = RouteInfoCollector(from_crs=plain_crs, to_crs=geo_crs, convert_prj_sys=True, convert_type='gc-84',
                                  convert_loc=False)
    data_col.collect_trajectory(car.get_trajectory_info())
    data_col.collect_gps(car.get_gps_loc_info())
    data_col.save_trajectory(file_type='geojson', out_fldr=r'./data/output/trajectory/', file_name=agent_id)
    data_col.save_gps_info(file_type='geojson', out_fldr=r'./data/output/gps/', file_name=agent_id)
    data_col.save_mix_info(file_type='geojson', out_fldr=r'./data/output/mix/', file_name=agent_id)




