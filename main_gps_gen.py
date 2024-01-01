# -- coding: utf-8 --
# @Time    : 2023/12/9 8:29
# @Author  : TangKai
# @Team    : ZheChengData


import datetime
import pandas as pd
from src.map.Net import Net
from src.generation.GpsGen import Car
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
    my_net.to_plane_prj()  # 转平面投影
    # _time_step = 0.1  # 仿真步长, s
    # agent_id = 2
    # o_node, d_node = 5953, 8528
    #
    # # 2.新建一个车对象, 配备一个电子地图net, 仿真步长为{_time_step}s
    # car = Car(agent_id=agent_id, speed_miu=12.0, speed_sigma=3.6,
    #           net=my_net, time_step=_time_step,
    #           loc_frequency=2.0, loc_error_sigma=10.0, loc_error_miu=0.0,
    #           start_time=datetime.datetime.now())
    # # 依据起终结点获得route
    # car.acquire_route_by_od(o_node=o_node, d_node=d_node)
    # # 开始行车
    # car.start_drive()
    # # 存储GPS数据
    # car.gps_device.export_data(convert_loc=True, from_crs='EPSG:32650', to_crs='EPSG:4326',
    #                            out_fldr=r'./data/output/gps/', file_name=rf'agent_{agent_id}')

    # 3.读取GPS文件
    gps_df = pd.read_csv(r'./data/output/gps/agent_1.csv')
    gps_obj = GpsPointsGdf(gps_points_df=gps_df, lat_field=gps_field.LAT_FIELD, lng_field=gps_field.LNG_FIELD,
                           time_format="%Y-%m-%d %H:%M:%S.%f", buffer=80.0, geo_crs='EPSG:4326', plane_crs='EPSG:32650')

    # 初始化一个隐马尔可夫模型
    hhm_obj = HiddenMarkov(net=my_net, gps_points=gps_obj, beta=31.2, gps_sigma=10.0)
    hhm_obj.generate_markov_para()
    hhm_obj.solve()
    hhm_obj.acquire_res()




