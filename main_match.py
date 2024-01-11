# -- coding: utf-8 --
# @Time    : 2023/12/9 8:29
# @Author  : TangKai
# @Team    : ZheChengData

import geopandas as gpd
from src.map.Net import Net
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

    # 3.读取GPS文件
    gps_df = gpd.read_file(r'./data/output/gps/car_1.geojson')
    gps_obj = GpsPointsGdf(gps_points_df=gps_df, lat_field=gps_field.LAT_FIELD, lng_field=gps_field.LNG_FIELD,
                           time_format="%Y-%m-%d %H:%M:%S.%f", buffer=80.0, geo_crs='EPSG:4326', plane_crs='EPSG:32650')
    # 降频
    gps_obj.lower_frequency(n=5)
    sub_net = my_net.create_computational_net(gps_array_buffer=gps_obj.get_gps_array_buffer(buffer=200.0))
    # 初始化一个隐马尔可夫模型
    hhm_obj = HiddenMarkov(net=sub_net, gps_points=gps_obj, beta=31.2, gps_sigma=10.0)
    hhm_obj.generate_markov_para()
    hhm_obj.solve()
    hhm_obj.acquire_res()




