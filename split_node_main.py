# -- coding: utf-8 --
# @Time    : 2024/2/27 23:07
# @Author  : TangKai
# @Team    : ZheChengData


import geopandas as gpd
from src.gotrackit.map.Net import Net

if __name__ == "__main__":
    # 1.新建一个路网对象, 并且使用平面坐标
    plain_crs = 'EPSG:32649'
    geo_crs = 'EPSG:4326'
    my_net = Net(link_path=r'data/input/net/xian/link.shp',
                 node_path=r'data/input/net/xian/node.shp',
                 weight_field='length', geo_crs=geo_crs, plane_crs=plain_crs)

    # 初始化
    my_net.init_net()

    my_net.split_link(p=(308758.1, 3784493.5), target_link=344327)

    my_net.export_net(export_crs='EPSG:32649', file_type='shp', flag_name='split', out_fldr=r'./data/input/net/xian')