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
    my_net = Net(link_path=r'data/input/net/conn_test/link.shp',
                 node_path=r'data/input/net/conn_test/node.shp',
                 weight_field='length', geo_crs=geo_crs, plane_crs=plain_crs)

    # 初始化
    my_net.init_net()

    my_net.check()
