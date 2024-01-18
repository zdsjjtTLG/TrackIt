# -- coding: utf-8 --
# @Time    : 2024/1/11 13:21
# @Author  : TangKai
# @Team    : ZheChengData


"""测试创建子net"""

import geopandas as gpd
from src.gotrackit.map.Net import Net
from src.gotrackit.gps.LocGps import GpsPointsGdf
from src.gotrackit.GlobalVal import NetField, GpsField

net_field = NetField()
gps_field = GpsField()


if __name__ == '__main__':
    # 1.新建一个路网对象, 并且使用平面坐标
    my_net = Net(link_path=r'data/input/net/cc/FinalLink.shp',
                 node_path=r'data/input/net/cc/FinalNode.shp',
                 weight_field='length', geo_crs='EPSG:4326', plane_crs='EPSG:32650')
    # 初始化
    my_net.init_net()

    # 创建GPS对象
    gps_gdf = gpd.read_file(r'./data/output/gps/car_1.geojson')
    gps_gdf[gps_field.LNG_FIELD] = gps_gdf['geometry'].apply(lambda x: x.x)
    gps_gdf[gps_field.LAT_FIELD] = gps_gdf['geometry'].apply(lambda x: x.y)

    gps_obj = GpsPointsGdf(lng_field=gps_field.LNG_FIELD, lat_field=gps_field.LAT_FIELD, buffer=200.0,
                           plane_crs='EPSG:32650', geo_crs='EPSG:4326', gps_points_df=gps_gdf)

    sub_net = my_net.create_computational_net(gps_array_buffer=gps_obj.get_gps_array_buffer(buffer=200.0))

    print(sub_net.get_shortest_path_length(o_node=9917, d_node=2938))

