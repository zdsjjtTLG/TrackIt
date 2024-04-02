# -- coding: utf-8 --
# @Time    : 2024/3/30 8:43
# @Author  : TangKai
# @Team    : ZheChengData

import pyproj

import geopandas as gpd
from shapely.geometry import Point
from src.gotrackit.netxfer.SumoConvert import SumoConvert
from gotrackit.netreverse.RoadNet.save_file import save_file

if __name__ == '__main__':
    sc = SumoConvert()
    edge_gdf, junction_gdf, lane_polygon_gdf, avg_center_gdf = sc.get_net_shp(
        net_path=r'./data/input/net/test/MICRO-2024-03-30/osm.net.xml',
        prj4_str="+proj=utm +zone=49 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
        core_num=5)

    save_file(data_item=avg_center_gdf, out_fldr=r'./data/input/net/test/MICRO-2024-03-30/',
              file_type='shp', file_name='center')
    print(edge_gdf.columns)
    print(junction_gdf.columns)
    print(lane_polygon_gdf.columns)
    print(avg_center_gdf.columns)
    save_file(data_item=edge_gdf, out_fldr=r'./data/input/net/test/MICRO-2024-03-30/',
              file_type='shp', file_name='edge')
    save_file(data_item=junction_gdf, out_fldr=r'./data/input/net/test/MICRO-2024-03-30/',
              file_type='shp', file_name='junction')
    save_file(data_item=lane_polygon_gdf, out_fldr=r'./data/input/net/test/MICRO-2024-03-30/',
              file_type='shp', file_name='lane_area')
