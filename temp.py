# -- coding: utf-8 --
# @Time    : 2024/1/13 13:34
# @Author  : TangKai
# @Team    : ZheChengData
import os
import geopandas as gpd
import datetime
from src.gotrackit.tools.coord_trans import LngLatTransfer
con = LngLatTransfer()


if __name__ == '__main__':
    link = gpd.read_file(r'./data/input/net/xian/link_a.shp')
    # node = gpd.read_file(r'./data/input/net/xian/node_a.shp')
    # link = link.to_crs('EPSG:4326')
    # node = node.to_crs('EPSG:4326')
    # print(link['geometry'])
    # print(node['geometry'])
    # used_node = set(link['from_node']) | set(link['to_node'])
    #
    # node = node[node['node_id'].isin(list(used_node))].copy()
    # node.reset_index(inplace=True, drop=True)
    #
    #
    # link['geometry'] = link['geometry'].apply(lambda x: con.obj_convert(geo_obj=x, con_type='gc-84'))
    # node['geometry'] = node['geometry'].apply(lambda x: con.obj_convert(geo_obj=x, con_type='gc-84'))
    #
    # node.to_file(r'./data/input/net/xian/node.shp', encoding='gbk')
    # link.to_file(r'./data/input/net/xian/link.shp', encoding='gbk')


    # for fldr in [r'gps', 'mix', 'trajectory']:
    #     print(fldr)
    #     for file in os.listdir(rf'./data/output/{fldr}/'):
    #         print(file)
    #         gdf = gpd.read_file(rf'./data/output/{fldr}/{file}')
    #
    #         gdf['geometry'] = gdf['geometry'].apply(lambda x: con.obj_convert(geo_obj=x, con_type='gc-84'))
    #         gdf['lng'] = gdf['geometry'].apply(lambda x: x.x)
    #         gdf['lat'] = gdf['geometry'].apply(lambda x: x.y)
    #
    #         gdf.to_file(rf'./data/output/{fldr}/{file}', encoding='gbk', driver='GeoJSON')

    x = datetime.datetime(year=2022, month=5, day=12, hour=16, minute=14, second=0)
    print(x)

    for i, a in enumerate([1,2,3,6]):
        print(i, a)