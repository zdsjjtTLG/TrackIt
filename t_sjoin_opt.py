# -- coding: utf-8 --
# @Time    : 2024/6/2 20:46
# @Author  : TangKai
# @Team    : ZheChengData
import pickle
import time

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from src.gotrackit.tools.grid import get_grid_data

def t():
    grid_df = get_grid_data(polygon_gdf=gpd.GeoDataFrame(geometry=[Polygon([(113, 22), (114, 22),
                                                                            (114, 23), (113, 23)])],
                                                         crs='EPSG:4326'), meter_step=1000)
    print(grid_df)


if __name__ == '__main__':

    # gps_df = pd.read_csv(r'./data/input/net/test/0516BUG/车辆轨迹V3.csv', encoding='ANSI')
    #
    # link = gpd.read_file(r'./data/input/net/test/0516BUG/shp/Final_link_net.shp')

    # link = pd.concat([link, link])
    # link.reset_index(inplace=True, drop=True)

    # print(link)
    # bound_link = link.bounds
    # min_lng, min_lat, max_lng, max_lat = bound_link['minx'].min(), bound_link['miny'].min(), bound_link['maxx'].max(), bound_link['maxy'].max()
    # # print(Polygon([(min_lng, min_lat), (max_lng, min_lat), (max_lng, max_lat), (min_lng, max_lat)]))
    # grid_df = get_grid_data(polygon_gdf=gpd.GeoDataFrame(geometry=[Polygon([(min_lng, min_lat), (max_lng, min_lat),
    #                                                                         (max_lng, max_lat), (min_lng, max_lat)])],
    #                                                      crs='EPSG:4326'), meter_step=10000)
    #
    # grid_df.to_file('./data/input/net/test/0516BUG/shp/grid.shp')
    #
    # grid_df = gpd.read_file('./data/input/net/test/0516BUG/shp/grid.shp')
    # grid_df['grid_id'] = [i for i in range(1, len(grid_df) + 1)]
    #
    # sjoin_cache = gpd.sjoin(grid_df, link[['link_id', 'geometry']])
    #
    # used_grid = set(sjoin_cache['grid_id'])
    #
    # grid_df = grid_df[grid_df['grid_id'].isin(used_grid)]
    #
    # grid_df.to_file('./data/input/net/test/0516BUG/shp/grid.shp')
    #
    # with open('./data/input/net/test/0516BUG/shp/sjoin_cache', 'wb') as f:
    #     pickle.dump(sjoin_cache[['grid_id', 'link_id']], f)

    # with open('./data/input/net/test/0516BUG/shp/sjoin_cache', 'rb') as f:
    #     sjoin_cache = pickle.load(f)
    # print(sjoin_cache)
    # grid_df = gpd.read_file('./data/input/net/test/0516BUG/shp/grid.shp')
    #
    # from shapely.ops import unary_union
    # gps_df['geometry'] = gps_df[['lng', 'lat']].apply(lambda row: Point(row), axis=1)
    # gps_df = gpd.GeoDataFrame(gps_df, geometry='geometry', crs='EPSG:4326')
    # gps_df = gps_df.to_crs('EPSG:32648')
    # gps_df['geometry'] = gps_df['geometry'].buffer(500.0)
    # gps_df = gps_df.to_crs('EPSG:4326')
    #
    # aaa = time.time()
    # rou = unary_union(gps_df['geometry'].to_list())
    #
    # gps_array_buffer_gdf = gpd.GeoDataFrame({'geometry': [rou]}, geometry='geometry',
    #                                         crs='EPSG:4326')
    # print(time.time() - aaa)
    # print(gps_array_buffer_gdf)
    # x = time.time()
    # x1 = gpd.sjoin(gps_array_buffer_gdf, link)
    # print(time.time() - x)
    # print(x1['link_id'])
    # print(len(x1['link_id'].unique()))
    #
    # z = time.time()
    # c = time.time()
    # j1 = gpd.sjoin(gps_array_buffer_gdf, grid_df)
    # print(time.time() - c)
    # used_grid = set(j1['grid_id'])
    # res = set(sjoin_cache[sjoin_cache['grid_id'].isin(used_grid)]['link_id'])
    # link_a = link[link['link_id'].isin(res)]
    # r = gpd.sjoin(link_a, gps_array_buffer_gdf)
    # print(time.time() - z)
    # res = r['link_id'].unique()
    # print(len(res))

    t()
    # get_grid_data(polygon_gdf=gpd.GeoDataFrame({'geometry': [Polygon([(70.0, ), (),
    #                                                                   (), ()])]}), meter_step=1000, is_geo_coord=True)