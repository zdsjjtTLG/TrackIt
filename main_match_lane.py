# -- coding: utf-8 --
# @Time    : 2024/2/29 19:56
# @Author  : TangKai
# @Team    : ZheChengData
import time

import geopandas as gpd
import pandas as pd
from src.gotrackit.map.Net import Net
from src.gotrackit.gps.LocGps import GpsPointsGdf
from src.gotrackit.model.Markov import HiddenMarkov
from src.gotrackit.GlobalVal import NetField, GpsField
from src.gotrackit.visualization import VisualizationCombination
import src.gotrackit.netreverse.NetGen as ng


def generate_net_from_lane():
    lane_center_gdf = gpd.read_file(r'./data/input/net/lane/edge_3857.shp')
    print(lane_center_gdf)
    print(lane_center_gdf['id'].unique())
    lane_center_gdf = lane_center_gdf.to_crs('EPSG:4326')

    nv = ng.NetReverse()
    link_gdf, node_gdf, node_group_status_gdf = nv.create_node_from_link(link_gdf=lane_center_gdf,
                                                                         update_link_field_list=['link_id', 'from_node',
                                                                                                 'to_node',
                                                                                                 'dir', 'length'],
                                                                         execute_modify=True,
                                                                         modify_minimum_buffer=0.35,
                                                                         fill_dir=1,
                                                                         plain_prj='EPSG:32650',
                                                                         out_fldr=r'./data/input/net/lane/')


def match():
    link_gdf, node_gdf, trajectory_gdf = clean()

    # 1.新建一个路网对象, 并且使用平面坐标
    plain_crs = 'EPSG:32650'
    geo_crs = 'EPSG:4326'
    my_net = Net(link_gdf=link_gdf,
                 node_gdf=node_gdf,
                 weight_field='length',
                 geo_crs=geo_crs, plane_crs=plain_crs)

    match_res = pd.DataFrame()

    # 初始化
    my_net.init_net()
    used_agent = list(trajectory_gdf.sample(n=20)['agent_id'].unique())
    for agent_id, tra_gdf in trajectory_gdf.groupby('agent_id'):
        print(agent_id)
        # if agent_id == 43:
        #     a = 1
        gps_obj = GpsPointsGdf(gps_points_df=tra_gdf, time_format='%Y-%m-%d %H:%M:%S.%f', geo_crs='EPSG:4326',
                               plane_crs=plain_crs, buffer=8.1)

        # 初始化一个隐马尔可夫模型
        hmm_obj = HiddenMarkov(net=my_net, gps_points=gps_obj, beta=10.2, gps_sigma=5.0)
        hmm_obj.generate_markov_para()
        hmm_obj.solve()

        x = hmm_obj.acquire_res()
        # hmm_obj.acquire_geo_res(out_fldr=r'./data/output/match_visualization/lane/', flag_name=rf'lane_{agent_id}')

        x[['prj_lng', 'prj_lat']] = x.apply(lambda item: (item['prj_geo'].x, item['prj_geo'].y), axis=1, result_type='expand')
        x[['lng', 'lat']] = x.apply(lambda item: (item['geometry'].x, item['geometry'].y), axis=1, result_type='expand')
        x = x[['agent_id', 'veh_type', 'speed', 'time', 'lng', 'lat', 'prj_lng', 'prj_lat', 'link_id']].copy()
        x = pd.merge(x, link_gdf[['link_id', 'id', 'index']], on='link_id', how='left')
        x.rename(columns={'id': 'edge_id', 'index': 'lane_index'}, inplace=True)
        x.drop(columns=['link_id'], axis=1, inplace=True)
        match_res = pd.concat([match_res, x])

    match_res.reset_index(inplace=True, drop=True)
    match_res.to_csv(r'./data/output/match_visualization/lane/trajectory_lane_match.csv',
                     encoding='utf_8_sig', index=False)
    print(match_res)


def clean():
    link_gdf = gpd.read_file(r'./data/input/net/lane/LinkAfterModify.shp')
    node_gdf = gpd.read_file(r'./data/input/net/lane/NodeAfterModify.shp')

    trajectory_gdf = gpd.read_file(r'./data/output/gps/lane/trajectory_3857.shp')

    trajectory_gdf.rename(columns={'id': 'agent_id'}, inplace=True)
    trajectory_gdf['time'] = pd.to_datetime(trajectory_gdf['time'], unit='ms')
    trajectory_gdf = trajectory_gdf.to_crs('EPSG:4326')

    trajectory_gdf[['lng', 'lat']] = trajectory_gdf.apply(lambda item: (item['geometry'].x, item['geometry'].y),
                                                          result_type='expand', axis=1)
    return link_gdf, node_gdf, trajectory_gdf


if __name__ == '__main__':
    # generate_net_from_lane()

    match()