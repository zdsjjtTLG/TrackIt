# -- coding: utf-8 --
# @Author  : TangKai
# @Team    : ZheChengData

"""使用原 Python 版 gotrackit 跑 data/input 下的实际匹配案例。"""
import datetime
import os
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from gotrackit.MapMatch import MapMatch
from gotrackit.MatchResAna import dense_res_based_on_net, generate_check_file
from gotrackit.WrapsFunc import function_time_cost
from gotrackit.gps.LocGps import GpsPointsGdf
from gotrackit.gps.Trajectory import TrajectoryPoints
from gotrackit.map.Net import Net
from gotrackit.model.Para import ParaGrid

def t_cq_match():
    gps_df = gpd.read_file(rf'./data/input/cq/gps.shp')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                          result_type='expand')
    gps_df['link_id'] = 999
    gps_df.drop(columns=['geometry'], axis=1, inplace=True)
    my_net = Net(link_path=r'./data/input/cq/modifiedConn_link.shp',
                 node_path=r'./data/input/cq/modifiedConn_node.shp', not_conn_cost=1200, fmm_cache=True,
                 cache_cn=3, prj_cache=True,
                 recalc_cache=False, fmm_cache_fldr=r'./data/input/cq/', is_hierarchical=True)
    my_net.init_net()
    pgd = ParaGrid(beta_list=[1.0, 2.0,6.0], gps_sigma_list=[10.0, 20.0, 120.0])
    # match
    mpm = MapMatch(net=my_net,
                   is_rolling_average=False, flag_name='cq_test',
                   export_html=True, export_geo_res=True,
                   out_fldr=r'./data/output/match_visualization/cq', dense_gps=False,
                   use_sub_net=True, multi_core_save=True, instant_output=True, use_para_grid=True, para_grid=pgd,
                   user_field_list=['link_id'], heading_vec_len=25.0)
    match_res, warn_info, error_info = mpm.execute(gps_df=gps_df)
    print(warn_info)
    print(match_res)
    print(error_info)
    print(pgd.search_res)

@function_time_cost
def t_0327sample_match():

    # 读取GPS数据
    # 这是一个有10辆车的GPS数据的文件
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    gps_df = pd.read_csv(r'./data/input/0327sample/0327sample.csv')
    tp = TrajectoryPoints(gps_points_df=gps_df, plain_crs='EPSG:32649')
    tp.lower_frequency(2).kf_smooth(o_deviation=0.4)

    gps_df = tp.trajectory_data(_type='df')
    print(gps_df)

    # gps_df = gps_df[gps_df['agent_id'].isin(['xa_car_8'])]
    l = gpd.read_file(r'./data/input/0327sample/modifiedConn_link.shp')
    n = gpd.read_file(r'./data/input/0327sample/modifiedConn_node.shp')

    my_net = Net(link_gdf=l, prj_cache=True,
                 node_gdf=n, fmm_cache=False, fmm_cache_fldr=r'./data/input/0327sample/', recalc_cache=False, cache_cn=1,
                 cache_slice=6, grid_len=2000, is_hierarchical=False)
    my_net.init_net()  # net初始化
    # my_net = Net(link_gdf=l,
    #              node_gdf=n)
    # my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_buffer=100, flag_name='xa_sample',
                   use_sub_net=False, use_heading_inf=True, omitted_l=6.0,
                   del_dwell=False, dwell_l_length=50.0, dwell_n=0,
                   is_lower_f=True, lower_n=1, user_field_list=['agent_id', 'seq'],
                   export_html=True, export_geo_res=True, use_gps_source=True,
                   export_all_agents=True, instant_output=False, visualization_cache_times=20,
                   is_rolling_average=False, window=3,
                   out_fldr=r'./data/output/match_visualization/xa_sample', dense_gps=False,
                   gps_radius=10.0)
    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的agent的相关信息({agent_id1: pd.DataFrame(), agent_id2: pd.DataFrame()...})
    # 第三个是匹配出错的agent的id列表(GPS点经过预处理(或者原始数据)后点数量不足2个)
    # mpm.execute()

    match_res, warn_info, error_info = mpm.execute(gps_df=gps_df)
    print(warn_info)
    print(match_res)
    print(error_info)
    match_res.to_csv(fr'./data/output/match_visualization/xa_sample/match_res.csv', encoding='utf_8_sig', index=False)

@function_time_cost
def t_0424sample_match():

    # 读取GPS数据
    # 这是一个有10辆车的GPS数据的文件
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    gps_df = gpd.read_file(r'./data/input/0424sample/0424sample.geojson')
    gps_df.drop(columns=['geometry'], axis=1, inplace=True)
    gps_df['from_node'] = 563
    # gps_df = gps_df[gps_df['agent_id'].isin(['xa_car_22'])]
    print(gps_df)
    l = gpd.read_file(r'./data/input/0424sample/modifiedConn_link.shp')
    n = gpd.read_file(r'./data/input/0424sample/modifiedConn_node.shp')
    n.crs = 'epsg:4326'
    # l = gpd.read_file(r'./data/input/QuickStart-Match-1/LinkAfterModify.shp')
    # n = gpd.read_file(r'./data/input/QuickStart-Match-1/NodeAfterModify.shp')
    my_net = Net(link_gdf=l, prj_cache=True,
                 node_gdf=n, fmm_cache=False, fmm_cache_fldr=r'./data/input/0424sample/', recalc_cache=False, cache_cn=3,
                 cache_slice=6, cut_off=1000, not_conn_cost=1200.0, is_hierarchical=True)
    my_net.init_net()  # net初始化
    a = time.time()
    mpm = MapMatch(net=my_net, gps_buffer=500, flag_name='xa_sample_0424', speed_threshold=120,
                   use_sub_net=True, use_heading_inf=True,
                   omitted_l=6.0, del_dwell=True, dwell_l_length=25.0, dwell_n=1,
                   lower_n=2, is_lower_f=False, top_k=20,
                   is_rolling_average=False, window=3,
                   export_html=True, export_geo_res=False, use_gps_source=False,
                   out_fldr=r'./data/output/0424sample/', dense_gps=True,
                   gps_radius=20.0, multi_core_save=True, instant_output=True, user_field_list=['from_node'],
                   heading_vec_len=50)

    # match_res, warn_info, error_info = mpm.execute()
    # match_res, warn_info, error_info = mpm.multi_core_execute(gps_df=gps_df, core_num=2)
    match_res, warn_info, error_info = mpm.execute(gps_df=gps_df, thread_num=2)
    print(time.time() - a)
    print(error_info)
    print(warn_info)

def t_denseexample_match():

    gps_df = gpd.read_file(r'./data/input/denseexample/test999.geojson')
    gps_df['heading12'] = [i for i in range(len(gps_df))]
    gps_df['sub_seq'] = 969126
    gps_df['agent_id'] = 'car_pop'
    link = gpd.read_file(r'./data/input/denseexample/modifiedConn_link.shp')
    node = gpd.read_file(r'./data/input/denseexample/modifiedConn_node.shp')
    # del link['speed']
    print(link.dtypes)
    print(sys.getsizeof(link))
    my_net = Net(link_gdf=link,
                 node_gdf=node, fmm_cache=False,
                 recalc_cache=True, fmm_cache_fldr=r'./data/input/denseexample', grid_len=1000, cache_cn=1, prj_cache=True,
                 is_hierarchical=True, not_conn_cost=1750.23)
    my_net.init_net()
    pgd = ParaGrid(use_heading_inf_list=[False, True], beta_list=[0.1, ], gps_sigma_list=[0.1, 0.001])
    # match
    mpm = MapMatch(net=my_net, is_rolling_average=False, window=2, flag_name='denseexample',
                   export_html=True, export_geo_res=True,
                   gps_buffer=400,
                   out_fldr=r'./data/output/dense_example/',
                   dense_gps=True, use_st=False,
                   use_sub_net=True, dense_interval=50.0, use_gps_source=False, use_heading_inf=True,
                   gps_radius=15.0, use_para_grid=False, para_grid=pgd, heading_vec_len=15,
                   user_field_list=['sub_seq', 'agent_id', 'lng'])
    match_res_df, warn_info, error_info = mpm.execute(gps_df=gps_df)
    # print(gps_df.columns)
    # print(match_res_df.columns)
    print(match_res_df)
    # print(res)
    # print(warn_info)
    # print(error_info)
    # print(pd.DataFrame(pgd.search_res))
    match_res_df.to_csv(r'./data/output/match_visualization/dense_example/match_res.csv', encoding='utf_8_sig', index=False)
    # print(gpd.__version__)

    dense_res = dense_res_based_on_net(net=my_net, match_res_df=match_res_df)

    dense_res.to_csv(r'./data/output/match_visualization/dense_example/dense_res.csv', encoding='utf_8_sig', index=False)

def t_0326taxi_match():
    gps_df = pd.read_csv(r'./data/input/0326taxi/TaxiData-Sample-a.csv')
    gps_df['lng'] = np.around(gps_df['lng'], 6)
    gps_df['lat'] = np.around(gps_df['lat'], 6)

    link = gpd.read_file(r'./data/input/0326taxi/new_link.shp')
    link['speed'] = [np.random.randint(20, 60) for i in range(len(link))]
    my_net = Net(link_gdf=link,
                 node_path=r'./data/input/0326taxi/modifiedConn_node.shp',
                 not_conn_cost=2000, fmm_cache=False, recalc_cache=False, cut_off=800.0, prj_cache=True,
                 fmm_cache_fldr=r'./data/input/0326taxi/', cache_cn=6, is_hierarchical=True)
    my_net.init_net()
    pgd = ParaGrid(gps_sigma_list=[10, 20, 30], beta_list=[10, 20, 30])
    # match
    gps_df['rkkkl'] = [i for i in range(len(gps_df))]
    gps_df['prj_lng'] = [i + 1 for i in range(len(gps_df))]
    mpm = MapMatch(net=my_net,
                   is_rolling_average=False, window=2, flag_name='0326_taxi',
                   export_html=True, export_geo_res=True, gps_sigma=10,
                   gps_buffer=500, out_fldr=r'./data/output/0326_taxi',
                   dense_gps=False, dense_interval=200, use_heading_inf=False,
                   top_k=40, dwell_n=0,
                   use_sub_net=False, use_gps_source=False, gps_radius=20, instant_output=True, use_para_grid=False,
                   para_grid=pgd, heading_vec_len=80, user_field_list=['rkkkl', 'prj_lng'])
    res, may_error_info, error_info = mpm.execute(gps_df=gps_df)

def t_check0325_match():
    # 某快速路匹配示例
    my_net = Net(link_path=r'./data/input/check0325/G15_links.shp',
                 node_path=r'./data/input/check0325/G15_gps_node.shp', fmm_cache=True, cut_off=1000.0, recalc_cache=False,
                 fmm_cache_fldr=r'./data/input/check0325/', prj_cache=True)
    my_net.init_net()

    gps_df = pd.read_csv(r'./data/input/check0325/car_gps_test_noheading.csv')
    gps_df = gps_df.loc[0:260, :].copy()
    print(gps_df)
    mpm = MapMatch(net=my_net,
                   is_rolling_average=False, window=2,
                   flag_name='check0325', export_html=True, export_geo_res=True,
                   out_fldr=r'./data/output/match_visualization/sample', instant_output=True)
    res, may_error_info, error_info = mpm.execute(gps_df=gps_df)
    print(res)
    print(may_error_info)
    print(error_info)

def t_bug0329_match():
    from shapely.geometry import Point
    link = gpd.read_file(r'./data/input/bug0329/LinkAfterModify.shp')
    node = gpd.read_file(r'./data/input/bug0329/NodeAfterModify.shp')
    link = link.to_crs('EPSG:4326')
    node = node.to_crs('EPSG:4326')

    print(link)
    print(node)
    gps_df = gpd.read_file(r'./data/input/bug0329/trajectory_helmert.shp')
    print(gps_df.crs)
    gps_df = gps_df.to_crs('EPSG:4326')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                          result_type='expand')

def t_bug0516_match():
    # 1.读取GPS数据
    # 这是一个有10辆车的GPS数据的文件, 已经做过了数据清洗以及行程切分
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    gps_df = pd.read_csv(r'./data/input/bug0516/车辆轨迹V3.csv', encoding='gbk')
    gps_df = gps_df[['agent_id', 'time', 'lng', 'lat']].copy()
    print(gps_df)
    # gps_df = gpd.read_file(r'./data/input/0516BUG/gps_test_0605.shp')
    # gps_df = gps_df[gps_df['id'] == 123]
    # gps_df['agent_id'] = 1111
    # print(gps_df)
    # gps_df['time'] = [datetime.datetime.now().timestamp() + i for i in range(len(gps_df))]
    # gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
    #                                       result_type='expand')
    # del gps_df['geometry']
    link = gpd.read_file(r'./data/input/bug0516/link.shp')
    node = gpd.read_file(r'./data/input/bug0516/Final_node_net.shp')
    

    # link = link.to_crs('EPSG:32648')
    # link['geometry'] = link['geometry'].simplify(20.0)
    # link = link.to_crs('EPSG:4326')
    # link.to_file(r'./data/input/0516BUG/shp/link.shp')
    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_gdf=link,
                 node_gdf=node, not_conn_cost=2000.0, cut_off=1000.0, grid_len=4000, is_hierarchical=True, prj_cache=True)
    my_net.init_net()  # net初始化
    # para_grid = ParaGrid(gps_sigma_list=[i for i in range(1, 40, 10)],
    #                      beta_list=[0.1, 6.0], use_heading_inf_list=[False, True])
    mpm = MapMatch(net=my_net, gps_buffer=700.0, flag_name='id1',
                   use_sub_net=True, use_heading_inf=False,
                   is_rolling_average=False, window=2, dense_interval=400.0, dense_gps=True,
                   omitted_l=20.1, del_dwell=False, dwell_l_length=15.0,
                   export_html=True, export_geo_res=True, top_k=20, is_lower_f=False, lower_n=2,
                   out_fldr=r'./data/output/bug0516/',
                   use_gps_source=False, gps_radius=20.0, instant_output=True)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = mpm.execute(gps_df=gps_df)
    match_res.to_csv(r'./data/output/bug0516/match_res.csv', encoding='utf_8_sig', index=False)
    print(warn_info)

    generate_check_file(my_net, warn_info_dict=warn_info, out_fldr=r'./data/output/bug0516/', )

def t_agents150_match():
    # 1.读取GPS数据
    gps_df = pd.read_csv(r'./data/input/agents150/150_agents.csv')
    my_net = Net(link_path=r'./data/input/agents150/new_link.shp',
                 node_path=r'./data/input/agents150/modifiedConn_node.shp', not_conn_cost=999.0,
                 fmm_cache=True, fmm_cache_fldr=r'./data/input/agents150/', cache_cn=5, prj_cache=True,
                 recalc_cache=False, is_hierarchical=True)
    my_net.init_net()  # net初始化

    a = time.time()
    para_grid = ParaGrid(gps_sigma_list=[1.0, 2.0, 3.0],
                         beta_list=[500, 1000], use_heading_inf_list=[False])
    a = time.time()
    mpm = MapMatch(net=my_net, gps_buffer=120, flag_name='id1',
                   use_sub_net=False, use_heading_inf=True,
                   is_rolling_average=True, window=2, dense_interval=80, dense_gps=False,
                   omitted_l=20.1, del_dwell=True, dwell_l_length=15.0,
                   export_html=True, export_geo_res=False, top_k=20,
                   out_fldr=r'./data/output/agents150/', instant_output=True,
                   use_gps_source=True, gps_radius=10.0, use_para_grid=False, para_grid=para_grid,
                   visualization_cache_times=2)
    # match_res, warn_info, error_info = mpm.execute()
    match_res, warn_info, error_info = mpm.multi_core_execute(gps_df=gps_df, core_num=3)
    print(time.time() - a)
    print(match_res)
    print(warn_info)
    print(error_info)
    print(para_grid.search_res)

@function_time_cost
def t_route2gps_match():

    gps_df = pd.read_csv(r'./data/input/route2gps/gps.csv')
    # gps_df = gps_df[gps_df['agent_id'].isin(['11-3', '15-1'])]
    gps_df['heading1'] = 123
    gps_df['my_field'] = 'apolq'
    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_path=r'./data/input/route2gps/FinalLink.shp',
                 node_path=r'./data/input/route2gps/FinalNode.shp', not_conn_cost=1500.0, fmm_cache=False,
                 cut_off=1500, recalc_cache=False, fmm_cache_fldr=r'./data/input/route2gps/')
    my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_buffer=50, flag_name='cd_route2gps',
                   use_sub_net=False, use_heading_inf=True,
                   is_rolling_average=False, is_lower_f=False, lower_n=2, dense_gps=False,
                   time_format='"%Y-%m-%d %H:%M:%S"', dense_interval=30.0,
                   omitted_l=2.0, del_dwell=True, dwell_l_length=20.0, dwell_n=0,
                   export_html=True, export_geo_res=False, top_k=5, instant_output=False,
                   out_fldr=r'./data/output/route2gps/',
                   use_gps_source=False, gps_radius=10.0, export_all_agents=True, multi_core_save=False,
                   user_field_list=['heading1', 'my_field'])

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, _ = mpm.execute(gps_df=gps_df,)
    match_res.to_csv(r'./data/output/route2gps/match_res.csv',
                     encoding='utf_8_sig', index=False)
    print(match_res)
    print(match_res.columns)
    print(_)

def t_0419test_match():
    gps = gpd.read_file(r'./data/input/0419test/gps.shp')
    gps['agent_id'] = 1
    gps['time'] = [datetime.datetime.now().timestamp() + i for i in range(len(gps))]

    gps[['lng', 'lat']] = gps.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1, result_type='expand')
    gps.drop(columns=['geometry'], axis=1, inplace=True)
    # gps = gps.loc[[0, 1], :].copy()

    my_net = Net(link_path=r'./data/input/0419test/LinkAfterModify.shp',
                 node_path=r'./data/input/0419test/NodeAfterModify.shp', not_conn_cost=1500,
                 prj_cache=True)

    my_net.init_net()  # net初始化
    mpm = MapMatch(net=my_net, gps_buffer=500, flag_name='cd_route2gps',
                   use_sub_net=False, use_heading_inf=True,
                   is_rolling_average=False, window=3, is_lower_f=True, lower_n=2, dense_gps=False,
                   omitted_l=2.0, del_dwell=True,
                   gps_sigma=30,
                   export_html=True, export_geo_res=True, top_k=10,
                   out_fldr=r'./data/output/match_visualization/0419test',
                   use_gps_source=False,
                   gps_radius=10.0)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = mpm.execute(gps_df=gps,)
    match_res.to_csv(r'./data/output/match_visualization/0318cd_route2gps/match_res.csv', encoding='utf_8_sig')
    print(warn_info)
    print(match_res)
    print(error_info)
    for may_error_agent in warn_info.keys():
        print(may_error_agent)
        print(warn_info[may_error_agent])

def leeds_yg_test():
    # origin_link = gpd.read_file(r'./data/input/0506yg/origin_link.shp')
    # origin_node = gpd.read_file(r'./data/input/0506yg/origin_node.shp')
    #
    # node_map = {ori: new for ori, new in zip(origin_node['TOID'], range(1, len(origin_node) + 1))}
    # origin_link['dir'] = 1
    # origin_link.loc[origin_link['directiona'] == 'both directions', 'dir'] = 0
    # origin_link.loc[origin_link['directiona'] == 'in direction', 'dir'] = 1
    # origin_link.loc[origin_link['directiona'] == 'in opposite direction', 'dir'] = -1
    # origin_link.rename(columns={'endnode_hr': 'to_node',
    #                             'startnode_': 'from_node'}, inplace=True)
    #
    #
    # neg_index = origin_link['dir'] == -1
    # neg_link = origin_link[neg_index].copy()
    # neg_link[['from_node', 'to_node']] = neg_link[['to_node', 'from_node']]
    # from shapely.geometry import LineString
    # neg_link['geometry'] = neg_link['geometry'].apply(
    #     lambda geo: LineString(list(geo.coords)[::-1]))
    # neg_link['dir'] = 1
    # origin_link.drop(index=origin_link[neg_index].index, axis=0, inplace=True)
    # origin_link = pd.concat([origin_link, neg_link])
    # origin_link.reset_index(inplace=True, drop=True)
    #
    # origin_node['node_id'] = origin_node['TOID'].map(node_map)
    #
    # print(len(origin_link))
    # origin_link['from_node'] = origin_link['from_node'].map(node_map)
    # origin_link['to_node'] = origin_link['to_node'].map(node_map)
    # origin_link.dropna(subset=['from_node', 'to_node'], inplace=True, how='any')
    # origin_link.reset_index(inplace=True, drop=True)
    # print(len(origin_link))
    # origin_link['link_id'] = [i for i in range(1, len(origin_link) + 1)]
    # origin_link['from_node'] = origin_link['from_node'].astype(int)
    # origin_link['to_node'] = origin_link['to_node'].astype(int)
    # origin_link['geometry'] = origin_link['geometry'].remove_repeated_points(1.5)
    # origin_link['l'] = origin_link['geometry'].length
    # origin_link = origin_link.to_crs('EPSG:4326')
    # origin_node = origin_node.to_crs('EPSG:4326')
    # origin_link[['TOID', 'link_id', 'l', 'from_node', 'to_node', 'dir', 'length', 'geometry']].to_file(
    #     './data/input/0506yg/link.shp', encoding='gbk')
    #
    # origin_node[['TOID', 'node_id', 'geometry']].to_file(
    #     './data/input/0506yg/node.shp', encoding='gbk')
    #

    gps_df = pd.read_csv(r'./data/input/0506yg/gps.csv')
    gps_df['agent_id'] = gps_df['agent_id'].astype(int)
    # gps_df = gpd.read_file(r'./data/input/0506yg/zf_gps.shp')
    # gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
    #                                       result_type='expand')
    # del gps_df['geometry']
    # gps_df['agent_id'] = 123
    #
    gps_df['time'] = [datetime.datetime.now().timestamp() + i for i in range(0, len(gps_df))]
    gps_df['speed'] = 123
    x = gpd.read_file(r'./data/input/0506yg/aaa_link.shp')
    print(x)
    my_net = Net(link_path='./data/input/0506yg/aaa_link.shp', prj_cache=True,
                 node_path='./data/input/0506yg/aaa_node.shp', is_hierarchical=True)
    my_net.init_net()  # net初始化
    mpm = MapMatch(net=my_net,  gps_buffer=60.0, flag_name='yg1',
                   use_sub_net=True, use_heading_inf=False,
                   is_rolling_average=False, window=2, dense_interval=80, dense_gps=False,
                   omitted_l=20.1, del_dwell=False, dwell_l_length=15.0, time_unit='s',
                   export_html=True, export_geo_res=True, top_k=20, is_lower_f=False, lower_n=2,
                   instant_output=True,
                   out_fldr=r'./data/output/match_visualization/0506yg/', user_field_list=['speed'],
                   use_gps_source=True, gps_radius=10.0)
    # mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=60.0, flag_name='yg1', beta=5.6,
    #                use_sub_net=False, use_heading_inf=True, max_increment_times=2, increment_buffer=100,
    #                is_rolling_average=False, window=2, dense_interval=80, dense_gps=False,
    #                omitted_l=20.1, del_dwell=False, dwell_l_length=15.0,
    #                export_html=True, export_geo_res=False, top_k=20, is_lower_f=False, lower_n=2,
    #                out_fldr=r'./data/output/match_visualization/0506yg/',
    #                use_gps_source=True, gps_radius=10.0, instant_output=True)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = mpm.execute(gps_df=gps_df,)
    # match_res.to_csv(r'./data/output/match_visualization/0506yg/match_res.csv', encoding='utf_8_sig')
    print(warn_info)
    print(match_res)
    print(error_info)

def xishu_0507():
    # 1.读取GPS数据
    # 这是一个有10辆车的GPS数据的文件, 已经做过了数据清洗以及行程切分
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    p = ParaGrid(beta_list=[1, 4, 6, 14, 20], gps_sigma_list=[10, 20, 50, 120, 150], omitted_l_list=[1, 5, 10])
    gps_df = gpd.read_file(r'./data/input/0507xishu/163.csv')
    # gps_df = gpd.read_file(r'./data/input/0507xishu/163 - 副本.csv')
    print(gps_df.columns)
    gps_df['time'] = [datetime.datetime.now().timestamp() + i for i in range(0, len(gps_df))]
    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_path=r'./data/input/0507xishu/FinalLink.shp',
                 node_path=r'./data/input/0507xishu/FinalNode.shp', not_conn_cost=3200.0,
                 fmm_cache=False, fmm_cache_fldr=r'./data/input/0507xishu', cache_cn=6,
                 recalc_cache=False, cut_off=2500.0, prj_cache=True)
    my_net.init_net()  # net初始化
    mpm = MapMatch(net=my_net, gps_buffer=500.0, flag_name='0507_xishu_120dense',
                   use_sub_net=True, use_heading_inf=True, dense_gps=False,
                   dense_interval=400, beta=20,
                   omitted_l=5.1, del_dwell=True, dwell_l_length=25.0, dwell_n=1,
                   export_html=True, export_geo_res=True, top_k=25,
                   out_fldr=r'./data/output/match_visualization/稀疏/',
                   use_gps_source=False, gps_radius=10.0, instant_output=True,
                   user_field_list=['lng', 'seq'])

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = mpm.execute(gps_df=gps_df)
    print(match_res)
    print(pd.DataFrame(p.search_res))

def t_carla0605_match():
    gps_df = gpd.read_file(r'./data/input/carla0605/tra.geojson')
    gps_df = gps_df.to_crs('EPSG:4326')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda item: (item['geometry'].x, item['geometry'].y),
                                          result_type='expand', axis=1)
    gps_df = pd.DataFrame(gps_df)
    gps_df['aq'] = 1
    # gps_df = gps_df.loc[1000:1400, :]
    del gps_df['geometry']
    print(gps_df)

    l = gpd.read_file(r'./data/input/carla0605/LinkAfterModify.shp')
    n = gpd.read_file(r'./data/input/carla0605/NodeAfterModify.shp')
    l = l.to_crs('EPSG:3857')
    l['geometry'] = l['geometry'].simplify(2.0)
    l = l.to_crs('EPSG:4326')
    n = n.to_crs('EPSG:4326')
    my_net = Net(link_gdf=l, node_gdf=n, prj_cache=True)

    my_net.init_net()  # net初始化

    a = time.time()
    mpm = MapMatch(net=my_net, gps_buffer=80.0, flag_name='carla_0605', speed_threshold=150,
                   use_sub_net=False, use_heading_inf=True,
                   omitted_l=6.0, del_dwell=False, dwell_l_length=15.0, dwell_n=2,
                   is_lower_f=False, top_k=20,
                   is_rolling_average=False,
                   export_html=False, export_geo_res=True, use_gps_source=False,
                   out_fldr=r'./data/output/carla0605', dense_gps=False, dense_interval=10.0,
                   gps_radius=10.0, multi_core_save=True, instant_output=True, heading_vec_len=3, user_field_list=['aq', 'lng'])

    match_res, warn_info, error_info = mpm.execute(gps_df=gps_df)
    print(time.time() - a)
    print(warn_info.keys())
    print(match_res)
    print(error_info)

def t_zdsy_match():
    gps_df = gpd.read_file(r'./data/input/zdsy/gps.csv')

    # gps_buffer = gps_df.copy()
    # gps_buffer['geometry'] = gpd.points_from_xy(gps_df['lng'], gps_df['lat'], crs='EPSG:4326')
    # gps_buffer = gpd.GeoDataFrame(gps_buffer, geometry='geometry', crs='EPSG:4326')
    # gps_buffer = gps_buffer.to_crs('EPSG:32650')
    # gps_buffer['geometry'] = gps_buffer['geometry'].buffer(150)
    # gps_buffer = gps_buffer.to_crs('EPSG:4326')
    # gps_buffer.to_file(r'./data/input/zdsy/gps_buffer.shp')
    # from src.gotrackit.tools.grid import get_grid_data
    # from shapely.geometry import Polygon, LineString
    # from shapely.ops import unary_union
    # min_x, min_y, max_x, max_y = \
    #     gps_df['lng'].min(), gps_df['lat'].min(), gps_df['lng'].max(), gps_df['lat'].max()
    # gird_gdf = get_grid_data(polygon_gdf=gpd.GeoDataFrame({'geometry': [Polygon([(min_x, min_y),
    #                                                                              (max_x, min_y),
    #                                                                              (max_x, max_y),
    #                                                                              (min_x, max_y)])]}, crs='EPSG:4326'), is_geo_coord=True,
    #                          meter_step=1000)
    # gird_gdf.to_file(r'./data/input/zdsy/grid.shp')
    # cor_grid = gpd.sjoin(gird_gdf, gps_buffer)
    # cor_grid.drop_duplicates(subset=['grid_id'], keep='first', inplace=True)
    # cor_grid.to_file(r'./data/input/zdsy/cor_grid.shp')
    #
    # route_path_buffer = unary_union(LineString(gps_buffer['geometry'].centroid).buffer(0.002))
    # gpd.GeoDataFrame({'geometry': [route_path_buffer]}, crs='EPSG:4326').to_file(r'./data/input/zdsy/route_buffer.shp')

    l = gpd.read_file(r'./data/input/zdsy/FinalLink_dir.shp')
    n = gpd.read_file(r'./data/input/zdsy/FinalNode.shp')
    l = l.to_crs('EPSG:32650')
    l['geometry'] = l['geometry'].simplify(5.0)
    l = l.to_crs('EPSG:4326')
    l['speed'] = [np.random.randint(20, 60) for i in range(len(l))]
    l[['from_node', 'to_node']] = l[['to_node', 'from_node']]
    l['geometry'] = l['geometry'].reverse()
    l.loc[l['dir'] == 1, 'dir'] = -1
    print(len(l))
    l[['from_node', 'to_node']].to_csv(r'F:\PyPrj\pythonProject2\edge.csv', encoding='utf_8_sig', index=False)
    my_net = Net(link_gdf=l,
                 node_gdf=n, grid_len=2000, is_hierarchical=True, cut_off=400, prj_cache=True)

    my_net.init_net()  # net初始化

    a = time.time()
    mpm = MapMatch(net=my_net, gps_buffer=80.0, flag_name='zdsy_test',
                   omitted_l=6.0, del_dwell=False, dwell_l_length=15.0, dwell_n=2,
                   is_lower_f=False, top_k=8, dense_gps=False,
                   is_rolling_average=False,
                   export_html=True, export_geo_res=True, use_gps_source=False,
                   out_fldr=r'./data/output/match_visualization/zdsy_test/',
                   gps_radius=10.0, multi_core_save=True, instant_output=True)

    match_res, warn_info, error_info = mpm.execute(gps_df=gps_df)
    print(time.time() - a)
    print(warn_info.keys())
    print(match_res)
    print(error_info)

def t_ldsobug_match():

    # import gotrackit.netreverse.NetGen as ng
    # link = gpd.read_file(r'./data/input/ldso0703/split_osm_test.geojson')
    # # print(link)
    # nv = ng.NetReverse.create_node_from_link(link_gdf=link,
    #                                          update_link_field_list=['link_id', 'from_node', 'to_node', 'dir', 'length'],
    #                                          fill_dir=0, out_fldr=r'./data/input/ldso0703')
    # # print(node)
    link = gpd.read_file(r'./data/input/ldso0703/LinkAfterModify.shp')
    node = gpd.read_file(r'./data/input/ldso0703/NodeAfterModify.shp')

    link['dir'] = 1

    gps_df = pd.read_csv(r'./data/input/ldso0703/tra_.csv')
    my_net = Net(link_gdf=link, node_gdf=node, prj_cache=True)
    my_net.init_net()
    # mpm = MapMatch(net=my_net, gps_df=gps_df, use_sub_net=False, export_html=True, time_format='%Y/%m/%d %H:%M:%S.%f',
    #                top_k=5, omitted_l=6,  del_dwell=True, dwell_l_length=0.5,dwell_n=1,
    #                out_fldr=r'./data/output/match_visualization/ldso0703/', export_geo_res=True)
    mpm = MapMatch(net=my_net, gps_buffer=30, flag_name='syc_test', gps_route_buffer_gap=0,
                   time_format='%Y/%m/%d %H:%M:%S.%f', use_sub_net=True, use_heading_inf=True, top_k=5,
                   omitted_l=6.0, del_dwell=True, dwell_l_length=0.5, dwell_n=1,  # 相邻点的间距小于dwell_l_length的gps会被删除，不进入匹配
                   lower_n=2, is_lower_f=False,
                   is_rolling_average=False, window=3,
                   dense_gps=False,
                   export_html=True, export_geo_res=True, use_gps_source=False,  # export_html 设置为true 警告信息才会打印全部字段
                   out_fldr=r'./data/output/match_visualization/ldso0703/',  # 结果输出存放文件夹
                   gps_radius=1.0)
    res, warn, _ = mpm.execute(gps_df=gps_df,)

    generate_check_file(my_net, warn_info_dict=warn, out_fldr=r'./data/output/match_visualization/ldso0703/')
    print(res)

def t_gjy0828_match():
    input_fldr = r'./data/input/gjy0828'
    output_fldr = r'./data/output/gjy0828'

    gps_df = pd.read_csv(os.path.join(input_fldr, '公交线路模拟GPS点.csv'))
    gps_df.rename(columns={'stopnodeid': 'node_id'}, inplace=True)
    gps_df.reset_index(inplace=True, drop=True)
    print(len(gps_df))
    # gps_df['node_id'] = None
    # gps_df.loc[10, 'node_id'] = None
    link = gpd.read_file(os.path.join(input_fldr, 'link.shp'))
    node = gpd.read_file(os.path.join(input_fldr, 'node.shp'))
    # link['speed'] = [np.random.randint(20, 60) for i in range(len(link))]
    my_net = Net(link_gdf=link, node_gdf=node, fmm_cache=False, fmm_cache_fldr=r'./data/input/gjy0828',
                 recalc_cache=False, prj_cache=True, cut_off=800)
    my_net.init_net()
    mpm = MapMatch(net=my_net, flag_name='0808_gjy',
                   export_html=True, export_geo_res=False,
                   gps_buffer=75, out_fldr=output_fldr, dense_gps=False, is_lower_f=False,
                   dense_interval=30, use_st=False,
                   top_k=10, dwell_n=0, del_dwell=False, use_heading_inf=True,
                   use_sub_net=True, use_gps_source=False, gps_radius=6, instant_output=False, use_node_restrict=True)
    # gps_df['eq_id'] = [i for i in range(len(gps_df))]
    # eq_df = pd.DataFrame({'eq_id': [i for i in range(len(gps_df))]})
    # eq_df['from_node'] = 1872
    # eq_df['to_node'] = 1855
    x = time.time()
    mpm.execute(gps_df=gps_df)
    print(time.time() - x)

def quick_start_match_1():
    # gps_df1 = pd.read_csv(r'./data/input/QuickStart-Match-1/0327sample.csv')
    #
    # gps_df3 = pd.DataFrame()
    # for file in os.listdir(r'./data/output/sample_gps/'):
    #     if 'gps' in file and '0527' in file and 'xishu' not in file:
    #         gps_df3 = pd.concat([gps_df3, gpd.read_file(rf'./data/output/sample_gps/{file}')])
    #
    # gps_df1[['agent_id', 'time', 'lng', 'lat']].to_csv(r'./data/input/QuickStart-Match-1/example_gps.csv',
    #                                                    encoding='utf_8_sig', index=False)
    #
    # gps_df3[['agent_id', 'time', 'lng', 'lat']].to_csv(r'./data/input/QuickStart-Match-1/example_sparse_gps.csv',
    #                                                    encoding='utf_8_sig', index=False)

    # general
    gps_df = pd.read_csv(r'./data/input/QuickStart-Match-1/example_gps.csv')
    #
    # 利用gps数据构建TrajectoryPoints, 并且对数据进行清洗
    # 是否需要进行该步操作视实际情况而定
    tp = TrajectoryPoints(gps_points_df=gps_df, plain_crs='EPSG:32649')
    tp.lower_frequency(lower_n=2).kf_smooth(o_deviation=0.3).rolling_average(rolling_window=3)  # 由于样例数据定位频率高且有一定的误差，因此先做间隔采样然后执行滤波平滑
    gps_df = tp.trajectory_data(_type='df')
    tp.export_html(out_fldr=r'./data/output/match_visualization/QuickStart-Match-1/', file_name='general_gps')

    link = gpd.read_file(r'./data/input/QuickStart-Match-1/modifiedConn_link.shp')
    node = gpd.read_file(r'./data/input/QuickStart-Match-1/modifiedConn_node.shp')
    my_net = Net(link_gdf=link, node_gdf=node, not_conn_cost=1200)
    my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_buffer=120, flag_name='general_sample', time_format=r'%Y-%m-%d %H:%M:%S',
                   use_heading_inf=True, omitted_l=6.0, export_html=True, del_dwell=False, export_geo_res=True,
                   out_fldr=r'./data/output/match_visualization/QuickStart-Match-1', dense_gps=True,
                   gps_radius=20.0)
    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的agent的相关信息({agent_id1: pd.DataFrame(), agent_id2: pd.DataFrame()...})
    # 第三个是匹配出错的agent的id列表(GPS点经过预处理(或者原始数据)后点数量不足2个)

    match_res, warn_info, error_info = mpm.execute(gps_df=gps_df)
    match_res.to_csv(fr'./data/output/match_visualization/QuickStart-Match-1/general_match_res.csv',
                     encoding='utf_8_sig', index=False)
    print(warn_info)
    print(error_info)

def quick_start_match_2():

    # general
    gps_df = pd.read_csv(r'./data/input/QuickStart-Match-1/example_gps.csv')

    # 利用gps数据构建TrajectoryPoints, 并且对数据进行清洗
    # 是否需要进行该步操作视实际情况而定
    tp = TrajectoryPoints(gps_points_df=gps_df, plain_crs='EPSG:32649')
    tp.lower_frequency(lower_n=2).kf_smooth(o_deviation=0.3).rolling_average(rolling_window=3)  
    gps_df = tp.trajectory_data(_type='df')
    tp.export_html(out_fldr=r'./data/output/match_visualization/QuickStart-Match-1/', file_name='general_gps')

    link = gpd.read_file(r'./data/input/QuickStart-Match-1/modifiedConn_link.shp')
    node = gpd.read_file(r'./data/input/QuickStart-Match-1/modifiedConn_node.shp')
    my_net = Net(link_gdf=link, node_gdf=node, not_conn_cost=1200)
    my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_buffer=120, flag_name='general_sample', time_format=r'%Y-%m-%d %H:%M:%S',
                   use_heading_inf=True, omitted_l=6.0, export_html=True, del_dwell=False, export_geo_res=True,
                   out_fldr=r'./data/output/match_visualization/QuickStart-Match-1', dense_gps=True,
                   gps_radius=20.0)
    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的agent的相关信息({agent_id1: pd.DataFrame(), agent_id2: pd.DataFrame()...})
    # 第三个是匹配出错的agent的id列表(GPS点经过预处理(或者原始数据)后点数量不足2个)

    match_res, warn_info, error_info = mpm.execute(gps_df=gps_df)
    match_res.to_csv(fr'./data/output/match_visualization/QuickStart-Match-1/general_match_res.csv',
                     encoding='utf_8_sig', index=False)
    print(warn_info)
    print(error_info)

def ldso_0710():
    import gotrackit.netreverse.NetGen as ng
    link = gpd.read_file(r'./data/input/ldso0710/link_test.geojson')
    link = link[['geometry']].copy()
    from gotrackit.tools.coord_trans import LngLatTransfer
    c = LngLatTransfer()
    link = c.gdf_convert(link, con_type='None', ignore_z=True)
    link, node, _ = ng.NetReverse.create_node_from_link(link_gdf=link, update_link_field_list=['link_id',
                                                                                               'from_node', 'to_node', 'length',
                                                                                               'dir'],
                                                        out_fldr=r'./data/input/ldso0710/', fill_dir=1)
    gps_df = gpd.read_file(r'./data/input/ldso0710/traj_test.geojson')
    gps_df['lng'] = gps_df['geometry'].x
    gps_df['lat'] = gps_df['geometry'].y
    gps_df['agent_id'] = 1
    del gps_df['geometry']
    gps_df.rename(columns={'timestamp': 'time'}, inplace=True)

    my_net = Net(link_gdf=link, node_gdf=node,  prj_cache=True, cut_off=2000)
    my_net.init_net()
    # mpm = MapMatch(net=my_net, flag_name='ldso0710',
    #                export_html=True, export_geo_res=True,
    #                gps_buffer=100, out_fldr=r'./data/input/ldso0710/',
    #                top_k=10, dwell_n=0, del_dwell=False, use_heading_inf=False,
    #                use_sub_net=False, use_gps_source=False, gps_radius=6,
    #                instant_output=True)
    mpm = MapMatch(net=my_net, flag_name='ldso0710',
                   export_html=True, export_geo_res=True,
                   gps_buffer=100, out_fldr=r'./data/input/ldso0710/',
                   top_k=10, dwell_n=0, del_dwell=False,
                   use_heading_inf=True, heading_para_array=np.array([1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001]),
                   omitted_l=1.0,
                   use_sub_net=False, use_gps_source=False, gps_radius=6,
                   instant_output=True)
    mpm.execute(gps_df=gps_df)
