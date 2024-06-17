# -- coding: utf-8 --
# @Time    : 2024/3/6 9:31
# @Author  : TangKai
# @Team    : ZheChengData

"""博主的测试代码, 相关文件都在本地, 所以不要运行该文件"""
import datetime
import os
import pickle
import time

import numpy as np
import pandas as pd
import geopandas as gpd
from gotrackit.map.Net import Net
from gotrackit.MapMatch import MapMatch
from gotrackit.model.Para import ParaGrid
# from src.gotrackit.map.Net import Net
# from src.gotrackit.MapMatch import MapMatch
# from src.gotrackit.model.Para import ParaGrid
from src.gotrackit.WrapsFunc import function_time_cost

# test 1

def t_cq_match():
    gps_df = gpd.read_file(rf'./data/output/gps/cq/gps.shp')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                          result_type='expand')
    gps_df.drop(columns=['geometry'], axis=1, inplace=True)
    my_net = Net(link_path=r'./data/input/net/test/cq/modifiedConn_link.shp',
                 node_path=r'./data/input/net/test/cq/modifiedConn_node.shp', not_conn_cost=1200, fmm_cache=True,
                 cache_cn=3,
                 recalc_cache=False, fmm_cache_fldr=r'./data/input/net/test/cq/', is_hierarchical=True)
    my_net.init_net()
    pgd = ParaGrid(beta_list=[1.0, 2.0,6.0], gps_sigma_list=[10.0, 20.0, 120.0])
    # match
    mpm = MapMatch(net=my_net, gps_df=gps_df,
                   is_rolling_average=False, flag_name='cq_test',
                   export_html=True, export_geo_res=True,
                   out_fldr=r'./data/output/match_visualization/cq', dense_gps=False,
                   use_sub_net=True, multi_core_save=True, instant_output=True, use_para_grid=True, para_grid=pgd)
    match_res, warn_info, error_info = mpm.execute()
    print(warn_info)
    print(match_res)
    print(error_info)
    print(pgd.search_res)

@function_time_cost
def t_sample_match():

    # 读取GPS数据
    # 这是一个有10辆车的GPS数据的文件
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    gps_df = pd.read_csv(r'./data/output/gps/sample/0327sample.csv')
    print(gps_df)
    # gps_df = gps_df[~gps_df['agent_id'].isin(['xa_car_11', 'xa_car_12'])]
    l = gpd.read_file(r'./data/input/net/xian/modifiedConn_link.shp')
    n = gpd.read_file(r'./data/input/net/xian/modifiedConn_node.shp')
    # 构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    # my_net = Net(link_path=r'./data/input/net/xian/modifiedConn_link.shp',
    #              node_path=r'./data/input/net/xian/modifiedConn_node.shp')
    my_net = Net(link_gdf=l,
                 node_gdf=n, fmm_cache=False, fmm_cache_fldr=r'./data/input/net/xian/', recalc_cache=False, cache_cn=1,
                 cache_slice=6, grid_len=2000, is_hierarchical=False)
    my_net.init_net()  # net初始化
    # my_net = Net(link_gdf=l,
    #              node_gdf=n)
    # my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=100, flag_name='xa_sample',
                   use_sub_net=True, use_heading_inf=True,
                   omitted_l=6.0, del_dwell=True, dwell_l_length=25.0, dwell_n=1,
                   lower_n=2, is_lower_f=True,
                   is_rolling_average=True, window=3,
                   export_html=True, export_geo_res=False, use_gps_source=False,
                   export_all_agents=True,
                   out_fldr=r'./data/output/match_visualization/xa_sample', dense_gps=False,
                   gps_radius=20.0)
    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的agent的相关信息({agent_id1: pd.DataFrame(), agent_id2: pd.DataFrame()...})
    # 第三个是匹配出错的agent的id列表(GPS点经过预处理(或者原始数据)后点数量不足2个)
    # mpm.execute()

    match_res, warn_info, error_info = mpm.execute()
    print(warn_info)
    print(match_res)
    print(error_info)
    match_res.to_csv(fr'./data/output/match_visualization/xa_sample/match_res.csv', encoding='utf_8_sig', index=False)

@function_time_cost
def t_sample_0424_match():

    # 读取GPS数据
    # 这是一个有10辆车的GPS数据的文件
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    gps_df = gpd.read_file(r'./data/output/gps/sample/0424sample.geojson')
    gps_df.drop(columns=['geometry'], axis=1, inplace=True)
    # gps_df = gps_df[gps_df['agent_id'].isin(['xa_car_22'])]
    print(gps_df)
    l = gpd.read_file(r'./data/input/net/xian/modifiedConn_link.shp')
    n = gpd.read_file(r'./data/input/net/xian/modifiedConn_node.shp')
    n.crs = 'epsg:4326'
    # l = gpd.read_file(r'./data/input/net/xian/LinkAfterModify.shp')
    # n = gpd.read_file(r'./data/input/net/xian/NodeAfterModify.shp')
    my_net = Net(link_gdf=l,
                 node_gdf=n, fmm_cache=True, fmm_cache_fldr=r'./data/input/net/xian/', recalc_cache=False, cache_cn=3,
                 cache_slice=6, cut_off=1000, not_conn_cost=1200.0, is_hierarchical=True)
    my_net.init_net()  # net初始化
    a = time.time()
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=500, flag_name='xa_sample_0424',
                   use_sub_net=True, use_heading_inf=True,
                   omitted_l=6.0, del_dwell=True, dwell_l_length=25.0, dwell_n=1,
                   lower_n=2, is_lower_f=False, top_k=20,
                   is_rolling_average=False, window=3,
                   export_html=True, export_geo_res=False, use_gps_source=False,
                   out_fldr=r'./data/output/match_visualization/xa_sample', dense_gps=True,
                   gps_radius=20.0, multi_core_save=True, instant_output=True)

    # match_res, warn_info, error_info = mpm.execute()
    match_res, warn_info, error_info = mpm.multi_core_execute(core_num=2)
    print(time.time() - a)
    # print(warn_info.keys())
    # mpm.multi_core_execute(core_num=2)
    # print(warn_info)
    # print(match_res)
    # print(error_info)
    # match_res.to_csv(fr'./data/output/match_visualization/xa_sample/match_res.csv', encoding='utf_8_sig', index=False)


def t_sample_xa_xishu_match():

    # 读取GPS数据
    # 这是一个有10辆车的GPS数据的文件
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    gps_df = gpd.read_file(r'./data/output/gps/sample/0329xa_xishu.geojson')
    gps_df.drop(columns=['geometry'], axis=1, inplace=True)
    print(gps_df)
    # gps_df = gps_df[gps_df['agent_id'] == 'xa_car_4']
    l = gpd.read_file(r'./data/input/net/xian/modifiedConn_link.shp')
    n = gpd.read_file(r'./data/input/net/xian/modifiedConn_node.shp')
    # 构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    # my_net = Net(link_path=r'./data/input/net/xian/modifiedConn_link.shp',
    #              node_path=r'./data/input/net/xian/modifiedConn_node.shp')
    my_net = Net(link_gdf=l,
                 node_gdf=n)
    my_net.init_net()  # net初始化

    # 新建一个地图匹配对象, 指定其使用net对象, gps数据
    # gps_buffer: 单位米, 意为只选取每个GPS点附近100米范围内的路段作为候选路段
    # use_sub_net: 是否使用子网络进行计算
    # use_heading_inf: 是否使用GPS差分航向角来修正发射概率
    # 按照上述参数进行匹配: 匹配程序会报warning, 由于GPS的定位误差较大, 差分航向角的误差也很大
    # mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=100, flag_name='xa_sample',
    #                use_sub_net=True, use_heading_inf=True,
    #                export_html=True, export_geo_res=True,
    #                out_fldr=r'./data/output/match_visualization/xa_sample',
    #                use_gps_source=True,
    #                geo_res_fldr=r'./data/output/match_visualization/xa_sample', dense_gps=False)

    # 这个地图匹配对象, 指定一些额外的参数, 可以全部匹配成功
    # is_rolling_average=True, 启用了滑动窗口平均来对GPS数据进行降噪
    # window=3, 滑动窗口大小为3
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=250, flag_name='xa_sample_xishu',
                   use_sub_net=True, use_heading_inf=True,
                   dense_gps=True, dense_interval=40,
                   is_rolling_average=True, window=3,
                   export_html=True, export_geo_res=True,
                   use_gps_source=True,
                   out_fldr=r'./data/output/match_visualization/xa_sample')
    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, _ = mpm.execute()
    print(warn_info)
    print(match_res)
    match_res.to_csv(r'./data/output/match_visualization/xa_sample/xishu_match_res.csv', encoding='utf_8_sig',
                     index=False)


def dense_example():
    gps_df = gpd.read_file(r'./data/output/gps/dense_example/test999.geojson')
    my_net = Net(link_path=r'./data/input/net/xian/modifiedConn_link.shp',
                 node_path=r'./data/input/net/xian/modifiedConn_node.shp', fmm_cache=True,
                 recalc_cache=False, fmm_cache_fldr=r'./data/input/net/xian', grid_len=1000, cache_cn=1,
                 is_hierarchical=True)

    my_net.init_net()
    pgd = ParaGrid(use_heading_inf_list=[False, True], beta_list=[0.1, 1.0], gps_sigma_list=[1.0, 5.0])
    # match
    mpm = MapMatch(net=my_net, gps_df=gps_df, is_rolling_average=True, window=2, flag_name='dense_example',
                   export_html=True, export_geo_res=True,
                   gps_buffer=400,
                   out_fldr=r'./data/output/match_visualization/dense_example',
                   dense_gps=True,
                   use_sub_net=False, dense_interval=50.0, use_gps_source=False, use_heading_inf=True,
                   gps_radius=15.0, use_para_grid=True, para_grid=pgd)
    res, warn_info, error_info = mpm.execute()
    print(res)
    print(warn_info)
    print(error_info)
    print(pd.DataFrame(pgd.search_res))
    res.to_csv(r'./data/output/match_visualization/dense_example/match_res.csv', encoding='utf_8_sig', index=False)

def t_0326_taxi():
    gps_df = pd.read_csv(r'./data/input/net/test/0326fyx/gps/part/TaxiData-Sample.csv')
    # my_net = Net(link_path=r'./data/input/net/test/0402BUG/load/new_link.shp',
    #              node_path=r'./data/input/net/test/0402BUG/load/modifiedConn_node.shp',
    #              not_conn_cost=2000)
    # my_net.init_net()

    my_net = Net(link_path=r'./data/input/net/test/0402BUG/load/new_link.shp',
                 node_path=r'./data/input/net/test/0402BUG/load/modifiedConn_node.shp',
                 not_conn_cost=2000, fmm_cache=True, recalc_cache=False, cut_off=500.0,
                 fmm_cache_fldr=r'./data/input/net/test/0402BUG/load/', cache_cn=6, is_hierarchical=True)
    my_net.init_net()
    pgd = ParaGrid(gps_sigma_list=[10, 20, 30], beta_list=[10, 20, 30])
    # match
    mpm = MapMatch(net=my_net, gps_df=gps_df,
                   is_rolling_average=True, window=2, flag_name='0326_taxi',
                   export_html=True, export_geo_res=True,
                   gps_buffer=500, out_fldr=r'./data/output/match_visualization/0326_taxi',
                   heading_para_array=np.array([1.0, 1.0, 0.1, 0.1, 0.00001, 0.00001, 0.000001, 0.0000001, 0.0000001]),
                   dense_gps=True, dense_interval=120, use_heading_inf=False,
                   top_k=60, dwell_n=0,
                   use_sub_net=False, use_gps_source=False, gps_radius=20, instant_output=True, use_para_grid=False,
                   para_grid=pgd)
    res, may_error_info, error_info = mpm.execute()
    print(res)

def check_0325():
    # 某快速路匹配示例
    my_net = Net(link_path=r'./data/input/net/test/0325/G15_links.shp',
                 node_path=r'./data/input/net/test/0325/G15_gps_node.shp', fmm_cache=True, cut_off=1000.0,
                 fmm_cache_fldr=r'./data/input/net/test/0325/')
    my_net.init_net()

    gps_df = pd.read_csv(r'./data/input/net/test/0325/car_gps_test_noheading.csv')
    gps_df = gps_df.loc[0:260, :].copy()
    print(gps_df)
    mpm = MapMatch(net=my_net, gps_df=gps_df,
                   is_rolling_average=False, window=2,
                   flag_name='check_0325_source', export_html=True, export_geo_res=True,
                   out_fldr=r'./data/output/match_visualization/sample')
    res, may_error_info, error_info = mpm.execute()
    print(res)
    print(may_error_info)
    print(error_info)

def bug_0329():
    from shapely.geometry import Point
    link = gpd.read_file(r'./data/input/net/test/0329BUG/new_link/LinkAfterModify.shp')
    node = gpd.read_file(r'./data/input/net/test/0329BUG/new_link/NodeAfterModify.shp')
    link = link.to_crs('EPSG:4326')
    node = node.to_crs('EPSG:4326')

    print(link)
    print(node)
    gps_df = gpd.read_file(r'./data/input/net/test/0329BUG/helmert/trajectory_helmert.shp')
    print(gps_df.crs)
    gps_df = gps_df.to_crs('EPSG:4326')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                          result_type='expand')
    pd.to_datetime()
    #
    # gps_df['seq'] = [i for i in range(0, len(gps_df))]
    # gps_df.to_file(r'aaa.shp')
    # print(link)
    # print(node)
    # print(gps_df)
    # my_net = Net(link_gdf=link, node_gdf=node)
    # my_net.init_net()
    # mpm = MapMatch(net=my_net, gps_df=gps_df, dense_gps=False,
    #                is_rolling_average=False, window=2,
    #                lower_n=3, is_lower_f=True, gps_buffer=50,
    #                use_sub_net=False,
    #                flag_name='0329_bug', export_html=True, export_geo_res=True,
    #                out_fldr=r'./data/output/match_visualization/0329_bug',
    #                geo_res_fldr=r'./data/output/match_visualization/0329_bug')
    # res_df, label_list = mpm.execute()
    # print(label_list)
    # print(res_df)


def bug_0402():
    # 1.读取GPS数据
    # 这是一个有10辆车的GPS数据的文件, 已经做过了数据清洗以及行程切分
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    gps_df = gpd.read_file(r'./data/input/net/test/0402BUG/gps/gps_CORRECT.geojson')
    # gps_df = gpd.read_file(r'./data/input/net/test/0402BUG/gps/new_gps.geojson')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                          result_type='expand')
    gps_df.drop(columns=['geometry'], axis=1, inplace=True)
    gps_df['agent_id'] = gps_df['agent_id'].astype(int)
    gps_df['time'] = [datetime.datetime.now().timestamp() * 1000 + i for i in range(0, len(gps_df))]
    # gps_df['time'] = \
    #     pd.to_datetime(gps_df['time'], unit='ms')
    # print(gps_df)
    # gps_df_a = gps_df.copy()
    # gps_df_a['agent_id'] = 12
    # gps_df = pd.concat([gps_df_a, gps_df])
    print(gps_df)

    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    # my_net = Net(link_path=r'./data/input/net/test/0402BUG/load/opt_link_10.shp',
    #              node_path=r'./data/input/net/test/0402BUG/load/opt_node.shp')
    my_net = Net(link_path=r'./data/input/net/test/0402BUG/load/new_link.shp',
                 node_path=r'./data/input/net/test/0402BUG/load/modifiedConn_node.shp', not_conn_cost=1000.0,
                 fmm_cache=True, fmm_cache_fldr=r'./data/input/net/test/0402BUG/load/', cache_cn=6,
                 recalc_cache=False, cut_off=1200.0, grid_len=2000, is_hierarchical=True)
    my_net.init_net()  # net初始化
    # my_net = Net(link_path=r'./data/input/net/test/0402BUG/load/new_link.shp',
    #              node_path=r'./data/input/net/test/0402BUG/load/modifiedConn_node.shp', not_conn_cost=1500)
    # my_net.init_net()  # net初始化
    para_grid = ParaGrid(gps_sigma_list=[1.0, 2.0, 3.0],
                         beta_list=[500, 1000], use_heading_inf_list=[True, False])
    # mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=120, flag_name='id1',
    #                use_sub_net=False, use_heading_inf=True, max_increment_times=2, increment_buffer=100,
    #                is_rolling_average=True, window=2, dense_interval=80, dense_gps=False,
    #                omitted_l=20.1, del_dwell=True, dwell_l_length=15.0,
    #                export_html=True, export_geo_res=False, top_k=20, is_lower_f=False, lower_n=2,
    #                out_fldr=r'./data/output/match_visualization/0402BUG/',
    #                use_gps_source=True, gps_radius=10.0)
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=120, flag_name='id1', time_unit='ms',
                   use_sub_net=False, use_heading_inf=False,
                   is_rolling_average=True, window=2, dense_interval=80, dense_gps=False,
                   omitted_l=1.0, del_dwell=True, dwell_l_length=15.0,
                   export_html=True, export_geo_res=False, top_k=20, is_lower_f=False, lower_n=2,
                   out_fldr=r'./data/output/match_visualization/0402BUG/',
                   use_gps_source=True, gps_radius=10.0, use_para_grid=False, para_grid=para_grid)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = mpm.execute()
    match_res.to_csv(r'./data/output/match_visualization/0402BUG/match_res.csv', encoding='utf_8_sig')
    print(warn_info)
    print(match_res)
    print(pd.DataFrame(para_grid.search_res))

def bug_0516():
    # 1.读取GPS数据
    # 这是一个有10辆车的GPS数据的文件, 已经做过了数据清洗以及行程切分
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    # gps_df = pd.read_csv(r'./data/input/net/test/0516BUG/车辆轨迹V3.csv', encoding='gbk')
    # gps_df = gps_df[['agent_id', 'time', 'lng', 'lat']].copy()
    # print(gps_df)
    gps_df = gpd.read_file(r'./data/input/net/test/0516BUG/gps_test_0605.shp')
    gps_df = gps_df[gps_df['id'] == 123]
    gps_df['agent_id'] = 1111
    print(gps_df)
    gps_df['time'] = [datetime.datetime.now().timestamp() + i for i in range(len(gps_df))]
    gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                          result_type='expand')
    del gps_df['geometry']
    # link = gpd.read_file(r'./data/input/net/test/0516BUG/shp/link.shp')
    # node = gpd.read_file(r'./data/input/net/test/0516BUG/shp/Final_node_net.shp')
    #
    # with open(r'./data/input/net/test/0516BUG/shp/link', 'wb') as f:
    #     pickle.dump(link, f)
    # with open(r'./data/input/net/test/0516BUG/shp/node', 'wb') as f:
    #     pickle.dump(node, f)

    with open(r'./data/input/net/test/0516BUG/shp/link', 'rb') as f:
        link = pickle.load(f)
    with open(r'./data/input/net/test/0516BUG/shp/node', 'rb') as f:
        node = pickle.load(f)
    # link = link.to_crs('EPSG:32648')
    # link['geometry'] = link['geometry'].simplify(20.0)
    # link = link.to_crs('EPSG:4326')
    # link.to_file(r'./data/input/net/test/0516BUG/shp/link.shp')
    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_gdf=link,
                 node_gdf=node, not_conn_cost=2000.0, cut_off=1000.0, grid_len=4000, is_hierarchical=True)
    my_net.init_net()  # net初始化
    # para_grid = ParaGrid(gps_sigma_list=[i for i in range(1, 40, 10)],
    #                      beta_list=[0.1, 6.0], use_heading_inf_list=[False, True])
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=700.0, flag_name='id1',
                   use_sub_net=True, use_heading_inf=False,
                   is_rolling_average=False, window=2, dense_interval=400.0, dense_gps=True,
                   omitted_l=20.1, del_dwell=False, dwell_l_length=15.0,
                   export_html=True, export_geo_res=True, top_k=20, is_lower_f=False, lower_n=2,
                   out_fldr=r'./data/output/match_visualization/0516BUG/',
                   use_gps_source=False, gps_radius=20.0, instant_output=True)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = mpm.execute()
    match_res.to_csv(r'./data/output/match_visualization/0516BUG/match_res.csv', encoding='utf_8_sig', index=False)
    print(warn_info)
    # print(match_res)


def bug_0614():
    # 1.读取GPS数据
    # 这是一个有10辆车的GPS数据的文件, 已经做过了数据清洗以及行程切分
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    # gps_df = pd.read_excel(r'./data/input/net/test/0614BUG/车辆轨迹_川ACF929_0613.xlsx')
    # gps_df.rename(columns={'时间': 'time', '经度': 'lng', '纬度': 'lat'}, inplace=True)
    # gps_df = gps_df[['agent_id', 'time', 'lng', 'lat']].copy()
    # print(gps_df)
    gps_df = pd.read_csv(r'./data/input/net/test/0614BUG/20240320_to_20240320_data_chunk_0.csv')
    gps_df = gps_df[gps_df['plateNo'] == '川ABQ128'].copy()
    # gps_df = gps_df[gps_df['plateNo'] == '川ADK592'].copy()
    gps_df.rename(columns={'posTime': 'time', 'new_longitude': 'lng', 'new_latitude': 'lat'}, inplace=True)
    gps_df['agent_id'] = 113
    # plateNo,plateColor,posTime,posDate,longitude,latitude,new_latitude,new_longitude,gpsSpeed
    print(gps_df)

    # link = gpd.read_file(r'F:\PyPrj\TrackIt\data\input\net\test\0614BUG\net_final\merge_FinalLink.shp', encoding='gbk')
    # node = gpd.read_file(r'F:\PyPrj\TrackIt\data\input\net\test\0614BUG\net_final\merge_FinalNode.shp', encoding='gbk')
    #
    # my_net = Net(link_gdf=link,
    #              node_gdf=node, not_conn_cost=2500.0, cut_off=1200.0, grid_len=4000, is_hierarchical=True)
    # my_net.init_net()  # net初始化
    #
    # with open(r'F:\PyPrj\TrackIt\data\input\net\test\0614BUG\net_final\net', 'wb') as f:
    #     pickle.dump(my_net, f)

    with open(r'F:\PyPrj\TrackIt\data\input\net\test\0614BUG\net_final\net', 'rb') as f:
        my_net = pickle.load(f)

    p = ParaGrid(gps_sigma_list=[30, 40, 50, 60, 70, 80], beta_list=[4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=900.0, flag_name='id1',
                   use_sub_net=True, use_heading_inf=True, time_unit='ms',
                   is_rolling_average=True, window=2, dense_interval=400.0, dense_gps=True,
                   omitted_l=20.1, del_dwell=False, dwell_l_length=15.0,
                   export_html=True, export_geo_res=True, top_k=50, is_lower_f=False, lower_n=2,
                   out_fldr=r'./data/output/match_visualization/0614BUG/',
                   use_gps_source=False, gps_radius=20.0, instant_output=True, para_grid=p, use_para_grid=False)
    a, b, c = mpm.execute()
    print(b.keys())
    print(b)

def agents_150_test():
    # 1.读取GPS数据
    gps_df = pd.read_csv(r'./data/input/net/test/0402BUG/gps/150_agents.csv')
    my_net = Net(link_path=r'./data/input/net/test/0402BUG/load/new_link.shp',
                 node_path=r'./data/input/net/test/0402BUG/load/modifiedConn_node.shp', not_conn_cost=999.0,
                 fmm_cache=True, fmm_cache_fldr=r'./data/input/net/test/0402BUG/load/', cache_cn=5,
                 recalc_cache=False, is_hierarchical=True)
    my_net.init_net()  # net初始化

    a = time.time()
    para_grid = ParaGrid(gps_sigma_list=[1.0, 2.0, 3.0],
                         beta_list=[500, 1000], use_heading_inf_list=[False])
    a = time.time()
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=120, flag_name='id1',
                   use_sub_net=False, use_heading_inf=True,
                   is_rolling_average=True, window=2, dense_interval=80, dense_gps=False,
                   omitted_l=20.1, del_dwell=True, dwell_l_length=15.0,
                   export_html=True, export_geo_res=False, top_k=20,
                   out_fldr=r'./data/output/match_visualization/0402BUG/', instant_output=False,
                   use_gps_source=True, gps_radius=10.0, use_para_grid=False, para_grid=para_grid)
    # match_res, warn_info, error_info = mpm.execute()
    match_res, warn_info, error_info = mpm.multi_core_execute(core_num=6)
    print(time.time() - a)
    print(match_res)
    print(warn_info)
    print(error_info)
    print(para_grid.search_res)


@function_time_cost
def route2gps():

    gps_df = pd.read_csv(r'./data/output/route2gps/gps.csv')
    # gps_df = gps_df[gps_df['agent_id'].isin(['11-3', '15-1'])]

    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_path=r'./data/output/reverse/0318cd/FinalLink.shp',
                 node_path=r'./data/output/reverse/0318cd/FinalNode.shp', not_conn_cost=1500.0, fmm_cache=True,
                 max_cut_off=2000.0, cut_off=1500, recalc_cache=True, fmm_cache_fldr=r'./data/output/reverse/0318cd')
    my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=50, flag_name='cd_route2gps',
                   use_sub_net=False, use_heading_inf=True,
                   is_rolling_average=False, is_lower_f=False, lower_n=2, dense_gps=False,
                   time_format='"%Y-%m-%d %H:%M:%S"', dense_interval=30.0,
                   omitted_l=2.0, del_dwell=True, dwell_l_length=20.0, dwell_n=0,
                   export_html=False, export_geo_res=False, top_k=5, instant_output=True,
                   out_fldr=r'./data/output/match_visualization/0318cd_route2gps/',
                   use_gps_source=False, gps_radius=10.0, export_all_agents=True, multi_core_save=True)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, _ = mpm.execute()
    # match_res.to_csv(r'./data/output/match_visualization/0318cd_route2gps/match_res.csv', encoding='utf_8_sig')
    print(match_res)
    print(warn_info)
    print(match_res)
    for may_error_agent in warn_info.keys():
        print(may_error_agent)
        print(warn_info[may_error_agent])


def simple_0419_net():
    gps = gpd.read_file(r'./data/input/net/test/0419test/gps.shp')
    gps['agent_id'] = 1
    gps['time'] = [datetime.datetime.now().timestamp() + i for i in range(len(gps))]

    gps[['lng', 'lat']] = gps.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1, result_type='expand')
    gps.drop(columns=['geometry'], axis=1, inplace=True)
    gps = gps.loc[[0, 1], :].copy()

    my_net = Net(link_path=r'./data/input/net/test/0419test/LinkAfterModify.shp',
                 node_path=r'./data/input/net/test/0419test/NodeAfterModify.shp', not_conn_cost=1500)

    my_net.init_net()  # net初始化
    mpm = MapMatch(net=my_net, gps_df=gps, gps_buffer=500, flag_name='cd_route2gps',
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
    match_res, warn_info, error_info = mpm.execute()
    match_res.to_csv(r'./data/output/match_visualization/0318cd_route2gps/match_res.csv', encoding='utf_8_sig')
    print(warn_info)
    print(match_res)
    print(error_info)
    for may_error_agent in warn_info.keys():
        print(may_error_agent)
        print(warn_info[may_error_agent])


def aaa(b: list[bool] = True):
    b[0] = False
    print(b)


@function_time_cost
def route2gps_break_test():

    gps_df = pd.read_csv(r'./data/output/route2gps/gps.csv')
    gps_df = gps_df[gps_df['agent_id'].isin(['11-3'])]
    gps_df.reset_index(inplace=True, drop=True)
    print(gps_df)
    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_path=r'./data/output/reverse/0318cd/FinalLink.shp',
                 node_path=r'./data/output/reverse/0318cd/FinalNode.shp', not_conn_cost=1500.0, fmm_cache=True,
                 max_cut_off=2000.0, cut_off=6500, recalc_cache=False, fmm_cache_fldr=r'./data/output/reverse/0318cd')
    my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=5500, flag_name='cd_route2gps_break',
                   use_sub_net=False, use_heading_inf=True,
                   is_rolling_average=False, is_lower_f=False, lower_n=2, dense_gps=False,
                   time_format='"%Y-%m-%d %H:%M:%S"', dense_interval=30.0,
                   omitted_l=2.0, del_dwell=True, dwell_l_length=20.0, dwell_n=0,
                   export_html=True, export_geo_res=False, top_k=10,
                   out_fldr=r'./data/output/match_visualization/0318cd_route2gps/',
                   use_gps_source=False, gps_radius=10.0, export_all_agents=True, multi_core_save=True)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, _ = mpm.execute()
    match_res.to_csv(r'./data/output/match_visualization/0318cd_route2gps/match_res.csv', encoding='utf_8_sig')
    print(warn_info)
    print(match_res)
    for may_error_agent in warn_info.keys():
        print(may_error_agent)
        print(warn_info[may_error_agent])


def bug_0429_a():
    gps_df = pd.read_csv(r'./data/input/net/test/bug0429/gps数据/match-05.csv')
    gps_df.reset_index(inplace=True, drop=True)
    l = gpd.read_file(r'./data/input/net/test/bug0429/shp文件/LinkAfterModify.shp')
    n = gpd.read_file(r'./data/input/net/test/bug0429/shp文件/NodeAfterModify.shp')
    del gps_df['time']
    gps_df.rename(columns={'时间戳': 'time'}, inplace=True)
    print(gps_df)
    l['dir'] = 0
    gps_df['time'] = gps_df['time'] / 1000
    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_gdf=l,
                 node_gdf=n, not_conn_cost=1500.0,
                 cut_off=500)
    my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=600, flag_name='0429_bug',
                   use_sub_net=False, use_heading_inf=False,
                   is_rolling_average=False, is_lower_f=False, lower_n=2, dense_gps=False,
                   time_format='%Y-%m-%d %H:%M:%S', dense_interval=30.0,
                   omitted_l=2.0, del_dwell=False, dwell_l_length=20.0, dwell_n=0,
                   export_html=True, export_geo_res=False, top_k=20,
                   out_fldr=r'./data/input/net/test/bug0429',
                   use_gps_source=True, gps_radius=10.0, export_all_agents=False, multi_core_save=True)
    mpm.execute()


def leeds_yg_test():
    # origin_link = gpd.read_file(r'./data/input/net/test/0506yg/origin_link.shp')
    # origin_node = gpd.read_file(r'./data/input/net/test/0506yg/origin_node.shp')
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
    #     './data/input/net/test/0506yg/link.shp', encoding='gbk')
    #
    # origin_node[['TOID', 'node_id', 'geometry']].to_file(
    #     './data/input/net/test/0506yg/node.shp', encoding='gbk')
    #

    gps_df = pd.read_csv(r'./data/input/net/test/0506yg/gps.csv')
    gps_df['agent_id'] = gps_df['agent_id'].astype(int)
    # gps_df = gpd.read_file(r'./data/input/net/test/0506yg/zf_gps.shp')
    # gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
    #                                       result_type='expand')
    # del gps_df['geometry']
    # gps_df['agent_id'] = 123
    #
    gps_df['time'] = [datetime.datetime.now().timestamp() + i for i in range(0, len(gps_df))]
    my_net = Net(link_path='./data/input/net/test/0506yg/aaa_link.shp',
                 node_path='./data/input/net/test/0506yg/aaa_node.shp', is_hierarchical=True)
    my_net.init_net()  # net初始化
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=60.0, flag_name='yg1',
                   use_sub_net=True, use_heading_inf=True,
                   is_rolling_average=False, window=2, dense_interval=80, dense_gps=False,
                   omitted_l=20.1, del_dwell=False, dwell_l_length=15.0, time_unit='s',
                   export_html=True, export_geo_res=False, top_k=20, is_lower_f=False, lower_n=2,
                   out_fldr=r'./data/output/match_visualization/0506yg/',
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
    match_res, warn_info, error_info = mpm.execute()
    # match_res.to_csv(r'./data/output/match_visualization/0506yg/match_res.csv', encoding='utf_8_sig')
    print(warn_info)
    print(match_res)
    print(error_info)

def xishu_0507():
    # 1.读取GPS数据
    # 这是一个有10辆车的GPS数据的文件, 已经做过了数据清洗以及行程切分
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    p = ParaGrid(beta_list=[1, 4, 6, 14, 20], gps_sigma_list=[10, 20, 50, 120, 150], omitted_l_list=[1, 5, 10])
    gps_df = gpd.read_file(r'./data/input/net/test/0507xishu/163.csv')

    gps_df['time'] = [datetime.datetime.now().timestamp() + i for i in range(0, len(gps_df))]
    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_path=r'./data/input/net/test/0507xishu/FinalLink.shp',
                 node_path=r'./data/input/net/test/0507xishu/FinalNode.shp', not_conn_cost=3200.0,
                 fmm_cache=True, fmm_cache_fldr=r'./data/input/net/test/0507xishu', cache_cn=6,
                 recalc_cache=False, cut_off=1000.0)
    my_net.init_net()  # net初始化
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=500.0, flag_name='0507_xishu_120dense',
                   use_sub_net=False, use_heading_inf=True,
                   is_rolling_average=False, window=2, dense_interval=100.0, dense_gps=True,
                   omitted_l=5.1, del_dwell=True, dwell_l_length=25.0, dwell_n=1,
                   beta=15.0, gps_sigma=130.0,
                   export_html=True, export_geo_res=True, top_k=60, is_lower_f=False, lower_n=2,
                   out_fldr=r'./data/output/match_visualization/稀疏/',
                   use_gps_source=False, gps_radius=10.0, use_para_grid=False, para_grid=p, instant_output=True)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = mpm.execute()
    print(match_res)
    print(pd.DataFrame(p.search_res))

def sz_rand_route_match():
    gps_df = pd.read_csv(r'./data/output/0508rand/gps.csv')
    print(gps_df)
    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_path=r'./data/output/reverse/0508rand/FinalLink.shp',
                 node_path=r'./data/output/reverse/0508rand/FinalNode.shp', not_conn_cost=1500.0, fmm_cache=True,
                 cut_off=1200, recalc_cache=True, fmm_cache_fldr=r'./data/output/reverse/0508rand')
    my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=100, flag_name='cd_route2gps_break',
                   use_sub_net=False, use_heading_inf=True,
                   is_rolling_average=False, is_lower_f=False, lower_n=2, dense_gps=False,
                   time_format='"%Y-%m-%d %H:%M:%S"', dense_interval=30.0,
                   omitted_l=2.0, del_dwell=True, dwell_l_length=20.0, dwell_n=0,
                   export_html=True, export_geo_res=True, top_k=10,
                   out_fldr=r'./data/output/match_visualization/0508rand/',
                   use_gps_source=False, gps_radius=10.0, export_all_agents=True, multi_core_save=True, instant_output=True)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    # match_res, warn_info, _ = mpm.execute()
    match_res, warn_info, _ = mpm.multi_core_execute(core_num=3)
    match_res.to_csv(r'./data/output/match_visualization/0318cd_route2gps/match_res.csv', encoding='utf_8_sig')
    print(warn_info)
    print(match_res)
    for may_error_agent in warn_info.keys():
        print(may_error_agent)
        print(warn_info[may_error_agent])


def check_p(a: dict = None, flag=None):
    print(flag, id(a))
    print(flag, a[1], id( a[1]))
    a[1] = list()
    print(flag, a[1])
def m_core_execute():
    import multiprocessing
    n = 2
    print(f'using multiprocessing - {n} cores')
    pool = multiprocessing.Pool(processes=n)
    result_list = []
    t = {1: [12,3,4], 2: [123,111]}
    for i in range(0, n):
        result = pool.apply_async(check_p,
                                  args=(t, i))
        result_list.append(result)
    pool.close()
    pool.join()
    print(id(t))
    print(id(t[1]))


def t_simplify():
    link = gpd.read_file(r'./data/input/net/test/0516BUG/shp/Final_link_net.shp')
    node = gpd.read_file(r'./data/input/net/test/0516BUG/shp/Final_node_net.shp')

    link = link.to_crs('EPSG:32648')
    link['geometry'] = link['geometry'].simplify(1.0)
    link = link.to_crs('EPSG:4326')

    link.to_file(r'./data/input/net/test/0516BUG/shp/xxx.shp')

def xishu_sample_xian():
    gps_df = gpd.GeoDataFrame()
    for file in os.listdir(r'./data/output/sample_gps/'):
        if 'gps' in file and '0527' in file:
            gps_df = pd.concat([gps_df, gpd.read_file(rf'./data/output/sample_gps/{file}')])
    gps_df.reset_index(inplace=True, drop=True)
    del gps_df['geometry'], gps_df['heading']
    gps_df = pd.DataFrame(gps_df)

    l = gpd.read_file(r'./data/input/net/xian/modifiedConn_link.shp')
    n = gpd.read_file(r'./data/input/net/xian/modifiedConn_node.shp')
    my_net = Net(link_gdf=l,
                 node_gdf=n, fmm_cache=True, fmm_cache_fldr=r'./data/input/net/xian/', recalc_cache=False, cache_cn=1,
                 cache_slice=6, cut_off=2500)
    my_net.init_net()  # net初始化

    a = time.time()
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=700, flag_name='xishu',
                   use_sub_net=True, use_heading_inf=True,
                   omitted_l=6.0, del_dwell=True, dwell_l_length=25.0, dwell_n=1,
                   is_lower_f=False,
                   is_rolling_average=False,
                   export_html=True, export_geo_res=True, use_gps_source=False,
                   out_fldr=r'./data/output/sample_gps/', dense_gps=True, dense_interval=100.0,
                   gps_radius=12.0)

    match_res, warn_info, error_info = mpm.execute()
    print(time.time() - a)
    print(warn_info)
    print(match_res)
    print(error_info)
    match_res.to_csv(fr'./data/output/match_visualization/xa_sample/match_res.csv', encoding='utf_8_sig', index=False)


def sz_real_match():
    # gps_df = pd.read_csv(r'./data/output/gps/real_sz/gps_trip.csv')

    # agent = set(gps_df.sample(n=15)['agent_id'])
    # agent = {1824, 1058, 1635, 5028, 5573, 2271, 4937, 4243, 1268, 2943, 3160, 3353, 2876, 1917, 4415}
    # print(agent)
    # gps_df = gps_df[gps_df['agent_id'].isin(agent)]


    gps_df = gpd.read_file(r'./data/output/gps/real_sz/test_gps.shp')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda item: (item['geometry'].x, item['geometry'].y),
                                          result_type='expand', axis=1)
    gps_df = pd.DataFrame(gps_df)
    gps_df['lng'] = 114.067361
    gps_df['time'] = [i + 200 for i in range(3)]
    del gps_df['geometry']
    print(gps_df)

    l = gpd.read_file(r'./data/output/gps/real_sz/net/base_link.shp')
    n = gpd.read_file(r'./data/output/gps/real_sz/net/base_node.shp')
    my_net = Net(link_gdf=l,
                 node_gdf=n, fmm_cache_fldr=r'./data/output/gps/real_sz/net/', cache_cn=1, cut_off=1200,
                 cache_slice=6)

    my_net.init_net()  # net初始化

    a = time.time()
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=500.0, flag_name='sz_real',
                   use_sub_net=True, use_heading_inf=True,
                   omitted_l=6.0, del_dwell=True, dwell_l_length=15.0, dwell_n=2,
                   is_lower_f=False, top_k=20,
                   is_rolling_average=False,
                   export_html=True, export_geo_res=False, use_gps_source=False,
                   out_fldr=r'./data/output/match_visualization/real_sz', dense_gps=True, dense_interval=200.0,
                   gps_radius=12.0, multi_core_save=True)

    match_res, warn_info, error_info = mpm.execute()
    print(time.time() - a)
    print(warn_info.keys())
    print(match_res)
    print(error_info)
    # match_res.to_csv(fr'./data/output/match_visualization/xa_sample/match_res.csv', encoding='utf_8_sig', index=False)



def carla_0605():
    gps_df = gpd.read_file(r'./data/input/net/test/carla0605/trajectory/tra.geojson')
    gps_df = gps_df.to_crs('EPSG:4326')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda item: (item['geometry'].x, item['geometry'].y),
                                          result_type='expand', axis=1)
    gps_df = pd.DataFrame(gps_df)
    # gps_df = gps_df.loc[1000:1400, :]
    del gps_df['geometry']
    print(gps_df)

    l = gpd.read_file(r'./data/input/net/test/carla0605/town07_map/LinkAfterModify.shp')
    n = gpd.read_file(r'./data/input/net/test/carla0605/town07_map/NodeAfterModify.shp')
    l = l.to_crs('EPSG:4326')
    n = n.to_crs('EPSG:4326')
    my_net = Net(link_gdf=l,
                 node_gdf=n, fmm_cache=True, fmm_cache_fldr=r'./data/input/net/test/carla0605/town07_map/',
                 cache_slice=3, recalc_cache=False, is_hierarchical=True)

    my_net.init_net()  # net初始化

    a = time.time()
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=80.0, flag_name='carla_0605',
                   use_sub_net=False, use_heading_inf=True,
                   omitted_l=6.0, del_dwell=False, dwell_l_length=15.0, dwell_n=2,
                   is_lower_f=False, top_k=30,
                   is_rolling_average=False,
                   export_html=False, export_geo_res=True, use_gps_source=False,
                   out_fldr=r'./data/output/match_visualization/carla0605', dense_gps=False, dense_interval=10.0,
                   gps_radius=10.0, multi_core_save=True, instant_output=True)

    match_res, warn_info, error_info = mpm.execute()
    print(time.time() - a)
    print(warn_info.keys())
    print(match_res)
    print(error_info)
    # match_res.to_csv(fr'./data/output/match_visualization/xa_sample/match_res.csv', encoding='utf_8_sig', index=False)


def zdsy_test():
    gps_df = gpd.read_file(r'./data/input/net/test/zdsy/gps.csv')

    l = gpd.read_file(r'./data/input/net/test/zdsy/FinalLink_dir.shp')
    n = gpd.read_file(r'./data/input/net/test/zdsy/FinalNode.shp')
    my_net = Net(link_gdf=l,
                 node_gdf=n, grid_len=2000, is_hierarchical=True, fmm_cache=False, recalc_cache=False,
                 fmm_cache_fldr=r'./data/input/net/test/zdsy/', cache_cn=6, cut_off=1000,
                 cache_slice=15)

    my_net.init_net()  # net初始化

    a = time.time()
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=80.0, flag_name='zdsy_test',
                   use_sub_net=False, use_heading_inf=True,
                   omitted_l=6.0, del_dwell=False, dwell_l_length=15.0, dwell_n=2,
                   is_lower_f=False, top_k=10,
                   is_rolling_average=False,
                   export_html=True, export_geo_res=True, use_gps_source=False,
                   out_fldr=r'./data/output/match_visualization/zdsy_test/',
                   gps_radius=10.0, multi_core_save=True, instant_output=True)

    match_res, warn_info, error_info = mpm.execute()
    print(time.time() - a)
    print(warn_info.keys())
    print(match_res)
    print(error_info)


if __name__ == '__main__':

    # t_cq_match()
    t_sample_match()
    # t_sample_0424_match()

    # check_0325()
    # dense_example()
    # bug_0402()
    # t_0326_taxi()

    # route2gps()

    # agents_150_test()
    # simple_0419_net()
    #
    # route2gps_break_test()
    # leeds_yg_test()
    # xishu_0507()
    # sz_rand_route_match()

    # bug_0516()
    # m_core_execute()
    # t_simplify()

    # xishu_sample_xian()

    # sz_real_match()
    # print(segmentize(s_loc=(1, 12), e_loc=(0, -90), n=5))

    # carla_0605()
    # zdsy_test()
    # bug_0614()
    # from shapely.geometry import LineString
    # a = gpd.GeoSeries(LineString([(0,0), (0.5,0.1), (5,5)]))
    # x = a.simplify(0.5)
    # print(list(x[0].coords))

# 0.3.1: cos nan fill 1, remove dup