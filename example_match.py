# -- coding: utf-8 --
# @Time    : 2024/3/6 9:31
# @Author  : TangKai
# @Team    : ZheChengData

"""博主的测试代码, 相关文件都在本地, 所以不要运行该文件"""

import os
import pandas as pd
import geopandas as gpd
from src.gotrackit.map.Net import Net
from src.gotrackit.MapMatch import MapMatch
# from gotrackit.map.Net import Net
# from gotrackit.MapMatch import MapMatch


# test 1
def t_lane_match():
    """车道匹配测试"""

    # def generate_net_from_lane():
    #     lane_center_gdf = gpd.read_file(r'data/input/net/test/lane/edge_3857.shp')
    #     print(lane_center_gdf)
    #     print(lane_center_gdf['id'].unique())
    #     lane_center_gdf = lane_center_gdf.to_crs('EPSG:4326')
    #
    #     nv = ng.NetReverse()
    #     link_gdf, node_gdf, node_group_status_gdf = nv.create_node_from_link(link_gdf=lane_center_gdf,
    #                                                                          update_link_field_list=['link_id',
    #                                                                                                  'from_node',
    #                                                                                                  'to_node',
    #                                                                                                  'dir', 'length'],
    #                                                                          execute_modify=True,
    #                                                                          modify_minimum_buffer=0.35,
    #                                                                          fill_dir=1,
    #                                                                          plain_prj='EPSG:32650',
    #                                                                          out_fldr=r'./data/input/net/lane/')

    # clean

    link_gdf = gpd.read_file(r'data/input/net/test/lane/LinkAfterModify.shp')
    node_gdf = gpd.read_file(r'data/input/net/test/lane/NodeAfterModify.shp')
    my_net = Net(link_gdf=link_gdf, node_gdf=node_gdf)
    my_net.init_net()

    trajectory_gdf = gpd.read_file(r'./data/output/gps/lane/trajectory_3857.shp')
    trajectory_gdf.rename(columns={'id': 'agent_id'}, inplace=True)

    # 抽样测试
    select_agent = list(trajectory_gdf.sample(frac=0.0001)['agent_id'].unique())
    print(rf'{len(select_agent)} selected agents......')
    trajectory_gdf = trajectory_gdf[trajectory_gdf['agent_id'].isin(select_agent)]
    trajectory_gdf.reset_index(inplace=True, drop=True)

    trajectory_gdf['time'] = pd.to_datetime(trajectory_gdf['time'], unit='ms')
    trajectory_gdf = trajectory_gdf.to_crs('EPSG:4326')

    trajectory_gdf[['lng', 'lat']] = trajectory_gdf.apply(lambda item: (item['geometry'].x, item['geometry'].y),
                                                          result_type='expand', axis=1)

    # match
    mpm = MapMatch(net=my_net, gps_df=trajectory_gdf,
                   is_rolling_average=False, window=2, gps_buffer=12, use_sub_net=False,
                   flag_name='check_0325', export_html=True, export_geo_res=True,
                   html_fldr=r'./data/output/match_visualization/lane',
                   use_heading_inf=False,
                   geo_res_fldr=r'./data/output/match_visualization/lane', dense_gps=False)
    res, _ = mpm.execute()

    res[['prj_lng', 'prj_lat']] = res.apply(lambda item: (item['prj_geo'].x, item['prj_geo'].y), axis=1,
                                            result_type='expand')
    res[['lng', 'lat']] = res.apply(lambda item: (item['geometry'].x, item['geometry'].y), axis=1, result_type='expand')
    res = res[['agent_id', 'veh_type', 'speed', 'time', 'lng', 'lat', 'prj_lng', 'prj_lat', 'link_id']].copy()
    res = pd.merge(res, link_gdf[['link_id', 'id', 'index']], on='link_id', how='left')
    res.rename(columns={'id': 'edge_id', 'index': 'lane_index'}, inplace=True)
    res.drop(columns=['link_id'], axis=1, inplace=True)

    res.to_csv(r'./data/output/match_visualization/lane/trajectory_lane_match.csv',
               encoding='utf_8_sig', index=False)


def t_cq_match():
    gps_df = gpd.read_file(rf'./data/output/gps/cq/gps.shp')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                          result_type='expand')
    gps_df.drop(columns=['geometry'], axis=1, inplace=True)
    my_net = Net(link_path=r'./data/input/net/test/cq/modifiedConn_link.shp',
                 node_path=r'./data/input/net/test/cq/modifiedConn_node.shp')
    my_net.init_net()

    # match
    mpm = MapMatch(net=my_net, gps_df=gps_df,
                   is_rolling_average=False, flag_name='cq_test',
                   export_html=True, export_geo_res=True,
                   html_fldr=r'./data/output/match_visualization/cq',
                   geo_res_fldr=r'./data/output/match_visualization/cq', dense_gps=False,
                   use_sub_net=True)
    res, _ = mpm.execute()


def t_sample_match():

    # 读取GPS数据
    # 这是一个有10辆车的GPS数据的文件
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    gps_df = pd.read_csv(r'./data/output/gps/sample/0327sample.csv')
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
    #                html_fldr=r'./data/output/match_visualization/xa_sample',
    #                use_gps_source=True,
    #                geo_res_fldr=r'./data/output/match_visualization/xa_sample', dense_gps=False)

    # 这个地图匹配对象, 指定一些额外的参数, 可以全部匹配成功
    # is_rolling_average=True, 启用了滑动窗口平均来对GPS数据进行降噪
    # window=3, 滑动窗口大小为3
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=100, flag_name='xa_sample',
                   use_sub_net=True, use_heading_inf=False,
                   is_rolling_average=True, window=3,
                   export_html=True, export_geo_res=True,
                   html_fldr=r'./data/output/match_visualization/xa_sample',
                   geo_res_fldr=r'./data/output/match_visualization/xa_sample', dense_gps=False)
    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info = mpm.execute()
    print(warn_info)
    print(match_res)
    match_res.to_csv(r'./data/output/match_visualization/xa_sample/match_res.csv', encoding='utf_8_sig', index=False)

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
    #                html_fldr=r'./data/output/match_visualization/xa_sample',
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
                   html_fldr=r'./data/output/match_visualization/xa_sample',
                   geo_res_fldr=r'./data/output/match_visualization/xa_sample',)
    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info = mpm.execute()
    print(warn_info)
    print(match_res)
    match_res.to_csv(r'./data/output/match_visualization/xa_sample/xishu_match_res.csv', encoding='utf_8_sig', index=False)


def dense_example():

    gps_df = gpd.read_file(r'./data/output/gps/dense_example/test999.geojson')
    my_net = Net(link_path=r'./data/input/net/xian/modifiedConn_link.shp',
                 node_path=r'./data/input/net/xian/modifiedConn_node.shp')
    my_net.init_net()

    # match
    mpm = MapMatch(net=my_net, gps_df=gps_df, is_rolling_average=True, window=2, flag_name='dense_example',
                   export_html=True, export_geo_res=True,
                   gps_buffer=400,
                   html_fldr=r'./data/output/match_visualization/dense_example',
                   geo_res_fldr=r'./data/output/match_visualization/dense_example', dense_gps=True,
                   use_sub_net=True, dense_interval=30, use_gps_source=False, use_heading_inf=True)
    res, _ = mpm.execute()
    print(res)
    print(_)

def t_0326_taxi():
    gps_df = pd.read_csv(r'./data/input/net/test/0326fyx/gps/part/TaxiData-Sample.csv')
    my_net = Net(link_path=r'./data/input/net/test/0326fyx/load/modifiedConn_link1.shp',
                 node_path=r'./data/input/net/test/0326fyx/load/modifiedConn_node.shp',
                 not_conn_cost=2000)
    my_net.init_net()

    # match
    mpm = MapMatch(net=my_net, gps_df=gps_df,
                   is_rolling_average=True, window=2, flag_name='0326_taxi_a',
                   export_html=True, export_geo_res=True,
                   gps_buffer=500, gps_sigma=30, beta=5.0,
                   html_fldr=r'./data/output/match_visualization/0326_taxi',
                   geo_res_fldr=r'./data/output/match_visualization/0326_taxi', dense_gps=True,
                   use_sub_net=True, dense_interval=60, use_gps_source=True)
    res, _ = mpm.execute()
    print(res)

def check_0325():
    # 某快速路匹配示例
    my_net = Net(link_path=r'./data/input/net/test/0325/G15_links.shp',
                 node_path=r'./data/input/net/test/0325/G15_gps_node.shp')
    my_net.init_net()

    gps_df = pd.read_csv(r'./data/input/net/test/0325/car_gps_test_noheading.csv')
    gps_df = gps_df.loc[0:260, :].copy()
    print(gps_df)
    mpm = MapMatch(net=my_net, gps_df=gps_df,
                   is_rolling_average=False, window=2,
                   flag_name='check_0325_source', export_html=True, export_geo_res=True,
                   html_fldr=r'./data/output/match_visualization/sample',
                   geo_res_fldr=r'./data/output/match_visualization/sample')
    res_df, label_list = mpm.execute()
    print(label_list)
    print(res_df)


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

    gps_df['seq'] = [i for i in range(0, len(gps_df))]
    gps_df.to_file(r'aaa.shp')
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
    #                html_fldr=r'./data/output/match_visualization/0329_bug',
    #                geo_res_fldr=r'./data/output/match_visualization/0329_bug')
    # res_df, label_list = mpm.execute()
    # print(label_list)
    # print(res_df)


def bug_0402():
    # 1.读取GPS数据
    # 这是一个有10辆车的GPS数据的文件, 已经做过了数据清洗以及行程切分
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    # gps_df = pd.read_csv(r'./data/input/net/test/0402BUG/gps/id1v1.csv')
    gps_df = gpd.read_file(r'./data/input/net/test/0402BUG/gps/new_gps.geojson')
    gps_df.drop(columns=['geometry'], axis=1, inplace=True)
    # gps_df['agent_id'] = gps_df['agent_id'].fillna(22223)
    # gps_df['agent_id'] = gps_df['agent_id'].astype(int)
    # gps_df['agent_id'] = gps_df['agent_id'].astype(str)
    print(gps_df)
    # gps_df.to_csv(r'./data/input/net/test/0402BUG/gps/gps.csv', encoding='utf_8_sig')
    # gps_df = gps_df[gps_df['agent_id'] == 'xa_car_4']

    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_path=r'./data/input/net/test/0402BUG/load/new_link.shp',
                 node_path=r'./data/input/net/test/0402BUG/load/modifiedConn_node.shp', not_conn_cost=1500)
    my_net.init_net()  # net初始化

    # 3.新建一个地图匹配对象, 指定其使用net对象, gps数据
    # 按照以下参数进行匹配: 匹配程序会报warning, 由于GPS的定位误差较大, 差分航向角的误差也很大
    # mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=100, flag_name='xa_sample',
    #                use_sub_net=True, use_heading_inf=True,
    #                export_html=True, export_geo_res=True,
    #                html_fldr=r'./data/output/match_visualization/xa_sample',
    #                use_gps_source=True,
    #                geo_res_fldr=r'./data/output/match_visualization/xa_sample', dense_gps=False)

    # 3.这个地图匹配对象, 指定一些额外的参数, 可以全部匹配成功
    # is_rolling_average=True, 启用了滑动窗口平均来对GPS数据进行降噪
    # window=3, 滑动窗口大小为3
    # mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=300, flag_name='id1',
    #                is_lower_f=True, lower_n=2,
    #                beta=10,
    #                use_sub_net=True, use_heading_inf=True,
    #                max_increment_times=2, increment_buffer=200,
    #                is_rolling_average=True, window=2,
    #                export_html=True, export_geo_res=True,
    #                html_fldr=r'./data/output/match_visualization/0402BUG/',
    #                geo_res_fldr=r'./data/output/match_visualization/0402BUG/', dense_gps=True, dense_interval=50)

    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=300, flag_name='id1',
                   use_sub_net=True, use_heading_inf=True, max_increment_times=2, increment_buffer=100,
                   is_rolling_average=True, window=2, dense_interval=50,
                   export_html=True, export_geo_res=True, top_k=25,
                   html_fldr=r'./data/output/match_visualization/0402BUG/',
                   geo_res_fldr=r'./data/output/match_visualization/0402BUG/', dense_gps=True)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info = mpm.execute()
    print(warn_info)
    print(match_res)
    # match_res.to_csv(r'/home/omnisky/Desktop/FYX/TrackIt-main/mine/output/id1/match_res.csv', encoding='utf_8_sig', index=False)


if __name__ == '__main__':
    # t_lane_match()

    # t_cq_match()

    # t_sample_match()
    # check_0325()
    # dense_example()
    # t_0326_taxi()
    # t_sample_xa_xishu_match()
    # bug_0329()
    bug_0402()

    # l = gpd.read_file(r'./data/input/net/test/0402BUG/load/link.shp')
    # l = l.to_crs('EPSG:32650')
    # l['geometry'] = l['geometry'].remove_repeated_points(0.5)
    # l = l.to_crs('EPSG:4326')
    # l.to_file(r'./data/input/net/test/0402BUG/load/new_link.shp', encoding='gbk')
