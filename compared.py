# -- coding: utf-8 --
# @Time    : 2024/4/29 14:58
# @Author  : TangKai
# @Team    : ZheChengData
import pickle
import time

import geopandas as gpd
import pandas as pd
from gotrackit.netreverse.RoadNet.Split.SplitPath import split_path
import gotrackit.netreverse.NetGen as ng
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
from src.gotrackit.map.Net import Net
from src.gotrackit.MapMatch import MapMatch

def format_seg_net(link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None):
    link = split_path(path_gdf=link_gdf)
    link.drop(columns=['from_node', 'to_node', 'link_id', 'ft_loc'], axis=1, inplace=True)
    link = ng.NetReverse.create_node_from_link(link_gdf=link,
                                               update_link_field_list=['link_id', 'from_node', 'to_node'],
                                               out_fldr=r'./data/input/net/xian/')
    return link


def build_lvm_net(link_data=None, node_df=None):
    # 创建一个 InMemMap 对象
    my_map = InMemMap('my_net', use_latlon=True, use_rtree=True, index_edges=True)

    for idx, row in node_df.iterrows():
        node_id = row['node_id']
        node_lon, node_lat = row['geometry'].x, row['geometry'].y
        print(node_id, (node_lat, node_lon))
        my_map.add_node(node_id, (node_lat, node_lon))

    for idx, row in link_data.iterrows():

        _dir = int(row['dir'])
        from_node = row['from_node']
        to_node = row['to_node']
        print(from_node, to_node)
        if _dir == 0:
            my_map.add_edge(to_node, from_node)
        my_map.add_edge(from_node, to_node)

    a = time.time()
    my_map.purge()
    print(time.time() - a)
    return my_map


def build_lvm_gps(gps=None):
    return [[lat, lng] for lat, lng in zip(gps['lat'], gps['lng'])]


def lvm_match(used_map=None, gps=None, i=1, out_fldr=None):
    print(gps)
    j = 0
    for max_dist in [100, 200, 300, 400, 500, 600]:
        for min_prob_norm in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
            for non_emitting_states in [True, False]:
                j += 1
                a = time.time()
                # 构建地图匹配工具
                try:
                    matcher = DistanceMatcher(used_map,
                                              max_dist=max_dist * 1.0,
                                              min_prob_norm=min_prob_norm,
                                              obs_noise=36,
                                              obs_noise_ne=50,
                                              dist_noise=50,
                                              max_lattice_width=60,
                                              non_emitting_states=non_emitting_states)
                    # 进行地图匹配
                    states, _ = matcher.match(gps, unique=False)
                    cost = time.time() - a
                    print(rf'{cost} seconds!')
                    print(states)
                    print(max_dist, min_prob_norm, non_emitting_states)
                    # 绘制底图匹配结果
                    mmviz.plot_map(used_map, matcher=matcher, path=out_fldr,
                                   show_labels=False, show_matching=True, show_graph=False,
                                   filename=f'sz_lvm-{j}')
                    print(max_dist, min_prob_norm, non_emitting_states)
                    print('****************')
                except Exception as e:
                    print(repr(e))



def xa_simple_test_lvm():

    l = gpd.read_file(r'./data/input/net/xian/modifiedConn_link.shp')
    n = gpd.read_file(r'./data/input/net/xian/modifiedConn_node.shp')


    gps_df = gpd.read_file(r'./data/output/gps/sample/0424sample.geojson')
    gps_df.drop(columns=['geometry'], axis=1, inplace=True)
    map_con = build_lvm_net(l, n)
    cost_all = 0
    a = 0
    for agent_id, _gps_df in gps_df.groupby('agent_id'):
        a += 1
        gps_array = build_lvm_gps(gps=_gps_df)
        try:
            cost_all += lvm_match(used_map=map_con, gps=gps_array, i=a)
        except KeyError:
            pass
    print(cost_all)

def sz_test_lvm():

    import datetime
    gps_df = pd.read_csv(r'./data/input/net/test/0402BUG/dense_sz.csv')
    gps_df['agent_id'] = gps_df['agent_id'].astype(int)
    gps_df['time'] = [i for i in range(0, len(gps_df))]
    print(gps_df)
    gps_df.to_csv(r'temp.csv', encoding='utf_8_sig')
    l = gpd.read_file(r'./data/input/net/test/0402BUG/load/new_link.shp')
    n = gpd.read_file(r'./data/input/net/test/0402BUG/load/modifiedConn_node.shp')
    map_con = build_lvm_net(l, n)

    cost_all = 0
    a = 0
    for agent_id, _gps_df in gps_df.groupby('agent_id'):
        a += 1
        print(_gps_df)
        gps_array = build_lvm_gps(gps=_gps_df)
        lvm_match(used_map=map_con, gps=gps_array, i=a,
                  out_fldr=r'./data/output/match_visualization/0402BUG/')

    print(cost_all)

def sz_test():
    import datetime
    gps_df = pd.read_csv(r'./data/input/net/test/0402BUG/dense_sz.csv')
    gps_df['agent_id'] = gps_df['agent_id'].astype(int)
    gps_df['time'] = [datetime.datetime.now().timestamp() + i for i in range(0, len(gps_df))]
    print(gps_df)

    my_net = Net(link_path=r'./data/input/net/test/0402BUG/load/new_link.shp',
                 node_path=r'./data/input/net/test/0402BUG/load/modifiedConn_node.shp', not_conn_cost=1000.0,
                 fmm_cache=True, fmm_cache_fldr=r'./data/input/net/test/0402BUG/load/', cache_cn=6,
                 recalc_cache=False, cut_off=1999.0)
    my_net.init_net()  # net初始化
    a = time.time()
    mpm = MapMatch(net=my_net, gps_df=gps_df, gps_buffer=120, flag_name='id1',
                   use_sub_net=False, use_heading_inf=True, max_increment_times=2, increment_buffer=100,
                   is_rolling_average=False, window=2, dense_interval=80, dense_gps=False,
                   omitted_l=20.1, del_dwell=False, dwell_l_length=15.0,
                   is_lower_f=False, lower_n=3,
                   export_html=False, export_geo_res=False, top_k=20,
                   html_fldr=r'./data/output/match_visualization/0402BUG/',
                   use_gps_source=False, gps_radius=10.0)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = mpm.execute()
    print(time.time() - a)
    match_res.to_csv(r'./data/output/match_visualization/0402BUG/match_res.csv', encoding='utf_8_sig')
    print(warn_info)
    print(match_res)


if __name__ == '__main__':
    sz_test()
    # xa_simple_test_lvm()
    # sz_test_lvm()