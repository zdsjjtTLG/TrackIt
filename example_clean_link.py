# -- coding: utf-8 --
# @Time    : 2024/3/5 14:20
# @Author  : TangKai
# @Team    : ZheChengData

"""
例子: 使用format_single_geo函数, 清洗用户的线层文件:
1. 将geometry列的Multi拆分为single类型
2. 去除z坐标
3. 去除重叠点要素
"""

import time
import geopandas as gpd
# import src.gotrackit.netreverse.NetGen as ng
import gotrackit.netreverse.NetGen as ng


def func1():
    # 读取数据
    df = gpd.read_file(r'./data/output/request/0304/道路双线20230131_84.shp')

    # 处理geometry
    link_gdf = ng.NetReverse.clean_link_geo(gdf=df, plain_crs='EPSG:32649')

    # 创建点层
    nv = ng.NetReverse(net_out_fldr=r'./data/input/net/test/cq', plain_prj='EPSG:32649', conn_buffer=0.6)
    l, n, group = nv.create_node_from_link(link_gdf=link_gdf,
                                           update_link_field_list=['link_id', 'from_node', 'to_node', 'dir', 'length'],
                                           fill_dir=1, save_streets_after_modify_minimum=True,
                                           out_fldr=r'./data/input/net/test/cq')

    group.to_file('./data/input/net/test/cq/group.shp')

    nv.modify_conn(link_gdf=l, node_gdf=n)


def func2():
    link = gpd.read_file(r'./data/input/net/test/0308/edge/edge/edge.shp')

    print(link)
    print(link.crs)

    nv = ng.NetReverse()
    l, n, _ = nv.create_node_from_link(link_gdf=link,
                                       update_link_field_list=['link_id', 'from_node', 'to_node', 'dir', 'length'],
                                       fill_dir=1, out_fldr=r'./data/output/0308/', plain_prj='EPSG:32649')
    print(l)
    print(n)


def remap_id_of_link_node():
    l = gpd.read_file(r'./data/input/net/xian/modifiedConn_link.shp')
    n = gpd.read_file(r'./data/input/net/xian/modifiedConn_node.shp')
    print(l[['link_id', 'from_node', 'to_node']])
    print(n[['node_id']])
    nv = ng.NetReverse()
    nv.remapping_link_node_id(l, n)
    print(l[['link_id', 'from_node', 'to_node']])
    print(n[['node_id']])
    print(n['node_id'].max())

def clean():
    l = gpd.read_file(r'./data/input/net/test/0326fyx/load/modifiedConn_link.shp')
    l.loc[l['oneway'] == 'B', 'dir'] = 0
    l.loc[l['oneway'] == 'F', 'dir'] = 1
    l.loc[l['oneway'] == 'T', 'dir'] = 1

    l.to_file(r'./data/input/net/test/0326fyx/load/modifiedConn_link1.shp')


def simplify_trace():
    trace = gpd.read_file(r'./data/input/net/test/simplify_trace/trace.shp')
    trace = trace.to_crs('EPSG:32649')

    trace_a = trace.copy()
    a = time.time()
    trace_a['geometry'] = trace_a['geometry'].simplify(30.0)
    print(time.time() - a)
    trace_a = trace_a.to_crs('EPSG:4326')

    trace_a.to_file(r'./data/input/net/test/simplify_trace/simplify_trace.shp')

    trace_b = trace.copy()
    b = time.time()
    trace_b['geometry'] = trace_b['geometry'].remove_repeated_points(30.0)
    print(time.time() - b)
    trace_b = trace_b.to_crs('EPSG:4326')

    trace_b.to_file(r'./data/input/net/test/simplify_trace/drop_dup_trace.shp')


def redivide_link_node():
    # 读取数据
    origin_link = gpd.read_file(r'./data/input/net/test/0402BUG/load/test_link.geojson')
    # origin_link = gpd.read_file(r'./data/input/net/test/0317/divide_link.geojson')
    print(origin_link)

    # multi_core_merge=True表示启用多进程进行拓扑优化
    # merge_core_num表示启用两个核
    origin_link = ng.NetReverse.clean_link_geo(gdf=origin_link, l_threshold=1.0, plain_crs='EPSG:32650')
    nv = ng.NetReverse(net_out_fldr=r'./data/input/net/test/0402BUG/redivide',
                       plain_prj='EPSG:32650', flag_name='new_divide', multi_core_merge=True,
                       merge_core_num=2, save_streets_after_modify_minimum=True)

    # 处理geometry
    nv.redivide_link_node(link_gdf=origin_link)


def t_merge_multi():
    from src.gotrackit.netreverse.RoadNet.MultiCoreMerge.merge_links_multi import merge_links_multi
    origin_link = gpd.read_file(r'./data/input/net/test/0317/divide_link.geojson')
    origin_node = gpd.read_file(r'./data/input/net/test/0317/divide_node.geojson')
    origin_link = origin_link.to_crs('EPSG:4326')
    origin_node = origin_node.to_crs('EPSG:4326')
    origin_node['node_id'] = origin_node['node_id'].astype(int)
    l, n, _ = merge_links_multi(link_gdf=origin_link, node_gdf=origin_node, limit_col_name='name',
                                accu_l_threshold=90, core_num=3)
    print(l.columns)
    print(n)
    l.to_file(r'./data/input/net/test/0317/multi_merge_link.shp')
    n.to_file(r'./data/input/net/test/0317/multi_merge_node.shp')


def t_create_node():
    link = gpd.read_file(r'C:\Users\Administrator\Downloads\load_5\load_5\modifiedConn_link.shp')
    link = link[['dir', 'geometry']].copy()

    nv = ng.NetReverse()
    l, n, _ = nv.create_node_from_link(link_gdf=link,
                                       update_link_field_list=['link_id', 'from_node', 'to_node', 'dir', 'length'],
                                       fill_dir=1, out_fldr=r'C:\Users\Administrator\Downloads\load_5',
                                       plain_prj='EPSG:32650')
    print(l)
    print(n)


if __name__ == '__main__':
    # func2()
    # remap_id_of_link_node()
    # clean()
    # simplify_trace()
    redivide_link_node()
    # t_merge_multi()
    # import itertools
    # my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # result = itertools.islice(my_list, 2, 6)
    # print(len(result))
    # for item in result:
    #     # print(item)
    #     print('aaa')
    # t_create_node()
