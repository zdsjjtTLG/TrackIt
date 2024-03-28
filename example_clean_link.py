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

def clean():
    l = gpd.read_file(r'./data/input/net/test/0326fyx/load/modifiedConn_link.shp')
    l.loc[l['oneway'] == 'B', 'dir'] = 0
    l.loc[l['oneway'] == 'F', 'dir'] = 1
    l.loc[l['oneway'] == 'T', 'dir'] = 1

    l.to_file(r'./data/input/net/test/0326fyx/load/modifiedConn_link1.shp')


if __name__ == '__main__':
    # func2()
    # remap_id_of_link_node()
    clean()

