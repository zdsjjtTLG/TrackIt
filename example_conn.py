# -- coding: utf-8 --
# @Time    : 2024/2/11 21:51
# @Author  : TangKai
# @Team    : ZheChengData

"""修复联通性的例子"""

import geopandas as gpd
# import src.gotrackit.netreverse.NetGen as ng
import src.gotrackit.netreverse.NetGen as ng


def xa_test():
    link_gdf = gpd.read_file(r'./data/input/net/xian/link.shp')
    node_gdf = gpd.read_file(r'./data/input/net/xian/node.shp')
    nv = ng.NetReverse(plain_prj='EPSG:32649', conn_buffer=0.8, net_out_fldr=r'./data/input/net/xian/')

    new_link_gdf, new_node_gdf = nv.modify_conn(link_gdf=link_gdf, node_gdf=node_gdf, generate_mark=True,
                                                book_mark_name='xa_test')

    new_link_gdf.to_file(r'./data/input/net/xian/new_link.shp', crs='EPSG:4326')
    new_node_gdf.to_file(r'./data/input/net/xian/new_node.shp', crs='EPSG:4326')



def t_xa_bug():
    nv = ng.NetReverse(plain_prj='EPSG:32649', flag_name='all_xian',
                       net_out_fldr=r'./data/output/xa_bug/', save_tpr_link=True, save_done_topo=True,
                       is_multi_core=True,
                       used_core_num=7)
    nv.generate_net_from_pickle(binary_path_fldr=r'./data/input/net/test/xa_bug/path')


def sz_osm():
    link_gdf = gpd.read_file(r'./data/input/net/test/0326fyx/load/create_node/LinkAfterModify.shp')
    node_gdf = gpd.read_file(r'./data/input/net/test/0326fyx/load/create_node/NodeAfterModify.shp')
    nv = ng.NetReverse(plain_prj='EPSG:32650', conn_buffer=0.8, net_out_fldr=r'./data/input/net/test/0326fyx/load/',
                       flag_name='sz_osm')

    new_link_gdf, new_node_gdf = nv.modify_conn(link_gdf=link_gdf, node_gdf=node_gdf, generate_mark=True,
                                                book_mark_name='sz_osm')



def aaa():
    link_gdf = gpd.read_file(r'./data/input/net/xian/link.shp')
    node_gdf = gpd.read_file(r'./data/input/net/xian/node.shp')
    link_gdf = link_gdf.to_crs('EPSG:4326')
    node_gdf = node_gdf.to_crs('EPSG:4326')
    nv = ng.NetReverse(plain_prj='EPSG:32649', conn_buffer=0.8, net_out_fldr=r'./data/input/net/xian/',
                       multi_core_merge=True, core_num=2)

    new_link_gdf, new_node_gdf = nv.modify_conn(link_gdf=link_gdf, node_gdf=node_gdf, generate_mark=True,
                                                book_mark_name='xa_test')

    new_link_gdf.to_file(r'./data/input/net/xian/new_link.shp', crs='EPSG:4326')
    new_node_gdf.to_file(r'./data/input/net/xian/new_node.shp', crs='EPSG:4326')

    new_link_gdf = gpd.read_file(r'./data/input/net/xian/new_link.shp')
    new_node_gdf = gpd.read_file(r'./data/input/net/xian/new_node.shp')

    print(new_node_gdf.crs)
    print(new_link_gdf.crs)
    nv.topology_optimization(link_gdf=new_link_gdf, node_gdf=new_node_gdf)


if __name__ == '__main__':
    xa_test()
    # t_xa_bug()
    # sz_osm()
    # aaa()
