# -- coding: utf-8 --
# @Time    : 2024/2/11 21:51
# @Author  : TangKai
# @Team    : ZheChengData

"""修复联通性的例子"""

import geopandas as gpd
import src.gotrackit.netreverse.NetGen as ng


def xa_test():
    link_gdf = gpd.read_file(r'./data/input/net/xian/link.shp')
    node_gdf = gpd.read_file(r'./data/input/net/xian/node.shp')
    nv = ng.NetReverse(plain_prj='EPSG:32649', conn_buffer=0.8, net_out_fldr=r'./data/input/net/xian/')

    new_link_gdf, new_node_gdf = nv.modify_conn(link_gdf=link_gdf, node_gdf=node_gdf, generate_mark=True,
                                                book_mark_name='xa_test')


def t_xa_bug():
    nv = ng.NetReverse(plain_prj='EPSG:32649',
                       net_out_fldr=r'./data/output/xa_bug/', save_tpr_link=True, save_done_topo=True,
                       is_multi_core=False,
                       used_core_num=7)
    nv.generate_net_from_pickle(binary_path_fldr=r'./data/input/net/test/xa_bug/path')


if __name__ == '__main__':
    # xa_test()
    t_xa_bug()
