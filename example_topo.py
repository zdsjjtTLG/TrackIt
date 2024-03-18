# -- coding: utf-8 --
# @Time    : 2024/3/18 9:56
# @Author  : TangKai
# @Team    : ZheChengData

"""拓扑优化的例子"""

import geopandas as gpd
import src.gotrackit.netreverse.NetGen as ng


def t_xa_bug():
    link_gdf = gpd.read_file(r'./data/output/xa_bug/FinalLink.shp')
    node_gdf = gpd.read_file(r'./data/output/xa_bug/FinalNode.shp')
    nv = ng.NetReverse(plain_prj='EPSG:32649', flag_name='all_xian', accu_l_threshold=250, min_length=50,
                       allow_ring=False,
                       ignore_dir=False, restrict_angle=True, restrict_length=True, angle_threshold=30)
    link_gdf, node_gdf, _ = nv.topology_optimization(link_gdf=link_gdf, node_gdf=node_gdf, out_fldr=r'./data/output/xa_bug')


if __name__ == '__main__':
    # xa_test()
    t_xa_bug()
