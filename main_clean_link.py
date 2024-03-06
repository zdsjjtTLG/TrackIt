# -- coding: utf-8 --
# @Time    : 2024/3/5 14:20
# @Author  : TangKai
# @Team    : ZheChengData

import geopandas as gpd
import src.gotrackit.netreverse.NetGen as ng
from src.gotrackit.tools.geo_process import format_single_geo


if __name__ == '__main__':
    df = gpd.read_file(r'./data/output/request/0304/道路双线20230131_84.shp')
    link_gdf = format_single_geo(gdf=df)

    nv = ng.NetReverse(net_out_fldr=r'./data/input/net/0304', plain_prj='EPSG:32649', conn_buffer=0.6)
    l, n, group = nv.create_node_from_link(link_gdf=link_gdf,
                                           update_link_field_list=['link_id', 'from_node', 'to_node', 'dir', 'length'],
                                           fill_dir=1, save_streets_after_modify_minimum=True,
                                           out_fldr=r'./data/input/net/0304')
    group.to_file('./data/input/net/0304/group.shp')
    nv.modify_conn(link_gdf=l, node_gdf=n)
