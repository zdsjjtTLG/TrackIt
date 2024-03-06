# -- coding: utf-8 --
# @Time    : 2024/3/5 16:11
# @Author  : TangKai
# @Team    : ZheChengData
import geopandas as gpd
import src.gotrackit.netreverse.NetGen as ng
from src.gotrackit.tools.geo_process import format_single_geo


if __name__ == '__main__':
    df = gpd.read_file(r'./data/input/net/conntest_bug/link.geojson')
    df = format_single_geo(gdf=df)

    df = df[['fff', 'geometry']].copy()
    print(df)
    nv = ng.NetReverse(net_out_fldr=r'./data/input/net/conntest_bug', plain_prj='EPSG:32649', conn_buffer=0.6)
    l, n, group = nv.create_node_from_link(link_gdf=df,
                                           update_link_field_list=['link_id', 'from_node', 'to_node', 'dir', 'length'],
                                           fill_dir=1, save_streets_after_modify_minimum=True,
                                           out_fldr=r'./data/input/net/conntest_bug')

    nv.modify_conn(link_gdf=l, node_gdf=n)