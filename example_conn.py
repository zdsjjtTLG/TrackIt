# -- coding: utf-8 --
# @Time    : 2024/2/11 21:51
# @Author  : TangKai
# @Team    : ZheChengData

"""修复联通性的例子"""


from src.gotrackit.map.Net import Net
import geopandas as gpd
from src.gotrackit.netreverse.RoadNet.conn import Conn
import src.gotrackit.netreverse.NetGen as ng


if __name__ == '__main__':

    # n = Net(link_path=r'./data/input/net/xian/link.shp',
    #         node_path=r'./data/input/net/xian/node.shp',
    #         plane_crs='EPSG:32649', create_single=False)
    #
    # conn = Conn(net=n, check_buffer=0.8)
    # conn.execute(generate_mark=True, out_fldr=r'./data/output/conn/', file_name='conn-xa')
    #
    # n.export_net(out_fldr=r'./data/input/net/xian/', flag_name='conn_done', file_type='shp')

    # n = Net(link_path=r'./data/input/net/test/sz/FinalLink.shp',
    #         node_path=r'./data/input/net/test/sz/FinalNode.shp',
    #         plane_crs='EPSG:32650', create_single=False)
    #
    # conn = Conn(net=n, check_buffer=0.8)
    # conn.execute(generate_mark=True, out_fldr=r'./data/output/conn/sz', file_name='conn-sz')
    # n.export_net(out_fldr=r'./data/output/conn/sz', flag_name='conn_sz', file_type='shp')

    link_gdf = gpd.read_file(r'./data/input/net/test/sz/FinalLink.shp')
    node_gdf = gpd.read_file(r'./data/input/net/test/sz/FinalNode.shp')

    nv = ng.NetReverse(net_file_type='shp', conn_buffer=0.8, net_out_fldr=r'./data/input/net/test/sz/')
    new_link_gdf, new_node_gdf = nv.modify_conn(link_gdf=link_gdf, node_gdf=node_gdf, book_mark_name='sz_conn_test')

    print(new_link_gdf)
    print(new_node_gdf)
