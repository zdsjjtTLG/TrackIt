# -- coding: utf-8 --
# @Time    : 2024/2/11 21:51
# @Author  : TangKai
# @Team    : ZheChengData


from src.gotrackit.netreverse.RoadNet.conn import Conn
from src.gotrackit.map.Net import Net
from src.gotrackit.netreverse.RoadNet.SaveStreets.streets import modify_minimum
import geopandas as gpd

if __name__ == '__main__':

    n = Net(link_path=r'./data/input/net/xian/link.shp',
            node_path=r'./data/input/net/xian/node.shp',
            plane_crs='EPSG:32649', create_single=False)

    conn = Conn(net=n, check_buffer=0.8)
    conn.execute(generate_mark=True, out_fldr=r'./data/output/conn/', file_name='conn-xa')

    n.export_net(out_fldr=r'./data/input/net/xian/', flag_name='conn_done', file_type='shp')