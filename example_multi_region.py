# -- coding: utf-8 --
# @Time    : 2024/6/8 20:11
# @Author  : TangKai
# @Team    : ZheChengData
import os

import geopandas as gpd
from src.gotrackit.tools.geo_process import rn_partition_alpha
import src.gotrackit.netreverse.NetGen as ng

if __name__ == '__main__':
    # link = gpd.read_file(r'./data/input/net/xian/modifiedConn_link.shp')
    # print(link.columns)
    # link = rn_partition_alpha(split_path_gdf=link, partition_num=5, is_geo_coord=True)
    # print(link.columns)
    # link.to_file(r'./data/output/temp/link_group.shp', encoding='gbk')
    # fldr = r'./data/output/multi_nanjing/net/'
    fldr = r'./data/output/shanghai/net/'
    net_list = []
    for i in range(0, 2):
        l = gpd.read_file(os.path.join(fldr, f'region-{i}', 'FinalLink.shp'))
        net_list.append([l,gpd.read_file(os.path.join(fldr, f'region-{i}', 'FinalNode.shp'))])

    print(net_list)
    l, n = ng.NetReverse.merge_net(net_list=net_list, conn_buffer=0.2,
                                   out_fldr=r'./data/output/multi_nanjing/net/merge')
    l.to_file(r'./data/output/shanghai/net/merge/link.shp', encoding='gbk')
    n.to_file(r'./data/output/shanghai/net/merge/node.shp', encoding='gbk')
