# -- coding: utf-8 --
# @Time    : 2024/3/18 20:17
# @Author  : TangKai
# @Team    : ZheChengData

import geopandas as gpd
import src.gotrackit.netreverse.NetGen as ng
# import gotrackit.netreverse.NetGen as ng

def redivide_link_0502():
    link = gpd.read_file(r'./data/input/net/test/0502BUG/路网/modifiedConn_link.shp')
    node = gpd.read_file(r'./data/input/net/test/0502BUG/路网/modifiedConn_node.shp')
    print('done reading')
    print(len(link))
    link.drop_duplicates(subset=['from_node', 'to_node'], inplace=True, keep='first')
    link.reset_index(inplace=True, drop=True)
    print(len(link))
    nv = ng.NetReverse()

    link = nv.clean_link_geo(gdf=link, plain_crs='EPSG:32649', l_threshold=10.0)
    print(r'done cleaning')
    # 执行划分路网
    # divide_l: 所有长度大于divide_l的路段都将按照divide_l进行划分
    # min_l: 划分后如果剩下的路段长度小于min_l, 那么此次划分将不被允许
    new_link, new_node = nv.divide_links(link_gdf=link, node_gdf=node, divide_l=4000, min_l=5.0)
    print('done redividing')
    new_link.to_file(r'./data/input/net/test/0502BUG/路网/link.shp')
    new_node.to_file(r'./data/input/net/test/0502BUG/路网/node.shp')


def redivide_test():
    ######   将数据处理为标准格式    ######
    # link_gdf = gpd.read_file(r'./data/input/net/test/0317/link.geojson')
    # node_gdf = gpd.read_file(r'./data/input/net/test/0317/node.geojson')
    # node_gdf.rename(columns={'osmid': 'node_id'}, inplace=True)
    # node_gdf['node_id'] = node_gdf['node_id'].astype(int)
    # node_map = {k: v for k, v in zip(node_gdf['node_id'], range(1, len(node_gdf) + 1))}
    # node_gdf['node_id'] = node_gdf.apply(lambda row: node_map[row['node_id']], axis=1)
    # link_gdf['from'] = link_gdf['from'].astype(int)
    # link_gdf['to'] = link_gdf['to'].astype(int)
    #
    # link_gdf.rename(columns={'fid': 'link_id', 'from': 'from_node', 'to': 'to_node'}, inplace=True)
    # link_gdf['from_node'] = link_gdf.apply(lambda row: node_map[row['from_node']], axis=1)
    # link_gdf['to_node'] = link_gdf.apply(lambda row: node_map[row['to_node']], axis=1)
    # link_gdf['dir'] = 0
    # link_gdf = link_gdf.to_crs('EPSG:4326')
    # node_gdf = node_gdf.to_crs('EPSG:4326')
    # link_gdf.to_file(r'./data/input/net/test/0317/link1.geojson', encoding='gbk', driver='GeoJSON')
    # node_gdf.to_file(r'./data/input/net/test/0317/node1.geojson', encoding='gbk', driver='GeoJSON')
    ######   将数据处理为标准格式    ######

    link = gpd.read_file(r'./data/input/net/test/0317/link1.geojson')
    node = gpd.read_file(r'./data/input/net/test/0317/node1.geojson')

    nv = ng.NetReverse()
    # 执行划分路网
    # divide_l: 所有长度大于divide_l的路段都将按照divide_l进行划分
    # min_l: 划分后如果剩下的路段长度小于min_l, 那么此次划分将不被允许
    new_link, new_node = nv.divide_links(link_gdf=link, node_gdf=node, divide_l=50, min_l=5.0)

    new_link.to_file(r'./data/input/net/test/0317/divide_link.geojson', driver='GeoJSON')
    new_node.to_file(r'./data/input/net/test/0317/divide_node.geojson', driver='GeoJSON')


if __name__ == '__main__':
    redivide_link_0502()

