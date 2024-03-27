# -- coding: utf-8 --
# @Time    : 2024/3/18 20:17
# @Author  : TangKai
# @Team    : ZheChengData

from src.gotrackit.map.Net import Net

if __name__ == '__main__':

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

    my_net = Net(link_path=r'./data/input/net/test/0317/link1.geojson',
                 node_path=r'./data/input/net/test/0317/node1.geojson')
    print(my_net.geo_crs, my_net.planar_crs)

    # 执行划分路网
    # divide_l: 所有长度大于divide_l的路段都将按照divide_l进行划分
    # min_l: 划分后如果剩下的路段长度小于min_l, 那么此次划分将不被允许
    # is_init_link: 划分后是否重新初始化路网对象
    # method: alpha 或者 beta, 前一种方法可保留与划分前的link的映射关系(_parent_link字段)
    my_net.divide_links(divide_l=50, min_l=5.0, is_init_link=True, method='alpha')
    my_net.export_net(out_fldr=r'./data/input/net/test/0317/', file_type='geojson', flag_name='divide')

