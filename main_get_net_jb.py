# -- coding: utf-8 --
# @Time    : 2024/2/20 10:32
# @Author  : TangKai
# @Team    : ZheChengData


import pickle
import geopandas as gpd
import matplotlib.pyplot as plt
from src.gotrackit.netreverse.NetGen import NetReverse


if __name__ == '__main__':

    # 读取接驳路径数据
    with open(r'data/input/daily_prj/jn_jb/丁家庄_path_gdf', 'rb') as f:
        path_df = pickle.load(f)
    print(path_df)
    print(path_df[['od_id', 'path_id', 'link_id', 'road_name']])
    path_gdf = gpd.GeoDataFrame(path_df, geometry='geometry', crs='EPSG:4326')

    del path_df

    path_gdf.rename(columns={'path_id': 'scheme', 'link_id': 'seq', 'cost': 'time_cost'}, inplace=True)

    nv = NetReverse(plain_prj='EPSG:32650', flag_name='jn_jb', net_out_fldr=r'./data/output/reverse/jn_jb/',
                    accu_l_threshold=120, angle_threshold=15)
    nv.generate_net_from_path_gdf(path_gdf=path_gdf)
