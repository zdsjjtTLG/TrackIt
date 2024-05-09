# -- coding: utf-8 --
# @Time    : 2024/4/16 10:19
# @Author  : TangKai
# @Team    : ZheChengData

import pickle
import pandas as pd
import geopandas as gpd
from src.gotrackit.netreverse.Parse.gd_car_path import parse_path_from_gd
from src.gotrackit.gps.GpsXfer import Route2Gps
from datetime import datetime

if __name__ == '__main__':
    # with open(r'./data/output/request/0318cd/11_test_0318cd_gd_path_1', 'rb') as f:
    #     json_data = pickle.load(f)
    # path = gpd.GeoDataFrame()
    # for k in json_data.keys():
    #     print(k)
    #     if k <= 20:
    #         _path = parse_path_from_gd(json_data=json_data[k], check=False, parse_num=3)
    #         _path['path_id'] = _path.apply(lambda row: str(k) + '-' + str(row['scheme']), axis=1)
    #         path = pd.concat([path, _path])
    # path.reset_index(inplace=True, drop=True)
    # path.to_file(r'./data/output/route2gps/path.shp', encoding='gbk')
    # print(path)
    # r2g = Route2Gps(path_gdf=path, path_o_time_df=pd.DataFrame({'path_id': list(path['path_id'].unique()),
    #                                                             'o_time': [datetime.now()] * len(list(path['path_id'].unique()))}))
    # gps_df = r2g.xfer()
    # print(gps_df)
    # gps_df.to_csv(r'./data/output/route2gps/gps.csv', encoding='utf_8_sig', index=False)

    with open(r'./data/output/request/0508rand/16_0508_rand_od_gd_path_1', 'rb') as f:
        json_data = pickle.load(f)
    path = gpd.GeoDataFrame()
    for k in json_data.keys():
        print(k)
        if k <= 40:
            _path = parse_path_from_gd(json_data=json_data[k], check=False, parse_num=3)
            _path['path_id'] = _path.apply(lambda row: str(k) + '-' + str(row['scheme']), axis=1)
            path = pd.concat([path, _path])
    path.reset_index(inplace=True, drop=True)
    path.to_file(r'./data/output/0508rand/path.shp', encoding='gbk')
    print(path)
    r2g = Route2Gps(path_gdf=path, path_o_time_df=pd.DataFrame({'path_id': list(path['path_id'].unique()),
                                                                'o_time': [datetime.now()] * len(
                                                                    list(path['path_id'].unique()))}))
    gps_df = r2g.xfer()
    print(gps_df)
    gps_df.to_csv(r'./data/output/0508rand/gps.csv', encoding='utf_8_sig', index=False)
