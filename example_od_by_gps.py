# -- coding: utf-8 --
# @Time    : 2024/3/7 13:55
# @Author  : TangKai
# @Team    : ZheChengData

import geopandas as gpd
from src.gotrackit.netreverse.PublicTools.od import extract_od_by_gps


if __name__ == '__main__':
    sz_gps_df = gpd.read_file(r'./data/output/gps/real_sz/TaxiData2.csv')
    print(sz_gps_df)