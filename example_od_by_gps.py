# -- coding: utf-8 --
# @Time    : 2024/3/7 13:55
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
import geopandas as gpd
from src.gotrackit.netreverse.PublicTools.od import extract_od_by_gps


if __name__ == '__main__':
    sz_gps_df = pd.read_csv(r'./data/output/gps/real_sz/TaxiData2.csv', nrows=5000)
    print(sz_gps_df)