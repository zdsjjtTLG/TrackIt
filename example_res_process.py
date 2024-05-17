# -- coding: utf-8 --
# @Time    : 2024/5/9 12:54
# @Author  : TangKai
# @Team    : ZheChengData


import pandas as pd
from src.gotrackit.gps.LocGps import GpsPointsGdf


if __name__ == '__main__':
    match_res = pd.read_csv(r'./data/output/match_visualization/0402BUG/match_res.csv')
    match_res['next_single'] = match_res['single_link_id'].shift(1).fillna(-1).astype(int)
    print(match_res)
    match_res['label'] = match_res['single_link_id'] - match_res['next_single']
    GpsPointsGdf.del_consecutive_zero(df=match_res, col='label', n=0)
    print(match_res)


