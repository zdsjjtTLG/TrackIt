# -- coding: utf-8 --
# @Time    : 2024/1/20 21:27
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
import numpy as np
from src.gotrackit.tools.coord_trans import LngLatTransfer

con = LngLatTransfer()

if __name__ == '__main__':
    df = pd.read_csv(r'./TaxiData2.csv')
    a_df = df.sample(frac=0.1)
    used_car_id = list(a_df['VehicleNum'].unique())[0]

    print(used_car_id)

    used_gps_df = df[df['VehicleNum'] == used_car_id].copy()
    print(used_gps_df['Speed'].unique())

    used_gps_df['time'] = pd.to_datetime(used_gps_df['timestamp'], unit='s')

    used_gps_df['hh'] = used_gps_df['time'].apply(lambda x: x.hour)
    used_gps_df = used_gps_df[used_gps_df['hh'].isin([16])].copy()



    used_gps_df.sort_values(by='time', ascending=True, inplace=True)
    used_gps_df.reset_index(inplace=True, drop=True)
    print(used_gps_df)

    used_gps_df.drop(columns=['timestamp'], axis=1, inplace=True)
    used_gps_df[['longitude', 'latitude']] = used_gps_df[['longitude', 'latitude']].apply(
        lambda item: con.loc_convert(lng=item[0], lat=item[1], con_type='84-gc'), axis=1, result_type='expand')
    used_gps_df.rename(columns={'VehicleNum': 'agent_id'}, inplace=True)

    used_gps_df['seq'] = [i for i in range(1, len(used_gps_df) + 1)]

    used_gps_df.to_csv(r'sz_gps.csv', encoding='utf_8_sig', index=False)
