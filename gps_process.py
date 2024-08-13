# -- coding: utf-8 --
# @Time    : 2024/8/7 18:11
# @Author  : TangKai
# @Team    : ZheChengData

import geopandas as gpd
import datetime
from src.gotrackit.gps.LocGps import TrajectoryPoints

if __name__ == '__main__':
    gps_df = gpd.read_file(r'./data/input/net/test/0402BUG/gps/gps_CORRECT.geojson')
    # gps_df = gpd.read_file(r'./data/input/net/test/0402BUG/gps/new_gps.geojson')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                          result_type='expand')
    gps_df.drop(columns=['geometry'], axis=1, inplace=True)
    gps_df['agent_id'] = gps_df['agent_id'].astype(int)
    gps_df['time'] = [datetime.datetime.now().timestamp() * 1000 + i for i in range(0, len(gps_df))]
    gps_df['seq'] = 12
    tp = TrajectoryPoints(gps_points_df=gps_df, time_unit='ms')
    tp.simplify_trajectory(l_threshold=10.0).del_dwell_points()
    x = tp.trajectory_data()
    x.to_file(r'./data/input/net/test/0402BUG/gps/gps_simplify.geojson', driver='GeoJSON')