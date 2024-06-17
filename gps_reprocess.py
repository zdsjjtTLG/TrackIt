# -- coding: utf-8 --
# @Time    : 2024/5/30 10:01
# @Author  : TangKai
# @Team    : ZheChengData


import pandas as pd
import geopandas as gpd
# from src.gotrackit.gps.GpsTrip import GpsPreProcess
from gotrackit.gps.GpsTrip import GpsPreProcess

def gps_od_sc():
    from src.gotrackit.tools.coord_trans import LngLatTransfer
    con = LngLatTransfer()
    sz_test_gps_gdf = pd.read_csv(r'./data/input/net/test/0516BUG/车辆轨迹V3.csv', encoding='ANSI')
    # sz_test_gps_gdf = pd.read_csv(r'./data/input/net/test/0402BUG/gps/gps.csv')
    sz_test_gps_gdf.rename(
        columns={'VehicleNum': 'agent_id', 'longitude': 'lng', 'latitude': 'lat', 'timestamp': 'time'}, inplace=True)

    sz_test_gps_gdf[['lng', 'lat']] = sz_test_gps_gdf.apply(lambda row: con.loc_convert(lng=row['lng'],
                                                                                        lat=row['lat'],
                                                                                        con_type='84-gc'), axis=1,
                                                            result_type='expand')
    _ = sz_test_gps_gdf.copy()
    max_age = sz_test_gps_gdf['agent_id'].max()
    _['agent_id'] = _['agent_id'] + max_age
    sz_test_gps_gdf = pd.concat([sz_test_gps_gdf, _])
    print(sz_test_gps_gdf)

    # grp = GpsPreProcess(gps_df=sz_test_gps_gdf)
    # gps_trip = grp.trip_segmentations(group_gap_threshold=1800, plain_crs='EPSG:32648')
    # gps_trip.to_file(r'./data/output/gps/real_sc/gps_trip.geojson', encoding='gbk')

    # grp = GpsPreProcess(gps_df=sz_test_gps_gdf)
    # gps_od, od_line = grp.generate_od_by_gps(way_points_num=2, group_gap_threshold=1800, plain_crs='EPSG:32648')
    # gps_od.to_csv(r'./data/output/gps/real_sc/gps_od_sc.csv', encoding='utf_8_sig', index=False)
    # od_line.to_file(r'./data/output/gps/real_sc/gps_od_sc.shp')

    right_df = gpd.read_file(r'./data/output/gps/real_sc/gps_trip.geojson')
    right_df = pd.DataFrame(right_df)
    del right_df['geometry']
    print(right_df)
    grp = GpsPreProcess(gps_df=right_df)
    gps_od, od_line = grp.sampling_waypoints_od(way_points_num=2)
    gps_od.to_csv(r'./data/output/gps/real_sc/gps_od_sc_alpha.csv', encoding='utf_8_sig', index=False)
    od_line.to_file(r'./data/output/gps/real_sc/gps_od_sc_alpha.shp')


def t_od():

    gps_gdf = gpd.read_file(r'data/output/gps/example/trip_example.shp')
    gps_gdf = pd.DataFrame(gps_gdf)
    gps_gdf[['lng', 'lat']] = gps_gdf.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                            result_type='expand')
    del gps_gdf['geometry']
    print(gps_gdf)

    grp = GpsPreProcess(gps_df=gps_gdf)
    gps_trip = grp.trip_segmentations(group_gap_threshold=1800, plain_crs='EPSG:32650',
                                      min_distance_threshold=10.0)
    print(gps_trip)
    gps_trip.to_csv(r'./data/output/gps/example/gps_trip.csv', encoding='utf_8_sig', index=False)
    #
    # grp = GpsPreProcess(gps_df=gps_gdf)
    # gps_od, od_line = grp.generate_od_by_gps(way_points_num=5, group_gap_threshold=1800, plain_crs='EPSG:32650', min_distance_threshold=-1)
    # gps_od.to_csv(r'./data/output/gps/example/gps_od.csv', encoding='utf_8_sig', index=False)
    # od_line.to_file(r'./data/output/gps/example/gps_od.shp')

    # right_df = gpd.read_file(r'./data/output/gps/real_sc/gps_trip.geojson')
    # right_df = pd.DataFrame(right_df)
    # del right_df['geometry']
    # print(right_df)
    # grp = GpsPreProcess(gps_df=right_df)
    # gps_od, od_line = grp.sampling_waypoints_od(way_points_num=2)
    # gps_od.to_csv(r'./data/output/gps/real_sc/gps_od_sc_alpha.csv', encoding='utf_8_sig', index=False)
    # od_line.to_file(r'./data/output/gps/real_sc/gps_od_sc_alpha.shp')


def jdqk():
    pass
    gps_df = pd.DataFrame({'agent_id': ['aa', 'bb', 'cc'],
                           'time': [123, 12345, 122222],
                           'lng': [114.123, 114.111, 113.456],
                           'lat': [22.12, 22.11, 22.11]})
    print(gps_df)
    grp = GpsPreProcess(gps_df=gps_df)
    x = grp.trip_segmentations(time_unit='s')
    z, y = grp.sampling_waypoints_od()
    print(x)
    print(z)
    print(y)


if __name__ == '__main__':
    gps_od_sc()
    # t_od()
    # jdqk()