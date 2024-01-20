# -- coding: utf-8 --
# @Time    : 2023/12/9 8:29
# @Author  : TangKai
# @Team    : ZheChengData

import geopandas as gpd
from src.gotrackit.map.Net import Net
from src.gotrackit.gps.LocGps import GpsPointsGdf
from src.gotrackit.model.Markov import HiddenMarkov
from src.gotrackit.GlobalVal import NetField, GpsField
from src.gotrackit.visualization import VisualizationCombination

net_field = NetField()
gps_field = GpsField()


if __name__ == '__main__':
    # 1.新建一个路网对象, 并且使用平面坐标
    plain_crs = 'EPSG:32649'
    geo_crs = 'EPSG:4326'
    my_net = Net(link_path=r'data/input/net/xian/link.shp',
                 node_path=r'data/input/net/xian/node.shp',
                 weight_field='length', geo_crs=geo_crs, plane_crs=plain_crs)
    # 初始化
    my_net.init_net()

    # 3.读取GPS文件
    gps_df = gpd.read_file(r'./data/output/gps/test120.geojson')

    # 4.初始化一个匹配结果管理器
    vc = VisualizationCombination(use_gps_source=False)

    # 对每辆车的轨迹进行匹配
    for agent_id, gps_df in gps_df.groupby(gps_field.AGENT_ID_FIELD):
        _gps_df = gps_df[gps_df[gps_field.AGENT_ID_FIELD] == agent_id].copy()
        _gps_df.reset_index(inplace=True, drop=True)

        # 创建按一个gps_obj对象
        gps_obj = GpsPointsGdf(gps_points_df=_gps_df, lat_field=gps_field.LAT_FIELD, lng_field=gps_field.LNG_FIELD,
                               time_format="%Y-%m-%d %H:%M:%S", buffer=80.0, geo_crs=geo_crs, plane_crs=plain_crs,
                               max_increment_times=1)
        # 降频处理
        gps_obj.lower_frequency(n=2)
        # gps_obj.lower_frequency(n=9)

        # 做一次滑动窗口平均
        gps_obj.rolling_average(window=2)

        # now gps
        # now_gps_gdf = gps_obj.gps_gdf
        # print(now_gps_gdf)
        # now_gps_gdf = now_gps_gdf.to_crs('EPSG:4326')
        # now_gps_gdf.to_file(r'temp_gps.geojson', driver='GeoJSON')

        # 依据当前的GPS数据(源数据)做一个子网络
        sub_net = my_net.create_computational_net(gps_array_buffer=gps_obj.get_gps_array_buffer(buffer=200.0))

        # 初始化一个隐马尔可夫模型
        hmm_obj = HiddenMarkov(net=sub_net, gps_points=gps_obj, beta=31.2, gps_sigma=10.0)
        hmm_obj.generate_markov_para()
        hmm_obj.solve()
        # gps_link_gdf, base_link_gdf, base_node_gdf = hhm_obj.acquire_visualization_res(use_gps_source=True)

        vc.collect_hmm(hmm_obj)

    vc.visualization(zoom=15, out_fldr=r'./data/output/match_visualization/',
                     file_name='test120')

    # generate_html(mix_gdf=gps_link_gdf, out_fldr=r'./data/output/match_visualization', file_name=r'test2',
    #               link_gdf=base_link_gdf, node_gdf=base_node_gdf)
