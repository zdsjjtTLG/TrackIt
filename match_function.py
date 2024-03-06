# -- coding: utf-8 --
# @Time    : 2024/3/6 9:31
# @Author  : TangKai
# @Team    : ZheChengData

"""博主的测试代码, 相关文件都在本地, 所以不要运行该文件"""


import os
import pandas as pd
import geopandas as gpd
from src.gotrackit.map.Net import Net
import src.gotrackit.netreverse.NetGen as ng
from src.gotrackit.gps.LocGps import GpsPointsGdf
from src.gotrackit.model.Markov import HiddenMarkov
from src.gotrackit.GlobalVal import NetField, GpsField
from src.gotrackit.visualization import VisualizationCombination

net_field = NetField()
gps_field = GpsField()


def match(plain_crs: str = 'EPSG:32650', geo_crs: str = 'EPSG:4326', search_method: str = 'dijkstra',
          link_path: str = None, node_path: str = None, link_gdf=None, node_gdf=None,
          weight_field='length', gps_df: pd.DataFrame = None, time_format: str = "%Y-%m-%d %H:%M:%S",
          is_lower_f: bool = False, lower_n: int = 2, is_rolling_average: bool = False, window: int = 2,
          gps_buffer: float = 90, use_sub_net: bool = True, buffer_for_sub_net: float = 110,
          beta: float = 20.2, gps_sigma: float = 20.0, flag_name: str = 'test',
          export_html: bool = False, export_geo_res: bool = False, geo_res_fldr: str = None, html_fldr: str = None):
    print(fr'using {search_method}....')
    # 1.新建一个路网对象, 并且使用平面坐标
    my_net = Net(link_path=link_path,
                 node_path=node_path,
                 link_gdf=link_gdf, node_gdf=node_gdf,
                 weight_field=weight_field, geo_crs=geo_crs, plane_crs=plain_crs, search_method=search_method)

    # 初始化
    my_net.init_net()

    match_res_df = pd.DataFrame()

    # 对每辆车的轨迹进行匹配
    for agent_id, gps_df in gps_df.groupby(gps_field.AGENT_ID_FIELD):
        file_name = '-'.join([flag_name, str(agent_id)])
        print(rf'agent: {agent_id}')
        _gps_df = gps_df[gps_df[gps_field.AGENT_ID_FIELD] == agent_id].copy()
        _gps_df.reset_index(inplace=True, drop=True)
        gps_obj = GpsPointsGdf(gps_points_df=_gps_df, time_format=time_format,
                               buffer=gps_buffer,
                               geo_crs=geo_crs, plane_crs=plain_crs, max_increment_times=5)
        # 降频处理
        if is_lower_f:
            print(rf'lower {lower_n}-frequency')
            gps_obj.lower_frequency(n=lower_n)

        if is_rolling_average:
            print(rf'rolling average by window {window}')
            gps_obj.rolling_average(window=window)

        # 依据当前的GPS数据(源数据)做一个子网络
        if use_sub_net:
            print(rf'using sub net')
            sub_net = my_net.create_computational_net(gps_array_buffer=gps_obj.get_gps_array_buffer(buffer=buffer_for_sub_net))
            # 初始化一个隐马尔可夫模型
            hmm_obj = HiddenMarkov(net=sub_net, gps_points=gps_obj, beta=beta, gps_sigma=gps_sigma)
        else:
            print(rf'using whole net')
            hmm_obj = HiddenMarkov(net=my_net, gps_points=gps_obj, beta=beta, gps_sigma=gps_sigma)

        # 求解参数
        hmm_obj.generate_markov_para()
        hmm_obj.solve()
        _match_res_df = hmm_obj.acquire_res()
        if hmm_obj.is_warn:
            print(r'重新计算...')
            gps_obj.rolling_average(window=2)
            hmm_obj = HiddenMarkov(net=sub_net, gps_points=gps_obj, beta=beta, gps_sigma=gps_sigma)
            hmm_obj.generate_markov_para()
            hmm_obj.solve()
            _match_res_df = hmm_obj.acquire_res()
            print(hmm_obj.is_warn)
        else:
            print('no warning...')

        if export_geo_res:
            hmm_obj.acquire_geo_res(out_fldr=geo_res_fldr,
                                    flag_name=file_name)

        match_res_df = pd.concat([match_res_df, _match_res_df])

        if export_html:
            if export_html:
                # 4.初始化一个匹配结果管理器
                vc = VisualizationCombination(use_gps_source=False)
                vc.collect_hmm(hmm_obj)
                vc.visualization(zoom=15, out_fldr=html_fldr,
                                 file_name=file_name)
                del vc
    match_res_df.reset_index(inplace=True, drop=True)
    return match_res_df


# test 1
def t_lane_match():
    """车道匹配测试"""

    # def generate_net_from_lane():
    #     lane_center_gdf = gpd.read_file(r'data/input/net/test/lane/edge_3857.shp')
    #     print(lane_center_gdf)
    #     print(lane_center_gdf['id'].unique())
    #     lane_center_gdf = lane_center_gdf.to_crs('EPSG:4326')
    #
    #     nv = ng.NetReverse()
    #     link_gdf, node_gdf, node_group_status_gdf = nv.create_node_from_link(link_gdf=lane_center_gdf,
    #                                                                          update_link_field_list=['link_id',
    #                                                                                                  'from_node',
    #                                                                                                  'to_node',
    #                                                                                                  'dir', 'length'],
    #                                                                          execute_modify=True,
    #                                                                          modify_minimum_buffer=0.35,
    #                                                                          fill_dir=1,
    #                                                                          plain_prj='EPSG:32650',
    #                                                                          out_fldr=r'./data/input/net/lane/')

    # clean

    link_gdf = gpd.read_file(r'data/input/net/test/lane/LinkAfterModify.shp')
    node_gdf = gpd.read_file(r'data/input/net/test/lane/NodeAfterModify.shp')

    trajectory_gdf = gpd.read_file(r'./data/output/gps/lane/trajectory_3857.shp')
    trajectory_gdf.rename(columns={'id': 'agent_id'}, inplace=True)

    # 抽样测试
    select_agent = list(trajectory_gdf.sample(frac=0.001)['agent_id'].unique())
    print(rf'{len(select_agent)} selected agents......')
    trajectory_gdf = trajectory_gdf[trajectory_gdf['agent_id'].isin(select_agent)]
    trajectory_gdf.reset_index(inplace=True, drop=True)

    trajectory_gdf['time'] = pd.to_datetime(trajectory_gdf['time'], unit='ms')
    trajectory_gdf = trajectory_gdf.to_crs('EPSG:4326')

    trajectory_gdf[['lng', 'lat']] = trajectory_gdf.apply(lambda item: (item['geometry'].x, item['geometry'].y),
                                                          result_type='expand', axis=1)

    # match
    res = match(plain_crs='EPSG:32650', geo_crs='EPSG:4326', link_gdf=link_gdf, node_gdf=node_gdf,
                gps_df=trajectory_gdf, use_sub_net=False, gps_buffer=8.1, export_html=False, export_geo_res=True,
                geo_res_fldr='./data/output/match_visualization/lane', flag_name='lane', lower_n=3, is_lower_f=True)

    res[['prj_lng', 'prj_lat']] = res.apply(lambda item: (item['prj_geo'].x, item['prj_geo'].y), axis=1,
                                            result_type='expand')
    res[['lng', 'lat']] = res.apply(lambda item: (item['geometry'].x, item['geometry'].y), axis=1, result_type='expand')
    res = res[['agent_id', 'veh_type', 'speed', 'time', 'lng', 'lat', 'prj_lng', 'prj_lat', 'link_id']].copy()
    res = pd.merge(res, link_gdf[['link_id', 'id', 'index']], on='link_id', how='left')
    res.rename(columns={'id': 'edge_id', 'index': 'lane_index'}, inplace=True)
    res.drop(columns=['link_id'], axis=1, inplace=True)

    res.to_csv(r'./data/output/match_visualization/lane/trajectory_lane_match.csv',
               encoding='utf_8_sig', index=False)


def t_cq_match():
    gps_df = gpd.read_file(rf'./data/output/gps/cq/gps.shp')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                          result_type='expand')
    print(gps_df)
    match(plain_crs='EPSG:32649', geo_crs='EPSG:4326',
          link_path=r'./data/input/net/test/cq/modifiedConn_link.shp',
          node_path=r'./data/input/net/test/cq/modifiedConn_node.shp', use_sub_net=True, gps_df=gps_df,
          flag_name='cq_test', export_html=True, export_geo_res=True,
          geo_res_fldr=r'./data/output/match_visualization/cq',
          gps_sigma=26,
          html_fldr=r'./data/output/match_visualization/cq')


def t_sample_match():
    gps_df = pd.DataFrame()
    fldr = r'./data/output/gps/sample/'
    for file in os.listdir(fldr):
        _ = gpd.read_file(os.path.join(fldr, file))
        gps_df = pd.concat([gps_df, _])
    gps_df.reset_index(inplace=True, drop=True)
    print(gps_df)
    # gps_df = gps_df[gps_df['agent_id'] == 'xa_car_3'].copy()

    match(plain_crs='EPSG:32649', geo_crs='EPSG:4326', link_path=r'./data/input/net/xian/conn_done_link.shp',
          node_path=r'./data/input/net/xian/conn_done_node.shp', use_sub_net=True, gps_buffer=60,
          buffer_for_sub_net=170, gps_df=gps_df,
          is_rolling_average=False, window=2,
          flag_name='xian_sample', export_html=True, export_geo_res=True,
          html_fldr=r'./data/output/match_visualization/sample',
          geo_res_fldr=r'./data/output/match_visualization/sample')


if __name__ == '__main__':

    # t_lane_match()

    t_cq_match()

    # t_sample_match()
