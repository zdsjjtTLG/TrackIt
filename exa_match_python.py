# -- coding: utf-8 --
# @Author  : TangKai
# @Team    : ZheChengData

"""使用原 Python 版 gotrackit 跑 data/input 下的实际匹配案例。"""
import datetime
import json
import os
import sys
import time
import traceback

import geopandas as gpd
import numpy as np
import pandas as pd

from gotrackit.MapMatch import MapMatch
from gotrackit.MatchResAna import dense_res_based_on_net, generate_check_file
from gotrackit.WrapsFunc import function_time_cost
from gotrackit.gps.LocGps import GpsPointsGdf
from gotrackit.gps.Trajectory import TrajectoryPoints
from gotrackit.map.Net import Net
from gotrackit.model.Para import ParaGrid

os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

OUTPUT_ROOT = os.path.join(".", "data", "output", "python")
CASE_NAMES = (
    "cq",
    "0327sample",
    "0424sample",
    "denseexample",
    "0326taxi",
    "check0325",
    "bug0329",
    "bug0516",
    "agents150",
    "route2gps",
    "0402bug",
    "0419test",
    "0506yg",
    "0507xishu",
    "carla0605",
    "zdsy",
    "ldso0703",
    "gjy0828",
    "QuickStart-Match-1",
    "ldso0710",
    "0729hz",
    "250415bug",
    "bug0213",
    "250114",
    "1126",
    "reverse1121",
    "test1104",
    "0614bug",
)


def case_output_dir(case_name: str) -> str:
    out_fldr = os.path.join(OUTPUT_ROOT, case_name)
    os.makedirs(out_fldr, exist_ok=True)
    return out_fldr


def ensure_all_output_dirs() -> None:
    for case_name in CASE_NAMES:
        case_output_dir(case_name)


def kf_smooth_compat(
    trajectory: TrajectoryPoints,
    *,
    p_deviation: float = 0.01,
    o_deviation: float = 0.1,
) -> TrajectoryPoints:
    """兼容 Pandas 3：补回 groupby.apply 不再保留的 agent_id。"""
    gps_attr = "_GpsPointsGdf__gps_points_gdf"
    before = getattr(trajectory, gps_attr)
    grouped_agent_ids = [
        agent_id
        for agent_id, group in before.groupby("agent_id")
        for _ in range(len(group))
    ]
    trajectory.kf_smooth(
        p_deviation=p_deviation,
        o_deviation=o_deviation,
    )
    after = getattr(trajectory, gps_attr)
    if "agent_id" not in after.columns:
        after["agent_id"] = grouped_agent_ids
    return trajectory


def timed_execute(
    case_name: str,
    matcher: MapMatch,
    gps_df: pd.DataFrame | gpd.GeoDataFrame,
    *,
    multi_core: bool = False,
    core_num: int = 1,
) -> tuple[pd.DataFrame, dict, list]:
    """统一计时匹配执行，并落盘 match_res.csv / metadata.json。"""
    out_fldr = case_output_dir(case_name)
    started = time.time()
    if multi_core:
        match_res, warn_info, error_info = matcher.multi_core_execute(
            gps_df=gps_df, core_num=core_num
        )
    else:
        match_res, warn_info, error_info = matcher.execute(gps_df=gps_df)
    seconds = time.time() - started

    if match_res is not None:
        match_res.to_csv(
            os.path.join(out_fldr, "match_res.csv"),
            encoding="utf_8_sig",
            index=False,
        )

    match_rows = int(len(match_res)) if match_res is not None else 0
    warn_agents = int(len(warn_info)) if warn_info is not None else 0
    error_agents = [str(agent_id) for agent_id in (error_info or [])]
    metadata = {
        "case": case_name,
        "seconds": seconds,
        "match_rows": match_rows,
        "warn_agents": warn_agents,
        "error_agents": error_agents,
        "multi_core": multi_core,
    }
    if multi_core:
        metadata["core_num"] = int(core_num)
    with open(os.path.join(out_fldr, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(
        f"{case_name}: rows={match_rows}, warnings={warn_agents}, "
        f"errors={len(error_agents)}, seconds={seconds:.3f}",
        flush=True,
    )
    return match_res, warn_info, error_info


def t_cq_match():
    gps_df = gpd.read_file(rf'./data/input/cq/gps.shp')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                          result_type='expand')
    gps_df['link_id'] = 999
    gps_df.drop(columns=['geometry'], inplace=True)
    my_net = Net(link_path=r'./data/input/cq/modifiedConn_link.shp',
                 node_path=r'./data/input/cq/modifiedConn_node.shp', not_conn_cost=1200, fmm_cache=False,
                 cache_cn=3, prj_cache=False,
                 recalc_cache=False, fmm_cache_fldr=r'./data/input/cq/', is_hierarchical=True)
    my_net.init_net()
    pgd = ParaGrid(beta_list=[1.0, 2.0,6.0], gps_sigma_list=[10.0, 20.0, 120.0])
    # match
    mpm = MapMatch(net=my_net,
                   is_rolling_average=False, flag_name='cq_test',
                   export_html=False, export_geo_res=False,
                   out_fldr=case_output_dir("cq"), dense_gps=False,
                   use_sub_net=True, multi_core_save=True, instant_output=False, use_para_grid=True, para_grid=pgd,
                   user_field_list=['link_id'], heading_vec_len=25.0)
    match_res, warn_info, error_info = timed_execute("cq", mpm, gps_df)
    print(warn_info)
    print(match_res)
    print(error_info)
    print(pgd.search_res)

@function_time_cost
def t_0327sample_match():

    # 读取GPS数据
    # 这是一个有10辆车的GPS数据的文件
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    gps_df = pd.read_csv(r'./data/input/0327sample/0327sample.csv')
    tp = TrajectoryPoints(gps_points_df=gps_df, plain_crs='EPSG:32649')
    tp.lower_frequency(2)
    kf_smooth_compat(tp, o_deviation=0.4)

    gps_df = tp.trajectory_data(_type='df')
    print(gps_df)

    # gps_df = gps_df[gps_df['agent_id'].isin(['xa_car_8'])]
    l = gpd.read_file(r'./data/input/0327sample/modifiedConn_link.shp')
    n = gpd.read_file(r'./data/input/0327sample/modifiedConn_node.shp')

    my_net = Net(link_gdf=l, prj_cache=False,
                 node_gdf=n, fmm_cache=False, fmm_cache_fldr=r'./data/input/0327sample/', recalc_cache=False, cache_cn=1,
                 cache_slice=6, grid_len=2000, is_hierarchical=False)
    my_net.init_net()  # net初始化
    # my_net = Net(link_gdf=l,
    #              node_gdf=n)
    # my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_buffer=100, flag_name='xa_sample',
                   use_sub_net=False, use_heading_inf=True, omitted_l=6.0,
                   del_dwell=False, dwell_l_length=50.0, dwell_n=0,
                   is_lower_f=True, lower_n=1, user_field_list=['agent_id', 'seq'],
                   export_html=False, export_geo_res=False, use_gps_source=True,
                   export_all_agents=True, instant_output=False, visualization_cache_times=20,
                   is_rolling_average=False, window=3,
                   out_fldr=case_output_dir("0327sample"), dense_gps=False,
                   gps_radius=10.0)
    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的agent的相关信息({agent_id1: pd.DataFrame(), agent_id2: pd.DataFrame()...})
    # 第三个是匹配出错的agent的id列表(GPS点经过预处理(或者原始数据)后点数量不足2个)
    # mpm.execute()

    match_res, warn_info, error_info = timed_execute("0327sample", mpm, gps_df)
    print(warn_info)
    print(match_res)
    print(error_info)

@function_time_cost
def t_0424sample_match():

    # 读取GPS数据
    # 这是一个有10辆车的GPS数据的文件
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    gps_df = gpd.read_file(r'./data/input/0424sample/0424sample.geojson')
    gps_df.drop(columns=['geometry'], inplace=True)
    gps_df['from_node'] = 563
    # gps_df = gps_df[gps_df['agent_id'].isin(['xa_car_22'])]
    print(gps_df)
    l = gpd.read_file(r'./data/input/0424sample/modifiedConn_link.shp')
    n = gpd.read_file(r'./data/input/0424sample/modifiedConn_node.shp')
    n.crs = 'epsg:4326'
    # l = gpd.read_file(r'./data/input/QuickStart-Match-1/LinkAfterModify.shp')
    # n = gpd.read_file(r'./data/input/QuickStart-Match-1/NodeAfterModify.shp')
    my_net = Net(link_gdf=l, prj_cache=False,
                 node_gdf=n, fmm_cache=False, fmm_cache_fldr=r'./data/input/0424sample/', recalc_cache=False, cache_cn=3,
                 cache_slice=6, cut_off=1000, not_conn_cost=1200.0, is_hierarchical=True)
    my_net.init_net()  # net初始化
    mpm = MapMatch(net=my_net, gps_buffer=500, flag_name='xa_sample_0424', speed_threshold=120,
                   use_sub_net=True, use_heading_inf=True,
                   omitted_l=6.0, del_dwell=True, dwell_l_length=25.0, dwell_n=1,
                   lower_n=2, is_lower_f=False, top_k=20,
                   is_rolling_average=False, window=3,
                   export_html=False, export_geo_res=False, use_gps_source=False,
                   out_fldr=case_output_dir("0424sample"), dense_gps=True,
                   gps_radius=20.0, multi_core_save=True, instant_output=False, user_field_list=['from_node'],
                   heading_vec_len=50)

    # match_res, warn_info, error_info = mpm.execute()
    # match_res, warn_info, error_info = mpm.multi_core_execute(gps_df=gps_df, core_num=2)
    match_res, warn_info, error_info = timed_execute("0424sample", mpm, gps_df)
    print(error_info)
    print(warn_info)

def t_denseexample_match():

    gps_df = gpd.read_file(r'./data/input/denseexample/test999.geojson')
    gps_df['heading12'] = [i for i in range(len(gps_df))]
    gps_df['sub_seq'] = 969126
    gps_df['agent_id'] = 'car_pop'
    link = gpd.read_file(r'./data/input/denseexample/modifiedConn_link.shp')
    node = gpd.read_file(r'./data/input/denseexample/modifiedConn_node.shp')
    # del link['speed']
    print(link.dtypes)
    print(sys.getsizeof(link))
    my_net = Net(link_gdf=link,
                 node_gdf=node, fmm_cache=False,
                 recalc_cache=False, fmm_cache_fldr=r'./data/input/denseexample', grid_len=1000, cache_cn=1, prj_cache=False,
                 is_hierarchical=True, not_conn_cost=1750.23)
    my_net.init_net()
    pgd = ParaGrid(use_heading_inf_list=[False, True], beta_list=[0.1, ], gps_sigma_list=[0.1, 0.001])
    # match
    mpm = MapMatch(net=my_net, is_rolling_average=False, window=2, flag_name='denseexample',
                   export_html=False, export_geo_res=False,
                   gps_buffer=400,
                   out_fldr=case_output_dir("denseexample"),
                   dense_gps=True, use_st=False,
                   use_sub_net=True, dense_interval=50.0, use_gps_source=False, use_heading_inf=True,
                   gps_radius=15.0, use_para_grid=False, para_grid=pgd, heading_vec_len=15,
                   user_field_list=['sub_seq', 'agent_id', 'lng'])
    match_res_df, warn_info, error_info = timed_execute("denseexample", mpm, gps_df)
    # print(gps_df.columns)
    # print(match_res_df.columns)
    print(match_res_df)
    # print(res)
    # print(warn_info)
    # print(error_info)
    # print(pd.DataFrame(pgd.search_res))
    # print(gpd.__version__)

    dense_res = dense_res_based_on_net(net=my_net, match_res_df=match_res_df)

    dense_res.to_csv(os.path.join(case_output_dir("denseexample"), "dense_res.csv"),
                     encoding='utf_8_sig', index=False)

def t_0326taxi_match():
    gps_df = pd.read_csv(r'./data/input/0326taxi/TaxiData-Sample-a.csv')
    gps_df['lng'] = np.around(gps_df['lng'], 6)
    gps_df['lat'] = np.around(gps_df['lat'], 6)

    link = gpd.read_file(r'./data/input/0326taxi/new_link.shp')
    link['speed'] = [np.random.randint(20, 60) for i in range(len(link))]
    my_net = Net(link_gdf=link,
                 node_path=r'./data/input/0326taxi/modifiedConn_node.shp',
                 not_conn_cost=2000, fmm_cache=False, recalc_cache=False, cut_off=800.0, prj_cache=False,
                 fmm_cache_fldr=r'./data/input/0326taxi/', cache_cn=6, is_hierarchical=True)
    my_net.init_net()
    pgd = ParaGrid(gps_sigma_list=[10, 20, 30], beta_list=[10, 20, 30])
    # match
    gps_df['rkkkl'] = [i for i in range(len(gps_df))]
    gps_df['prj_lng'] = [i + 1 for i in range(len(gps_df))]
    mpm = MapMatch(net=my_net,
                   is_rolling_average=False, window=2, flag_name='0326_taxi',
                   export_html=False, export_geo_res=False, gps_sigma=10,
                   gps_buffer=500, out_fldr=case_output_dir("0326taxi"),
                   dense_gps=False, dense_interval=200, use_heading_inf=False,
                   top_k=40, dwell_n=0,
                   use_sub_net=False, use_gps_source=False, gps_radius=20, instant_output=False, use_para_grid=False,
                   para_grid=pgd, heading_vec_len=80, user_field_list=['rkkkl', 'prj_lng'])
    res, may_error_info, error_info = timed_execute("0326taxi", mpm, gps_df)

def t_check0325_match():
    # 某快速路匹配示例
    my_net = Net(link_path=r'./data/input/check0325/G15_links.shp',
                 node_path=r'./data/input/check0325/G15_gps_node.shp', fmm_cache=False, cut_off=1000.0, recalc_cache=False,
                 fmm_cache_fldr=r'./data/input/check0325/', prj_cache=False)
    my_net.init_net()

    gps_df = pd.read_csv(r'./data/input/check0325/car_gps_test_noheading.csv')
    gps_df = gps_df.loc[0:260, :].copy()
    print(gps_df)
    mpm = MapMatch(net=my_net,
                   is_rolling_average=False, window=2,
                   flag_name='check0325', export_html=False, export_geo_res=False,
                   out_fldr=case_output_dir("check0325"), instant_output=False)
    res, may_error_info, error_info = timed_execute("check0325", mpm, gps_df)
    print(res)
    print(may_error_info)
    print(error_info)

def t_bug0329_match():
    from shapely.geometry import Point
    link = gpd.read_file(r'./data/input/bug0329/LinkAfterModify.shp')
    node = gpd.read_file(r'./data/input/bug0329/NodeAfterModify.shp')
    link = link.to_crs('EPSG:4326')
    node = node.to_crs('EPSG:4326')

    print(link)
    print(node)
    gps_df = gpd.read_file(r'./data/input/bug0329/trajectory_helmert.shp')
    print(gps_df.crs)
    gps_df = gps_df.to_crs('EPSG:4326')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
                                          result_type='expand')

def t_bug0516_match():
    # 1.读取GPS数据
    # 这是一个有10辆车的GPS数据的文件, 已经做过了数据清洗以及行程切分
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    gps_df = pd.read_csv(r'./data/input/bug0516/车辆轨迹V3.csv', encoding='gbk')
    gps_df = gps_df[['agent_id', 'time', 'lng', 'lat']].copy()
    print(gps_df)
    # gps_df = gpd.read_file(r'./data/input/0516BUG/gps_test_0605.shp')
    # gps_df = gps_df[gps_df['id'] == 123]
    # gps_df['agent_id'] = 1111
    # print(gps_df)
    # gps_df['time'] = [datetime.datetime.now().timestamp() + i for i in range(len(gps_df))]
    # gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
    #                                       result_type='expand')
    # del gps_df['geometry']
    link = gpd.read_file(r'./data/input/bug0516/link.shp')
    node = gpd.read_file(r'./data/input/bug0516/Final_node_net.shp')
    

    # link = link.to_crs('EPSG:32648')
    # link['geometry'] = link['geometry'].simplify(20.0)
    # link = link.to_crs('EPSG:4326')
    # link.to_file(r'./data/input/0516BUG/shp/link.shp')
    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_gdf=link,
                 node_gdf=node, not_conn_cost=2000.0, cut_off=1000.0, grid_len=4000,
                 is_hierarchical=True, fmm_cache=False, prj_cache=False)
    my_net.init_net()  # net初始化
    # para_grid = ParaGrid(gps_sigma_list=[i for i in range(1, 40, 10)],
    #                      beta_list=[0.1, 6.0], use_heading_inf_list=[False, True])
    mpm = MapMatch(net=my_net, gps_buffer=700.0, flag_name='id1',
                   use_sub_net=True, use_heading_inf=False,
                   is_rolling_average=False, window=2, dense_interval=400.0, dense_gps=True,
                   omitted_l=20.1, del_dwell=False, dwell_l_length=15.0,
                   export_html=False, export_geo_res=False, top_k=20, is_lower_f=False, lower_n=2,
                   out_fldr=case_output_dir("bug0516"),
                   use_gps_source=False, gps_radius=20.0, instant_output=False)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = timed_execute("bug0516", mpm, gps_df)
    print(warn_info)

    generate_check_file(my_net, warn_info_dict=warn_info, out_fldr=case_output_dir("bug0516"))

def t_agents150_match():
    # 1.读取GPS数据
    gps_df = pd.read_csv(r'./data/input/agents150/150_agents.csv')
    my_net = Net(link_path=r'./data/input/agents150/new_link.shp',
                 node_path=r'./data/input/agents150/modifiedConn_node.shp', not_conn_cost=999.0,
                 fmm_cache=False, fmm_cache_fldr=r'./data/input/agents150/', cache_cn=5, prj_cache=False,
                 recalc_cache=False, is_hierarchical=True)
    my_net.init_net()  # net初始化
    
    mpm = MapMatch(net=my_net, gps_buffer=120, flag_name='id1',
                   use_sub_net=False, use_heading_inf=True,
                   is_rolling_average=True, window=2, dense_interval=80, dense_gps=False,
                   omitted_l=20.1, del_dwell=True, dwell_l_length=15.0,
                   export_html=False, export_geo_res=False, top_k=20,
                   out_fldr=case_output_dir("agents150"), instant_output=False,
                   use_gps_source=True, gps_radius=10.0, use_para_grid=False,
                   visualization_cache_times=2)
    # match_res, warn_info, error_info = mpm.execute()
    match_res, warn_info, error_info = timed_execute(
        "agents150", mpm, gps_df, multi_core=True, core_num=3
    )
    print(match_res)
    print(warn_info)
    print(error_info)

@function_time_cost
def t_route2gps_match():

    gps_df = pd.read_csv(r'./data/input/route2gps/gps.csv')
    # gps_df = gps_df[gps_df['agent_id'].isin(['11-3', '15-1'])]
    gps_df['heading1'] = 123
    gps_df['my_field'] = 'apolq'
    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_path=r'./data/input/route2gps/FinalLink.shp',
                 node_path=r'./data/input/route2gps/FinalNode.shp', not_conn_cost=1500.0, fmm_cache=False,
                 cut_off=1500, recalc_cache=False, fmm_cache_fldr=r'./data/input/route2gps/',
                 prj_cache=False)
    my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_buffer=50, flag_name='cd_route2gps',
                   use_sub_net=False, use_heading_inf=True,
                   is_rolling_average=False, is_lower_f=False, lower_n=2, dense_gps=False,
                   time_format='"%Y-%m-%d %H:%M:%S"', dense_interval=30.0,
                   omitted_l=2.0, del_dwell=True, dwell_l_length=20.0, dwell_n=0,
                   export_html=False, export_geo_res=False, top_k=5, instant_output=False,
                   out_fldr=case_output_dir("route2gps"),
                   use_gps_source=False, gps_radius=10.0, export_all_agents=True, multi_core_save=False,
                   user_field_list=['heading1', 'my_field'])

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, _ = timed_execute("route2gps", mpm, gps_df)
    print(match_res)
    print(match_res.columns)
    print(_)

def t_0419test_match():
    gps = gpd.read_file(r'./data/input/0419test/gps.shp')
    gps['agent_id'] = 1
    gps['time'] = [datetime.datetime.now().timestamp() + i for i in range(len(gps))]

    gps[['lng', 'lat']] = gps.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1, result_type='expand')
    gps.drop(columns=['geometry'], inplace=True)
    # gps = gps.loc[[0, 1], :].copy()

    # 0419test 只保留了 GPS；其深圳路网与 0402BUG/agents150/0326taxi 使用的是同一份数据。
    my_net = Net(link_path=r'./data/input/0419test/new_link.shp',
                 node_path=r'./data/input/0419test/modifiedConn_node.shp', not_conn_cost=1500,
                 fmm_cache=False, prj_cache=False)

    my_net.init_net()  # net初始化
    mpm = MapMatch(net=my_net, gps_buffer=500, flag_name='cd_route2gps',
                   use_sub_net=False, use_heading_inf=True,
                   is_rolling_average=False, window=3, is_lower_f=True, lower_n=2, dense_gps=False,
                   omitted_l=2.0, del_dwell=True,
                   gps_sigma=30,
                   export_html=False, export_geo_res=False, top_k=10,
                   out_fldr=case_output_dir("0419test"),
                   use_gps_source=False,
                   gps_radius=10.0)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = timed_execute("0419test", mpm, gps)
    print(warn_info)
    print(match_res)
    print(error_info)
    for may_error_agent in warn_info.keys():
        print(may_error_agent)
        print(warn_info[may_error_agent])

def t_0506yg_match():

    gps_df = pd.read_csv(r'./data/input/0506yg/gps.csv')
    gps_df['agent_id'] = gps_df['agent_id'].astype(int)
    # gps_df = gpd.read_file(r'./data/input/0506yg/zf_gps.shp')
    # gps_df[['lng', 'lat']] = gps_df.apply(lambda row: (row['geometry'].x, row['geometry'].y), axis=1,
    #                                       result_type='expand')
    # del gps_df['geometry']
    # gps_df['agent_id'] = 123
    #
    gps_df['time'] = [datetime.datetime.now().timestamp() + i for i in range(0, len(gps_df))]
    gps_df['speed'] = 123
    x = gpd.read_file(r'./data/input/0506yg/aaa_link.shp')
    print(x)
    my_net = Net(link_path='./data/input/0506yg/aaa_link.shp', prj_cache=False,
                 node_path='./data/input/0506yg/aaa_node.shp', is_hierarchical=True,
                 fmm_cache=False)
    my_net.init_net()  # net初始化
    mpm = MapMatch(net=my_net,  gps_buffer=60.0, flag_name='yg1',
                   use_sub_net=True, use_heading_inf=False,
                   is_rolling_average=False, window=2, dense_interval=80, dense_gps=False,
                   omitted_l=20.1, del_dwell=False, dwell_l_length=15.0, time_unit='s',
                   export_html=False, export_geo_res=False, top_k=20, is_lower_f=False, lower_n=2,
                   instant_output=False,
                   out_fldr=case_output_dir("0506yg"), user_field_list=['speed'],
                   use_gps_source=True, gps_radius=10.0)

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = timed_execute("0506yg", mpm, gps_df)
    # match_res.to_csv(r'./data/output/match_visualization/0506yg/match_res.csv', encoding='utf_8_sig')
    print(warn_info)
    print(match_res)
    print(error_info)

def t_0507xishu_match():
    # 1.读取GPS数据
    # 这是一个有10辆车的GPS数据的文件, 已经做过了数据清洗以及行程切分
    # 用于地图匹配的GPS数据需要用户自己进行清洗以及行程切分
    p = ParaGrid(beta_list=[1, 4, 6, 14, 20], gps_sigma_list=[10, 20, 50, 120, 150], omitted_l_list=[1, 5, 10])
    gps_df = gpd.read_file(r'./data/input/0507xishu/163.csv')
    # gps_df = gpd.read_file(r'./data/input/0507xishu/163 - 副本.csv')
    print(gps_df.columns)
    gps_df['time'] = [datetime.datetime.now().timestamp() + i for i in range(0, len(gps_df))]
    # 2.构建一个net, 要求路网线层和路网点层必须是WGS-84, EPSG:4326 地理坐标系
    my_net = Net(link_path=r'./data/input/0507xishu/FinalLink.shp',
                 node_path=r'./data/input/0507xishu/FinalNode.shp', not_conn_cost=3200.0,
                 fmm_cache=False, fmm_cache_fldr=r'./data/input/0507xishu', cache_cn=6,
                 recalc_cache=False, cut_off=2500.0, prj_cache=False)
    my_net.init_net()  # net初始化
    mpm = MapMatch(net=my_net, gps_buffer=500.0, flag_name='0507_xishu_120dense',
                   use_sub_net=True, use_heading_inf=True, dense_gps=False,
                   dense_interval=400, beta=20,
                   omitted_l=5.1, del_dwell=True, dwell_l_length=25.0, dwell_n=1,
                   export_html=False, export_geo_res=False, top_k=25,
                   out_fldr=case_output_dir("0507xishu"),
                   use_gps_source=False, gps_radius=10.0, instant_output=False,
                   user_field_list=['lng', 'seq'])

    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的路段节点编号
    match_res, warn_info, error_info = timed_execute("0507xishu", mpm, gps_df)
    print(match_res)
    print(pd.DataFrame(p.search_res))

def t_carla0605_match():
    gps_df = gpd.read_file(r'./data/input/carla0605/tra.geojson')
    gps_df = gps_df.to_crs('EPSG:4326')
    gps_df[['lng', 'lat']] = gps_df.apply(lambda item: (item['geometry'].x, item['geometry'].y),
                                          result_type='expand', axis=1)
    gps_df = pd.DataFrame(gps_df)
    gps_df['aq'] = 1
    # gps_df = gps_df.loc[1000:1400, :]
    del gps_df['geometry']
    print(gps_df)

    l = gpd.read_file(r'./data/input/carla0605/LinkAfterModify.shp')
    n = gpd.read_file(r'./data/input/carla0605/NodeAfterModify.shp')
    l = l.to_crs('EPSG:3857')
    l['geometry'] = l['geometry'].simplify(2.0)
    l = l.to_crs('EPSG:4326')
    n = n.to_crs('EPSG:4326')
    my_net = Net(link_gdf=l, node_gdf=n, fmm_cache=False, prj_cache=False)

    my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_buffer=80.0, flag_name='carla_0605', speed_threshold=150,
                   use_sub_net=False, use_heading_inf=True,
                   omitted_l=6.0, del_dwell=False, dwell_l_length=15.0, dwell_n=2,
                   is_lower_f=False, top_k=20,
                   is_rolling_average=False,
                   export_html=False, export_geo_res=False, use_gps_source=False,
                   out_fldr=case_output_dir("carla0605"), dense_gps=False, dense_interval=10.0,
                   gps_radius=10.0, multi_core_save=True, instant_output=False, heading_vec_len=3, user_field_list=['aq', 'lng'])

    match_res, warn_info, error_info = timed_execute("carla0605", mpm, gps_df)
    print(warn_info.keys())
    print(match_res)
    print(error_info)

def t_zdsy_match():
    gps_df = gpd.read_file(r'./data/input/zdsy/gps.csv')

    l = gpd.read_file(r'./data/input/zdsy/FinalLink_dir.shp')
    n = gpd.read_file(r'./data/input/zdsy/FinalNode.shp')
    l = l.to_crs('EPSG:32650')
    l['geometry'] = l['geometry'].simplify(5.0)
    l = l.to_crs('EPSG:4326')
    l['speed'] = [np.random.randint(20, 60) for i in range(len(l))]
    l[['from_node', 'to_node']] = l[['to_node', 'from_node']]
    l['geometry'] = l['geometry'].reverse()
    l.loc[l['dir'] == 1, 'dir'] = -1
    print(len(l))

    my_net = Net(link_gdf=l, node_gdf=n, grid_len=2000, is_hierarchical=True,
                 cut_off=400, fmm_cache=False, prj_cache=False)

    my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_buffer=80.0, flag_name='zdsy_test',
                   omitted_l=6.0, del_dwell=False, dwell_l_length=15.0, dwell_n=2,
                   is_lower_f=False, top_k=8, dense_gps=False,
                   is_rolling_average=False,
                   export_html=False, export_geo_res=False, use_gps_source=False,
                   out_fldr=case_output_dir("zdsy"),
                   gps_radius=10.0, multi_core_save=True, instant_output=False)
    match_res, warn_info, error_info = timed_execute("zdsy", mpm, gps_df)
    print(warn_info.keys())
    print(match_res)
    print(error_info)

def t_ldsobug_match():

    # import gotrackit.netreverse.NetGen as ng
    # link = gpd.read_file(r'./data/input/ldso0703/split_osm_test.geojson')
    # # print(link)
    # nv = ng.NetReverse.create_node_from_link(link_gdf=link,
    #                                          update_link_field_list=['link_id', 'from_node', 'to_node', 'dir', 'length'],
    #                                          fill_dir=0, out_fldr=r'./data/input/ldso0703')
    # # print(node)
    link = gpd.read_file(r'./data/input/ldso0703/LinkAfterModify.shp')
    node = gpd.read_file(r'./data/input/ldso0703/NodeAfterModify.shp')

    link['dir'] = 1

    gps_df = pd.read_csv(r'./data/input/ldso0703/tra_.csv')
    my_net = Net(link_gdf=link, node_gdf=node, fmm_cache=False, prj_cache=False)
    my_net.init_net()
    # mpm = MapMatch(net=my_net, gps_df=gps_df, use_sub_net=False, export_html=True, time_format='%Y/%m/%d %H:%M:%S.%f',
    #                top_k=5, omitted_l=6,  del_dwell=True, dwell_l_length=0.5,dwell_n=1,
    #                out_fldr=r'./data/output/match_visualization/ldso0703/', export_geo_res=True)
    mpm = MapMatch(net=my_net, gps_buffer=30, flag_name='syc_test', gps_route_buffer_gap=0,
                   time_format='%Y/%m/%d %H:%M:%S.%f', use_sub_net=True, use_heading_inf=True, top_k=5,
                   omitted_l=6.0, del_dwell=True, dwell_l_length=0.5, dwell_n=1,  # 相邻点的间距小于dwell_l_length的gps会被删除，不进入匹配
                   lower_n=2, is_lower_f=False,
                   is_rolling_average=False, window=3,
                   dense_gps=False,
                   export_html=False, export_geo_res=False, use_gps_source=False,  # export_html 设置为true 警告信息才会打印全部字段
                   out_fldr=case_output_dir("ldso0703"),  # 结果输出存放文件夹
                   gps_radius=1.0)
    res, warn, _ = timed_execute("ldso0703", mpm, gps_df)

    generate_check_file(my_net, warn_info_dict=warn, out_fldr=case_output_dir("ldso0703"))
    print(res)

def t_gjy0828_match():
    input_fldr = r'./data/input/gjy0828'
    output_fldr = case_output_dir("gjy0828")

    gps_df = pd.read_csv(os.path.join(input_fldr, '公交线路模拟GPS点.csv'))
    gps_df.rename(columns={'stopnodeid': 'node_id'}, inplace=True)
    gps_df.reset_index(inplace=True, drop=True)
    print(len(gps_df))
    # gps_df['node_id'] = None
    # gps_df.loc[10, 'node_id'] = None
    link = gpd.read_file(os.path.join(input_fldr, 'link.shp'))
    node = gpd.read_file(os.path.join(input_fldr, 'node.shp'))
    # link['speed'] = [np.random.randint(20, 60) for i in range(len(link))]
    my_net = Net(link_gdf=link, node_gdf=node, fmm_cache=False, fmm_cache_fldr=r'./data/input/gjy0828',
                 recalc_cache=False, prj_cache=False, cut_off=800)
    my_net.init_net()
    mpm = MapMatch(net=my_net, flag_name='0808_gjy',
                   export_html=False, export_geo_res=False,
                   gps_buffer=75, out_fldr=output_fldr, dense_gps=False, is_lower_f=False,
                   dense_interval=30, use_st=False,
                   top_k=10, dwell_n=0, del_dwell=False, use_heading_inf=True,
                   use_sub_net=True, use_gps_source=False, gps_radius=6, instant_output=False, use_node_restrict=True)
    # gps_df['eq_id'] = [i for i in range(len(gps_df))]
    # eq_df = pd.DataFrame({'eq_id': [i for i in range(len(gps_df))]})
    # eq_df['from_node'] = 1872
    # eq_df['to_node'] = 1855
    timed_execute("gjy0828", mpm, gps_df)

def quick_start_match_1():

    gps_df = pd.read_csv(r'./data/input/QuickStart-Match-1/example_gps.csv')
    #
    # 利用gps数据构建TrajectoryPoints, 并且对数据进行清洗
    # 是否需要进行该步操作视实际情况而定
    tp = TrajectoryPoints(gps_points_df=gps_df, plain_crs='EPSG:32649')
    tp.lower_frequency(lower_n=2)
    kf_smooth_compat(tp, o_deviation=0.3)
    tp.rolling_average(rolling_window=3)  # 由于样例数据定位频率高且有一定的误差，因此先做间隔采样然后执行滤波平滑
    gps_df = tp.trajectory_data(_type='df')
    # tp.export_html(out_fldr=case_output_dir("QuickStart-Match-1"), file_name='general_gps')

    link = gpd.read_file(r'./data/input/QuickStart-Match-1/modifiedConn_link.shp')
    node = gpd.read_file(r'./data/input/QuickStart-Match-1/modifiedConn_node.shp')
    my_net = Net(link_gdf=link, node_gdf=node, not_conn_cost=1200,
                 fmm_cache=False, prj_cache=False)
    my_net.init_net()  # net初始化

    mpm = MapMatch(net=my_net, gps_buffer=120, flag_name='general_sample', time_format=r'%Y-%m-%d %H:%M:%S',
                   use_heading_inf=True, omitted_l=6.0, export_html=False, del_dwell=False, export_geo_res=False,
                   out_fldr=case_output_dir("QuickStart-Match-1"), dense_gps=True,
                   gps_radius=20.0)
    # 第一个返回结果是匹配结果表
    # 第二个是发生警告的agent的相关信息({agent_id1: pd.DataFrame(), agent_id2: pd.DataFrame()...})
    # 第三个是匹配出错的agent的id列表(GPS点经过预处理(或者原始数据)后点数量不足2个)

    match_res, warn_info, error_info = timed_execute("QuickStart-Match-1", mpm, gps_df)
    print(warn_info)
    print(error_info)

def t_ldso0710_match():
    import gotrackit.netreverse.NetGen as ng
    link = gpd.read_file(r'./data/input/ldso0710/link_test.geojson')
    link = link[['geometry']].copy()
    from gotrackit.tools.coord_trans import LngLatTransfer
    c = LngLatTransfer()
    link = c.gdf_convert(link, con_type='None', ignore_z=True)
    link, node, _ = ng.NetReverse.create_node_from_link(link_gdf=link, update_link_field_list=['link_id',
                                                                                               'from_node', 'to_node', 'length',
                                                                                               'dir'],
                                                        out_fldr=r'./data/input/ldso0710/', fill_dir=1)
    gps_df = gpd.read_file(r'./data/input/ldso0710/traj_test.geojson')
    gps_df['lng'] = gps_df['geometry'].x
    gps_df['lat'] = gps_df['geometry'].y
    gps_df['agent_id'] = 1
    del gps_df['geometry']
    gps_df.rename(columns={'timestamp': 'time'}, inplace=True)

    my_net = Net(link_gdf=link, node_gdf=node, fmm_cache=False,
                 prj_cache=False, cut_off=2000)
    my_net.init_net()
    # mpm = MapMatch(net=my_net, flag_name='ldso0710',
    #                export_html=True, export_geo_res=True,
    #                gps_buffer=100, out_fldr=r'./data/input/ldso0710/',
    #                top_k=10, dwell_n=0, del_dwell=False, use_heading_inf=False,
    #                use_sub_net=False, use_gps_source=False, gps_radius=6,
    #                instant_output=True)
    mpm = MapMatch(net=my_net, flag_name='ldso0710',
                   export_html=False, export_geo_res=False,
                   gps_buffer=100, out_fldr=case_output_dir("ldso0710"),
                   top_k=10, dwell_n=0, del_dwell=False,
                   use_heading_inf=True, heading_para_array=np.array([1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001]),
                   omitted_l=1.0,
                   use_sub_net=False, use_gps_source=False, gps_radius=6,
                   instant_output=False)
    timed_execute("ldso0710", mpm, gps_df)


def t_0402bug_match():
    input_fldr = r'./data/input/0402bug'
    gps_df = gpd.read_file(os.path.join(input_fldr, 'gps_CORRECT.geojson'))
    gps_df[['lng', 'lat']] = gps_df.apply(
        lambda row: (row['geometry'].x, row['geometry'].y),
        axis=1,
        result_type='expand',
    )
    gps_df.drop(columns=['geometry'], inplace=True)
    gps_df['agent_id'] = gps_df['agent_id'].astype(int)
    base_ms = int(datetime.datetime(2024, 1, 1).timestamp() * 1000)
    gps_df['time'] = [base_ms + i for i in range(len(gps_df))]
    gps_df['aaaa'] = 456

    my_net = Net(
        link_path=os.path.join(input_fldr, 'new_link.shp'),
        node_path=os.path.join(input_fldr, 'modifiedConn_node.shp'),
        not_conn_cost=1000.0,
        cut_off=1200.0,
        grid_len=2000,
        is_hierarchical=True,
        prj_cache=True,
        fmm_cache=False,
    )
    my_net.init_net()
    mpm = MapMatch(
        net=my_net,
        gps_buffer=100,
        flag_name='bug_0402',
        time_unit='ms',
        use_sub_net=False,
        use_heading_inf=True,
        omitted_l=1.0,
        del_dwell=True,
        dwell_l_length=15.0,
        lower_n=2,
        is_lower_f=False,
        is_rolling_average=False,
        window=2,
        dense_gps=False,
        dense_interval=80.0,
        top_k=20,
        export_html=False,
        export_geo_res=False,
        out_fldr=case_output_dir('0402bug'),
        use_para_grid=False,
        user_field_list=['aaaa'],
        gps_radius=10.0,
        instant_output=False,
    )
    match_res, warn_info, error_info = timed_execute("0402bug", mpm, gps_df)
    print(warn_info)
    print(match_res)
    print(error_info)


def t_0729hz_match():
    input_fldr = r'./data/input/0729hz'
    gps_df = gpd.read_file(os.path.join(input_fldr, 'matcher_input_gps.geojson'))
    gps_df.drop(columns=['geometry'], inplace=True)

    link = gpd.read_file(os.path.join(input_fldr, 'matcher_input_network.geojson'))
    link['link_id'] = link['link_id'].astype('int64')
    link['from_node'] = link['from_node'].astype('int64')
    link['to_node'] = link['to_node'].astype('int64')
    node = gpd.read_file(os.path.join(input_fldr, 'matcher_input_nodes.geojson'))
    node['node_id'] = node['node_id'].astype('int64')

    my_net = Net(
        link_gdf=link,
        node_gdf=node,
        cut_off=1200.0,
        not_conn_cost=1500.0,
        is_hierarchical=True,
        grid_len=2000,
        fmm_cache=False,
        prj_cache=False,
    )
    my_net.init_net()
    mpm = MapMatch(
        net=my_net,
        gps_buffer=60.0,
        flag_name='0729hz',
        use_sub_net=True,
        use_heading_inf=True,
        omitted_l=6.0,
        del_dwell=False,
        dwell_l_length=15.0,
        dwell_n=1,
        top_k=10,
        gps_sigma=1.0,
        dense_gps=False,
        export_html=False,
        export_geo_res=False,
        out_fldr=case_output_dir('0729hz'),
        gps_radius=10.0,
        instant_output=False,
    )
    match_res, warn_info, error_info = timed_execute("0729hz", mpm, gps_df)
    print(warn_info)
    print(match_res)
    print(error_info)


def t_250415bug_match():
    input_fldr = r'./data/input/250415bug'
    gps_df = gpd.read_file(os.path.join(input_fldr, 'input.csv'))
    link = gpd.read_file(os.path.join(input_fldr, 'LinkAfterModify.shp'))
    node = gpd.read_file(os.path.join(input_fldr, 'NodeAfterModify.shp'))
    my_net = Net(
        link_gdf=link,
        node_gdf=node,
        prj_cache=True,
        is_hierarchical=True,
        recalc_cache=False,
        cut_off=600,
        fmm_cache=False,
    )
    my_net.init_net()
    mpm = MapMatch(
        net=my_net,
        gps_buffer=100,
        flag_name='250415',
        top_k=20,
        use_sub_net=True,
        use_heading_inf=True,
        omitted_l=6.0,
        gps_sigma=1,
        del_dwell=False,
        dwell_l_length=50.0,
        dwell_n=0,
        export_html=False,
        export_geo_res=False,
        use_gps_source=False,
        out_fldr=case_output_dir('250415bug'),
        dense_gps=False,
        gps_radius=5.0,
        instant_output=False,
    )
    match_res, warn_info, error_info = timed_execute("250415bug", mpm, gps_df)
    print(warn_info)
    print(match_res)
    print(error_info)


def t_bug0213_match():
    from gotrackit.gps.GpsTrip import GpsPreProcess
    from gotrackit.tools.coord_trans import LngLatTransfer

    input_fldr = r'./data/input/bug0213'
    gps_gdf = pd.read_json(os.path.join(input_fldr, '车辆数据.json'))
    gps_gdf = gps_gdf.rename(
        columns={'timestamp': 'time', 'pos_wgs84_x': 'lng', 'pos_wgs84_y': 'lat'}
    )
    gps_gdf = gps_gdf.drop(
        columns=['wheel_angle', 'transmission', 'undirected_acc', 'pos_utm_y', 'pos_utm_x'],
        errors='ignore',
    )
    gps_gdf['agent_id'] = 0
    gps_gdf['time'] = (
        pd.to_datetime(gps_gdf['time'], unit='ms')
        .dt.tz_localize('UTC')
        .dt.tz_convert('Asia/Shanghai')
        .dt.strftime('%Y-%m-%d %H:%M:%S')
    )
    gps_gdf = gps_gdf.sort_values(by='time').reset_index(drop=True)

    grp = GpsPreProcess(gps_df=gps_gdf, use_multi_core=False)
    gps_trip = grp.trip_segmentations(
        group_gap_threshold=1800,
        plain_crs='EPSG:32651',
        min_distance_threshold=-1,
        dwell_accu_time=60,
        n=5,
    )
    gps_trip.drop_duplicates(subset=['agent_id', 'time'], keep='first', inplace=True)
    gps_trip.reset_index(inplace=True, drop=True)

    con = LngLatTransfer()
    gps_trip['lng'], gps_trip['lat'] = con.loc_convert(
        lng=gps_trip['lng'], lat=gps_trip['lat'], con_type='84-gc'
    )

    tp = TrajectoryPoints(gps_points_df=gps_trip, time_unit='s', plain_crs='EPSG:32650')
    tp.lower_frequency(lower_n=7)
    tp.del_dwell_points(dwell_n=15, dwell_l_length=20)
    process_df = tp.trajectory_data(_type='df')
    # tp.export_html(out_fldr=case_output_dir('bug0213'), file_name='sample', radius=3.0)

    link = gpd.read_file(os.path.join(input_fldr, 'FinalLink.shp'))
    node = gpd.read_file(os.path.join(input_fldr, 'FinalNode.shp'))
    link['dir'] = 0
    link = link.to_crs('EPSG:4326')
    node = node.to_crs('EPSG:4326')
    my_net = Net(
        link_gdf=link,
        node_gdf=node,
        plane_crs='EPSG:32651',
        cut_off=1200.0,
        prj_cache=True,
        fmm_cache=False,
    )
    my_net.init_net()
    mpm = MapMatch(
        net=my_net,
        flag_name='shanghai',
        use_sub_net=False,
        gps_buffer=100,
        top_k=20,
        dense_gps=True,
        use_heading_inf=False,
        omitted_l=6.0,
        del_dwell=False,
        export_html=False,
        export_geo_res=False,
        use_gps_source=True,
        gps_radius=15.0,
        export_all_agents=False,
        out_fldr=case_output_dir('bug0213'),
        instant_output=False,
    )
    match_res, warn_info, error_info = timed_execute("bug0213", mpm, process_df)
    print(match_res)
    print(warn_info)
    print(error_info)


def t_250114_match():
    from datetime import datetime
    from shapely.geometry import Point

    input_fldr = r'./data/input/250114'
    plain_crs = 'EPSG:32650'
    traj_df = pd.read_csv(os.path.join(input_fldr, 'town12_trajectory.csv'))
    geo_list, lng_list, lat_list, time_list = [], [], [], []
    for i in range(len(traj_df)):
        u_x = float(traj_df['x'][i])
        u_y = float(traj_df['y'][i])
        timestamp = float(traj_df['hmm_time'][i])
        dt_object = datetime.utcfromtimestamp(timestamp)
        formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S.%f')
        geo_list.append(Point(u_x, u_y))
        lng_list.append(u_x)
        lat_list.append(u_y)
        time_list.append(formatted_date)

    traj_df['lng'] = lng_list
    traj_df['lat'] = lat_list
    traj_df['time'] = time_list
    traj_df['agent_id'] = '10001'
    traj_df = gpd.GeoDataFrame(traj_df, geometry=geo_list, crs=plain_crs)
    traj_df = traj_df.to_crs('EPSG:4326')
    traj_df['lng'] = traj_df['geometry'].x
    traj_df['lat'] = traj_df['geometry'].y
    traj_df = traj_df.loc[1100:5300, ['agent_id', 'lng', 'lat', 'time']].copy()

    link = gpd.read_file(os.path.join(input_fldr, 'LinkAfterModify.shp'))
    node = gpd.read_file(os.path.join(input_fldr, 'NodeAfterModify.shp'))
    link = link.to_crs('EPSG:3857')
    link['geometry'] = link['geometry'].simplify(1.5)
    link = link.to_crs('EPSG:4326')
    node = node.to_crs('EPSG:4326')

    my_net = Net(
        link_gdf=link,
        node_gdf=node,
        prj_cache=True,
        cut_off=800,
        not_conn_cost=3000,
        fmm_cache=False,
        recalc_cache=True,
    )
    my_net.init_net()
    mpm = MapMatch(
        net=my_net,
        gps_buffer=200,
        flag_name='250114',
        use_sub_net=False,
        use_heading_inf=True,
        heading_para_array=np.array(
            [1.0, 1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01]
        ),
        omitted_l=6.0,
        del_dwell=False,
        dwell_l_length=10.0,
        dwell_n=1,
        lower_n=5,
        is_lower_f=False,
        beta=3,
        gps_sigma=120,
        is_rolling_average=False,
        window=3,
        time_format='%Y-%m-%d %H:%M:%S.%f',
        dense_gps=False,
        top_k=70,
        export_html=False,
        export_geo_res=False,
        use_gps_source=False,
        out_fldr=case_output_dir('250114'),
        instant_output=False,
    )
    match_res, warn_info, error_info = timed_execute("250114", mpm, traj_df)
    print(warn_info)
    print(match_res)
    print(error_info)


def t_1126_match():
    input_fldr = r'./data/input/1126'
    gps_df = gpd.read_file(os.path.join(input_fldr, 'gps.shp'))
    gps_df['lng'] = gps_df['geometry'].x
    gps_df['lat'] = gps_df['geometry'].y
    gps_df['agent_id'] = 'car1'
    gps_df['time'] = [
        datetime.datetime.now().timestamp() * 1000 + i * 800 for i in range(len(gps_df))
    ]
    gps_df = pd.DataFrame(gps_df.drop(columns=['geometry']))

    link = gpd.read_file(os.path.join(input_fldr, 'LinkAfterModify.shp'))
    node = gpd.read_file(os.path.join(input_fldr, 'NodeAfterModify.shp'))
    link['dir'] = 0
    my_net = Net(
        link_gdf=link,
        node_gdf=node,
        fmm_cache=False,
        prj_cache=True,
        recalc_cache=False,
        grid_len=1000,
        cache_cn=1,
        is_hierarchical=True,
        not_conn_cost=1750.23,
    )
    my_net.init_net()
    mpm = MapMatch(
        net=my_net,
        flag_name='1126',
        export_html=False,
        export_geo_res=False,
        gps_sigma=10,
        gps_buffer=100,
        time_unit='ms',
        out_fldr=case_output_dir('1126'),
        dense_gps=False,
        instant_output=False,
        use_sub_net=True,
        use_gps_source=False,
        use_heading_inf=True,
        gps_radius=15.0,
        use_para_grid=False,
        heading_vec_len=15,
    )
    match_res, warn_info, error_info = timed_execute("1126", mpm, gps_df)
    print(match_res)
    print(warn_info)
    print(error_info)


def t_reverse1121_match():
    input_fldr = r'./data/input/reverse1121'
    gps_df = gpd.read_file(os.path.join(input_fldr, 'gps.shp'))
    gps_df['lng'] = gps_df['geometry'].x
    gps_df['lat'] = gps_df['geometry'].y
    gps_df['agent_id'] = 'car1'
    gps_df['time'] = [
        datetime.datetime.now().timestamp() * 1000 + i * 800 for i in range(len(gps_df))
    ]
    gps_df = pd.DataFrame(gps_df.drop(columns=['geometry']))

    link = gpd.read_file(os.path.join(input_fldr, 'LinkAfterModify.shp'))
    node = gpd.read_file(os.path.join(input_fldr, 'NodeAfterModify.shp'))
    link['dir'] = 0
    my_net = Net(
        link_gdf=link,
        node_gdf=node,
        prj_cache=True,
        fmm_cache=False,
        recalc_cache=False,
        grid_len=1000,
        cache_cn=1,
        is_hierarchical=True,
        not_conn_cost=1750.23,
    )
    my_net.init_net()
    mpm = MapMatch(
        net=my_net,
        is_rolling_average=False,
        window=2,
        flag_name='1121test',
        export_html=False,
        export_geo_res=False,
        gps_buffer=400,
        time_unit='ms',
        out_fldr=case_output_dir('reverse1121'),
        dense_gps=False,
        instant_output=False,
        use_sub_net=True,
        dense_interval=50.0,
        use_gps_source=False,
        use_heading_inf=True,
        gps_radius=15.0,
        use_para_grid=False,
        heading_vec_len=15,
    )
    match_res, warn_info, error_info = timed_execute("reverse1121", mpm, gps_df)
    print(match_res)
    print(warn_info)
    print(error_info)


def t_test1104_match():
    input_fldr = r'./data/input/test1104'
    output_fldr = case_output_dir('test1104')
    gps_df = pd.read_csv(os.path.join(input_fldr, 'newtest.csv'))
    link = gpd.read_file(os.path.join(input_fldr, 'modifiedConnLink.shp'))
    node = gpd.read_file(os.path.join(input_fldr, 'modifiedConnNode.shp'))
    my_net = Net(link_gdf=link, node_gdf=node, cut_off=1200, prj_cache=True)
    my_net.init_net()
    mpm = MapMatch(
        net=my_net,
        flag_name='1104',
        export_html=False,
        export_geo_res=False,
        gps_buffer=250,
        out_fldr=output_fldr,
        top_k=10,
        dwell_n=0,
        del_dwell=False,
        use_heading_inf=True,
        dense_gps=True,
        dense_interval=250,
        use_sub_net=True,
        use_gps_source=False,
        gps_radius=6,
        instant_output=False,
    )
    match_res, warn_info, error_info = timed_execute("test1104", mpm, gps_df)
    print(match_res)
    print(warn_info)
    print(error_info)


def t_0614bug_match():
    input_fldr = r'./data/input/0614bug'
    gps_df = pd.read_csv(os.path.join(input_fldr, '20240320_to_20240320_data_chunk_0.csv'))
    gps_df = gps_df[gps_df['plateNo'] == '川ABQ128'].copy()
    gps_df.rename(
        columns={'posTime': 'time', 'new_longitude': 'lng', 'new_latitude': 'lat'},
        inplace=True,
    )
    gps_df['agent_id'] = 113

    link = gpd.read_file(os.path.join(input_fldr, 'merge_FinalLink.shp'), encoding='gbk')
    node = gpd.read_file(os.path.join(input_fldr, 'merge_FinalNode.shp'), encoding='gbk')
    my_net = Net(
        link_gdf=link,
        node_gdf=node,
        not_conn_cost=2500.0,
        cut_off=600.0,
        grid_len=4000,
        is_hierarchical=True,
    )
    my_net.init_net()

    p = ParaGrid(
        gps_sigma_list=[30, 40, 50, 60, 70, 80],
        beta_list=[4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    mpm = MapMatch(
        net=my_net,
        gps_buffer=900.0,
        flag_name='id1',
        speed_threshold=20000,
        use_sub_net=True,
        use_heading_inf=True,
        time_unit='ms',
        is_rolling_average=False,
        window=2,
        dense_interval=400.0,
        dense_gps=True,
        omitted_l=20.1,
        del_dwell=False,
        dwell_l_length=15.0,
        export_html=False,
        export_geo_res=False,
        top_k=30,
        is_lower_f=False,
        lower_n=2,
        out_fldr=case_output_dir('0614bug'),
        use_gps_source=False,
        gps_radius=20.0,
        instant_output=False,
        para_grid=p,
        use_para_grid=False,
        user_field_list=['plateNo'],
    )
    match_res, warn_info, error_info = timed_execute("0614bug", mpm, gps_df)
    print(warn_info.keys())
    print(warn_info)
    generate_check_file(
        my_net,
        warn_info_dict=warn_info,
        out_fldr=case_output_dir('0614bug'),
        file_name='0614bug',
    )


MATCH_FUNCTIONS = {
    "cq": t_cq_match,
    "0327sample": t_0327sample_match,
    "0424sample": t_0424sample_match,
    "denseexample": t_denseexample_match,
    "0326taxi": t_0326taxi_match,
    "check0325": t_check0325_match,
    "bug0329": t_bug0329_match,
    "bug0516": t_bug0516_match,
    "agents150": t_agents150_match,
    "route2gps": t_route2gps_match,
    "bug_0402": t_bug_0402_match,
    "0419test": t_0419test_match,
    "0506yg": t_0506yg_match,
    "0507xishu": t_0507xishu_match,
    "carla0605": t_carla0605_match,
    "zdsy": t_zdsy_match,
    "ldso0703": t_ldsobug_match,
    "gjy0828": t_gjy0828_match,
    "QuickStart-Match-1": quick_start_match_1,
    "ldso0710": t_ldso0710_match,
    "0729hz": t_0729hz_match,
    "250415bug": t_250415bug_match,
    "bug0213": t_bug0213_match,
    "250114": t_250114_match,
    "1126": t_1126_match,
    "reverse1121": t_reverse1121_match,
    "test1104": t_test1104_match,
    "0614bug": t_0614bug_match,
}


def run_all() -> dict[str, dict[str, object]]:
    ensure_all_output_dirs()
    summary: dict[str, dict[str, object]] = {}
    for case_name, match_func in MATCH_FUNCTIONS.items():
        print(f"\n===== START {case_name}: {match_func.__name__} =====", flush=True)
        start = time.time()
        try:
            match_func()
        except Exception:
            error_text = traceback.format_exc()
            summary[case_name] = {
                "status": "failed",
                "seconds": time.time() - start,
                "error": error_text,
            }
            print(error_text, flush=True)
        else:
            summary[case_name] = {
                "status": "passed",
                "seconds": time.time() - start,
            }
            meta_path = os.path.join(case_output_dir(case_name), "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as meta_file:
                    meta = json.load(meta_file)
                if "seconds" in meta:
                    summary[case_name]["execute_seconds"] = meta["seconds"]
        print(
            f"===== END {case_name}: {summary[case_name]['status']} "
            f"({summary[case_name]['seconds']:.3f}s) =====",
            flush=True,
        )
        if "execute_seconds" in summary[case_name]:
            print(
                f"===== execute_seconds={summary[case_name]['execute_seconds']:.3f}s =====",
                flush=True,
            )

    summary_path = os.path.join(OUTPUT_ROOT, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    run_result = run_all()
    failed_cases = [
        case_name
        for case_name, case_result in run_result.items()
        if case_result["status"] == "failed"
    ]
    if failed_cases:
        print(f"\nFAILED CASES: {', '.join(failed_cases)}", flush=True)
        raise SystemExit(1)
