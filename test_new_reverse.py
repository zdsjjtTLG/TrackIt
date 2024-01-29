# -- coding: utf-8 --
# @Time    : 2024/1/27 21:32
# @Author  : TangKai
# @Team    : ZheChengData

import geopandas as gpd
import pandas as pd

import src.gotrackit.netreverse.NetGen as ng

if __name__ == '__main__':
    ####  一、直接指定矩形区域, 基于矩形区域随机构造OD, 获取路网  ####
    # 必须要指定:
    # 1. 左下角点的经纬度坐标, 矩形区域的宽度和高度
    # 3. key_list
    # 4. 输出路网的存储文件目录net_out_fldr
    # 5. 请求的路径源文件的存储文件目录binary_path_fldr,
    # 6. 平面投影坐标系 plain_crs
    # 7. od_type='rand_od'
    # nv = ng.NetReverse(flag_name='test_rectangle', net_out_fldr=r'./data/output/reverse/test_rectangle/',
    #                    plain_prj='EPSG:32650', save_tpr_link=True, angle_threshold=40)
    #
    # nv.generate_net_from_request(key_list=['02a0764c920b2b248ca71e29bf673db9'],
    #                              log_fldr=r'./', save_log_file=False,
    #                              binary_path_fldr=r'./data/output/request/test_rectangle/',
    #                              w=2000, h=2000, min_lng=126.665019, min_lat=45.747539, od_type='rand_od',
    #                              od_num=1, gap_n=1000, min_od_length=800, ignore_hh=True)

    ####  二、自定义面域文件, 基于自定义区域随机构造OD, 获取路网  ####
    # 必须要指定:
    # 1. 自定义的区域文件
    # 3. key_list
    # 4. 输出路网的存储文件目录net_out_fldr
    # 5. 请求的路径源文件的存储文件目录binary_path_fldr,
    # 6. 平面投影坐标系 plain_crs
    # 7. od_type='rnd'
    nv = ng.NetReverse(flag_name='test_diy_region', net_out_fldr=r'./data/output/reverse/test_diy_region/',
                       plain_prj='EPSG:32650', save_tpr_link=True, angle_threshold=20)
    target_region_gdf = gpd.read_file(r'./data/input/region/diy_region.shp')
    print(target_region_gdf)
    nv.generate_net_from_request(key_list=['02a0764c920b2b248ca71e29bf673db9'],
                                 log_fldr=r'./', save_log_file=True,
                                 binary_path_fldr=r'./data/output/request/test_diy_region/',
                                 region_gdf=target_region_gdf, od_type='rand_od', boundary_buffer=1000, gap_n=1000,
                                 min_od_length=1200, od_num=20)
    #

    ####  三、自定义面域文件, 基于多个自定义区域构造区域-区域的OD, 获取路网  ####
    # 必须要指定:
    # 1. 自定义的区域文件
    # 3. key_list
    # 4. 输出路网的存储文件目录net_out_fldr
    # 5. 请求的路径源文件的存储文件目录binary_path_fldr,
    # 6. 平面投影坐标系 plain_crs
    # 7. od_type='region'
    nv = ng.NetReverse(flag_name='test_taz', net_out_fldr=r'./data/output/reverse/test_taz/',
                       plain_prj='EPSG:32650', save_tpr_link=True, angle_threshold=20)
    target_region_gdf = gpd.read_file(r'./data/input/region/simple_taz.shp')
    print(target_region_gdf)
    nv.generate_net_from_request(key_list=['02a0764c920b2b248ca71e29bf673db9'],
                                 log_fldr=r'./', save_log_file=True,
                                 binary_path_fldr=r'./data/output/request/test_taz/',
                                 region_gdf=target_region_gdf, od_type='region_od')

    ####  四、基于路径源文件, 获取路网  ####
    # 必须要指定:
    # 1. 路径源文件目录
    # 2. 路径源文件的名称列表
    # 3. key_list
    # 4. 输出路网的存储目录
    # 5. plain_crs
    nv = ng.NetReverse(flag_name='test_pickle', net_out_fldr=r'./data/output/reverse/test_pickle/',
                       plain_prj='EPSG:32650', save_tpr_link=True, angle_threshold=20)
    nv.generate_net_from_pickle(binary_path_fldr=r'./data/output/request/test_taz/',
                                pickle_file_name_list=['14_test_taz_gd_path_1'])

    ####  五、基于已有OD, 进行路径请求获取路网, 获取路网  ####
    # 必须要指定:
    # 1. 路径源文件目录
    # 2. 路径源文件的名称列表
    # 3. key_list
    # 4. 输出路网的存储目录
    # 5. plain_crs
    nv = ng.NetReverse(flag_name='test_diy_od', net_out_fldr=r'./data/output/reverse/test_diy_od/',
                       plain_prj='EPSG:32651', save_tpr_link=True, angle_threshold=20)
    nv.generate_net_from_request(binary_path_fldr=r'./data/output/request/test_diy_od/',
                                 key_list=['02a0764c920b2b248ca71e29bf673db9'],
                                 od_file_path=r'./data/output/od/苏州市.csv', od_type='diy_od')

    # 或者
    # diy_od_df = pd.read_csv(r'./data/output/od/苏州市.csv')
    # nv.generate_net_from_request(binary_path_fldr=r'./data/output/request/test_diy_od/',
    #                              key_list=['02a0764c920b2b248ca71e29bf673db9'],
    #                              od_df=diy_od_df,
    #                              od_type='diy_od')


    ####  六、生产点层  ####
    # 必须要指定:
    # 1. 路径源文件目录
    # 2. 路径源文件的名称列表
    # 3. key_list
    # 4. 输出路网的存储目录
    # 5. plain_crs
    nv = ng.NetReverse()
    link_gdf = gpd.read_file(r'./data/output/create_node/link.shp')
    print(link_gdf)
    link_gdf.drop(columns=['link_id', 'from_node', 'to_node', 'length'], axis=1, inplace=True)
    new_link_gdf, new_node_gdf, node_group_status_gdf = nv.create_node_from_link(link_gdf=link_gdf, using_from_to=False,
                                                                                 update_link_field_list=['link_id',
                                                                                                         'from_node',
                                                                                                         'to_node',
                                                                                                         'length'],
                                                                                 plain_prj='EPSG:32651',
                                                                                 modify_minimum_buffer=0.7,
                                                                                 execute_modify=True,
                                                                                 ignore_merge_rule=True,
                                                                                 out_fldr=r'./data/output/create_node/')
