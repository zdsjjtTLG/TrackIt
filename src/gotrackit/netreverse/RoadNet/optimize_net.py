# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData

"""拓扑优化, 先合并2度节点, 再处理重复link"""

import os
import geopandas as gpd
from ..book_mark import generate_book_mark
from .DupProcess.DupLinks import process_dup_link
from .Merge.merge_links import merge_two_degrees_node
from .MultiCoreMerge.merge_links_multi import merge_links_multi


def optimize(link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None, ignore_dir: bool = False,
             allow_ring: bool = False, limit_col_name: str = None, plain_prj: str = 'EPSG:32650',
             accu_l_threshold: float = 500.0, angle_threshold: float = 15.0, restrict_length: bool = True,
             restrict_angle: bool = True, save_preliminary: bool = True, out_fldr: str = None,
             is_process_dup_link: bool = True, process_dup_link_buffer: float = 0.75, min_length: float = 50.0,
             dup_link_buffer_ratio: float = 60.0, modify_minimum_buffer: float = 0.8, multi_core: bool = False,
             core_num: int = 3) -> \
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    """crs input: EPSG:4326
    拓扑优化, 先合并2度节点, 再处理重复link
    :param link_gdf: 路网线层
    :param node_gdf: 路网点层
    :param ignore_dir: 是否忽略行车方向进行合并
    :param limit_col_name: 限制字段名称
    :param allow_ring: 是否运行合并后出现环
    :param plain_prj: 平面投影坐标系
    :param accu_l_threshold: 合并后的link的长度的最大值(m)
    :param angle_threshold: 允许的最大转角(如果两个相邻的可合并link的转角超过这个角度则不允许合并)
    :param restrict_length: 是否启用合并后的长度限制
    :param restrict_angle:  是否启用转角限制
    :param save_preliminary: 是否保存2度节点合并后的路网
    :param out_fldr: 输出文件目录
    :param is_process_dup_link: 是否处理重复link
    :param process_dup_link_buffer: 处理重复link时的buffer(m)
    :param min_length: 路段最小长度
    :param dup_link_buffer_ratio: LinkBuffer重叠率阈值, 推荐60
    :param modify_minimum_buffer
    :param multi_core:
    :param core_num
    :return:
    """
    link_gdf.reset_index(inplace=True, drop=True)
    node_gdf.reset_index(inplace=True, drop=True)
    # 1.按照规则合并2度节点
    if multi_core:
        new_link, new_node, merge_info_dict = merge_links_multi(link_gdf=link_gdf, limit_col_name=limit_col_name,
                                                                ignore_dir=ignore_dir,
                                                                allow_ring=allow_ring,
                                                                node_gdf=node_gdf,
                                                                plain_prj=plain_prj, accu_l_threshold=accu_l_threshold,
                                                                angle_threshold=angle_threshold,
                                                                restrict_length=restrict_length,
                                                                restrict_angle=restrict_angle,
                                                                min_length=min_length, core_num=core_num)
    else:
        new_link, new_node, merge_info_dict = merge_two_degrees_node(link_gdf=link_gdf, limit_col_name=limit_col_name,
                                                                     ignore_dir=ignore_dir,
                                                                     allow_ring=allow_ring,
                                                                     node_gdf=node_gdf,
                                                                     plain_prj=plain_prj,
                                                                     accu_l_threshold=accu_l_threshold,
                                                                     angle_threshold=angle_threshold,
                                                                     restrict_length=restrict_length,
                                                                     restrict_angle=restrict_angle,
                                                                     min_length=min_length)
    if save_preliminary:
        generate_book_mark(input_fldr=out_fldr, name_loc_dict=merge_info_dict, prj_name=out_fldr.split('/')[-1])
        new_link.to_file(os.path.join(out_fldr, 'AfterTPMergeLink.shp'), encoding='gbk')
        new_node.to_file(os.path.join(out_fldr, 'AfterTPMergeNode.shp'), encoding='gbk')

    # 是否处理重复link
    if is_process_dup_link:
        origin_crs = new_link.crs
        new_link = new_link.to_crs(plain_prj)
        new_node = new_node.to_crs(plain_prj)
        print(r'##########   Remove Overlapping Road Segments')
        final_link, final_node, dup_info_dict = process_dup_link(link_gdf=new_link, node_gdf=new_node,
                                                                 buffer=process_dup_link_buffer,
                                                                 dup_link_buffer_ratio=dup_link_buffer_ratio,
                                                                 modify_minimum_buffer=modify_minimum_buffer)

        final_link = final_link.to_crs(origin_crs)
        final_node = final_node.to_crs(origin_crs)
        return final_link, final_node, dup_info_dict
    else:
        return new_link, new_node, dict()





