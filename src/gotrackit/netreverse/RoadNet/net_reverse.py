# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData

"""
依据路径规划请求结果逆向路网
"""

from .conn import Conn
import geopandas as gpd
from ...map.Net import Net
from .save_file import save_file
from ..GlobalVal import NetField
from ..RoadNet import optimize_net
from ..book_mark import generate_book_mark
from ..RoadNet.SaveStreets.streets import generate_node_from_link
from ..RoadNet.Tools.process import merge_double_link, convert_neg_to_pos

net_field = NetField()


length_field = net_field.LENGTH_FIELD
node_id_field = net_field.NODE_ID_FIELD
link_id_field = net_field.LINK_ID_FIELD
geometry_field = net_field.GEOMETRY_FIELD
to_node_id_field = net_field.TO_NODE_FIELD
direction_field = net_field.DIRECTION_FIELD
from_node_id_field = net_field.FROM_NODE_FIELD


def generate_net(path_gdf: gpd.GeoDataFrame = None, out_fldr: str = None,
                 save_split_link: bool = False, plain_prj: str = 'EPSG:32650', save_tpr_link: bool = False,
                 save_streets_before_modify_minimum: bool = True, restrict_angle: bool = True,
                 limit_col_name: str = 'road_name',
                 restrict_length: bool = True, accu_l_threshold: float = 150.0, angle_threshold: float = 15,
                 multi_core_merge: bool = False, core_num: int = 3,
                 modify_minimum_buffer: float = 0.8, flag_name: str = None,
                 save_streets_after_modify_minimum: bool = True, save_preliminary: bool = False,
                 save_done_topo: bool = False,
                 is_process_dup_link: bool = True, process_dup_link_buffer: float = 0.8, min_length: float = 50.0,
                 dup_link_buffer_ratio: float = 60.0, net_file_type: str = 'shp', modify_conn: bool = True,
                 conn_buffer: float = 0.8, conn_period: str = 'final'):
    """
    路网逆向主程序, 输入拆分好且去重的path_gdf(EPSG:4326), output: EPSG:4326
    :param path_gdf:
    :param flag_name: 标志字符
    :param plain_prj: 平面投影坐标系
    :param modify_minimum_buffer: 消除极小间隔点的buffer大小(m), 推荐, 0.1 ~ 0.7
    :param save_split_link: 是否保存路段拆分后的link
    :param save_streets_before_modify_minimum: 是否保存做完拓扑关联但是还没做极小间隔点优化的路网
    :param save_streets_after_modify_minimum: 是否保存做完拓扑关联和极小间隔点优化的路网
    :param save_tpr_link: 是否保存拓扑关联、极小间隔点优化、双向路段合并后的link
    :param save_preliminary: 是否保存合并2度节点后的link
    :param accu_l_threshold: 合并后的link的长度的最大值(m)
    :param angle_threshold: 允许的最大转角(如果两个相邻的可合并link的转角超过这个角度则不允许合并)
    :param multi_core_merge:
    :param core_num:
    :param min_length: 允许的最小link长度
    :param limit_col_name: 属性限制字段
    :param restrict_length: 是否启用合并后的长度限制
    :param restrict_angle:  是否启用转角限制
    :param save_preliminary: 是否保存2度节点合并后的路网
    :param save_done_topo: 是否保存拓扑优化之后的路网
    :param is_process_dup_link: 是否处理重叠link
    :param process_dup_link_buffer: 处理重叠link时的buffer取值
    :param dup_link_buffer_ratio:
    :param out_fldr: 输出路网的存储目录
    :param net_file_type: shp or geojson
    :param modify_conn: 是否修复联通性
    :param conn_buffer: 修复联通性的buffer
    :param conn_period: 在哪个阶段执行联通性修复? 取值 final or start, final表示在拓扑优化之后执行, start表示在拓扑优化之前执行
    :return:
    """
    if save_tpr_link or save_split_link or save_streets_before_modify_minimum or save_streets_after_modify_minimum:
        assert out_fldr is not None

    if modify_minimum_buffer <= conn_buffer:
        modify_minimum_buffer = conn_buffer + 0.1

    path_gdf_after_split = path_gdf

    if save_split_link:
        save_file(data_item=path_gdf_after_split, out_fldr=out_fldr, file_name='path_split', file_type=net_file_type)

    # 2.生成拓扑关联
    print(rf'##########   {flag_name} - Generate Topological Associations')
    new_link, new_node, node_group_status_gdf = \
        generate_node_from_link(link_gdf=path_gdf_after_split,
                                update_link_field_list=[link_id_field, length_field,
                                                        from_node_id_field, to_node_id_field, direction_field],
                                using_from_to=False,
                                ignore_merge_rule=True,
                                plain_prj=plain_prj,
                                fill_dir=1,
                                modify_minimum_buffer=modify_minimum_buffer,
                                out_fldr=out_fldr,
                                net_file_type=net_file_type,
                                save_streets_before_modify_minimum=save_streets_before_modify_minimum,
                                save_streets_after_modify_minimum=save_streets_after_modify_minimum)
    del path_gdf_after_split

    # 3. 消除dir为-1的link, 只修改线层
    link_gdf = convert_neg_to_pos(link_gdf=new_link)

    del new_link

    # 4. 目前dir都是1, 找出其中的双向道路, dir改为0, 只修改线层
    link_gdf = merge_double_link(link_gdf=link_gdf)

    if save_tpr_link:
        save_file(data_item=link_gdf, out_fldr=out_fldr, file_name='TprLink', file_type=net_file_type)
        save_file(data_item=new_node, out_fldr=out_fldr, file_name='TprNode', file_type=net_file_type)

    if modify_conn:
        if conn_period == 'start':
            print(rf'##########   {flag_name} - Repair Road Network Connectivity Before Topology Optimization')
            net = Net(link_gdf=link_gdf, node_gdf=new_node, create_single=False)
            conn = Conn(net=net, check_buffer=conn_buffer)
            # print(net.geo_crs, net.planar_crs)
            link_gdf, new_node = conn.execute(out_fldr=out_fldr, file_name=flag_name + '_conn', generate_mark=True)
            # net.export_net(export_crs='EPSG:4326', out_fldr=out_fldr, file_type='shp',
            #                flag_name='modifiedConn')
    # 5.拓扑优化
    print(rf'##########   {flag_name} - Topology Optimization')
    final_link, final_node, dup_info_dict = optimize_net.optimize(link_gdf=link_gdf, node_gdf=new_node,
                                                                  ignore_dir=False, limit_col_name=limit_col_name,
                                                                  save_preliminary=save_preliminary,
                                                                  out_fldr=out_fldr,
                                                                  plain_prj=plain_prj,
                                                                  restrict_angle=restrict_angle,
                                                                  restrict_length=restrict_length,
                                                                  accu_l_threshold=accu_l_threshold,
                                                                  angle_threshold=angle_threshold,
                                                                  process_dup_link_buffer=process_dup_link_buffer,
                                                                  is_process_dup_link=is_process_dup_link,
                                                                  allow_ring=False,
                                                                  modify_minimum_buffer=modify_minimum_buffer,
                                                                  min_length=min_length,
                                                                  dup_link_buffer_ratio=dup_link_buffer_ratio,
                                                                  multi_core=multi_core_merge, core_num=core_num)

    if save_done_topo:
        save_file(data_item=final_link, out_fldr=out_fldr, file_name='DoneTopoLink', file_type=net_file_type)
        save_file(data_item=final_node, out_fldr=out_fldr, file_name='DoneTopoNode', file_type=net_file_type)

    if modify_conn:
        if conn_period == 'final':
            print(rf'##########   {flag_name} - Repair Road Network Connectivity After Topology Optimization')
            net = Net(link_gdf=final_link, node_gdf=final_node, create_single=False)
            # print(net.geo_crs, net.planar_crs)
            conn = Conn(net=net, check_buffer=conn_buffer)
            final_link, final_node = conn.execute(out_fldr=out_fldr, file_name=flag_name + '_conn', generate_mark=True)
    try:
        final_link.loc[final_link['road_name'] == '路口转向', 'dir'] = 0
    except Exception as e:
        pass
    save_file(data_item=final_link, out_fldr=out_fldr, file_name='FinalLink', file_type=net_file_type)
    save_file(data_item=final_node, out_fldr=out_fldr, file_name='FinalNode', file_type=net_file_type)
    generate_book_mark(name_loc_dict=dup_info_dict, prj_name=flag_name, input_fldr=out_fldr)
    return final_link, final_node
