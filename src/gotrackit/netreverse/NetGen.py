# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData


"""生产/优化 路网的类和方法"""

import os.path
import pandas as pd
import multiprocessing
import geopandas as gpd
from ..map.Net import Net
from .RoadNet.conn import Conn
from .format_od import FormatOD
from .RoadNet import net_reverse
from .RoadNet.increment import increment
from .GlobalVal import NetField, GpsField
from .RoadNet.save_file import save_file
from .Request.request_path import CarPath
from .RoadNet.optimize_net import optimize
from .Parse.gd_car_path import ParseGdPath
from ..tools.common import avoid_duplicate_cols
from .RoadNet.Split.SplitPath import split_path
from .PublicTools.GeoProcess import generate_region
from .RoadNet.Split.SplitPath import split_path_main
from .RoadNet.Tools.process import merge_double_link, create_single_link
from .RoadNet.SaveStreets.streets import generate_node_from_link, modify_minimum
from ..tools.geo_process import clean_link_geo, remapping_id, rn_partition, rn_partition_alpha
from .RoadNet.SaveStreets.streets import drop_no_use_nodes

net_field = NetField()
gps_field = GpsField()

class Reverse(object):
    def __init__(self, flag_name: str = 'NetGen', plain_crs: str = None, net_out_fldr: str = r'./',
                 net_file_type: str = 'shp'):
        """

        Args:
            flag_name:
            plain_crs:
            net_out_fldr:
            net_file_type:
        """
        # overall
        self.flag_name = flag_name
        self.plain_crs = plain_crs
        self.net_out_fldr = net_out_fldr
        assert net_file_type in ['shp', 'geojson']
        self.net_file_type = net_file_type


class NetReverse(Reverse):
    def __init__(self, flag_name: str = 'NetGen', plain_crs: str = 'EPSG:32650', ignore_head_tail: bool = False,
                 cut_slice: bool = False, slice_num: int = 5, generate_rod: bool = False, min_rod_length: float = 5.0,
                 restrict_region_gdf: gpd.GeoDataFrame = None, save_split_link: bool = False,
                 modify_minimum_buffer: float = 0.8, save_streets_before_modify_minimum: bool = False,
                 save_streets_after_modify_minimum: bool = False, save_tpr_link: bool = False,
                 limit_col_name: str = 'road_name', ignore_dir: bool = False,
                 allow_ring: bool = False, restrict_angle: bool = True, restrict_length: bool = True,
                 accu_l_threshold: float = 200.0, angle_threshold: float = 35.0, min_length: float = 50.0,
                 multi_core_merge: bool = False, merge_core_num: int = 2,
                 save_preliminary: bool = False, save_done_topo: bool = False,
                 is_process_dup_link: bool = True, process_dup_link_buffer: float = 0.8,
                 dup_link_buffer_ratio: float = 60.0, net_out_fldr: str = r'./', net_file_type: str = 'shp',
                 is_modify_conn: bool = True, conn_buffer: float = 0.8, conn_period: str = 'final',
                 multi_core_parse: bool = False, parse_core_num: int = 2,
                 multi_core_reverse: bool = False, reverse_core_num: int = 2):
        """路网逆向NetReverse类：

         - 初始化

        Args:
            flag_name: [1]总体参数 - 项目名称
            net_out_fldr: [1]总体参数 - 输出路网的存储目录
            plain_crs: [1]总体参数 - 平面投影坐标系
            net_file_type: [1]总体参数 - 路网的输出文件类型，shp 或者 geojson
            multi_core_parse: [2]路径解析、拆分参数 - 是否启用多核对二进制路径文件进行解析
            parse_core_num: [2]路径解析、拆分参数 - 多核解析时使用的核数
            ignore_head_tail: [2]路径解析、拆分参数 - 是否忽略路径首尾的无名道路, 这种一般是小区内部道路
            cut_slice: [2]路径解析、拆分参数 - 拆分路段时，是否分片处理，内存不够时可以指定为True
            slice_num: [2]路径解析、拆分参数 - 拆分路段时，拆分为几个slice处理
            generate_rod: [2]路径解析、拆分参数 - 是否生成连杆, 只有当参数ignore_head_tail为false时该参数才生效
            min_rod_length: [2]路径解析、拆分参数 - 最小连杆长度，米
            restrict_region_gdf: [2]路径解析、拆分参数 - 限制区域gdf，若传入此区域，那么仅仅只对在区域范围内的路网进行逆向
            save_split_link: [2]路径解析、拆分参数 - 是否保存拆分路径后的link层文件
            modify_minimum_buffer: [3]拓扑生成参数 - 是极小间隔节点优化的buffer, m
            save_streets_before_modify_minimum: [3]拓扑生成参数 - 是否保存优化前的结果
            save_streets_after_modify_minimum: [3]拓扑生成参数 - 是否保存优化后的结果
            save_tpr_link: [3]拓扑生成参数 - 是否保存优化后且进行方向处理的文件
            limit_col_name: [4]拓扑优化参数 - 路段合并时，用于限制路段合并的线层属性字段，默认road_name，如果你要使用其他字段来限制合并，请自定义该参数
            ignore_dir: [4]拓扑优化参数 - 路段合并时，是否忽略行车方向
            allow_ring: [4]拓扑优化参数 - 是否允许路段合并后出现环
            restrict_angle: [4]拓扑优化参数 - 是否启用最大转角限制来约束路段合并
            restrict_length: [4]拓扑优化参数 - 是否启用最大路段长度限制来约束路段合并
            accu_l_threshold: [4]拓扑优化参数 - 允许的最长的路段长度，米
            angle_threshold: [4]拓扑优化参数 - 允许的最大的路段内转角，度
            min_length: [4]拓扑优化参数 - 允许的最小的路段长度，米
            save_preliminary: [4]拓扑优化参数 - 是否保留重复路段处理前的文件
            multi_core_merge: [4]拓扑优化参数 - 是否启用多进程进行路段合并
            merge_core_num: [4]拓扑优化参数 - 启用几个核进行路段合并
            save_done_topo: [4]拓扑优化参数 - 是否保存拓扑优化后的路网
            is_process_dup_link: [5]重叠路段处理参数 - 是否处理重复路
            process_dup_link_buffer: [5]重叠路段处理参数 - 处理重复路段所使用的buffer长度，米
            dup_link_buffer_ratio: [5]重叠路段处理参数 - dup_link_buffer_ratio
            is_modify_conn: [6]联通性修复参数 - 是否检查潜在的联通性问题并且进行修复
            conn_buffer: [6]联通性修复参数 - 检查联通性问题时使用的检测半径大小,单位米
            conn_period: [6]联通性修复参数 - 取值final或者start, final表示在拓扑优化之后修复联通性, start表示在拓扑优化之前修复联通性
            multi_core_reverse: [7]分区逆向参数 - 是否启用多进程对路网进行并行逆向计算
            reverse_core_num: [7]分区逆向参数 - 逆向并行计算要启用的核数
        """
        # overall
        super().__init__(flag_name, plain_crs, net_out_fldr, net_file_type)

        # split
        self.ignore_head_tail = ignore_head_tail
        self.cut_slice = cut_slice
        self.slice_num = slice_num
        self.generate_rod = generate_rod
        self.min_rod_length = min_rod_length
        self.save_split_link = save_split_link
        self.restrict_region_gdf = restrict_region_gdf

        # create node from link
        self.modify_minimum_buffer = \
            conn_buffer + 0.1 if modify_minimum_buffer <= conn_buffer else modify_minimum_buffer
        self.save_streets_before_modify_minimum = save_streets_before_modify_minimum
        self.save_streets_after_modify_minimum = save_streets_after_modify_minimum
        self.save_tpr_link = save_tpr_link

        # merge
        self.limit_col_name = limit_col_name
        self.ignore_dir = ignore_dir
        self.allow_ring = allow_ring
        self.restrict_angle = restrict_angle
        self.restrict_length = restrict_length
        self.accu_l_threshold = accu_l_threshold
        self.angle_threshold = angle_threshold
        self.min_length = min_length
        self.save_preliminary = save_preliminary
        self.save_done_topo = save_done_topo
        self.multi_core_merge = multi_core_merge
        self.merge_core_num = merge_core_num

        # process dup
        self.is_process_dup_link = is_process_dup_link
        self.process_dup_link_buffer = process_dup_link_buffer
        self.dup_link_buffer_ratio = dup_link_buffer_ratio

        # conn
        self.is_modify_conn = is_modify_conn
        self.conn_buffer = conn_buffer
        assert conn_period in ['start', 'final']
        self.conn_period = conn_period

        # attrs
        self.__od_df = pd.DataFrame()
        self.__region_gdf = gpd.GeoDataFrame()

        # if uses multi core
        self.multi_core_parse = multi_core_parse
        self.parse_core_num = parse_core_num

        # if uses multi core
        self.multi_core_reverse = multi_core_reverse
        self.reverse_core_num = reverse_core_num

    def generate_net_from_request(self, key_list: list[str], traffic_mode: str = 'car', binary_path_fldr: str = r'./',
                                  od_file_path: str = None, od_df: pd.DataFrame = None,
                                  region_gdf: gpd.GeoDataFrame = None, od_type='rand_od', boundary_buffer: float = 2000,
                                  cache_times: int = 300, ignore_hh: bool = True, remove_his: bool = True,
                                  log_fldr: str = None, save_log_file: bool = False,
                                  min_lng: float = None, min_lat: float = None, w: float = 2000, h: float = 2000,
                                  od_num: int = 100, gap_n: int = 1000, min_od_length: float = 1200.0,
                                  wait_until_recovery: bool = False,
                                  is_rnd_strategy: bool = False, strategy: str = '32') -> None:
        """NetReverse类方法 - generate_net_from_request：

         - 向开放平台请求路径后分析计算得到路网：构造OD -> 请求路径 -> 二进制存储 -> 路网生产

        Args:
            key_list: [1]请求设置参数 - 开发者key值列表，必需参数
            binary_path_fldr: [1]请求设置参数 - 存储请求路径源文件的目录
            traffic_mode: (v0.3.19即将支持) [1]请求设置参数 - 交通模式, 目前支持驾车(car)、骑行(bike)和步行(walk)
            wait_until_recovery: [1]请求设置参数 - 如果配额超限，是否一直等待直至配额恢复
            is_rnd_strategy: [1]请求设置参数 - 是否启用随机策略
            strategy: [1]请求设置参数 - 路径规划策略参数，若模式为驾车，取值请访问: https://lbs.amap.com/api/webservice/guide/api/newroute#s1；若模式为步行或骑行, 则取值为1、2、3，代表返回的方案数
            cache_times: [1]请求设置参数 - 路径文件缓存数，即每请求cache_times次缓存一次数据到binary_path_fldr下
            ignore_hh: [1]请求设置参数 - 是否忽略时段限制进行请求
            remove_his: [1]请求设置参数 - 是否对已经请求的OD重复(指的是在请求被意外中断的情况下，od_id为判断依据)请求
            save_log_file: [1]请求设置参数 - 是否保存日志文件
            log_fldr: [1]请求设置参数 - 日志文件的存储目录
            od_file_path: [2]OD构造参数 - 用于请求的od文件路径，可选参数
            od_df: [2]OD构造参数 - 用于请求的od数据，该参数和od_file_path任意指定一个即可，可选参数
            region_gdf: [2]OD构造参数 - 用于构造od的面域数据
            od_type: [2]OD构造参数 - 用于构造od的方法，rand_od、region_od、diy_od
            min_lng: [2]OD构造参数 - 矩形区域的左下角经度
            min_lat: [2]OD构造参数 - 矩形区域的左下角纬度
            w: [2]OD构造参数 - 矩形区域的宽度，米
            h: [2]OD构造参数 - 矩形区域的高度，米
            od_num: [2]OD构造参数 - 请求的od数，od数越多，请求的路径就越多，路网覆盖率就越完整，只有od_type为rand_od时起效
            gap_n: [2]OD构造参数 - 横纵向网格个数，只有od_type为rand_od时起效
            min_od_length: [2]OD构造参数 - od之间最短直线距离，只有od_type为rand_od时起效
            boundary_buffer: [2]OD构造参数 - 区域边界buffer，米，

        Returns:
            直接在net_out_fldr下生成路网
        """
        self.request_path(key_list=key_list, traffic_mode=traffic_mode, binary_path_fldr=binary_path_fldr,
                          od_file_path=od_file_path,
                          od_df=od_df, region_gdf=region_gdf, od_type=od_type,
                          boundary_buffer=boundary_buffer,
                          cache_times=cache_times, ignore_hh=ignore_hh,
                          remove_his=remove_his,
                          log_fldr=log_fldr, save_log_file=save_log_file,
                          min_lng=min_lng, min_lat=min_lat,
                          w=w, h=h, od_num=od_num, gap_n=gap_n,
                          min_od_length=min_od_length, wait_until_recovery=wait_until_recovery,
                          is_rnd_strategy=is_rnd_strategy, strategy=strategy)
        pickle_file_name_list = os.listdir(binary_path_fldr)
        self.generate_net_from_pickle(binary_path_fldr=binary_path_fldr,
                                      pickle_file_name_list=pickle_file_name_list)

    def generate_net_from_pickle(self, binary_path_fldr: str, pickle_file_name_list: list[str] = None) -> None:
        """NetReverse类方法 - generate_net_from_pickle：

        - 解析二进制路径文件, 然后生产路网
        
        Args:
            binary_path_fldr: 路径源文件的存储目录，必须参数
            pickle_file_name_list: 需要解析的路径文件名称列表，如果不传入则默认解析所有文件

        Returns:
            None, 直接在net_out_fldr下生成路网
        """
        attr_name_list = ['road_name']
        if pickle_file_name_list is None or not pickle_file_name_list:
            pickle_file_name_list = os.listdir(binary_path_fldr)
        pgd = ParseGdPath(binary_path_fldr=binary_path_fldr,
                          pickle_file_name_list=pickle_file_name_list,
                          flag_name=self.flag_name,
                          is_slice=self.cut_slice,
                          slice_num=self.slice_num,
                          restrict_region_gdf=self.restrict_region_gdf,
                          attr_name_list=attr_name_list,
                          ignore_head_tail=self.ignore_head_tail,
                          check=False, generate_rod=self.generate_rod,
                          min_rod_length=self.min_rod_length,
                          is_multi_core=self.multi_core_parse,
                          used_core_num=self.parse_core_num)

        split_path_gdf = pgd.parse_path_main_multi()
        self._generate_net_from_split_path(split_path_gdf=split_path_gdf)

    def generate_net_from_path_gdf(self, path_gdf: gpd.GeoDataFrame,
                                   slice_num: int = 1, attr_name_list: list = None,
                                   cut_slice: bool = False):
        """NetReverse类方法 - generate_net_from_path_gdf：

        - 从线层文件计算得到路网

        Args:
            path_gdf: 线层gdf数据，必须参数, 坐标系必须为EPSG:4326
            slice_num: 拆分路段时，拆分为几个slice处理
            attr_name_list: 限定字段列表
            cut_slice: 拆分路段时，是否分片处理，内存不够时可以指定为True

        Returns:
            直接在net_out_fldr下生成路网
        """
        print(rf'##########   {self.flag_name} - Split Path')
        if 'road_name' not in path_gdf.columns:
            path_gdf['road_name'] = ''
        attr_name_list = ['road_name'] if attr_name_list is None or len(attr_name_list) == 1 else attr_name_list
        split_path_gdf = split_path_main(path_gdf=path_gdf, restrict_region_gdf=self.restrict_region_gdf,
                                         slice_num=slice_num, attr_name_list=attr_name_list,
                                         cut_slice=cut_slice, drop_ft_loc=True)

        self._generate_net_from_split_path(split_path_gdf=split_path_gdf)

    @staticmethod
    def create_node_from_link(link_gdf: gpd.GeoDataFrame, update_link_field_list: list[str] = None,
                              using_from_to: bool = False, fill_dir: int = 0, plain_crs: str = 'EPSG:32650',
                              ignore_merge_rule: bool = True, modify_minimum_buffer: float = 0.8,
                              execute_modify: bool = True, auxiliary_judge_field: str = None,
                              out_fldr: str | None = r'./', save_streets_before_modify_minimum: bool = False,
                              save_streets_after_modify_minimum: bool = True, net_file_type: str = 'shp') -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """NetReverse类静态方法 - create_node_from_link：

        - 生产点线拓扑关联, 更新线层信息, 同时生产点层

        Args:
            link_gdf: 路网线层gdf数据，必需数据
            out_fldr: 输出文件的存储目录
            update_link_field_list: 需要更新的字段列表, 生产拓扑关联后需要更新的线层基本字段，从(link_id, from_node, to_node, dir, length)中选取
            using_from_to: 是否使用输入线层中的from_node字段和to_node字段
            fill_dir: 用于填充dir方向字段的值，如果update_link_field_list中包含dir字段，那么该参数需要传入值，允许的值为1或者0
            plain_crs: 所使用的平面投影坐标系
            ignore_merge_rule: 是否忽略极小间隔优化的规则
            execute_modify: 是否执行极小间隔节点优化
            modify_minimum_buffer: 极小间隔节点优化的buffer, 米
            auxiliary_judge_field: 用于判断是否可以合并的线层字段, 只有当ignore_merge_rule为False才起效
            save_streets_before_modify_minimum: 是否存储极小间隔优化前的数据
            save_streets_after_modify_minimum: 是否存储极小间隔优化后的数据
            net_file_type: 输出路网文件的存储类型，shp或者geojson

        Returns:
            线层gdf, 点层gdf
        """
        assert net_file_type in ['shp', 'geojson']
        assert '4326' not in plain_crs, \
            'mistakenly specifying planar_crs(plain_crs) as a geographic coordinate system(EPSG:4326)'
        link_gdf, node_gdf, node_group_status_gdf = \
            generate_node_from_link(link_gdf=link_gdf,
                                    update_link_field_list=update_link_field_list,
                                    using_from_to=using_from_to,
                                    fill_dir=fill_dir, plain_prj=plain_crs,
                                    ignore_merge_rule=ignore_merge_rule,
                                    modify_minimum_buffer=modify_minimum_buffer,
                                    execute_modify=execute_modify,
                                    auxiliary_judge_field=auxiliary_judge_field,
                                    out_fldr=out_fldr,
                                    net_file_type=net_file_type,
                                    save_streets_after_modify_minimum=save_streets_after_modify_minimum,
                                    save_streets_before_modify_minimum=save_streets_before_modify_minimum)
        return link_gdf, node_gdf, node_group_status_gdf

    def topology_optimization(self, link_gdf: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
        """NetReverse类方法 - topology_optimization：

        - 拓扑优化：对标准路网进行路段合并、重复路段消除

        Args:
            link_gdf: 线层gdf, 坐标系必须为EPSG:4326
            node_gdf: 点层gdf, 坐标系必须为EPSG:4326

        Returns:
            线层gdf, 点层gdf, 修复点位空间信息
        """
        add_built_in = False
        if self.limit_col_name not in link_gdf.columns:
            if self.restrict_length or self.restrict_angle:
                avoid_duplicate_cols(built_in_col_list=['x'], df=link_gdf)
                link_gdf['x'] = ''
                self.limit_col_name = 'x'
                add_built_in = True
            else:
                self.limit_col_name = None
        link_gdf, node_gdf, dup_info_dict = optimize(link_gdf=link_gdf, node_gdf=node_gdf,
                                                     ignore_dir=self.ignore_dir,
                                                     allow_ring=self.allow_ring,
                                                     limit_col_name=self.limit_col_name,
                                                     plain_prj=self.plain_crs,
                                                     accu_l_threshold=self.accu_l_threshold,
                                                     angle_threshold=self.angle_threshold,
                                                     restrict_length=self.restrict_length,
                                                     restrict_angle=self.restrict_angle,
                                                     save_preliminary=False,
                                                     out_fldr=self.net_out_fldr,
                                                     is_process_dup_link=False,
                                                     process_dup_link_buffer=self.process_dup_link_buffer,
                                                     min_length=self.min_length,
                                                     dup_link_buffer_ratio=self.dup_link_buffer_ratio,
                                                     multi_core=self.multi_core_merge, core_num=self.merge_core_num)
        if add_built_in:
            try:
                del link_gdf['x']
            except:
                pass
        if self.net_out_fldr is not None:
            save_file(data_item=link_gdf, out_fldr=self.net_out_fldr, file_type=self.net_file_type, file_name='opt_link')
            save_file(data_item=node_gdf, out_fldr=self.net_out_fldr, file_type=self.net_file_type, file_name='opt_node')
        return link_gdf, node_gdf, dup_info_dict

    def get_od_df(self) -> pd.DataFrame:
        return self.__od_df.copy()

    def check_conn(self):
        pass

    def modify_minimum(self):
        pass

    def increment_from_pickle(self, link_gdf: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame,
                              binary_path_fldr: str = None, increment_out_fldr: str = None,
                              cover_ratio_threshold: float = 60.0, cover_angle_threshold: float = 6.5,
                              save_times: int = 200, ignore_head_tail: bool = True,
                              save_new_split_link: bool = True, pickle_file_name_list: list = None,
                              check_path: bool = True, overlap_buffer_size: float = 0.3) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """依据路径源文件进行增量修改, 输入crs:EPSG:4326, 输出crs:EPSG:4326

        Args:
            link_gdf: 路网线层
            node_gdf: 路网点层
            save_times: 请求路径文件时单个文件的缓存数目
            binary_path_fldr: 二进制路径文件目录
            pickle_file_name_list: 二进制路径文件名称列表
            increment_out_fldr: 更新后的路网输出目录
            ignore_head_tail: 是否忽略路径首尾的无名道路, 这种一般是小区内部道路
            save_new_split_link: 是否保存新路径拆分后的文件
            check_path: 是否检查新路径
            overlap_buffer_size: 用于判断路段是否重合的buffer_size
            cover_ratio_threshold: 用重合率超过cover_ratio_threshold%就认为是重合(条件较为宽松), 宁愿少加也不多加
            cover_angle_threshold: 角度小于cover_angle_threshold度认为是重合(条件较为宽松), 宁愿少加也不多加

        Returns:

        """
        # 将新的轨迹都解析好存储到字段
        pgd = ParseGdPath(binary_path_fldr=binary_path_fldr,
                          check_fldr=increment_out_fldr,
                          pickle_file_name_list=pickle_file_name_list,
                          ignore_head_tail=ignore_head_tail,
                          check=check_path, generate_rod=False)
        new_path_gdf_dict = pgd.parse_path_main(out_type='dict', pickle_file_name_list=pickle_file_name_list)
        # 增量修改
        increment_link, increment_node = increment(link_gdf=link_gdf, node_gdf=node_gdf,
                                                   path_gdf_dict=new_path_gdf_dict, plain_crs=self.plain_crs,
                                                   out_fldr=increment_out_fldr, save_times=save_times,
                                                   save_new_split_link=save_new_split_link,
                                                   overlap_buffer_size=overlap_buffer_size,
                                                   cover_angle_threshold=cover_angle_threshold,
                                                   cover_ratio_threshold=cover_ratio_threshold,
                                                   net_file_type=self.net_file_type,
                                                   limit_col_name=self.limit_col_name)
        return increment_link, increment_node

    def request_path(self, key_list: list[str], traffic_mode: str = 'car', binary_path_fldr: str = r'./',
                     od_file_path: str = None, od_df: pd.DataFrame = None,
                     region_gdf: gpd.GeoDataFrame = None, od_type: str = 'rand_od', boundary_buffer: float = 2000,
                     cache_times: int = 300, ignore_hh: bool = True, remove_his: bool = True,
                     log_fldr: str = None, save_log_file: bool = False,
                     min_lng: float = None, min_lat: float = None, w: float = 2000, h: float = 2000,
                     od_num: int = 100, gap_n: int = 1000, min_od_length: float = 1200.0,
                     is_rnd_strategy: bool = False, strategy: str = '32', wait_until_recovery: bool = False) \
            -> tuple[bool, list[str]]:
        """NetReverse类方法 - request_path：

        - 请求路径存储为二进制文件：构造OD -> 请求 -> 二进制存储

        Args:
            key_list: 开发者key值列表，必需参数
            traffic_mode: (v0.3.19即将支持)交通模式, 目前支持驾车(car)、骑行(bike)和步行(walk)
            binary_path_fldr: 存储请求路径源文件的目录
            wait_until_recovery: 如果配额超限，是否一直等待直至配额恢复
            is_rnd_strategy: 是否启用随机策略
            strategy: [1]请求设置参数 - 路径规划策略参数，若模式为驾车，取值请访问: https://lbs.amap.com/api/webservice/guide/api/newroute#s1；若模式为步行或骑行, 则取值为1、2、3，代表返回的方案数
            cache_times: 路径文件缓存数，即每请求cache_times次缓存一次数据到binary_path_fldr下
            ignore_hh: 是否忽略时段限制进行请求
            remove_his: 是否对已经请求的OD重复(指的是在请求被意外中断的情况下，od_id为判断依据)请求
            save_log_file: 是否保存日志文件
            log_fldr: 日志文件的存储目录
            od_file_path: (2.OD构造参数)用于请求的od文件路径，可选参数
            od_df: (2.OD构造参数)用于请求的od数据，该参数和od_file_path任意指定一个即可，可选参数
            region_gdf: (2.OD构造参数)用于构造od的面域数据, EPSG:4326
            od_type: (2.OD构造参数)用于构造od的方法，rand_od、region_od、diy_od
            min_lng: (2.OD构造参数)矩形区域的左下角经度
            min_lat: (2.OD构造参数)矩形区域的左下角纬度
            w: (2.OD构造参数)矩形区域的宽度，米
            h: (2.OD构造参数)矩形区域的高度，米
            od_num: (2.OD构造参数)请求的od数，od数越多，请求的路径就越多，路网覆盖率就越完整，只有od_type为rand_od时起效
            gap_n: (2.OD构造参数)横纵向网格个数，只有od_type为rand_od时起效
            min_od_length: (2.OD构造参数)od之间最短直线距离，只有od_type为rand_od时起效
            boundary_buffer: (2.OD构造参数)区域边界buffer，米，

        Returns:
            if_end_request, new_file_list
        """
        assert od_type in ['rand_od', 'region_od', 'diy_od']
        fmod = FormatOD(plain_crs=self.plain_crs)
        if od_type == 'rand_od':
            if region_gdf is None or region_gdf.empty:
                region_gdf = generate_region(min_lng=min_lng, min_lat=min_lat, w=w, h=h, plain_crs=self.plain_crs)

            od_df = fmod.format_region_rnd_od(region_gdf=region_gdf, flag_name=self.flag_name, od_num=od_num,
                                              gap_n=gap_n, length_limit=min_od_length,
                                              boundary_buffer=boundary_buffer)
            self.__region_gdf = region_gdf
            od_file_path = None
        elif od_type == 'region_od':
            od_df = fmod.format_region_od(region_gdf=region_gdf)
            od_file_path = None
        elif od_type == 'diy_od':
            if od_df is None or od_df.empty:
                od_df = pd.read_csv(od_file_path)
                od_file_path = None

        self.__od_df = od_df

        path_request_obj = CarPath(key_list=key_list, input_file_path=od_file_path, od_df=od_df,
                                   cache_times=cache_times, ignore_hh=ignore_hh, out_fldr=binary_path_fldr,
                                   file_flag=self.flag_name, log_fldr=log_fldr, save_log_file=save_log_file,
                                   wait_until_recovery=wait_until_recovery)

        # 是否结束请求, 新生产的路网文件
        if_end_request, new_file_list = path_request_obj.get_path(traffic_mode=traffic_mode,
                                                                  remove_his=remove_his, strategy=strategy,
                                                                  is_rnd_strategy=is_rnd_strategy)

        return if_end_request, new_file_list

    def increment_from_path_gdf(self):
        pass

    def _generate_net_from_split_path(self, split_path_gdf: gpd.GeoDataFrame):
        if not self.multi_core_reverse:
            if split_path_gdf.empty:
                return None
            self.__generate_net_from_split_path(split_path_gdf=split_path_gdf)
        else:
            split_path_gdf = rn_partition_alpha(split_path_gdf=split_path_gdf, partition_num=self.reverse_core_num,
                                                is_geo_coord=True)
            if split_path_gdf.empty:
                return None
            split_path_gdf_dict = {region: gdf.reset_index(drop=True) for region, gdf in
                                   split_path_gdf.groupby('region_id')}
            self.__generate_net_from_split_path_parallel(split_path_gdf_dict=split_path_gdf_dict)
    def __generate_net_from_split_path(self, split_path_gdf: gpd.GeoDataFrame):
        """

        :param split_path_gdf:
        :return:
        """
        net_reverse.generate_net(path_gdf=split_path_gdf, out_fldr=self.net_out_fldr,
                                 plain_prj=self.plain_crs,
                                 flag_name=self.flag_name,
                                 limit_col_name=self.limit_col_name,
                                 restrict_angle=self.restrict_angle,
                                 restrict_length=self.restrict_length,
                                 accu_l_threshold=self.accu_l_threshold,
                                 min_length=self.min_length,
                                 angle_threshold=self.angle_threshold,
                                 save_split_link=self.save_split_link,
                                 save_tpr_link=self.save_tpr_link,
                                 modify_minimum_buffer=self.modify_minimum_buffer,
                                 save_preliminary=self.save_preliminary,
                                 save_done_topo=self.save_done_topo,
                                 save_streets_before_modify_minimum=self.save_streets_before_modify_minimum,
                                 save_streets_after_modify_minimum=self.save_streets_after_modify_minimum,
                                 is_process_dup_link=self.is_process_dup_link,
                                 process_dup_link_buffer=self.process_dup_link_buffer,
                                 dup_link_buffer_ratio=self.dup_link_buffer_ratio,
                                 net_file_type=self.net_file_type,
                                 modify_conn=self.is_modify_conn,
                                 conn_buffer=self.conn_buffer,
                                 conn_period=self.conn_period,
                                 multi_core_merge=self.multi_core_merge,
                                 core_num=self.merge_core_num)

    def __generate_net_from_split_path_parallel(self, split_path_gdf_dict: dict[int, gpd.GeoDataFrame]):
        """
        :param split_path_gdf_dict:
        :return:
        """
        core_num = len(split_path_gdf_dict)
        print(f'using multiprocessing - {core_num} cores')
        pool = multiprocessing.Pool(processes=core_num)
        result_list = []
        for i in split_path_gdf_dict.keys():
            core_out_fldr = os.path.join(self.net_out_fldr, rf'region-{i}')
            if os.path.exists(core_out_fldr):
                pass
            else:
                os.makedirs(core_out_fldr)

            result = pool.apply_async(net_reverse.generate_net,
                                      args=(split_path_gdf_dict[i], core_out_fldr, self.save_split_link, self.plain_crs,
                                            self.save_tpr_link, self.save_streets_before_modify_minimum,
                                            self.restrict_angle,
                                            self.limit_col_name, self.restrict_length, self.accu_l_threshold,
                                            self.angle_threshold,
                                            False, 1, self.modify_minimum_buffer, self.flag_name + f'-region{i}',
                                            self.save_streets_after_modify_minimum, self.save_preliminary,
                                            self.save_done_topo,
                                            self.is_process_dup_link, self.process_dup_link_buffer, self.min_length,
                                            self.dup_link_buffer_ratio,
                                            self.net_file_type, self.is_modify_conn, self.conn_buffer,
                                            self.conn_period))
            result_list.append(result)
        pool.close()
        pool.join()

    def modify_conn(self, link_gdf: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame,
                    book_mark_name: str = 'test', link_name_field: str = 'road_name', generate_mark: bool = False) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """NetReverse类方法 - modify_conn：

        - 联通性修复
        
        Args:
            link_gdf: 线层gdf, 要求输入必须为EPSG:4326
            node_gdf: 点层gdf, 要求输入必须为EPSG:4326
            book_mark_name: 空间书签名称
            generate_mark: 是否生成空间书签，在net_out_fldr下生成
            link_name_field: 参数暂未启用

        Returns:
            线层gdf, 点层gdf, 修复点位空间信息
        """
        link_gdf, node_gdf = self.fix_minimum_gap(node_gdf=node_gdf, link_gdf=link_gdf)
        net = Net(link_gdf=link_gdf, node_gdf=node_gdf, create_single=False, plane_crs=self.plain_crs,
                  delete_circle=False)
        conn = Conn(net=net, check_buffer=self.conn_buffer)
        link_gdf, node_gdf = conn.execute(out_fldr=self.net_out_fldr, file_name=book_mark_name,
                                          generate_mark=generate_mark)
        link_gdf.reset_index(inplace=True, drop=True)
        node_gdf.reset_index(inplace=True, drop=True)
        save_file(data_item=link_gdf, file_type=self.net_file_type, file_name='modifiedConnLink',
                  out_fldr=self.net_out_fldr)
        save_file(data_item=node_gdf, file_type=self.net_file_type, file_name='modifiedConnNode',
                  out_fldr=self.net_out_fldr)
        return link_gdf, node_gdf

    def fix_minimum_gap(self, node_gdf: gpd.GeoDataFrame = None, link_gdf: gpd.GeoDataFrame = None) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """

        Args:
            node_gdf:
            link_gdf:

        Returns:

        """
        # 去除没有link连接的节点
        drop_no_use_nodes(link_gdf, node_gdf)
        link_gdf, node_gdf, _ = modify_minimum(plain_prj=self.plain_crs, link_gdf=link_gdf, node_gdf=node_gdf,
                                               buffer=self.modify_minimum_buffer,
                                               ignore_merge_rule=True)
        return link_gdf, node_gdf

    @staticmethod
    def clean_link_geo(gdf: gpd.GeoDataFrame, plain_crs: str = 'EPSG:32650', l_threshold: float = 0.5) -> gpd.GeoDataFrame:
        """NetReverse类静态方法 - clean_link_geo：

         - 清洗线层：去除Z坐标、去除multi类型、拆分自相交对象、去除线层重叠折点
        
        Args:
            gdf: 线层gdf
            plain_crs: 平面投影坐标系
            l_threshold: 重叠折点检测阈值, 米

        Returns:
            线层gdf
        """
        return clean_link_geo(gdf=gdf, plain_crs=plain_crs, l_threshold=l_threshold)

    @staticmethod
    def remapping_link_node_id(link_gdf: gpd.GeoDataFrame or pd.DataFrame, node_gdf: gpd.GeoDataFrame or pd.DataFrame):
        """NetReverse类静态方法 - remapping_link_node_id：

        - ID重映射：为link、node层映射新的ID编号, 在原对象上直接修改

        Args:
            link_gdf: 线层gdf
            node_gdf: 线层gdf

        Returns:
            None
        """
        remapping_id(link_gdf=link_gdf, node_gdf=node_gdf)

    @staticmethod
    def divide_links(link_gdf: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame, plain_crs: str = 'EPSG:3857',
                     divide_l: float = 70.0, min_l: float = 1.0) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """NetReverse类静态方法 - divide_links：

        - 对标准路网执行划分(打断)：路段将长度大于divide_l的路段进行切分，同时更新点层

        Args:
            link_gdf: 线层gdf, 要求输入必须为EPSG:4326
            node_gdf: 点层gdf, 要求输入必须为EPSG:4326
            plain_crs: 平面投影坐标系
            divide_l: 所有长度(米)大于divide_l的路段都将被打断
            min_l: 某次打断后如果剩下的路段长度小于min_l, 那么此次打断将不被允许

        Returns:
            线层gdf, 点层gdf
        """
        link_gdf = merge_double_link(link_gdf=link_gdf)
        my_net = Net(link_gdf=link_gdf,
                     node_gdf=node_gdf, create_single=False, plane_crs=plain_crs)

        # 执行划分路网
        # divide_l: 所有长度大于divide_l的路段都将按照divide_l进行划分
        # min_l: 划分后如果剩下的路段长度小于min_l, 那么此次划分将不被允许
        # is_init_link: 划分后是否重新初始化路网对象
        # method: alpha 或者 beta, 前一种方法可保留与划分前的link的映射关系(_parent_link字段)
        my_net.divide_links(divide_l=divide_l, min_l=min_l, is_init_link=False, method='alpha')
        link, node = \
            my_net.get_bilateral_slink_data().reset_index(drop=True, inplace=False), \
            my_net.get_snode_data().reset_index(drop=True, inplace=False)
        link = link.to_crs('EPSG:4326')
        node = node.to_crs('EPSG:4326')
        return link, node

    @staticmethod
    def circle_process(link_gdf: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame, plain_crs: str = 'EPSG:3857') -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """NetReverse类静态方法 - circle_process：

        - 处理标准路网里面的环路：将环路进行打断

        Args:
            link_gdf: 线层gdf, 要求输入必须为EPSG:4326
            node_gdf: 点层gdf, 要求输入必须为EPSG:4326
            plain_crs: 平面投影坐标系

        Returns:
            线层gdf, 点层gdf
        """
        my_net = Net(link_gdf=link_gdf,
                     node_gdf=node_gdf, create_single=False, delete_circle=False, plane_crs=plain_crs)
        my_net.process_circle()
        link, node = \
            my_net.get_bilateral_slink_data().reset_index(drop=True, inplace=False), \
            my_net.get_snode_data().reset_index(drop=True, inplace=False)
        link = link.to_crs('EPSG:4326')
        node = node.to_crs('EPSG:4326')
        return link, node

    def redivide_link_node(self, link_gdf: gpd.GeoDataFrame):
        """NetReverse类方法 - redivide_link_node：

        - 路网重塑：对线层文件进行重塑(折点拆分 -> 拓扑优化 -> 重叠路段处理 -> 联通性修复)

        Args:
            link_gdf: 线层gdf

        Returns:
            None, 直接在net_out_fldr下生成路网
        """
        link_gdf = link_gdf.to_crs('EPSG:4326')
        link_gdf.dropna(axis=1, how='all', inplace=True)
        if link_gdf.empty:
            return None
        if net_field.DIRECTION_FIELD not in link_gdf.columns:
            print(rf'the link layer data lacks the dir field and is automatically filled with 0')
            link_gdf[net_field.DIRECTION_FIELD] = 0
        try:
            link_gdf[net_field.DIRECTION_FIELD] = link_gdf[net_field.DIRECTION_FIELD].astype(int)
        except Exception as e:
            print(rf'{repr(e)}: the dir field has an empty value.')
        try:
            del link_gdf[net_field.FROM_NODE_FIELD]
            del link_gdf[net_field.TO_NODE_FIELD]
            del link_gdf[net_field.LINK_ID_FIELD]
            del link_gdf[net_field.LENGTH_FIELD]
        except Exception as e:
            pass
        assert set(link_gdf[net_field.DIRECTION_FIELD]).issubset({0, 1}), \
            'there are abnormal values in the dir field. Only 0 and 1 are allowed.'

        # 创建single_link
        single_link_gdf = create_single_link(link_gdf=link_gdf)
        if self.limit_col_name not in single_link_gdf.columns:
            single_link_gdf[self.limit_col_name] = 'XX路'
        single_link_gdf = split_path(path_gdf=single_link_gdf)
        single_link_gdf.drop(columns=['ft_loc'], axis=1, inplace=True)
        del link_gdf
        self._generate_net_from_split_path(split_path_gdf=single_link_gdf)

    @staticmethod
    def merge_net(net_list: list[list[gpd.GeoDataFrame, gpd.GeoDataFrame]],
                  conn_buffer: float = 0.5, out_fldr: str = r'./', plain_crs: str = 'EPSG:3857') -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """NetReverse类静态方法 - merge_net：

        - 合并标准路网 ：对多个标准路网进行合并
        
        Args:
            net_list: 待合并的路网, crs必须为: EPSG:4326
            conn_buffer: 合并检测阈值(米)
            out_fldr: 存储文件目录
            plain_crs: 平面投影坐标系

        Returns:
            线层gdf, 点层gdf
        """
        max_link, max_node = 0, 0
        link_gdf, node_gdf = gpd.GeoDataFrame(), gpd.GeoDataFrame()
        for _link, _node in net_list:
            remapping_id(link_gdf=_link, node_gdf=_node, start_link_id=max_link + 1,
                         start_node_id=max_node + 1)
            link_gdf = pd.concat([link_gdf, _link])
            node_gdf = pd.concat([node_gdf, _node])
            max_link, max_node = link_gdf[net_field.LINK_ID_FIELD].max(), node_gdf[net_field.NODE_ID_FIELD].max()
        link_gdf.reset_index(inplace=True, drop=True)
        node_gdf.reset_index(inplace=True, drop=True)
        n = Net(link_gdf=link_gdf, node_gdf=node_gdf, create_single=False, plane_crs=plain_crs)
        conn = Conn(net=n, check_buffer=conn_buffer)
        link, node = conn.execute(out_fldr=out_fldr,
                                  file_name='NetMerge', generate_mark=True)
        save_file(out_fldr=out_fldr, file_name='MergeLink', file_type='shp', data_item=link)
        save_file(out_fldr=out_fldr, file_name='MergeNode', file_type='shp', data_item=node)
        return link, node
