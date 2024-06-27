# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData


"""生产路网的相关方法"""

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
from .RoadNet.Split.SplitPath import split_path
from .PublicTools.GeoProcess import generate_region
from .RoadNet.Split.SplitPath import split_path_main
from .RoadNet.Tools.process import merge_double_link, create_single_link
from .RoadNet.SaveStreets.streets import generate_node_from_link, modify_minimum
from ..tools.geo_process import clean_link_geo, remapping_id, rn_partition, rn_partition_alpha

net_field = NetField()
gps_field = GpsField()

class Reverse(object):
    def __init__(self, flag_name: str = 'NetGen', plain_crs: str = None, net_out_fldr: str = r'./',
                 net_file_type: str = 'shp'):
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
        """
        :param flag_name: 标志字符(项目名称)
        :param plain_crs: 平面投影坐标系
        :param net_out_fldr: 输出路网的存储目录
        :return:
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

    def generate_net_from_request(self, key_list: list[str] = None, binary_path_fldr: str = r'./',
                                  od_file_path: str = None, od_df: pd.DataFrame = None,
                                  region_gdf: gpd.GeoDataFrame = None, od_type='rand_od', boundary_buffer: float = 2000,
                                  cache_times: int = 300, ignore_hh: bool = True, remove_his: bool = True,
                                  log_fldr: str = None, save_log_file: bool = False,
                                  min_lng: float = None, min_lat: float = None, w: float = 2000, h: float = 2000,
                                  od_num: int = 100, gap_n: int = 1000, min_od_length: float = 1200.0,
                                  wait_until_recovery: bool = False) -> None:
        """构造OD -> 请求 -> 二进制存储 -> 路网生产"""
        self.request_path(key_list=key_list, binary_path_fldr=binary_path_fldr,
                          od_file_path=od_file_path,
                          od_df=od_df, region_gdf=region_gdf, od_type=od_type,
                          boundary_buffer=boundary_buffer,
                          cache_times=cache_times, ignore_hh=ignore_hh,
                          remove_his=remove_his,
                          log_fldr=log_fldr, save_log_file=save_log_file,
                          min_lng=min_lng, min_lat=min_lat,
                          w=w, h=h, od_num=od_num, gap_n=gap_n,
                          min_od_length=min_od_length, wait_until_recovery=wait_until_recovery)
        pickle_file_name_list = os.listdir(binary_path_fldr)
        self.generate_net_from_pickle(binary_path_fldr=binary_path_fldr,
                                      pickle_file_name_list=pickle_file_name_list)

    def generate_net_from_pickle(self, binary_path_fldr: str = None, pickle_file_name_list: list[str] = None) -> None:
        """
        从二进制路径文件进行读取, 然后生产路网
        :param binary_path_fldr:
        :param pickle_file_name_list:
        :return:
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

    def generate_net_from_path_gdf(self, path_gdf: gpd.GeoDataFrame = None,
                                   slice_num: int = 1, attr_name_list: list = None,
                                   cut_slice: bool = False):
        """
        input crs: EPSG:4326
        从路径gdf创建路网
        :param path_gdf:
        :param slice_num:
        :param attr_name_list:
        :param cut_slice:
        :return:
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
    def create_node_from_link(link_gdf: gpd.GeoDataFrame = None, update_link_field_list: list[str] = None,
                              using_from_to: bool = False, fill_dir: int = 0, plain_crs: str = 'EPSG:32650',
                              ignore_merge_rule: bool = True, modify_minimum_buffer: float = 0.8,
                              execute_modify: bool = True, auxiliary_judge_field: str = None,
                              out_fldr: str = None, save_streets_before_modify_minimum: bool = False,
                              save_streets_after_modify_minimum: bool = True, net_file_type: str = 'shp') -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        assert net_file_type in ['shp', 'geojson']
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

    def topology_optimization(self, link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None,
                              out_fldr: str = r'./') -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
        """

        :param link_gdf: EPSG:4326
        :param node_gdf: EPSG:4326
        :param out_fldr:
        :return: crs - EPSG:4326
        """
        if self.limit_col_name not in link_gdf.columns:
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
        if out_fldr is not None:
            save_file(data_item=link_gdf, out_fldr=out_fldr, file_type=self.net_file_type, file_name='opt_link')
            save_file(data_item=node_gdf, out_fldr=out_fldr, file_type=self.net_file_type, file_name='opt_node')
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
        """
        依据路径源文件进行增量修改, 输入crs:EPSG:4326, 输出crs:EPSG:4326
        :param link_gdf: 路网线层
        :param node_gdf: 路网点层
        :param save_times: 请求路径文件时单个文件的缓存数目
        :param binary_path_fldr: 二进制路径文件目录
        :param pickle_file_name_list: 二进制路径文件名称列表
        :param increment_out_fldr: 更新后的路网输出目录
        :param ignore_head_tail: 是否忽略路径首尾的无名道路, 这种一般是小区内部道路
        :param save_new_split_link: 是否保存新路径拆分后的文件
        :param check_path: 是否检查新路径
        :param overlap_buffer_size: 用于判断路段是否重合的buffer_size
        :param cover_ratio_threshold: 用重合率超过cover_ratio_threshold%就认为是重合(条件较为宽松), 宁愿少加也不多加
        :param cover_angle_threshold: 角度小于cover_angle_threshold度认为是重合(条件较为宽松), 宁愿少加也不多加
        :return:
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

    def request_path(self, key_list: list[str] = None, binary_path_fldr: str = r'./',
                     od_file_path: str = None, od_df: pd.DataFrame = None,
                     region_gdf: gpd.GeoDataFrame = None, od_type='rand_od', boundary_buffer: float = 2000,
                     cache_times: int = 300, ignore_hh: bool = True, remove_his: bool = True,
                     log_fldr: str = None, save_log_file: bool = False,
                     min_lng: float = None, min_lat: float = None, w: float = 2000, h: float = 2000,
                     od_num: int = 100, gap_n: int = 1000, min_od_length: float = 1200.0,
                     is_rnd_strategy: bool = True, strategy: str = '32', wait_until_recovery: bool = False) \
            -> tuple[bool, list[str]]:
        """构造OD -> 请求 -> 二进制存储, 要求输入的面域必须为EPSG:4326"""
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
        if_end_request, new_file_list = path_request_obj.get_path(remove_his=remove_his, strategy=strategy,
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

    def modify_conn(self, link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None,
                    book_mark_name: str = 'test', link_name_field: str = 'road_name', generate_mark: bool = False) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        要求输入必须为EPSG:4326
        :param link_gdf:
        :param node_gdf:
        :param book_mark_name:
        :param link_name_field:
        :param generate_mark
        :return:
        """
        link_gdf, node_gdf = self.fix_minimum_gap(node_gdf=node_gdf, link_gdf=link_gdf)
        net = Net(link_gdf=link_gdf, node_gdf=node_gdf, create_single=False, plane_crs=self.plain_crs)
        conn = Conn(net=net, check_buffer=self.conn_buffer)
        link_gdf, node_gdf = conn.execute(out_fldr=self.net_out_fldr, file_name=book_mark_name,
                                          generate_mark=generate_mark)
        link_gdf.reset_index(inplace=True, drop=True)
        node_gdf.reset_index(inplace=True, drop=True)
        save_file(data_item=link_gdf, file_type=self.net_file_type, file_name='modifiedConn',
                  out_fldr=self.net_out_fldr)
        save_file(data_item=node_gdf, file_type=self.net_file_type, file_name='modifiedConn',
                  out_fldr=self.net_out_fldr)
        return link_gdf, node_gdf

    def fix_minimum_gap(self, node_gdf: gpd.GeoDataFrame = None, link_gdf: gpd.GeoDataFrame = None) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        link_gdf, node_gdf, _ = modify_minimum(plain_prj=self.plain_crs, link_gdf=link_gdf, node_gdf=node_gdf,
                                               buffer=self.modify_minimum_buffer,
                                               ignore_merge_rule=True)
        return link_gdf, node_gdf

    @staticmethod
    def clean_link_geo(gdf: gpd.GeoDataFrame = None, plain_crs: str = 'EPSG:32650',
                       l_threshold: float = 0.5) -> gpd.GeoDataFrame:
        return clean_link_geo(gdf=gdf, plain_crs=plain_crs, l_threshold=l_threshold)

    @staticmethod
    def remapping_link_node_id(link_gdf: gpd.GeoDataFrame or pd.DataFrame, node_gdf: gpd.GeoDataFrame or pd.DataFrame):
        """
        :param link_gdf:
        :param node_gdf:
        :return:
        """
        remapping_id(link_gdf=link_gdf, node_gdf=node_gdf)

    @staticmethod
    def divide_links(link_gdf: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame = None, plain_crs: str = None,
                     divide_l: float = 70.0, min_l: float = 1.0) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """

        :param link_gdf: EPSG:4326
        :param node_gdf: EPSG:4326
        :param divide_l:
        :param min_l:
        :param plain_crs
        :return:
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
        link, node = my_net.get_bilateral_link_data().reset_index(drop=True), my_net.get_node_data().reset_index(drop=True)
        link = link.to_crs('EPSG:4326')
        node = node.to_crs('EPSG:4326')
        return link, node

    @staticmethod
    def circle_process(link_gdf: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame = None, plain_crs: str = None) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        :param link_gdf: EPSG:4326
        :param node_gdf: EPSG:4326
        :param plain_crs
        :return:
        """
        my_net = Net(link_gdf=link_gdf,
                     node_gdf=node_gdf, create_single=False, delete_circle=False, plane_crs=plain_crs)
        my_net.process_circle()
        link, node = my_net.get_bilateral_link_data().reset_index(drop=True), my_net.get_node_data().reset_index(
            drop=True)
        link = link.to_crs('EPSG:4326')
        node = node.to_crs('EPSG:4326')
        return link, node

    def redivide_link_node(self, link_gdf: gpd.GeoDataFrame = None):
        """
        对link进行线层和点层的重新划分
        :param link_gdf: gpd.GeoDataFrame, 要求至少有geometry字段
        """
        link_gdf = link_gdf.to_crs('EPSG:4326')

        if net_field.DIRECTION_FIELD not in link_gdf.columns:
            print(rf'link层数据缺少dir字段, 自动填充为0')
            link_gdf[net_field.DIRECTION_FIELD] = 0
        link_gdf[net_field.DIRECTION_FIELD] = link_gdf[net_field.DIRECTION_FIELD].astype(int)
        try:
            del link_gdf[net_field.FROM_NODE_FIELD]
            del link_gdf[net_field.TO_NODE_FIELD]
            del link_gdf[net_field.LINK_ID_FIELD]
            del link_gdf[net_field.LENGTH_FIELD]
        except Exception as e:
            print(repr(e))
        assert set(link_gdf[net_field.DIRECTION_FIELD]).issubset({0, 1}), 'dir字段中有异常值, 只允许0,1出现'

        # 创建single_link
        single_link_gdf = create_single_link(link_gdf=link_gdf)
        if self.limit_col_name not in single_link_gdf.columns:
            single_link_gdf[self.limit_col_name] = 'XX路'
        single_link_gdf = split_path(path_gdf=single_link_gdf)
        single_link_gdf.drop(columns=['ft_loc'], axis=1, inplace=True)
        del link_gdf
        self._generate_net_from_split_path(split_path_gdf=single_link_gdf)

    @staticmethod
    def merge_net(net_list: list[list[gpd.GeoDataFrame, gpd.GeoDataFrame]] = None,
                  conn_buffer: float = 0.5, out_fldr=r'./', plain_crs: str = None):
        """
        路网合并
        :param net_list: must EPSG:4326
        :param conn_buffer:
        :param out_fldr:
        :param plain_crs:
        :return:
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
