# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData


"""生产路网的相关方法"""


import os.path
import pandas as pd
import geopandas as gpd
from .GlobalVal import NetField
from .format_od import FormatOD
from .RoadNet import net_reverse
from .RoadNet.save_file import save_file
from .Request.request_path import CarPath
from .RoadNet.optimize_net import optimize
from .PublicTools.GeoProcess import generate_region
from .Parse.gd_car_path import parse_path_main_alpha
from .RoadNet.SaveStreets.streets import generate_node_from_link


net_field = NetField()


class Reverse(object):
    def __init__(self, flag_name: str = '深圳市', plain_prj: str = None, net_out_fldr: str = None):
        # overall
        self.flag_name = flag_name
        self.plain_prj = plain_prj
        self.net_out_fldr = net_out_fldr


class NetReverse(Reverse):
    def __init__(self, flag_name: str = '深圳市', plain_prj: str = 'EPSG:32650', ignore_head_tail: bool = False,
                 cut_slice: bool = False, slice_num: int = 5, generate_rod: bool = False, min_rod_length: float = 5.0,
                 restrict_region_gdf: gpd.GeoDataFrame = None, save_split_link: bool = False,
                 modify_minimum_buffer: float = 0.8, save_streets_before_modify_minimum: bool = False,
                 save_streets_after_modify_minimum: bool = False, save_tpr_link: bool = False, ignore_dir: bool = False,
                 allow_ring: bool = False, restrict_angle: bool = True, restrict_length: bool = True,
                 accu_l_threshold: float = 200.0, angle_threshold: float = 35.0, min_length: float = 50.0,
                 save_preliminary: bool = False, is_process_dup_link: bool = True, process_dup_link_buffer: float = 0.8,
                 dup_link_buffer_ratio: float = 60.0, net_out_fldr: str = None):
        """
        :param flag_name: 标志字符(项目名称)
        :param plain_prj: 平面投影坐标系
        :param net_out_fldr: 输出路网的存储目录
        :return:
        """
        # overall
        super().__init__(flag_name, plain_prj, net_out_fldr)

        # split
        self.ignore_head_tail = ignore_head_tail
        self.cut_slice = cut_slice
        self.slice_num = slice_num
        self.generate_rod = generate_rod
        self.min_rod_length = min_rod_length
        self.save_split_link = save_split_link
        self.restrict_region_gdf = restrict_region_gdf

        # create node from link
        self.modify_minimum_buffer = modify_minimum_buffer
        self.save_streets_before_modify_minimum = save_streets_before_modify_minimum
        self.save_streets_after_modify_minimum = save_streets_after_modify_minimum
        self.save_tpr_link = save_tpr_link

        # merge
        self.ignore_dir = ignore_dir
        self.allow_ring = allow_ring
        self.restrict_angle = restrict_angle
        self.restrict_length = restrict_length
        self.accu_l_threshold = accu_l_threshold
        self.angle_threshold = angle_threshold
        self.min_length = min_length
        self.save_preliminary = save_preliminary

        # process dup
        self.is_process_dup_link = is_process_dup_link
        self.process_dup_link_buffer = process_dup_link_buffer
        self.dup_link_buffer_ratio = dup_link_buffer_ratio

        # attrs
        self.__od_df = pd.DataFrame()
        self.__region_gdf = gpd.GeoDataFrame()

    def generate_net_from_request(self, key_list: list[str] = None, binary_path_fldr: str = None,
                                  od_file_path: str = None, od_df: pd.DataFrame = None,
                                  region_gdf: gpd.GeoDataFrame = None, od_type='rnd', boundary_buffer: float = 2000,
                                  cache_times: int = 300, ignore_hh: bool = True, remove_his: bool = True,
                                  log_fldr: str = None, save_log_file: bool = False,
                                  min_lng: float = None, min_lat: float = None, w: float = 2000, h: float = 2000,
                                  od_num: int = 100, gap_n: int = 1000, min_od_length: float = 1200.0) -> None:
        """请求 -> 二进制存储 -> 路网生产"""
        assert binary_path_fldr is not None

        assert od_type in ['rand_od', 'region_od', 'diy_od', 'gps_based']
        fmod = FormatOD(plain_crs=self.plain_prj)
        if od_type == 'rand_od':
            if region_gdf is None or region_gdf.empty:
                region_gdf = generate_region(min_lng=min_lng, min_lat=min_lat, w=w, h=h, plain_crs=self.plain_prj)

            od_df = fmod.format_region_rnd_od(region_gdf=region_gdf, flag_name=self.flag_name, od_num=od_num,
                                              gap_n=gap_n, length_limit=min_od_length,
                                              boundary_buffer=boundary_buffer)
            self.__region_gdf = region_gdf
        elif od_type == 'region_od':
            od_df = fmod.format_region_od(region_gdf=region_gdf)
        elif od_type == 'diy_od':
            if od_df is None or od_df.empty:
                od_df = pd.read_csv(od_file_path)
        elif od_type == 'gps_based':
            raise ValueError('Sorry! This function is under development! 这个函数正在开发中...')

        self.__od_df = od_df

        path_request_obj = CarPath(key_list=key_list, input_file_path=od_file_path, od_df=od_df,
                                   cache_times=cache_times, ignore_hh=ignore_hh, out_fldr=binary_path_fldr,
                                   file_flag=self.flag_name, log_fldr=log_fldr, save_log_file=save_log_file)
        path_request_obj.get_path(remove_his=remove_his)

        pickle_file_name_list = os.listdir(binary_path_fldr)
        self.generate_net_from_pickle(restrict_region_gdf=self.restrict_region_gdf,
                                      binary_path_fldr=binary_path_fldr,
                                      pickle_file_name_list=pickle_file_name_list)

    def generate_net_from_pickle(self, restrict_region_gdf: gpd.GeoDataFrame = None,
                                 binary_path_fldr: str = None, pickle_file_name_list: list[str] = None) -> None:
        """
        从二进制路径文件进行读取, 然后生产路网
        :return:
        """
        attr_name_list = ['road_name']
        split_path_gdf = parse_path_main_alpha(od_path_fldr=binary_path_fldr,
                                               pickle_file_name_list=pickle_file_name_list,
                                               flag_name=self.flag_name,
                                               is_slice=self.cut_slice,
                                               slice_num=self.slice_num,
                                               restrict_region_gdf=restrict_region_gdf,
                                               attr_name_list=attr_name_list,
                                               ignore_head_tail=self.ignore_head_tail,
                                               check=False, generate_rod=self.generate_rod,
                                               min_rod_length=self.min_rod_length)

        self.__generate_net_from_split_path(split_path_gdf=split_path_gdf)

    def generate_net_from_path_gdf(self, path_gdf: gpd.GeoDataFrame = None,
                                   restrict_region_gdf: gpd.GeoDataFrame = None,
                                   slice_num: int = 1, attr_name_list: list = None,
                                   cut_slice: bool = False):
        """
        input crs: EPSG:4326
        从路径gdf创建路网
        :param path_gdf:
        :param restrict_region_gdf:
        :param slice_num:
        :param attr_name_list:
        :param cut_slice:
        :return:
        """
        pass

    @staticmethod
    def create_node_from_link(link_gdf: gpd.GeoDataFrame = None, update_link_field_list: list[str] = None,
                              using_from_to: bool = False, fill_dir: int = 0, plain_prj: str = 'EPSG:32650',
                              ignore_merge_rule: bool = True, modify_minimum_buffer: float = 0.8,
                              execute_modify: bool = True, auxiliary_judge_field: str = None,
                              out_fldr: str = None, save_streets_before_modify_minimum: bool = False,
                              save_streets_after_modify_minimum: bool = True) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:

        link_gdf, node_gdf, node_group_status_gdf = \
            generate_node_from_link(link_gdf=link_gdf,
                                    update_link_field_list=update_link_field_list,
                                    using_from_to=using_from_to,
                                    fill_dir=fill_dir, plain_prj=plain_prj,
                                    ignore_merge_rule=ignore_merge_rule,
                                    modify_minimum_buffer=modify_minimum_buffer,
                                    execute_modify=execute_modify,
                                    auxiliary_judge_field=auxiliary_judge_field,
                                    out_fldr=out_fldr,
                                    save_streets_after_modify_minimum=save_streets_after_modify_minimum,
                                    save_streets_before_modify_minimum=save_streets_before_modify_minimum)
        return link_gdf, node_gdf, node_group_status_gdf

    def topology_optimization(self, link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None,
                              limit_col_name: str = 'road_name', out_fldr: str = None) -> \
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:

        link_gdf, node_gdf, dup_info_dict = optimize(link_gdf=link_gdf, node_gdf=node_gdf,
                                                     ignore_dir=self.ignore_dir,
                                                     allow_ring=self.allow_ring,
                                                     limit_col_name=limit_col_name,
                                                     plain_prj=self.plain_prj,
                                                     accu_l_threshold=self.accu_l_threshold,
                                                     angle_threshold=self.angle_threshold,
                                                     restrict_length=self.restrict_length,
                                                     restrict_angle=self.restrict_angle,
                                                     save_preliminary=False,
                                                     out_fldr=self.net_out_fldr,
                                                     is_process_dup_link=False,
                                                     process_dup_link_buffer=self.process_dup_link_buffer,
                                                     min_length=self.min_length,
                                                     dup_link_buffer_ratio=self.dup_link_buffer_ratio)
        if out_fldr is not None:
            save_file(data_item=link_gdf, out_fldr=out_fldr, file_type='shp', file_name='opt_link')
            save_file(data_item=node_gdf, out_fldr=out_fldr, file_type='shp', file_name='opt_node')
        return link_gdf, node_gdf, dup_info_dict

    def get_od_df(self) -> pd.DataFrame:
        return self.__od_df.copy()

    def check_conn(self):
        pass

    def modify_minimum(self):
        pass

    def __generate_net_from_split_path(self, split_path_gdf: gpd.GeoDataFrame):
        """

        :param split_path_gdf:
        :return:
        """
        net_reverse.generate_net(path_gdf=split_path_gdf, out_fldr=self.net_out_fldr,
                                 plain_prj=self.plain_prj,
                                 flag_name=self.flag_name,
                                 restrict_angle=self.restrict_angle,
                                 restrict_length=self.restrict_length,
                                 accu_l_threshold=self.accu_l_threshold,
                                 min_length=self.min_length,
                                 angle_threshold=self.angle_threshold,
                                 save_split_link=self.save_split_link,
                                 save_tpr_link=self.save_tpr_link,
                                 modify_minimum_buffer=self.modify_minimum_buffer,
                                 save_preliminary=self.save_preliminary,
                                 save_streets_before_modify_minimum=self.save_streets_before_modify_minimum,
                                 save_streets_after_modify_minimum=self.save_streets_after_modify_minimum,
                                 is_process_dup_link=self.is_process_dup_link,
                                 process_dup_link_buffer=self.process_dup_link_buffer,
                                 dup_link_buffer_ratio=self.dup_link_buffer_ratio)

