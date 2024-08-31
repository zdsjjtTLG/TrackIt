# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData


"""解析高德路径规划接口返回的数据"""


import os
import pickle
import pandas as pd
import multiprocessing
import geopandas as gpd
from geopy.distance import distance
from ...WrapsFunc import function_time_cost
from shapely.geometry import LineString, Point
from ..PublicTools.GeoProcess import point_distance
from ..RoadNet.Split.SplitPath import split_path_main
from ..PublicTools.IndexAna import find_continuous_repeat_index


class ParseGdPath(object):
    def __init__(self, used_core_num: int = 2, is_multi_core: bool = False, binary_path_fldr: str = None,
                 flag_name: str = None, restrict_region_gdf: gpd.GeoDataFrame = None,
                 slice_num: int = 1, is_slice: bool = False, attr_name_list: list = None,
                 pickle_file_name_list: list = None, check_fldr: str = None,
                 ignore_head_tail: bool = True, check: bool = False, generate_rod: bool = False,
                 min_rod_length: float = 1.5):
        """

        :param binary_path_fldr: 二进制路径文件所在的目录
        :param flag_name: str, 标志字符
        :param restrict_region_gdf: 区域GeoDataFrame
        :param slice_num: 在单独解析一个路径文件时的切片数量
        :param is_slice: 在单独解析一个路径文件时是否切片
        :param attr_name_list:
        :param pickle_file_name_list:
        :param ignore_head_tail:
        :param check:
        :param check_fldr:
        :param generate_rod: bool, 是否生成连杆
        :param min_rod_length
        """
        self. used_core_num =  used_core_num
        self.is_multi_core = is_multi_core
        self.binary_path_fldr = binary_path_fldr
        self.flag_name = flag_name
        self.restrict_region_gdf = restrict_region_gdf
        self.slice_num = slice_num
        self.is_slice = is_slice
        self.attr_name_list = attr_name_list
        self.pickle_file_name_list = pickle_file_name_list
        self.check_fldr = check_fldr
        self.ignore_head_tail = ignore_head_tail
        self.check: bool = check
        self.generate_rod = generate_rod
        self.min_rod_length = min_rod_length

    @function_time_cost
    def parse_path_main_multi(self) -> gpd.GeoDataFrame:
        all_split_path_gdf = gpd.GeoDataFrame()
        if self.is_multi_core:
            c_count = multiprocessing.cpu_count()
            n = self.used_core_num if self.used_core_num <= c_count else c_count
            file_num = len(self.pickle_file_name_list)
            n = file_num if file_num <= n else n
            file_name_item = cut_slice(item_list=self.pickle_file_name_list, slice_num=n)
            n = len(file_name_item)
            print(rf'启用{n}核...')
            pool = multiprocessing.Pool(processes=n)
            res_list = []
            for file_name_list in file_name_item:
                result = pool.apply_async(self.parse_path_main_alpha, args=(file_name_list, False))
                res_list.append(result)
            pool.close()
            pool.join()

            for _res in res_list:
                all_split_path_gdf = pd.concat([all_split_path_gdf, _res.get()])
            del res_list
            all_split_path_gdf.reset_index(inplace=True, drop=True)
            all_split_path_gdf.drop_duplicates(subset=['ft_loc'], keep='first', inplace=True)
            all_split_path_gdf.reset_index(inplace=True, drop=True)
            all_split_path_gdf.drop(columns=['ft_loc'], axis=1, inplace=True)
            return all_split_path_gdf
        else:
            return self.parse_path_main_alpha(pickle_file_name_list=self.pickle_file_name_list, drop_ft=True)

    def parse_path_main_alpha(self, pickle_file_name_list: list[str] = None, drop_ft: bool = False) -> gpd.GeoDataFrame:
        """
        给定目录, 读取目录下的二进制路径文件(读取一个文件马上解析并且拆分最小路段按照坐标去重复), 解析为gpd.GeoDataFrame(), crs:epsg:4326
        """
        if pickle_file_name_list is None:
            pickle_file_name_list = os.listdir(self.binary_path_fldr)

        q = 1
        all_split_link_gdf = gpd.GeoDataFrame()
        for file in pickle_file_name_list:
            try:
                with open(os.path.join(self.binary_path_fldr, file), 'rb') as f:
                    _ = pickle.load(f)
                print(file)
                path_route_dict = {}
                for k, v in _.items():
                    path_route_dict[q] = v
                    q += 1

                path_gdf = gpd.GeoDataFrame()
                print(rf'##########   {self.flag_name} parsing, a total of {len(path_route_dict)} paths')
                print(rf'##########   od_id start from {min(list(path_route_dict.keys()))}')
                for path_id in path_route_dict.keys():
                    o_loc = None
                    if self.generate_rod:
                        o_loc = tuple(map(float, path_route_dict[path_id]['route']['origin'].split(',')))
                    _path_gdf = parse_path_from_gd(json_data=path_route_dict[path_id],
                                                   check=self.check,
                                                   parse_num=10,
                                                   ignore_head_tail=self.ignore_head_tail,
                                                   crs='EPSG:4326',
                                                   flag_name=rf'path_{path_id}',
                                                   out_fldr=self.check_fldr,
                                                   generate_rod=self.generate_rod, o_loc=o_loc,
                                                   min_rod_length=self.min_rod_length)
                    if _path_gdf.empty:
                        continue
                    _path_gdf.drop(columns=['scheme', 'seq', 'time_cost', 'toll'], axis=1, inplace=True)
                    path_gdf = pd.concat([path_gdf, _path_gdf])
                path_gdf.set_geometry('geometry', inplace=True)
                path_gdf.reset_index(inplace=True, drop=True)

                # split
                print(rf'##########   {self.flag_name} - Split Path')
                split_path_gdf = split_path_main(path_gdf=path_gdf, restrict_region_gdf=self.restrict_region_gdf,
                                                 slice_num=self.slice_num, attr_name_list=self.attr_name_list,
                                                 cut_slice=self.is_slice, drop_ft_loc=False)
                all_split_link_gdf = pd.concat([all_split_link_gdf, split_path_gdf])
                all_split_link_gdf.reset_index(inplace=True, drop=True)
                all_split_link_gdf.drop_duplicates(subset=['ft_loc'], keep='first', inplace=True)
            except:
                print(rf'##########   Skip File {file}')
        all_split_link_gdf.reset_index(inplace=True, drop=True)
        if all_split_link_gdf.empty:
            return gpd.GeoDataFrame()
        all_split_link_gdf = gpd.GeoDataFrame(all_split_link_gdf, geometry='geometry', crs='EPSG:4326')
        if drop_ft:
            all_split_link_gdf.drop(columns=['ft_loc'], axis=1, inplace=True)
        return all_split_link_gdf

    def parse_path_main(self, out_type: str = 'dict', pickle_file_name_list: list[str] = None):
        """
        给定目录, 读取目录下的二进制路径文件, 读一个文件就解析一次, 存在内存中, 然后返回(返回字典或者gdf)
        :param out_type:
        :param pickle_file_name_list
        :return:
        """
        path_route_dict = {}
        q = 1
        print(rf'按照指定文件名读取...')
        for file in pickle_file_name_list:
            try:
                with open(os.path.join(self.binary_path_fldr, file), 'rb') as f:
                    _ = pickle.load(f)
                print(file)
                for k, v in _.items():
                    path_route_dict[q] = v
                    q += 1
            except:
                print(rf'{file}不读取......')

        if out_type == 'dict':
            path_gdf_dict = dict()
            for path_id in path_route_dict.keys():
                print(path_id)
                o_loc = None
                if self.generate_rod:
                    o_loc = tuple(map(float, path_route_dict[path_id]['origin'].split(',')))
                _new_path_gdf = parse_path_from_gd(json_data=path_route_dict[path_id],
                                                   check=self.check,
                                                   parse_num=3,
                                                   ignore_head_tail=self.ignore_head_tail,
                                                   crs='EPSG:4326',
                                                   flag_name=rf'path_{path_id}',
                                                   out_fldr=self.check_fldr,
                                                   generate_rod=self.generate_rod, o_loc=o_loc,
                                                   min_rod_length=self.min_rod_length)
                if _new_path_gdf.empty:
                    continue
                _new_path_gdf.drop(columns=['scheme', 'seq', 'time_cost', 'toll'], axis=1, inplace=True)
                path_gdf_dict[path_id] = _new_path_gdf
            return path_gdf_dict
        else:
            path_gdf = gpd.GeoDataFrame()
            for path_id in path_route_dict.keys():
                print(path_id)
                o_loc = None
                if self.generate_rod:
                    o_loc = tuple(map(float, path_route_dict[path_id]['origin'].split(',')))
                _path_gdf = parse_path_from_gd(json_data=path_route_dict,
                                               check=self.check,
                                               parse_num=10,
                                               ignore_head_tail=self.ignore_head_tail,
                                               crs='EPSG:4326',
                                               flag_name=rf'path_{path_id}',
                                               out_fldr=self.check_fldr, generate_rod=self.generate_rod, o_loc=o_loc,
                                               min_rod_length=self.min_rod_length)
                if _path_gdf.empty:
                    continue
                _path_gdf.drop(columns=['scheme', 'seq', 'time_cost', 'toll'], axis=1, inplace=True)
                path_gdf = pd.concat([path_gdf, _path_gdf])
            path_gdf.set_geometry('geometry', inplace=True)
            path_gdf.reset_index(inplace=True, drop=True)
            return path_gdf


def parse_path_from_gd(json_data: dict = None, parse_num: int = 1, check: bool = True, out_fldr: str = None,
                       flag_name: str = None, crs: str = 'EPSG:4326',
                       ignore_head_tail: bool = False, generate_rod: bool = False,
                       o_loc: tuple = None, min_rod_length: float = 1.5) -> gpd.GeoDataFrame:
    """
    从高德返回的分段路径得到path_line
    :param json_data: dict, 驾车路径规划返回的json数据
    :param parse_num: integer, 抽取几条路径(高德的驾车规划会返回<=3条路径)
    :param check: bool, 是否检查路径
    :param out_fldr: string
    :param flag_name: string, 保存文件时候, 用于命名
    :param crs: string
    :param ignore_head_tail: bool, 是否忽略路径首尾的无名道路, 这种一般是小区内部道路
    :param generate_rod: bool, 是否生成连杆
    :param o_loc: (x, y)
    :param min_rod_length
    :return: gpd.GeoDataFrame(), crs:EPSG:4326
    """
    all_path_gdf = gpd.GeoDataFrame()

    # 确定抽取的路径数(每个OD下)
    path_num = parse_num if parse_num <= len(json_data['route']['paths']) else len(json_data['route']['paths'])
    for i in range(0, path_num):
        _path_gdf = parse_single_path(json_data=json_data, path_seq=i, ignore_head_tail=ignore_head_tail,
                                      generate_rod=generate_rod, o_loc=o_loc, min_rod_length=min_rod_length)
        if _path_gdf.empty:
            continue
        _path_gdf['scheme'] = i + 1
        all_path_gdf = pd.concat([all_path_gdf, _path_gdf])

    if all_path_gdf.empty:
        return gpd.GeoDataFrame()

    all_path_gdf.reset_index(inplace=True, drop=True)

    all_path_gdf = gpd.GeoDataFrame(all_path_gdf, geometry='geometry', crs=crs)

    if check:
        try:
            all_path_gdf.to_file(os.path.join(out_fldr, rf'gd_path_{flag_name}.geojson'),
                                 driver='GeoJSON', encoding='gbk')
        except ValueError as e:
            print(e)
    return all_path_gdf


def parse_single_path(json_data: dict = None, path_seq: int = 0,
                      continuous_processing: bool = True, ignore_head_tail: bool = True,
                      generate_rod: bool = False, o_loc: tuple = None, min_rod_length: float = 1.5) -> gpd.GeoDataFrame:
    """
    从高德返回的路径规划结果中抽取第path_seq条路径
    注意: 同一条路径中, 会被切分为不同的段,一般来说, 同一路径, 相邻段之间是首尾相连的, 如果不相连, 这个就是路口的转向段.
    :param json_data: dict,
    :param path_seq: integer, 抽取OD对的第path_seq条路径
    :param continuous_processing: bool, 是否保持路径全程连续
    :param ignore_head_tail: bool, 是否忽略路径首尾的无名道路, 这种一般是小区内部道路
    :param generate_rod: bool, 是否生成连杆, 只有当参数ignore_head_tail为false时该参数才生效
    :param o_loc: 起点经纬度坐标
    :param min_rod_length
    :return: gpd.GeoDataFrame(), crs:EPSG:4326, scheme(路径方案编号), seq(路段序列), time_cost, toll, road_name, geometry
    """
    _coords = []
    path_line_list = []
    time_cost_list = []
    road_name_list = []
    tolls_list = []
    first_flag = True
    previous_path_item_end_point = None

    for item in json_data['route']['paths'][path_seq]['steps']:

        # 此分段的点序列
        point_list = [list(map(float, coords.split(','))) for coords in item['polyline'].split(';')]

        if first_flag:
            first_flag = False
            if not ignore_head_tail:
                if generate_rod:
                    assert o_loc is not None, '启用生成连杆模式要求传入起点经纬度坐标'
                    rod_l = point_distance(o_loc, point_list[0], crs_type='geo')
                    if rod_l >= min_rod_length:
                        path_line_list.append(LineString([o_loc, point_list[0]]))
                        time_cost_list.append(rod_l / 3.0)
                        road_name_list.append('道路连杆')
                        tolls_list.append(0.0)

        # 看看上个分段的终点和当前分段的起点距离, 如果距离较大, 说明上个分段和当前分段没有连续, 这部分一般都是路口转向线条
        if continuous_processing:
            if previous_path_item_end_point is not None:
                if distance((previous_path_item_end_point.y, previous_path_item_end_point.x),
                            (point_list[0][1], point_list[0][0])).m >= 0.5:
                    path_line_list.append(LineString([previous_path_item_end_point, Point(point_list[0])]))
                    time_cost_list.append(0)
                    road_name_list.append('路口转向')
                    tolls_list.append(0)
                else:
                    pass
        else:
            pass

        # 计算起点节点到终点节点的距离
        dis = distance((point_list[0][1], point_list[0][0]), (point_list[-1][1], point_list[-1][0])).m

        if dis <= 0.1:
            pass
        else:
            path_line = LineString(point_list)
            path_line_list.append(path_line)
            time_cost = item['cost']['duration']
            try:
                road_name = item['road_name']
            except KeyError as e:
                road_name = '无名道路'
            tolls = float(item['cost']['tolls'])

            time_cost_list.append(time_cost)
            road_name_list.append(road_name)
            tolls_list.append(tolls)

        # 记录此分段的最后一个点
        previous_path_item_end_point = Point(point_list[-1])

    # 原轨迹点
    path_gdf = gpd.GeoDataFrame({'time_cost': time_cost_list,
                                 'toll': tolls_list,
                                 'road_name': road_name_list,
                                 }, geometry=path_line_list, crs='EPSG:4326')
    if path_gdf.empty:
        pass
    else:
        origin_path_len = len(path_gdf)
        tail_del = False
        head_del = False

        if ignore_head_tail:
            # 如果全部是无名道路
            if len(path_gdf) == len(path_gdf[path_gdf['road_name'] == '无名道路']):
                return gpd.GeoDataFrame()

            # print(r'去除首尾的无名道路.....')
            target_index = find_continuous_repeat_index(road_name_list)
            if target_index is None or len(target_index) <= 1:
                pass
            else:
                # 从第一条路段开始就是无名道路, 且有连续值
                if target_index[0][0] == 0 and road_name_list[0] == '无名道路':
                    head_del = True
                    path_gdf.drop(index=target_index[0], inplace=True, axis=0)

                # 最后一条路段是无名道路, 且有连续值
                if target_index[-1][-1] == len(road_name_list) - 1 and road_name_list[-1] == '无名道路':
                    tail_del = True
                    path_gdf.drop(index=target_index[-1], inplace=True, axis=0)

            # 可能开头和尾部只有一个无名道路
            if not head_del and road_name_list[0] == '无名道路':
                path_gdf.drop(index=0, inplace=True, axis=0)

            if path_gdf.empty:
                pass
            else:
                if not tail_del and road_name_list[-1] == '无名道路':
                    path_gdf.drop(index=origin_path_len - 1, inplace=True, axis=0)

            path_gdf.reset_index(inplace=True, drop=True)
        path_gdf['seq'] = [i for i in range(1, len(path_gdf) + 1)]
        # scheme(路径方案编号), seq(路段序列), time_cost, toll, road_name, geometry
        # scheme唯一确定一条路径, (scheme, seq)唯一确定一个路段
    if not path_gdf.empty:
        try:
            path_gdf['geometry'] = path_gdf['geometry'].remove_repeated_points(1e-6)
        except:
            pass
    return path_gdf


def cut_slice(item_list: list[str] = None, slice_num: int = 2) -> list[list]:
    df = pd.DataFrame({'name': item_list})
    df['id'] = [i for i in range(len(df))]
    df['label'] = list(pd.cut(df['id'], bins=slice_num, labels=[i for i in range(0, slice_num)]))
    return [df[df['label'] == i]['name'].to_list() for i in range(0, slice_num)]
