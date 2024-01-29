# -- coding: utf-8 --
# @Time    : 2023/12/12 15:58
# @Author  : TangKai
# @Team    : ZheChengData

import os
import sys
import time
import pickle
import logging
import pandas as pd
from ..api.WebApi import BdTrafficSituation


class RequestTs(object):
    def __init__(self, key_list=None, input_file_path=None):
        self.key_list = key_list
        self.grid_df = pd.read_csv(input_file_path)

    def start_request(self, out_fldr=None, cache_times=2000, id_field=None,
                      file_flag=None, bounds_field='geometry', road_grade_field=None, rest_seconds=2000):
        """
        给一个od表, 按照时间进行请求
        :param out_fldr:
        :param cache_times:
        :param file_flag:
        :param id_field:
        :param bounds_field:
        :param road_grade_field:
        :param rest_seconds:
        :return:
        """
        # 1. 配置日志输出
        prj_root_fldr = os.getcwd()
        if 'log' not in os.listdir(os.path.join(prj_root_fldr, 'data/output/')):
            os.makedirs(os.path.join(prj_root_fldr, 'data/output/log/'))

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
        console_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(os.path.join(prj_root_fldr, fr'data/output/log/log_request_{file_flag}.log'),
                                           mode='a')
        file_handler.setFormatter(
            logging.Formatter('[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
        file_handler.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                            handlers=[file_handler, console_handler])
        logging.info(rf'{file_flag}_logging_info:.....')
        # 日志输出 #

        # 恢复请求状态
        num_list = []
        for file_name in os.listdir(out_fldr):
            if file_name.startswith(rf'{file_flag}'):
                num_list.append(int(file_name.split('_')[-1]))
        if num_list:
            last_num = max(num_list)
        else:
            last_num = 0

        # 读取已经请求的grid_id
        try:
            with open(os.path.join(out_fldr, 'already_request_list'), 'rb') as f:
                already_request_list = pickle.load(f)
        except:
            already_request_list = []

        logging.info(rf'请求{file_flag}, 第{last_num + 1}个文件...')
        logging.info(rf'already_request_list_len: {len(already_request_list)}')

        continue_request = True

        # 态势请求对象
        baidu_ts = BdTrafficSituation(ak_list=self.key_list)

        # 开始请求数据
        while continue_request:
            # 删除已经请求的......
            self.grid_df.drop(index=self.grid_df[self.grid_df[id_field].isin(already_request_list)].index, inplace=True,
                              axis=0)
            self.grid_df.reset_index(drop=True, inplace=True)
            logging.info(rf'尚且未请求的有{len(self.grid_df)}条')

            # 如果为空, 说明请求完毕了
            if self.grid_df.empty:
                logging.info('全部请求完毕...')
                return True

            # 开始请求
            grid_ts_dict, done_list, is_end = self.request_ts(request_df=self.grid_df, cache_times=cache_times,
                                                              bounds_field=bounds_field,
                                                              road_grade_field=road_grade_field,
                                                              id_field=id_field, request_ts_obj=baidu_ts)

            already_request_list.extend(done_list)
            last_num += 1
            self.save_file(out_fldr=out_fldr, file_flag=file_flag, last_num=last_num,
                           od_route_dict=grid_ts_dict, already_request_list=already_request_list)

            # 配额达到上限
            if is_end:
                logging.info(rf'达到配额上线, 程序停止{rest_seconds}秒...')
                time.sleep(rest_seconds)


    @staticmethod
    def request_ts(request_df=None, cache_times=None,
                   id_field=None, request_ts_obj=None, bounds_field=None, road_grade_field=None):
        """请求给定栅格, 每达到cache_times次(或者提前请求完或者达到配额上限)就返回"""
        _count = 0  # 用于计数
        res_dict = dict()  # 用于存储请求结果
        done_list = []  # 用于存储请求成功的grid_id

        limit_num = 0  # 用于记录达到上限的次数

        unknown_error_num = 0  # 记录非网络问题错误的grid数目
        unknown_error_list = []  # 记录非网络问题错误的grid_id

        http_error_num = 0  # 记录由于网络连接失败的grid数目
        http_error_list = []  # 记录由于网络连接失败的grid_id

        for _, row in request_df.iterrows():
            grid_id = int(row[id_field])
            bounds = row[bounds_field]
            road_grade = str(row[road_grade_field])

            # 请求
            json_data, info_code = request_ts_obj.rectangle_situation(
                bounds=bounds,
                coord_type_input="gcj02", id_label=grid_id,
                coord_type_output="gcj02",
                road_grade=road_grade)

            logging.info(rf'info_code: {info_code}, count: {_count}, grid_id: {grid_id}')

            # 解析数据
            if info_code is not None:
                # 成功返回
                if info_code == 0:
                    res_dict[(grid_id, int(road_grade))] = json_data
                    done_list.append(grid_id)
                    _count += 1
                elif info_code == 302:
                    limit_num += 1
                    if limit_num >= 30:
                        logging.info(r'达到配额上限,结束请求')
                        return res_dict, done_list, True
                elif info_code == 1:
                    logging.info(r'服务内部错误')
                    unknown_error_num += 1
                    unknown_error_list.append(grid_id)
                    # done_list.append(grid_id)
                    _count += 1
            else:
                # 网络错误
                http_error_num += 1
                http_error_list.append(grid_id)

            # 达到请求次数后缓存一次
            if _count >= cache_times:
                logging.info(rf'达到缓存次数..., 返回')
                logging.info(rf'非网络问题导致的失败od数: {len(unknown_error_list)}')
                logging.info(rf'https网络导致的失败od数: {len(http_error_list)}')
                return res_dict, done_list, False

        # 既没有达到缓存次数也没有时段超限
        logging.info(rf'请求完毕, 没有达到配额上限也没有达到缓存次数..., 返回')
        logging.info(rf'非网络问题导致的失败od数: {len(unknown_error_list)}')
        logging.info(rf'https网络导致的失败od数: {len(http_error_list)}')
        return res_dict, done_list, False

    @staticmethod
    def save_file(out_fldr=None, file_flag=None, last_num=None,
                  od_route_dict=None, already_request_list=None):
        with open(os.path.join(out_fldr, rf'{file_flag}_{last_num}'), 'wb') as f:
            pickle.dump(od_route_dict, f)
        with open(os.path.join(out_fldr, 'already_request_list'), 'wb') as f:
            pickle.dump(already_request_list, f)
