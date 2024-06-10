# -- coding: utf-8 --
# @Time    : 2024/1/27 17:40
# @Author  : TangKai
# @Team    : ZheChengData


"""请求路径数据"""

import pandas as pd
from ..GlobalVal import ODField
from .usage.gd_car_path import RequestOnTime

od_field = ODField()

od_id_field, o_x_field, o_y_field, d_x_field, d_y_field, hh_field, way_points_field = od_field.OD_ID_FIELD, \
    od_field.O_X_FIELD, od_field.O_Y_FIELD, od_field.D_X_FIELD, od_field.D_Y_FIELD, \
    od_field.HH_FIELD, od_field.WAY_POINTS_FIELD

class CarPath(object):
    def __init__(self, key_list: list[str] = None, input_file_path: str = None, od_df: pd.DataFrame = None,
                 cache_times: int = 300, ignore_hh: bool = True, out_fldr: str = None, file_flag: str = None,
                 log_fldr: str = None, save_log_file: bool = False, wait_until_recovery: bool = False):
        self.key_list = key_list
        self.input_file_path = input_file_path
        self.od_df = od_df
        self.cache_times = cache_times
        self.ignore_hh = ignore_hh
        self.out_fldr = out_fldr
        self.file_flag = file_flag
        self.log_fldr = log_fldr
        self.save_log_file = save_log_file
        self.wait_until_recovery = wait_until_recovery

    def get_path(self, remove_his: bool = True, is_rnd_strategy: bool = True, strategy: str = '32'):
        otr = RequestOnTime(key_list=self.key_list,
                            od_df=self.od_df,
                            input_file_path=self.input_file_path)
        if not self.ignore_hh:
            assert hh_field in self.od_df.columns, rf'如果启用了时段限制, 请确保od表中有{hh_field}字段, 否则请将ignore_hh参数设为True'
            self.od_df[hh_field] = self.od_df[hh_field].astype(int)
            assert set(self.od_df[hh_field]).issubset({i for i in range(0, 24)}), rf'{hh_field}字段的值有误!'
        key_info_dict = {k: 0 for k in self.key_list}
        while True:
            if_end_request, new_file_list = otr.start_request(out_fldr=self.out_fldr,
                                                              cache_times=self.cache_times,
                                                              id_field=od_id_field,
                                                              ignore_hh=self.ignore_hh,
                                                              file_flag=self.file_flag,
                                                              o_x_field=o_x_field,
                                                              o_y_field=o_y_field,
                                                              d_x_field=d_x_field, d_y_field=d_y_field,
                                                              way_points_field=way_points_field,
                                                              request_hh_field=hh_field,
                                                              log_fldr=self.log_fldr,
                                                              remove_his=remove_his,
                                                              save_log_file=self.save_log_file,
                                                              key_info_dict=key_info_dict,
                                                              is_rnd_strategy=is_rnd_strategy, strategy=strategy,
                                                              wait_until_recovery=self.wait_until_recovery)
            if if_end_request:
                break
            else:
                pass
        return if_end_request, new_file_list
