# -- coding: utf-8 --
# @Time    : 2024/5/13 15:13
# @Author  : TangKai
# @Team    : ZheChengData


class ParaGrid(object):
    def __init__(self, beta_list: list[float] = None,
                 gps_sigma_list: list[float] = None, use_heading_inf_list: list[bool] = None,
                 omitted_l_list: list[float] = None):
        self.beta_list = beta_list if beta_list is not None else [6.0, 10.0]
        self.gps_sigma_list = gps_sigma_list if gps_sigma_list is not None else [30.0, 40.0]
        self.use_heading_inf_list = sorted(use_heading_inf_list,
                                           reverse=True) if use_heading_inf_list is not None else [True, False]
        assert len(self.use_heading_inf_list) <= 2
        assert len(set(self.use_heading_inf_list)) == len(self.use_heading_inf_list)
        assert set(self.use_heading_inf_list).issubset({True, False})
        self.omitted_l_list = sorted(omitted_l_list) if omitted_l_list is not None else [6.0]
