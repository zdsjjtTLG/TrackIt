# -- coding: utf-8 --
# @Time    : 2024/5/13 15:13
# @Author  : TangKai
# @Team    : ZheChengData


class ParaGrid(object):
    def __init__(self, beta_list: list[float] = None,
                 gps_sigma_list: list[float] = None, use_heading_inf_list: list[bool] = None,
                 omitted_l_list: list[float] = None):
        self.beta_list = beta_list if beta_list is not None else [5.0, 6.0]
        self.gps_sigma_list = gps_sigma_list if gps_sigma_list is not None else [30.0, 50.0]
        self.use_heading_inf_list = sorted(use_heading_inf_list,
                                           reverse=True) if use_heading_inf_list is not None else [True, False]
        assert len(self.use_heading_inf_list) <= 2
        assert len(set(self.use_heading_inf_list)) == len(self.use_heading_inf_list)
        assert set(self.use_heading_inf_list).issubset({True, False})
        self.omitted_l_list = sorted(omitted_l_list) if omitted_l_list is not None else [6.0]
        self.__search_res = list()
        self.transit_res = dict()
        self.emission_res = dict()

        self.transit_res = {i: {'parameter': {'beta': beta}, 'res': {}} for i, beta in enumerate(self.beta_list)}
        if False in self.use_heading_inf_list:
            self.emission_res = {j: {'parameter': {'use_heading_inf': False,
                                                   'gps_sigma': gps_sigma,
                                                   'omitted_l': 1.0}, 'res': {}} for j, gps_sigma in
                                 enumerate(self.gps_sigma_list)}
        gap = set(self.use_heading_inf_list) - {False}
        if gap:
            self.emission_res.update({(j, m): {'parameter': {'use_heading_inf': True, 'gps_sigma': gps_sigma,
                                                             'omitted_l': omitted_l}, 'res': {}} for j, gps_sigma in
                                      enumerate(self.gps_sigma_list) for m, omitted_l in
                                      enumerate(self.omitted_l_list)})

    def init_para_grid(self):
        for k in self.transit_res.keys():
            self.transit_res[k]['res'] = dict()
        for k in self.emission_res.keys():
            self.emission_res[k]['res'] = dict()

    def update_res(self, res: dict = None):
        self.__search_res.append(res)

    @property
    def search_res(self) -> list:
        return self.__search_res

