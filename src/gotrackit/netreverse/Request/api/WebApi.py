# -- coding: utf-8 --
# @Time    : 2023/12/12 11:12
# @Author  : TangKai
# @Team    : ZheChengData

import json
import requests
import numpy as np

class GdRoutePlan(object):
    def __init__(self):
        pass

    @staticmethod
    def car_route_plan(od_id=None, origin=None, destination=None, key: str = None,
                       origin_id=None, destination_id=None,
                       origin_type=None, avoidpolygons=None,
                       waypoints_loc=None, strategy='32', is_rnd_strategy=False):
        """
        # 参数含义见: https://lbs.amap.com/api/webservice/guide/api/newroute
        :param key:
        :param origin:
        :param destination:
        :param origin_id:
        :param destination_id:
        :param origin_type:
        :param avoidpolygons:
        :param od_id:
        :param waypoints_loc:
        :param strategy:
        :param is_rnd_strategy: 是否启用随机策略
        :return:
        """
        api_url = 'https://restapi.amap.com/v5/direction/driving'
        strategy_list = ['0', '1', '2', '3', '32', '34', '35', '36', '37', '42']
        para_dict = {'key': key}
        if is_rnd_strategy:
            strategy = strategy_list[np.random.randint(0, len(strategy_list))]
        para_name = ['origin', 'destination', 'origin_id', 'destination_id', 'origin_type', 'avoidpolygons',
                     'waypoints', 'strategy']
        para_val = [origin, destination, origin_id, destination_id, origin_type, avoidpolygons, waypoints_loc,
                    str(strategy)]
        for name, val in zip(para_name, para_val):
            if para_val is not None:
                para_dict.update({name: val})
        para_dict.update({'show_fields': "cost,navi,tmcs,polyline"})
        # print(para_dict)
        # 请求
        try:
            r = requests.get(api_url, params=para_dict, timeout=10)
            json_data = json.loads(r.text)
            info_code = json_data['infocode']
        except:
            return None, None

        return json_data, int(info_code)

class BdTrafficSituation(object):
    def __init__(self, ak_list=None):
        self.ak_list = ak_list
        assert len(ak_list) >= 1, '至少有一个ak值'

    def rectangle_situation(self, bounds=None, coord_type_input=None, id_label=None,
                            coord_type_output=None, road_grade=None):
        """
        参数含义: https://lbsyun.baidu.com/faq/api?title=webapi/traffic-rectangleseek
        :param bounds:
        :param coord_type_input:
        :param coord_type_output:
        :param road_grade:
        :param id_label: str, 标记参数
        :return:
        """
        # 接口地址
        url = "https://api.map.baidu.com/traffic/v1/bound"
        ak = self.ak_list[np.random.randint(0, len(self.ak_list))]
        para_dict = {'ak': ak}
        para_name = ['bounds', 'coord_type_input', 'coord_type_output', 'road_grade']
        para_val = [bounds, coord_type_input, coord_type_output, road_grade]
        for name, val in zip(para_name, para_val):
            if para_val is not None:
                para_dict.update({name: val})
        print(para_dict)
        try:
            r = requests.get(url, params=para_dict, timeout=10)
            json_data = r.json()
            info_code = json_data['status']
        except:
            return None, None
        # info_code == 1, 服务内部错误
        # info_code == 302, 天配额超限，限制访问
        return json_data, info_code

