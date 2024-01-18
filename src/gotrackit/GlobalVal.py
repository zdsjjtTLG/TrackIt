# -- coding: utf-8 --
# @Time    : 2023/12/10 20:14
# @Author  : TangKai
# @Team    : ZheChengData

"""字段名称常量"""


class NetField(object):
    """路网字段"""
    def __init__(self):
        self.LINK_ID_FIELD = 'link_id'
        self.SINGLE_LINK_ID_FIELD = 'single_link_id'
        self.DIRECTION_FIELD = 'dir'
        self.FROM_NODE_FIELD = 'from_node'
        self.TO_NODE_FIELD = 'to_node'
        self.LENGTH_FIELD = 'length'
        self.GEOMETRY_FIELD = 'geometry'
        self.NODE_ID_FIELD = 'node_id'
        self.NODE_PATH_FIELD = 'node_path'
        self.COST_FIELD = 'cost'

class GpsField(object):
    """gps数据字段"""
    def __init__(self):
        self.POINT_SEQ_FIELD = 'seq'
        self.SUB_SEQ_FIELD = 'sub_seq'
        self.ORIGIN_POINT_SEQ_FIELD = 'origin_seq'
        self.TIME_FIELD = 'time'
        self.LNG_FIELD = 'lng'
        self.LAT_FIELD = 'lat'
        self.HEADING_FIELD = 'heading'
        self.AGENT_ID_FIELD = 'agent_id'
        self.TYPE_FIELD = 'type'
        self.NEXT_LINK_FIELD = 'next_link'


class MarkovField(object):
    """HMM模型字段"""
    def __init__(self):
        self.FROM_STATE = 'from_state'
        self.TO_STATE = 'to_state'
        self.ROUTE_LENGTH = 'route_l'
        self.ROUTE_ITEM = 'route_item'
        self.STRAIGHT_LENGTH = 'straight_l'
        self.DIS_GAP = 'dis_gap'
        self.PRJ_L = 'prj_dis'
