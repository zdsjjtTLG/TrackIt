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
        self.ROAD_NAME_FIELD = 'road_name'


class RegionField(object):
    """路网字段"""
    def __init__(self):
        self.REGION_ID_FIELD = 'region_id'
        self.GEO_FIELD = 'geometry'


class ODField(object):
    """路网字段"""
    def __init__(self):
        self.O_X_FIELD = 'o_x'
        self.O_Y_FIELD = 'o_y'
        self.D_X_FIELD = 'd_x'
        self.D_Y_FIELD = 'd_y'
        self.OD_ID_FIELD = 'od_id'
        self.WAY_POINTS_FIELD = 'way_points'
        self.HH_FIELD = 'hh'


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
        self.GEOMETRY_FIELD = 'geometry'

