# -- coding: utf-8 --
# @Time    : 2023/12/10 20:14
# @Author  : TangKai
# @Team    : ZheChengData

"""字段名称常量"""
import copy


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
        self.S_NODE = 'o_node'
        self.T_NODE = 'd_node'
        self.NODE_PATH_FIELD = 'path'
        self.COST_FIELD = 'cost'
        self.LINK_VEC_FIELD = 'dir_vec'
        self.X_DIFF = 'lv_dx'
        self.Y_DIFF = 'lv_dy'
        self.VEC_LEN = 'lvl'
        self.SEG_COUNT = 'seg_count'
        self.SEG_ACCU_LENGTH = 'seg_accu_length'
        self.GRID_ID = 'grid_id'

class GpsField(object):
    """gps数据字段"""
    def __init__(self):
        self.POINT_SEQ_FIELD = 'seq'
        self.LOC_TYPE = 'loc_type'
        self.SUB_SEQ_FIELD = 'sub_seq'
        self.ORIGIN_POINT_SEQ_FIELD = '__ori_seq'
        self.TIME_FIELD = 'time'
        self.LNG_FIELD = 'lng'
        self.LAT_FIELD = 'lat'
        self.HEADING_FIELD = 'heading'
        self.AGENT_ID_FIELD = 'agent_id'
        self.PRE_AGENT_ID_FIELD = 'pre_agent'
        self.ORIGIN_AGENT_ID_FIELD = 'origin_agent_id'
        self.TYPE_FIELD = 'type'
        self.NEXT_LINK_FIELD = 'next_link'
        self.NEXT_SINGLE = 'next_single'
        self.GEOMETRY_FIELD = 'geometry'
        self.FROM_GPS_SEQ = 'from_seq'
        self.TO_GPS_SEQ = 'to_seq'

        self.GROUP_FIELD = 'group'
        self.SUB_GROUP_FIELD = 'sub_group'
        self.NEXT_P = 'next_p'
        self.PRE_P = 'pre_p'
        self.NEXT_SEQ = 'next_seq'
        self.NEXT_TIME = 'next_time'
        self.PRE_TIME = 'pre_time'
        self.ADJ_TIME_GAP = 'time_gap'
        self.ADJ_DIS = 'gps_adj_dis'
        self.ADJ_X_DIS = 'gps_adj_xl'
        self.ADJ_Y_DIS = 'gps_adj_yl'
        self.ADJ_SPEED = 'adj_speed'

        self.DENSE_GEO = '__dens_geo__'
        self.N_SEGMENTS = '__n__'

        self.X_DIFF = 'gv_dx'
        self.Y_DIFF = 'gv_dy'
        self.VEC_LEN = 'gvl'

        self.PLAIN_X = 'prj_x'
        self.PLAIN_Y = 'prj_y'

        self.PRE_PLAIN_X = 'pre_prj_x'
        self.PRE_PLAIN_Y = 'pre_prj_y'

        self.SPEED_FIELD = 'speed'
        self.X_SPEED_FIELD = 'x_speed'
        self.Y_SPEED_FIELD = 'y_speed'


class RouteField(object):
    """"""
    def __init__(self):
        self.PATH_ID_FIELD = 'path_id'
        self.TIME_COST_FIELD = 'time_cost'
        self.SEQ_FIELD = 'seq'
        self.O_TIME_FIELD = 'o_time'


class MarkovField(object):
    """HMM模型字段"""
    def __init__(self):
        self.FROM_STATE = 'from_state'
        self.TO_STATE = 'to_state'
        self.FROM_SEQ = 'from_seq'
        self.TO_SEQ = 'to_seq'
        self.FROM_STATE_N = 'from_state_node'
        self.TO_STATE_N = 'to_state_node'
        self.ROUTE_LENGTH = 'route_l'
        self.ROUTE_ITEM = 'route_item'
        self.STRAIGHT_LENGTH = 'straight_l'
        self.DIS_GAP = 'dis_gap'
        self.PRJ_L = 'prj_dis'
        self.PRJ_GEO = 'prj_p'
        self.DIS_TO_NEXT = 'dis_to_next'
        self.HEADING_GAP = 'heading_gap'
        self.MATCH_HEADING = 'match_heading'
        self.USED_HEADING_GAP = 'used_heading_gap'
        self.EMISSION_P = 'emission_p'
        self.DRIVING_L = 'route_dis'


class OdField(object):
    """路网字段"""

    def __init__(self):
        self.OD_ID_FIELD = 'od_id'
        self.WAYPOINTS_FIELD = 'way_points'
        self.OX_FIELD = 'o_x'
        self.OY_FIELD = 'o_y'
        self.DX_FIELD = 'd_x'
        self.DY_FIELD = 'd_y'


class KeplerConfig(object):
    def __init__(self):

        self.__BASE_LAYER_CONFIG = {
            'id': 'kb29zg',
            'type': 'geojson',
            'config': {
                'dataId': 'mix',
                'label': 'mix',
                'color': [
                    18,
                    147,
                    154
                ],
                'highlightColor': [
                    252,
                    242,
                    26,
                    255
                ],
                'columns': {
                    'geojson': 'geometry'
                },
                'isVisible': True,
                'visConfig': {
                    'opacity': 0.8,
                    'strokeOpacity': 0.8,
                    'thickness': 0.1,
                    'strokeColor': [
                        221,
                        178,
                        124
                    ],
                    'colorRange': {},
                    'strokeColorRange': {
                        'name': 'Global Warming',
                        'type': 'sequential',
                        'category': 'Uber',
                        'colors': [
                            '#5A1846',
                            '#900C3F',
                            '#C70039',
                            '#E3611C',
                            '#F1920E',
                            '#FFC300'
                        ]
                    },
                    'radius': 3,
                    'sizeRange': [
                        0,
                        10
                    ],
                    'radiusRange': [
                        0,
                        50
                    ],
                    'heightRange': [
                        0,
                        500
                    ],
                    'elevationScale': 5,
                    'enableElevationZoomFactor': True,
                    'stroked': False,
                    'filled': True,
                    'enable3d': False,
                    'wireframe': False
                },
                'hidden': False,
                'textLabel': [
                    {
                        'field': None,
                        'color': [
                            255,
                            255,
                            255
                        ],
                        'size': 18,
                        'offset': [
                            0,
                            0
                        ],
                        'anchor': 'start',
                        'alignment': 'center'
                    }
                ]
            },
            'visualChannels': {
                'colorField': {},
                'colorScale': 'ordinal',
                'strokeColorField': None,
                'strokeColorScale': 'quantile',
                'sizeField': None,
                'sizeScale': 'linear',
                'heightField': None,
                'heightScale': 'linear',
                'radiusField': None,
                'radiusScale': 'linear'
            }
        }

        self.__GLB_MAP_CONFIG = {
            'version': 'v1',
            'config': {
                'visState': {
                    'filters': [],
                    'layers': [],
                    'interactionConfig': {
                        'tooltip': {
                            'fieldsToShow': {
                            },
                            'compareMode': False,
                            'compareType': 'absolute',
                            'enabled': True
                        },
                        'brush': {
                            'size': 2.7,
                            'enabled': False
                        },
                        'geocoder': {
                            'enabled': False
                        },
                        'coordinate': {
                            'enabled': True
                        }
                    },
                    'layerBlending': 'normal',
                    'splitMaps': [],
                    'animationConfig': {
                        'currentTime': None,
                        'speed': 0.5
                    }
                },
                'mapState': {
                    'bearing': 24,
                    'dragRotate': True,
                    'latitude': 34.23188690328708,
                    'longitude': 108.94371457924842,
                    'pitch': 50,
                    'zoom': 15,
                    'isSplit': False
                },
                'mapStyle': {
                    'styleType': 'dark',
                    'topLayerGroups': {},
                    'visibleLayerGroups': {
                        'label': True,
                        'road': False,
                        'border': False,
                        'building': True,
                        'water': True,
                        'land': True,
                        '3d building': False
                    },
                    'threeDBuildingColor': [
                        9.665468314072013,
                        17.18305478057247,
                        31.1442867897876
                    ],
                    'mapStyles': {}
                }
            }
        }

        self.__TIME_FILTER_CONFIG = {
            'id': 'fd18q2cbg',
            'dataId': [
                'mix'
            ],
            'name': [
                'time'
            ],
            'type': 'timeRange',
            'value': [
                1652372040.0,
                1652373265.0
            ],
            'enlarged': False,
            'plotType': 'histogram',
            'animationWindow': 'incremental',
            'yAxis': None,
            'speed': 0.2
        }

        self.BASE_LINK_NAME = 'base_link'
        self.BASE_NODE_NAME = 'base_node'
        self.ERROR_XFER = 'error_xfer'
        self.MIX_NAME = 'mix'
        self.GPS_NAME = 'gps'
        self.MATCH_LINK_NAME = 'match_link'
        self.TRAJECTORY_NAME = 'trajectory'

    def get_glb_map_config(self) -> dict:
        return copy.deepcopy(self.__GLB_MAP_CONFIG)

    def get_base_layer_config(self) -> dict:
        return copy.deepcopy(self.__BASE_LAYER_CONFIG)

    def get_time_filter_config(self) -> dict:
        return copy.deepcopy(self.__TIME_FILTER_CONFIG)


class PrjConst(object):
    def __init__(self):
        self.PRJ_CRS = 'EPSG:4326'


class ColorMap(object):
    def __init__(self):
        self.color = {'red': [255, 0, 0], 'yellow': [255, 255, 0], 'blue': [0, 0, 255],
                      'cyan': [0, 255, 255], 'pink': [255, 192, 203], 'purple': [255, 192, 203]}
