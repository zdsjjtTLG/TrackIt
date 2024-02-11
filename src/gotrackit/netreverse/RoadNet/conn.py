# -- coding: utf-8 --
# @Time    : 2024/1/10 10:14
# @Author  : TangKai
# @Team    : ZheChengData


"""路网联通性修正"""


import geopandas as gpd
from ...map.Net import Net
from ..GlobalVal import NetField

net_field = NetField()


dir_field = net_field.DIRECTION_FIELD
link_id_field = net_field.LINK_ID_FIELD
geometry_field = net_field.GEOMETRY_FIELD
from_node_field = net_field.FROM_NODE_FIELD
to_node_field = net_field.TO_NODE_FIELD
node_id_field = net_field.NODE_ID_FIELD

class Conn(object):
    def __init__(self, check_buffer: float = 0.3, net: Net = Net):
        self.net = net
        self.buffer = check_buffer
        self.not_conn_df = None

    def check(self):
        # get link data and node data
        self.net.to_plane_prj()

        link_gdf = self.net.get_bilateral_link_data()
        node_gdf = self.net.get_node_data()
        node_gdf.reset_index(inplace=True, drop=False)

        # node -> buffer
        node_gdf['buffer'] = node_gdf[geometry_field].apply(lambda geo: geo.buffer(self.buffer))
        node_gdf.set_geometry('buffer', inplace=True, crs=self.net.plane_crs)

        # sjoin with link and check
        join_df = gpd.sjoin(node_gdf, link_gdf, how='left')
        join_df.reset_index(inplace=True, drop=True)
        join_df['doubt'] = join_df.apply(
            lambda item: 1 if item[node_id_field] not in [item[from_node_field], item[to_node_field]] else 0, axis=1)
        join_df.drop(index=join_df[join_df['doubt'] == 0].index, inplace=True, axis=0)
        join_df.reset_index(inplace=True, drop=True)
        self.not_conn_df = join_df

    def corrective_conn(self):
        pass

