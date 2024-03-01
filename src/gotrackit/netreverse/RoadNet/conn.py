# -- coding: utf-8 --
# @Time    : 2024/1/10 10:14
# @Author  : TangKai
# @Team    : ZheChengData


"""路网联通性修正"""
import time
import swifter
import geopandas as gpd
from ...map.Net import Net
from ..GlobalVal import NetField
from ..book_mark import generate_book_mark

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

    def check(self, out_fldr: str = None, file_name: str = 'space_bookmarks', generate_mark: bool = False):
        # get link data and node data
        self.net.to_plane_prj()

        link_gdf = self.net.get_bilateral_link_data()
        node_gdf = self.net.get_node_data()

        # node -> buffer
        node_gdf['buffer'] = node_gdf[geometry_field].apply(lambda geo: geo.buffer(self.buffer))
        node_gdf.set_geometry('buffer', inplace=True, crs=self.net.plane_crs)

        # sjoin with link and check
        join_df = gpd.sjoin(node_gdf, link_gdf, how='left')
        join_df.reset_index(inplace=True, drop=True)
        a = time.time()
        join_df['doubt'] = join_df.apply(
            lambda item: 1 if item[node_id_field] not in [item[from_node_field], item[to_node_field]] else 0, axis=1)
        print(time.time() - a)
        time.sleep(100)
        join_df.drop(index=join_df[join_df['doubt'] == 0].index, inplace=True, axis=0)
        join_df.reset_index(inplace=True, drop=True)
        self.not_conn_df = join_df

        if generate_mark:
            node_gdf.set_geometry(geometry_field, inplace=True, crs=self.net.plane_crs)
            node_gdf = node_gdf.to_crs(self.net.geo_crs)
            agg_df = join_df.groupby(node_id_field).agg({link_id_field: list}).reset_index(drop=False)
            conn_dict = {str(node) + '-' + ','.join(list(map(str, link_list))): (
                node_gdf.at[node, geometry_field].x, node_gdf.at[node, geometry_field].y)
                for link_list, node in zip(agg_df[link_id_field], agg_df[node_id_field])}
            generate_book_mark(input_fldr=out_fldr, prj_name=file_name, _mode='replace', name_loc_dict=conn_dict)

    def corrective_conn(self):
        """"""
        # 遍历
        done_split_link = set()
        for split_node, n_link_gdf in self.not_conn_df.groupby(node_id_field):
            if len(n_link_gdf) == 1:
                # one link
                # just split this link
                # get the target link
                target_link_id = list(n_link_gdf[link_id_field])[0]
                split_node_geo = self.net.get_node_geo(split_node)

                split_ok, prj_p = self.net.split_link(p=split_node_geo, target_link=target_link_id, generate_node=False,
                                                      manually_specify_node_id=True, new_node_id=split_node,
                                                      omitted_length_threshold=0.5)
                if split_ok:
                    self.net.modify_node_gdf(node_id_list=[split_node], attr_field_list=[geometry_field],
                                             val_list=[[prj_p]])
                    done_split_link.add(target_link_id)
                else:
                    pass
            else:
                # more than one link
                pass

