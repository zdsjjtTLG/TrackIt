# -- coding: utf-8 --
# @Time    : 2024/1/10 10:14
# @Author  : TangKai
# @Team    : ZheChengData


"""路网联通性修正"""
from tqdm import tqdm
import geopandas as gpd
from ...map.Net import Net
from ..GlobalVal import NetField
from ..book_mark import generate_book_mark
from .Tools.process import merge_double_link
from .SaveStreets.streets import modify_minimum

net_field = NetField()

dir_field = net_field.DIRECTION_FIELD
link_id_field = net_field.LINK_ID_FIELD
geometry_field = net_field.GEOMETRY_FIELD
from_node_field = net_field.FROM_NODE_FIELD
to_node_field = net_field.TO_NODE_FIELD
node_id_field = net_field.NODE_ID_FIELD
length_field = net_field.LENGTH_FIELD


class Conn(object):
    def __init__(self, check_buffer: float = 0.3, net: Net = Net):
        """

        :param check_buffer:
        :param net:
        """
        self.net = net
        self.buffer = check_buffer
        self.not_conn_df = None

        # the link status
        self.done_split_link = dict()

    def check(self, out_fldr: str = None, file_name: str = 'space_bookmarks', generate_mark: bool = False) -> None:
        # get link data and node data
        self.net.to_plane_prj()

        link_gdf = self.net.get_bilateral_link_data()
        node_gdf = self.net.get_node_data()
        link_gdf.reset_index(inplace=True, drop=True)
        node_gdf.reset_index(inplace=True, drop=True)
        # node -> buffer
        join_df = self.get_doubt_item(node_gdf=node_gdf, link_gdf=link_gdf, buffer=self.buffer)
        self.not_conn_df = join_df

        if self.not_conn_df is None or self.not_conn_df.empty:
            return None

        if generate_mark:
            if out_fldr is None:
                return None
            node_gdf.set_geometry(geometry_field, inplace=True, crs=self.net.planar_crs)
            node_gdf = node_gdf.to_crs(self.net.geo_crs)
            agg_df = join_df.groupby(node_id_field).agg({link_id_field: list}).reset_index(drop=False)
            node_gdf.set_index(node_id_field, inplace=True)
            conn_dict = {str(node) + '-' + ','.join(list(map(str, link_list))): (
                node_gdf.at[node, geometry_field].x, node_gdf.at[node, geometry_field].y)
                for link_list, node in zip(agg_df[link_id_field], agg_df[node_id_field])}
            node_gdf.reset_index(drop=False, inplace=True)
            generate_book_mark(input_fldr=out_fldr, prj_name=file_name, _mode='replace', name_loc_dict=conn_dict)

    @staticmethod
    def get_doubt_item(node_gdf: gpd.GeoDataFrame = None, link_gdf: gpd.GeoDataFrame = None, plain_crs: str = None,
                       buffer: float = 0.6):
        """
        node_gdf will be modified inplace
        :param node_gdf:
        :param link_gdf:
        :param plain_crs:
        :param buffer:
        :return:
        """
        node_gdf['buffer'] = node_gdf[geometry_field].apply(lambda geo: geo.buffer(buffer))
        node_gdf.set_geometry('buffer', inplace=True, crs=plain_crs)
        join_df = gpd.sjoin(node_gdf, link_gdf, how='left')
        join_df.reset_index(inplace=True, drop=True)
        join_df['doubt'] = join_df.apply(
            lambda item: 1 if item[node_id_field] not in [item[from_node_field], item[to_node_field]] else 0, axis=1)
        join_df.drop(index=join_df[join_df['doubt'] == 0].index, inplace=True, axis=0)
        join_df.dropna(subset=[link_id_field], axis=0, inplace=True)
        join_df[link_id_field] = join_df[link_id_field].astype(int)
        # 按照link_name再筛选一次, 避免将本来不连通的给修正为联通的, to_do
        join_df.reset_index(inplace=True, drop=True)
        return join_df

    def corrective_conn(self):
        """"""
        # 遍历
        flag = 0
        if self.not_conn_df is None or self.not_conn_df.empty:
            return None
        total_len = len(set(self.not_conn_df[node_id_field]))
        for split_node, n_link_gdf in tqdm(self.not_conn_df.groupby(node_id_field), desc='modify conn',
                                           total=total_len, ncols=100):
            if 'index_right' in n_link_gdf.columns:
                n_link_gdf.drop(columns='index_right', axis=1, inplace=True)
            if split_node not in self.net.get_node_data()[node_id_field]:
                continue
            if len(n_link_gdf) == 1:
                target_link = n_link_gdf[link_id_field].to_list()[0]
                if target_link in self.done_split_link.keys() and self.done_split_link[target_link] <= 1:
                    temp_node_gdf = gpd.GeoDataFrame({node_id_field: [split_node], geometry_field: [
                        self.net.get_node_geo(split_node)]}, geometry=geometry_field, crs=self.net.planar_crs)

                    join_gdf = self.get_doubt_item(node_gdf=temp_node_gdf,
                                                   link_gdf=self.net.get_bilateral_link_data().reset_index(
                                                       inplace=False, drop=True),
                                                   buffer=self.buffer)

                    self.done_split_link[target_link] += 1
                    self._corrective_conn(n_link_gdf=join_gdf, split_node=split_node)
                    self.done_split_link[target_link] = 1
                else:
                    self.split_and_adjust(split_node=split_node, corr_link_gdf=n_link_gdf)
            else:
                # more than one link
                self._corrective_conn(n_link_gdf=n_link_gdf, split_node=split_node)
                for l in n_link_gdf[link_id_field]:
                    self.done_split_link[l] = 1
            flag += 1

    def _corrective_conn(self, n_link_gdf: gpd.GeoDataFrame = None, split_node: int = None):
        if n_link_gdf.empty:
            return None

        for target_link, corr_single_link_gdf in n_link_gdf.groupby(link_id_field):

            corr_single_link_gdf = \
                gpd.GeoDataFrame(corr_single_link_gdf, crs=self.net.planar_crs, geometry=geometry_field)
            if node_id_field in corr_single_link_gdf.columns:
                corr_single_link_gdf.drop(columns=[node_id_field], inplace=True, axis=1)
            if target_link in self.done_split_link.keys() and self.done_split_link[target_link] <= 1:
                temp_node_gdf = gpd.GeoDataFrame({node_id_field: [split_node], geometry_field: [
                    self.net.get_node_geo(split_node)]}, geometry=geometry_field, crs=self.net.planar_crs)

                join_gdf = self.get_doubt_item(node_gdf=temp_node_gdf,
                                               link_gdf=self.net.get_bilateral_link_data().reset_index(drop=True,
                                                                                                       inplace=False))

                self.done_split_link[target_link] += 1
                if join_gdf.empty:
                    self.done_split_link[target_link] = 1
                else:
                    self._corrective_conn(n_link_gdf=join_gdf, split_node=split_node)

            else:
                self.split_and_adjust(split_node=split_node, corr_link_gdf=corr_single_link_gdf)

    def split_and_adjust(self, split_node: int = None, corr_link_gdf: gpd.GeoDataFrame = None):
        """

        :param split_node:
        :param corr_link_gdf: len() = 1
        :return:
        """
        # just split this link and get the target link
        target_link_id = list(corr_link_gdf[link_id_field])[0]
        split_node_geo = self.net.get_node_geo(split_node)

        split_ok, prj_p, modified_link, res_type = self.net.split_link(p=split_node_geo,
                                                                       target_link=target_link_id,
                                                                       omitted_length_threshold=0.5)
        if split_ok:
            self.net.modify_link_gdf(link_id_list=[modified_link[0]], attr_field_list=[to_node_field],
                                     val_list=[[split_node]])
            self.net.modify_link_gdf(link_id_list=[modified_link[1]], attr_field_list=[from_node_field],
                                     val_list=[[split_node]])
            self.net.renew_link_tail_geo(link_list=[modified_link[0]])
            self.net.renew_link_head_geo(link_list=[modified_link[1]])
        else:
            if res_type == 'head_beyond':
                to_del_node = self.net.get_link_from_to(target_link_id)[0]
                self.net.modify_link_gdf(link_id_list=[target_link_id], attr_field_list=[from_node_field],
                                         val_list=[[split_node]])
                self.net.renew_link_head_geo(link_list=[target_link_id])
            else:
                to_del_node = self.net.get_link_from_to(target_link_id)[1]
                self.net.modify_link_gdf(link_id_list=[target_link_id], attr_field_list=[to_node_field],
                                         val_list=[[split_node]])
                self.net.renew_link_tail_geo(link_list=[target_link_id])
            # try:
            #     print(rf'del {to_del_node}')
            #     self.net.del_nodes(node_list=[to_del_node])
            # except KeyError:
            #     # this node has already been deleted
            #     pass

        self.done_split_link[target_link_id] = 1

    def execute(self, out_fldr: str = None, file_name: str = 'space_bookmarks', generate_mark: bool = False,
                link_name_field: str = 'road_name') -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """

        :param out_fldr:
        :param file_name:
        :param generate_mark:
        :param link_name_field:
        :return: crs - EPSG:4326
        """
        # check the conn problem
        self.check(out_fldr=out_fldr, file_name=file_name, generate_mark=generate_mark)

        # modify conn problem
        self.corrective_conn()

        self.net.check_ln_consistency()

        # drop dup road
        self.net.drop_dup_ft_road()

        # merger_double_link
        # self.net.merger_double_link()

        link_gdf, node_gdf = self.net.get_bilateral_link_data(), self.net.get_node_data()

        link_gdf.reset_index(inplace=True, drop=True)
        node_gdf.reset_index(inplace=True, drop=True)
        link_gdf, node_gdf, _ = modify_minimum(plain_prj=self.net.planar_crs, node_gdf=node_gdf,
                                               link_gdf=link_gdf, buffer=0.3, ignore_merge_rule=True)
        link_gdf = merge_double_link(link_gdf=link_gdf)
        link_gdf = link_gdf.to_crs(self.net.geo_crs)
        node_gdf = node_gdf.to_crs(self.net.geo_crs)

        link_gdf.reset_index(inplace=True, drop=True)
        node_gdf.reset_index(inplace=True, drop=True)

        return link_gdf, node_gdf
