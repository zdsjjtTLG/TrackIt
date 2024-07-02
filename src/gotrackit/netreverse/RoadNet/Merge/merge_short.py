# -- coding: utf-8 --
# @Time    : 2023/12/14 13:54
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
import networkx as nx
import geopandas as gpd
from ...GlobalVal import NetField
from .merge_links import merge_links
from shapely.geometry import Point, LineString
from .limit.direction_limit import limit_direction
from ...PublicTools.GeoProcess import calc_link_angle
from ..Merge.get_merged_link_seq import build_graph_from_link


glb_field = NetField()


class MergeShort(object):
    def __init__(self, link_gdf: gpd.GeoDataFrame = None, node_gdf: gpd.GeoDataFrame = None,
                 link_path: str = None, node_path: str = None, utm_crs: str = 'EPSG:32650',
                 judge_col_name: str = 'road_name',  threshold: float = 20.0, max_angle: float = 45.0):
        """

        :param threshold:
        :param link_gdf:
        :param node_gdf:
        :param link_path:
        :param node_path:
        :param utm_crs:
        :param judge_col_name:
        """
        self.threshold = threshold
        self.max_angle = max_angle
        self.link_gdf = link_gdf
        self.node_gdf = node_gdf
        if self.link_gdf is None:
            self.link_gdf = gpd.read_file(link_path)
        self.link_gdf.drop_duplicates(subset=[glb_field.FROM_NODE_FIELD, glb_field.TO_NODE_FIELD], keep='first', inplace=True)
        if self.node_gdf is None:
            self.node_gdf = gpd.read_file(node_path)
        for field in [glb_field.LINK_ID_FIELD, glb_field.DIRECTION_FIELD,
                      glb_field.FROM_NODE_FIELD, glb_field.TO_NODE_FIELD]:
            self.link_gdf[field] = self.link_gdf[field].astype(int)
        self.judge_col_name = judge_col_name
        self.merge_link_df = pd.DataFrame()
        self.utm_crs = utm_crs
        self.ud_graph, self.d_graph = None, None
        self.sort_ft_dir_dict = {tuple(sorted([f, t])): direction for f, t, direction in
                                 zip(self.link_gdf[glb_field.FROM_NODE_FIELD],
                                     self.link_gdf[glb_field.TO_NODE_FIELD],
                                     self.link_gdf[glb_field.DIRECTION_FIELD])}
        self.link_dir_dict = {link: direction for link, direction in
                              zip(self.link_gdf[glb_field.LINK_ID_FIELD],
                                  self.link_gdf[glb_field.DIRECTION_FIELD])}

    def get_merged_link_seq(self) -> None:
        # 先找出length小于threshold的link的索引
        short_link_df = self.link_gdf[(self.link_gdf[glb_field.LENGTH_FIELD] <= self.threshold) &
                                      (self.link_gdf['road_name'] != '路口转向')].copy()
        self.link_gdf.set_index(glb_field.LINK_ID_FIELD, inplace=True)
        self.node_gdf.set_index(glb_field.NODE_ID_FIELD, inplace=True)

        # 建无向图, 得到节点的度
        self.ud_graph, self.d_graph = build_graph_from_link(
            link_df=self.link_gdf[[glb_field.FROM_NODE_FIELD, glb_field.TO_NODE_FIELD, glb_field.DIRECTION_FIELD]],
            from_col_name=glb_field.FROM_NODE_FIELD, to_col_name=glb_field.TO_NODE_FIELD,
            ignore_dir=False, dir_col=glb_field.DIRECTION_FIELD)
        ud_degree_dict = dict(nx.degree(self.ud_graph))

        # 只合并非路口转向
        merge_group = []  # [(link1, link2), (link13, link21), ...]
        merge_group_seq_dict = dict()
        for _, short_link in short_link_df.iterrows():
            # 先检查short_link两端的节点的度
            # 节点两端至少有一个节点的无向图度为2
            if ud_degree_dict[short_link[glb_field.FROM_NODE_FIELD]] != 2 and ud_degree_dict[short_link[glb_field.TO_NODE_FIELD]] != 2:
                continue
            else:
                # 现在两端的节点至少有一个节点的度是2
                short_link_id = short_link[glb_field.LINK_ID_FIELD]
                if ud_degree_dict[short_link[glb_field.FROM_NODE_FIELD]] == 2:
                    merge_node = short_link[glb_field.FROM_NODE_FIELD]  # 和上游link合并
                    pre_link = set(self.link_gdf[(self.link_gdf[glb_field.FROM_NODE_FIELD] == merge_node) |
                                                 (self.link_gdf[glb_field.TO_NODE_FIELD] == merge_node)].index)
                    if len(pre_link) != 2:
                        continue
                    # 初筛方向
                    if set(self.link_gdf.loc[list(pre_link), :][glb_field.DIRECTION_FIELD]) == {0, 1}:
                        continue

                    pre_link = list(pre_link - {short_link_id})[0]
                    seq_list = self.judge_merge([pre_link, short_link_id])
                    if seq_list:
                        k = tuple(sorted((pre_link, short_link_id)))
                        merge_group.append(k)
                        merge_group_seq_dict.update({k: seq_list})
                        continue

                if ud_degree_dict[short_link[glb_field.TO_NODE_FIELD]] == 2:
                    merge_node = short_link[glb_field.TO_NODE_FIELD]  # 和下游节点合并
                    next_link = set(self.link_gdf[(self.link_gdf[glb_field.FROM_NODE_FIELD] == merge_node) |
                                                  (self.link_gdf[glb_field.TO_NODE_FIELD] == merge_node)].index)
                    if len(next_link) != 2:
                        print(next_link)
                        continue
                    # 初筛方向
                    if set(self.link_gdf.loc[list(next_link), :][glb_field.DIRECTION_FIELD]) == {0, 1}:
                        continue

                    next_link = list(next_link - {short_link_id})[0]
                    seq_list = self.judge_merge([short_link_id, next_link])
                    if seq_list:
                        k = tuple(sorted((short_link_id, next_link)))
                        merge_group.append(k)
                        merge_group_seq_dict.update({k: seq_list})
                        continue

        # list[set]
        merge_group = self.cluster_graph(edge_list=merge_group)

        # 确定走向
        final_group_seq_list = []
        group_id_list = []
        group_id = 0
        for group in merge_group:
            group_id += 1
            # 没有其他的短路段合并到同一个base_link上
            if len(group) == 2:
                group_seq = merge_group_seq_dict[tuple(sorted(group))]
                final_group_seq_list.append(group_seq)
            else:
                # 有其他的短路段合并到同一个base_link上
                node_seq = self.get_ud_chain_seq_list(
                    link_gdf=self.link_gdf.loc[list(group), :])
                group_seq = [tuple(sorted([node_seq[i], node_seq[i + 1]])) for i in range(0, len(node_seq) - 1)]
                final_group_seq_list.append(group_seq)
            group_id_list.append(group_id)
        merge_link_df = pd.DataFrame({'group': group_id_list})
        merge_link_df['link_seq'] = final_group_seq_list
        self.link_gdf.reset_index(inplace=True, drop=False)
        self.node_gdf.reset_index(inplace=True, drop=False)

        # 精筛方向
        merge_link_df = limit_direction(merged_df=merge_link_df,
                                        origin_graph_degree_dict=ud_degree_dict, link_df=self.link_gdf)

        self.merge_link_df = merge_link_df

    def judge_merge(self, link_list: list[int]) -> list[tuple]:
        """
        依据限制字段和转角是否可以合并
        :param link_list: 长度一定为2
        :return:
        """
        link1, link2 = link_list[0], link_list[1]

        # 1.属性名称限制
        if self.link_gdf.at[link1, self.judge_col_name] != self.link_gdf.at[link2, self.judge_col_name]:
            return []

        # 联通性限制
        edge_list = []
        d_g, u_g = nx.DiGraph(), nx.Graph()
        used_link_gdf = self.link_gdf.loc[link_list, :]
        for _, row in used_link_gdf.iterrows():
            if int(row[glb_field.DIRECTION_FIELD]) == 0:
                edge_list.append((row[glb_field.FROM_NODE_FIELD], row[glb_field.TO_NODE_FIELD]))
                edge_list.append((row[glb_field.TO_NODE_FIELD], row[glb_field.FROM_NODE_FIELD]))
            else:
                edge_list.append((row[glb_field.FROM_NODE_FIELD], row[glb_field.TO_NODE_FIELD]))
        d_g.add_edges_from(edge_list)
        u_g.add_edges_from(edge_list)
        u_g_dict = dict(u_g.degree)
        se_node = [i for i in u_g_dict.keys() if u_g_dict[i] == 1]
        if not nx.has_path(d_g, se_node[0], se_node[1]) and not nx.has_path(d_g, se_node[1], se_node[0]):
            return []
        seq_list = nx.dijkstra_path(u_g, se_node[0], se_node[1])

        if 332427 in seq_list:
            print(link_list)
        angle = self.calc_polyline_angle(link_geo1=self.link_gdf.at[link1, glb_field.GEOMETRY_FIELD],
                                         link_geo2=self.link_gdf.at[link2, glb_field.GEOMETRY_FIELD])

        # # 这样计算会有误差
        # if calc_link_angle(link_geo1=LineString([self.node_gdf.at[seq_list[0], glb_field.GEOMETRY_FIELD],
        #                                          self.node_gdf.at[seq_list[1], glb_field.GEOMETRY_FIELD]]),
        #                    link_geo2=LineString([self.node_gdf.at[seq_list[1], glb_field.GEOMETRY_FIELD],
        #                                          self.node_gdf.at[
        #                                              seq_list[2], glb_field.GEOMETRY_FIELD]])) >= self.max_angle:
        if 332427 in seq_list:
            print(link_list, angle)
        if angle >= self.max_angle:
            print(link_list, angle)
            return []
        return [tuple(sorted((seq_list[i], seq_list[i + 1]))) for i in range(0, len(seq_list) - 1)]

    @staticmethod
    def get_ud_chain_seq_list(link_gdf: gpd.GeoDataFrame) -> list[int]:
        """
        link_gdf一定是一条链
        :param link_gdf:
        :return:
        """
        edge_list = []
        u_g = nx.Graph()
        for _, row in link_gdf.iterrows():
            if row[glb_field.DIRECTION_FIELD] == 0:
                edge_list.append((row[glb_field.FROM_NODE_FIELD], row[glb_field.TO_NODE_FIELD]))
                edge_list.append((row[glb_field.TO_NODE_FIELD], row[glb_field.FROM_NODE_FIELD]))
            else:
                edge_list.append((row[glb_field.FROM_NODE_FIELD], row[glb_field.TO_NODE_FIELD]))
        u_g.add_edges_from(edge_list)
        u_g_dict = dict(u_g.degree)
        se_node = [i for i in u_g_dict.keys() if u_g_dict[i] == 1]
        return nx.dijkstra_path(u_g, se_node[0], se_node[1])

    @staticmethod
    def cluster_graph(edge_list: list[tuple] = None) -> list[set]:
        g = nx.Graph()
        g.add_edges_from(edge_list)
        return list(nx.connected_components(g))

    def calc_polyline_angle(self, link_geo1=None, link_geo2=None):
        """
        给出两条首尾相连的polyline, 计算在交点处的转角
        :param link_geo1:
        :param link_geo2:
        :return:
        """
        point_list1, point_list2 = [Point(xy) for xy in list(link_geo1.coords)], [Point(xy) for xy in
                                                                                  list(link_geo2.coords)]
        link1_node_df = gpd.GeoDataFrame({'id1': [i for i in range(1, len(point_list1) + 1)]},
                                         geometry=[Point(xy) for xy in list(link_geo1.coords)], crs='EPSG:4326')

        link2_node_df = gpd.GeoDataFrame({'id2': [i for i in range(1, len(point_list2) + 1)]},
                                         geometry=[Point(xy) for xy in list(link_geo2.coords)], crs='EPSG:4326')
        link1_node_df = link1_node_df.to_crs(self.utm_crs)
        link1_node_df['geometry'] = link1_node_df['geometry'].apply(lambda geo: geo.buffer(0.1))
        link1_node_df = link1_node_df.to_crs('EPSG:4326')

        # 肯定只有一个交点，有风险, geopandas 1.0.0不允许重复列
        join_df = gpd.sjoin(link1_node_df, link2_node_df)
        join_df.reset_index(inplace=True, drop=True)
        if join_df.at[0, 'id1'] == 1 and join_df.at[0, 'id2'] == 1:
            n1_s, n1_e, n2_s, n2_e = 2, 1, 1, 2
        elif join_df.at[0, 'id1'] == 1 and join_df.at[0, 'id2'] == len(point_list2):
            n1_s, n1_e, n2_s, n2_e = 1, 2, len(point_list2) - 1, len(point_list2)
        elif join_df.at[0, 'id1'] == len(point_list1) and join_df.at[0, 'id2'] == 1:
            n1_s, n1_e, n2_s, n2_e = len(point_list1) - 1, len(point_list1), 1, 2
        else:
            n1_s, n1_e, n2_s, n2_e = len(point_list1) - 1, len(point_list1), len(point_list2), len(point_list2) - 1

        link2_node_df.set_index('id2', inplace=True)
        link1_node_df.set_index('id1', inplace=True)

        return calc_link_angle(link_geo1=LineString([link1_node_df.at[n1_s, glb_field.GEOMETRY_FIELD].centroid,
                                                     link1_node_df.at[n1_e, glb_field.GEOMETRY_FIELD].centroid]),
                               link_geo2=LineString([link2_node_df.at[n2_s, glb_field.GEOMETRY_FIELD],
                                                     link2_node_df.at[n2_e, glb_field.GEOMETRY_FIELD]]))

    def merge(self) -> (gpd.GeoDataFrame, gpd.GeoDataFrame, dict):
        new_link, new_node, info_dict = merge_links(link_gdf=self.link_gdf, node_gdf=self.node_gdf,
                                                    merge_link_df=self.merge_link_df)
        return new_link, new_node, info_dict


if __name__ == '__main__':
    # gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[Point((113.21, 31.23))], crs='EPSG:4326')
    # print(gdf)
    # epsg = CRS.from_proj4("+proj=aeqd +lat_0=" + str(31.63) + " +lon_0=" + str(114.21) + " +datum=WGS84")
    # gdf = gdf.to_crs(epsg)
    # print(gdf)
    # print(gdf.crs)
    # pass
    #
    # g = nx.Graph()
    # g.add_edges_from([(1,2), (2,3), (3,4), (2,1), (3,2), (4, 3)])
    #
    # print(g.adj[2])
    # print(dict(g.degree))
    # print(nx.connected_components(g))
    # print(list(nx.connected_components(g)))

    l = gpd.read_file(r'F:\PyPrj\DataMallNet\data\output\reverse\长春市\FinalLink.shp')
    n = gpd.read_file(r'F:\PyPrj\DataMallNet\data\output\reverse\长春市\FinalNode.shp')
    for col in ['link_id', 'dir', 'from_node', 'to_node']:
        l[col] = l[col].astype(int)
    for col in ['node_id']:
        n[col] = n[col].astype(int)

    merge_short = MergeShort(link_gdf=l, node_gdf=n, threshold=50, max_angle=20)
    merge_short.get_merged_link_seq()
    l, n = merge_short.merge()

    l.to_file(r'./test_link.shp', encoding='gbk')
    n.to_file(r'./test_node.shp', encoding='gbk')

    # from shapely.ops import linemerge
    # a = LineString([(1,1), (2,2)])
    # b = LineString([(0, 0), (1.000000001,1)])
    #
    # print(linemerge([a, b]))


