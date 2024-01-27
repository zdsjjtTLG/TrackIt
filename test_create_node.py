# -- coding: utf-8 --
# @Time    : 2024/1/26 9:38
# @Author  : TangKai
# @Team    : ZheChengData

import geopandas as gpd
import src.gotrackit.map.NetTools as nt


if __name__ == '__main__':
    link = gpd.read_file(r'F:\PyPrj\TrackIt\data\input\net\xian\link.shp')
    link.drop(columns=['link_id', 'from_node', 'to_node'], axis=1, inplace=True)

    new_link, new_node, con_node = nt.create_node_from_link(link_gdf=link, using_from_to=False,
                                                            execute_modify=True,ignore_merge_rule=False,
                                                            modify_minimum_buffer=0.3,
                                                            plain_prj='EPSG:32649',
                                                            update_link_field_list=['link_id', 'from_node', 'to_node'])

    new_link.to_file(r'link.shp', encoding='gbk')
    new_node.to_file(r'node.shp', encoding='gbk')
    con_node.to_file(r'con_node.shp', encoding='gbk')
