# -- coding: utf-8 --
# @Time    : 2024/1/20 21:27
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
import numpy as np
import geopandas as gpd
from src.gotrackit.tools.coord_trans import LngLatTransfer

con = LngLatTransfer()

if __name__ == '__main__':

    link = gpd.read_file(r'./test_link.geojson')
    link['geometry'] = link['geometry'].apply(lambda x: list(x.geoms)[0])
    used_node = set(link['from_node']) | set(link['to_node'])
    node = gpd.read_file(r'./node.shp')
    node = node[node['node_id'].isin(used_node)]
    link['dir'] = link['dir'].astype(int)
    for col in link.columns:
        if col != 'geometry':
            # link[col] = link[col].astype(str)
            # link[col] = link[col].apply(lambda x: '\'' + x + '\'')
            pass

    for col in node.columns:
        if col != 'geometry':
            # node[col] = node[col].astype(str)
            # node[col] = node[col].apply(lambda x: '\'' + x + '\'')
            pass

    print(link['geometry'])
    print(node['geometry'])
    link.to_csv(r'./link.csv', encoding='utf_8_sig', index=False)
    node.to_csv(r'./node.csv', encoding='utf_8_sig', index=False)