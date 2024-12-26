# -- coding: utf-8 --
# @Time    : 2024/1/27 13:12
# @Author  : TangKai
# @Team    : ZheChengData


import os.path
import pandas as pd
import geopandas as gpd


def save_file(data_item: pd.DataFrame or gpd.GeoDataFrame = None, out_fldr: str = None, file_name: str = None,
              file_type: str = None) -> None:
    if data_item is None or data_item.empty:
        return None
    assert file_type in ['csv', 'shp', 'geojson']
    if out_fldr is None:
        pass
    else:
        if not os.path.exists(out_fldr):
            os.makedirs(out_fldr)
        if file_type == 'csv':
            data_item.to_csv(os.path.join(out_fldr, file_name + '.csv'), encoding='utf_8_sig', index=False)
        elif file_type == 'geojson':
            if isinstance(data_item, gpd.GeoDataFrame):
                data_item.to_file(os.path.join(out_fldr, file_name + '.geojson'), encoding='gbk')
            else:
                raise ValueError('only gpd.GeoDataFrame can be stored as geojson file')
        else:
            if isinstance(data_item, gpd.GeoDataFrame):
                data_item.to_file(os.path.join(out_fldr, file_name + '.shp'), encoding='gbk')
            else:
                raise ValueError('only gpd.GeoDataFrame can be stored as shape file')
