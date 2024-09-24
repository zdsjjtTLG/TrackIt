# -- coding: utf-8 --
# @Time    : 2024/8/18 16:11
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
import geopandas as gpd

def build_time_col(df: pd.DataFrame or gpd.GeoDataFrame = None, time_format: str = '%Y-%m-%d %H:%M:%S',
                   time_unit: str = 's', time_field: str = 'time') -> None:
    """
    change inplace
    :param df:
    :param time_format:
    :param time_unit:
    :param time_field:
    :return:
    """
    if df[time_field].dtype not in ['datetime64[ns]', 'datetime64[ms]', 'datetime64[s]']:
        try:
            df[time_field] = pd.to_datetime(df[time_field], format=time_format)
        except ValueError:
            print(rf'time column does not match format {time_format}, try using time-unit: {time_unit}')
            if df[time_field].dtype == object:
                df[time_field] = df[time_field].astype(float)
            df[time_field] = pd.to_datetime(df[time_field], unit=time_unit)
