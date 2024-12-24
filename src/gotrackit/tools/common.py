# -- coding: utf-8 --
# @Time    : 2024/12/17 22:42
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd
import geopandas as gpd

def avoid_duplicate_cols(built_in_col_list: list = None, df: pd.DataFrame | gpd.GeoDataFrame = None) -> dict:
    """重命名数据表中和内置名称冲突的字段

    Args:
        built_in_col_list: 要使用的内置名称字段列表
        df: 数据表

    Returns:
        dict
    """

    rename_dict = dict()

    # 数据表的所有列名称
    df_cols_list = list(df.columns)

    # 遍历每一个在函数内部需要使用的内置字段, 检查其是否已经存在数据表字段中
    for built_in_col in built_in_col_list:
        if built_in_col in df_cols_list:
            num = 1
            while '_'.join([built_in_col, str(num)]) in df_cols_list:
                num += 1
            rename_col = '_'.join([built_in_col, str(num)])
            rename_dict[built_in_col] = rename_col
        else:
            pass
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    return rename_dict
