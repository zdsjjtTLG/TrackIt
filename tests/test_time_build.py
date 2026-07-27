# -- coding: utf-8 --
# @Author  : TangKai
# @Team    : ZheChengData

import pandas as pd

from gotrackit.tools.time_build import build_time_col


def test_build_time_col_accepts_string_unix_seconds():
    time_df = pd.DataFrame({
        'time': pd.Series(['1710254640', '1710254644'], dtype='str'),
    })

    build_time_col(df=time_df, time_unit='s')

    assert pd.api.types.is_datetime64_any_dtype(time_df['time'].dtype)
    assert time_df['time'].tolist() == [
        pd.Timestamp('2024-03-12 14:44:00'),
        pd.Timestamp('2024-03-12 14:44:04'),
    ]


def test_build_time_col_keeps_datetime_values():
    original_time = pd.Series(pd.to_datetime([
        '2024-03-12 14:44:00',
        '2024-03-12 14:44:04',
    ]))
    time_df = pd.DataFrame({'time': original_time.copy()})

    build_time_col(df=time_df)

    pd.testing.assert_series_equal(time_df['time'], original_time, check_names=False)
