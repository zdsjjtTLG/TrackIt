# -- coding: utf-8 --
# @Time    : 2024/11/20 22:52
# @Author  : TangKai
# @Team    : ZheChengData
import time

import numpy as np
from geopy.distance import distance
import pandas as pd

def dis(lng_a: np.ndarray = None, lat_a: np.ndarray = None,
        lng_b: np.ndarray = None, lat_b: np.ndarray = None, ) -> np.ndarray:
    r = 6378.137
    lng_a, lat_a, lng_b, lat_b = np.radians(lng_a), np.radians(lat_a), np.radians(lng_b), np.radians(lat_b)
    print(lng_a, lat_a, lng_b, lat_b)
    d_lng, d_lat = lng_b - lng_a, lat_b - lat_a
    l = 2 * np.arcsin(np.sqrt(np.sin(d_lat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(d_lng / 2) ** 2)) * r
    return l * 1000


def bj_join():
    N = 5200
    df1 = pd.DataFrame({'g': [np.random.randint(1, 20) for i in range(N)], 'v1': [i for i in range(N)]})
    df2 = pd.DataFrame({'g': [np.random.randint(1, 20) for i in range(N)], 'v2': [i for i in range(N)]})
    print(df1)
    print(df2)

    s = time.time()
    res1 = pd.merge(df1, df2, on='g', how='outer')
    t1 = time.time()

    df1.set_index('g', inplace=True)
    df2.set_index('g', inplace=True)
    res2 = df1.join(df2, how='outer').reset_index(drop=False)
    t2 = time.time()

    print(t1 - s, t2 - t1)
    print(len(res2))

if __name__ == '__main__':
    # p1 = (-14.234, 0.122)
    # p2 = (114.234, 67.123)
    # l1 = distance((p1[1], p1[0]), (p2[1], p2[0])).m
    # l2 = dis(lng_a=np.array([p1[0]]), lat_a=np.array([p1[1]]), lng_b=np.array([p2[0]]), lat_b=np.array([p2[1]]))
    # print(l2 / 1000)
    # print(l1 / 1000)
    # np.arctan()
    # np.arctan2()
    bj_join()
