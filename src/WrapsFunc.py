# -- coding: utf-8 --
# @Time    : 2023/12/31 21:16
# @Author  : TangKai
# @Team    : ZheChengData

import time


def function_time_cost(f):
    def inner(*args, **kwargs):
        # time.time获取函数执行的时间
        s = time.time()  # func开始的时间
        res = f(*args, **kwargs)
        e = time.time()  # func结束的时间
        print(f"{f.__name__} costs :{e - s} seconds!")
        return res
    return inner


@function_time_cost
def aaa(a: int = None, b: int = None):
    print(a + b)
    time.sleep(12)
    return a + b


if __name__ == '__main__':
    aaa(13, 122)