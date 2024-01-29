# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData

def get_val(map_dict=None, k=None):
    try:
        return map_dict[k]
    except KeyError:
        return k