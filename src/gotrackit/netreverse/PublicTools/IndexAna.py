# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData

"""一些常用工具算法"""

from itertools import groupby


def find_continuous_repeat_index(input_list):
    """
    找出一个list中连续重复元素的索引
    :param input_list: list
    :return: 返回None代表没有连续重复的元素, 否则返回一个列表

    Examples
    --------
    >> a = [1, 2, 3]
    >> print(find_continuous_repeat_index(input_list=a))
    None

    >> b = [1, 1, 1, 2, 1, 1]
    >> print(find_continuous_repeat_index(input_list=b))
    [[0, 1, 2], [4, 5]]

    >> c = [1, 1, 1, 1, 1, 1]
    >> print(find_continuous_repeat_index(input_list=c))
    [[0, 1, 2, 3, 4, 5]]
    """

    # 所有元素都一样
    if len(input_list) == 1:
        return None

    if len(list(set(input_list))) == 1:
        return [[index for index in range(0, len(input_list))]]

    # 所有元素都不一样
    elif len(list(set(input_list))) == len(input_list):
        return None

    # 有部分元素一样
    else:
        dup_list = []

        index = 0
        for item_group in groupby(input_list):

            index_list = []

            for item in item_group[1]:
                index_list.append(index)
                index += 1

            if len(index_list) >= 2:
                dup_list.append(index_list)
            else:
                pass

        if not dup_list:
            return None
        else:
            return dup_list


if __name__ == '__main__':
    print(find_continuous_repeat_index([1, 1, 1, 2]))
