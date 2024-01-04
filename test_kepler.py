# -- coding: utf-8 --
# @Time    : 2024/1/4 22:33
# @Author  : TangKai
# @Team    : ZheChengData


import json
from keplergl import KeplerGl


if __name__ == '__main__':
    with open('./data/output/mix/test.geojson') as g:
        mix = json.load(g)
    # 生成KeplerGl对象s
    map1 = KeplerGl(height=400,
                    data={'car+gps': mix})  # data以图层名为键，对应的矢量数据为值
    map1.save_to_html(file_name='car.html')  # 导出到本地可编辑html文件
