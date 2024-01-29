# -- coding: utf-8 --
# @Time    : 2023/12/21 10:09
# @Author  : TangKai
# @Team    : ZheChengData


import os
import xml.etree.cElementTree as ET


def generate_book_mark(name_loc_dict: dict = None, prj_name: str = None, input_fldr: str = None, _mode='append'):
    if not name_loc_dict:
        return None
    if _mode == 'replace':
        try:
            os.remove(os.path.join(input_fldr, rf'{prj_name}.xml'))
        except FileNotFoundError:
            pass
    if rf'{prj_name}.xml' in os.listdir(input_fldr):
        # append
        tree = ET.parse(os.path.join(input_fldr, rf'{prj_name}.xml'))

        # 得到根节点Element
        root_ele = tree.getroot()

        # 拿到当前的已有的name
        already_have_name_list = []
        for child in root_ele:
            already_have_name_list.append(child.find('name').text)

        # 筛选
        name_loc_dict = {k: v for k, v in name_loc_dict.items() if k not in already_have_name_list}

        if not name_loc_dict:
            # print(rf'不新增书签')
            return None
        else:
            #  添加元素
            append_ele(name_loc_dict=name_loc_dict, prj_name=prj_name, root_ele=root_ele)

        # 存储文档
        tree.write(os.path.join(input_fldr, rf'{prj_name}.xml'))

    else:
        # 新建
        root_ele = ET.Element('qgis_bookmarks')

        # 以根节点创建文档树
        tree = ET.ElementTree(root_ele)

        append_ele(name_loc_dict=name_loc_dict, prj_name=prj_name, root_ele=root_ele)

        # 存储文档
        tree.write(os.path.join(input_fldr, rf'{prj_name}.xml'))


def append_ele(name_loc_dict: dict = None, prj_name: str = None,
               root_ele: ET.Element = None):

    # 迭代name_loc_dict创建新的子节点
    for name, loc in name_loc_dict.items():
        book_mark_ele = ET.Element('bookmark')
        cen_x, cen_y = loc[0], loc[1]
        # 依据loc生成x_min, y_min, x_max, y_max
        x_min, y_min, x_max, y_max = cen_x - 0.001, cen_y - 0.0008, \
                                     cen_x + 0.001, cen_y + 0.0008

        val_dict = {'id': str(name), 'name': str(name),
                    'project': prj_name, 'xmin': str(x_min), 'ymin': str(y_min), 'xmax': str(x_max), 'ymax': str(y_max),
                    'sr_id': '3452'}
        for ele_name in ['id', 'name', 'project', 'xmin', 'xmax', 'ymin', 'ymax', 'sr_id']:
            child_ele = ET.Element(ele_name)
            child_ele.text = val_dict[ele_name]
            book_mark_ele.append(child_ele)
        root_ele.append(book_mark_ele)

