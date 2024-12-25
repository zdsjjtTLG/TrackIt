# -- coding: utf-8 --
# @Time    : 2024/12/25 18:09
# @Author  : TangKai
# @Team    : ZheChengData

import numpy as np


class Registration(object):
    def __init__(self, method: str = 'helmert'):
        """Registration类

        - 初始化

        Args:
            method: 求解方法
        """
        self.method = method

        # 初始化仿射变换矩阵
        self.convert_mat = np.ndarray

    def generate_convert_mat(self, pixel_loc_array: np.ndarray = None, actual_loc_array: np.ndarray = None) -> None:
        """Registration类方法 - generate_convert_mat

        - 依据 像素坐标组 以及 对应的真实坐标组 计算仿射变换矩阵

        Args:
            pixel_loc_array: 像素坐标组, np.array([[x1, y1], [x2, y2], ...])
            actual_loc_array: 真实坐标组, np.array([[x1, y1], [x2, y2], ...])

        Returns:

        """
        assert pixel_loc_array.shape[0] == actual_loc_array.shape[0]

        # n组配对坐标
        n = pixel_loc_array.shape[0]
        assert n >= 3, 'at least 3 sets of matching point information'

        if self.method == 'helmert':
            self.helmert_method(pixel_loc_array=pixel_loc_array, actual_loc_array=actual_loc_array, n=n)
        else:
            self.six_parameter_method(pixel_loc_array=pixel_loc_array, actual_loc_array=actual_loc_array, n=n)

    def six_parameter_method(self, pixel_loc_array: np.ndarray = None, actual_loc_array: np.ndarray = None,
                             n: int = 1) -> None:
        """6参数法求解仿射矩阵

        Args:
            pixel_loc_array:
            actual_loc_array:
            n:

        Returns:

        """
        # 计算矩阵mat1
        p_xy_sum = sum([loc[0] * loc[1] for loc in pixel_loc_array])
        p_xx_sum = sum([loc[0] ** 2 for loc in pixel_loc_array])
        p_yy_sum = sum([loc[1] ** 2 for loc in pixel_loc_array])
        p_sum_array = np.sum(pixel_loc_array, axis=0)
        mat1 = np.array([[n, p_sum_array[0], p_sum_array[1]],
                         [p_sum_array[0], p_xx_sum, p_xy_sum],
                         [p_sum_array[1], p_xy_sum, p_yy_sum]])

        # 计算矩阵mat2
        a_sum_array = np.sum(actual_loc_array, axis=0)
        pa_xx_sum = sum([p_loc[0] * a_loc[0] for a_loc, p_loc in zip(actual_loc_array, pixel_loc_array)])
        pa_xy_sum = sum([p_loc[0] * a_loc[1] for a_loc, p_loc in zip(actual_loc_array, pixel_loc_array)])
        pa_yx_sum = sum([p_loc[1] * a_loc[0] for a_loc, p_loc in zip(actual_loc_array, pixel_loc_array)])
        pa_yy_sum = sum([p_loc[1] * a_loc[1] for a_loc, p_loc in zip(actual_loc_array, pixel_loc_array)])
        mat2 = np.array([[a_sum_array[0], a_sum_array[1]],
                         [pa_xx_sum, pa_xy_sum],
                         [pa_yx_sum, pa_yy_sum]])

        # mat1的逆矩阵(若没有逆则求伪逆) * mat2 得到仿射变换矩阵
        self.convert_mat = np.matmul(np.linalg.inv(mat1), mat2)

        # reshape
        self.convert_mat = np.array([[self.convert_mat[1][0], self.convert_mat[1][1], 0],
                                     [self.convert_mat[2][0], self.convert_mat[2][1], 0],
                                     [self.convert_mat[0][0], self.convert_mat[0][1], 0]])

    def helmert_method(self, pixel_loc_array: np.ndarray = None, actual_loc_array: np.ndarray = None,
                       n: int = 1) -> None:
        """helmert变换 求解仿射矩阵

        Args:
            pixel_loc_array:
            actual_loc_array:
            n:

        Returns:

        """
        p_sum = np.sum(pixel_loc_array, axis=0)
        a = p_sum[0]
        b = p_sum[1]

        a_sum = np.sum(actual_loc_array, axis=0)
        c = a_sum[0]
        d = a_sum[1]

        e = sum([p_loc[0] * a_loc[0] for a_loc, p_loc in zip(actual_loc_array, pixel_loc_array)])
        f = sum([p_loc[1] * a_loc[1] for a_loc, p_loc in zip(actual_loc_array, pixel_loc_array)])

        g = sum([loc[0] ** 2 for loc in pixel_loc_array])
        h = sum([loc[1] ** 2 for loc in pixel_loc_array])

        i = sum([p_loc[1] * a_loc[0] for a_loc, p_loc in zip(actual_loc_array, pixel_loc_array)])
        j = sum([p_loc[0] * a_loc[1] for a_loc, p_loc in zip(actual_loc_array, pixel_loc_array)])

        m_mat = np.array([[a, -b, n, 0],
                          [b, a, 0, n],
                          [g + h, 0, a, b],
                          [0, g + h, -b, a]])

        b_mat = np.array([[c, d, e + f, j - i]]).T

        # 求逆后矩阵乘法
        x = np.matmul(np.linalg.inv(m_mat), b_mat).T

        # x, y平移量
        dx, dy = x[0][2], x[0][3]

        # 反三角函数求旋转角度
        theta = np.arctan2(x[0][1], x[0][0])

        # xy轴缩放系数
        scale = np.sqrt(x[0][0] ** 2 + x[0][1] ** 2)

        self.convert_mat = np.array([[scale * np.cos(theta), scale * np.sin(theta), 0],
                                     [-scale * np.sin(theta), scale * np.cos(theta), 0],
                                     [dx, dy, 1]])

        print(dx, dy, scale, theta * 180 / np.pi)

    def coords_convert(self, x: float = None, y: float = None) -> np.ndarray:
        """
        将像素坐标转换为真实世界坐标
        :param x:
        :param y:
        :return:
        """
        return np.dot(np.array([[x, y, 1]]), self.convert_mat)[0][:2]


if __name__ == '__main__':
    # 像素坐标
    fig_loc = np.array([[998, -899],
                        [1526, -547],
                        [1030, -1497],
                        [1549, -1884]])

    # 地图真实坐标(这里的坐标系是EPSG:3857)
    map_loc = np.array([[13390508.490, 3711698.016],
                        [13390526.503, 3711702.823],
                        [13390498.451, 3711679.131],
                        [13390505.281, 3711656.220]])

    # 初始化求解类
    r = Registration()

    # 计算仿射变换矩阵
    r.generate_convert_mat(pixel_loc_array=fig_loc, actual_loc_array=map_loc)

    # 执行转换
    (real_x, real_y) = r.coords_convert(998, -899)

    print(r.convert_mat)
    print(real_x, real_y)