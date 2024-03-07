# GoTrackIt
[![Documentation Status](https://readthedocs.org/projects/gotrackit/badge/?version=latest)](https://gotrackit.readthedocs.io/en/latest/?badge=latest)
![PyPI - Version](https://img.shields.io/pypi/v/gotrackit)
![GitHub License](https://img.shields.io/github/license/zdsjjtTLG/Trackit)
![PyPI - Downloads](https://img.shields.io/pypi/dw/gotrackit)


作者: 唐铠, 794568794@qq.com, tangkai@zhechengdata.com


**v0.1.7已更新**
- 地图匹配效率提升；
- 修复概率连乘导致浮点数下溢的BUG；
- 新增路网联通性修复功能；
- 路网逆向模块优化。


## 1. 简介
本地图匹配包基于隐马尔可夫模型(HMM)实现了连续GPS点位的概率建模，利用这个包可以轻松对GPS数据进行地图匹配，本开源包的特点如下:

**数据无忧**
- 提供路网生产模块以及路网优化接口，您不需要准备任何路网和GPS数据即可玩转地图匹配；
- 提供GPS样例数据生产模块，解决没有GPS数据的难题；
- 提供GPS数据清洗接口，包括滑动窗口降噪、数据降频。

**文档齐全**

- 中文文档，有详细的操作指引；
- 算法原理讲解部分不涉及复杂的公式推导，使用动画形式剖析算法原理,简洁明了。

**匹配结果自动优化**
- 对基于HMM匹配的初步路径进行了优化，对于不连通的位置会自动补路，对于实际路网不连通的位置会输出警告，方便用户检查路网。



### 1.1. 如何安装gotrackit

#### __所需前置依赖__

- geopandas(0.14.1)
- geopy(2.4.1)
- gdal(3.4.3)
- networkx(3.2.1)
- shapely(2.0.2)
- pandas(2.0.3)
- numpy(1.26.2)
- pyproj(3.6.1)
- keplergl(0.3.2)

括号中为作者使用版本(基于python3.11), 仅供参考

geopandas为最新版本, 如果不是最新版本可能会报错(有个函数旧版本没有)

#### __使用pip安装__

安装：

``` shell
pip install -i https://pypi.org/simple/ gotrackit
```

更新：
``` shell
pip install --upgrade  -i https://pypi.org/simple/ gotrackit
```

### 1.2 用户手册

[用户文档链接](https://gotrackit.readthedocs.io/en/latest/)

## 2. 地图匹配问题

![car_gps.png](docs/_static/images/car_gps.png)

![where_car.png](docs/_static/images/whereIsCar.png)

__如何依据GPS数据推算车辆的实际路径？__

## 3. 地图匹配算法动画演示

想了解算法过程的可以参考B站视频:
[基于隐马尔可夫模型(HMM)的地图匹配算法动画版！学不会你来打我！](https://www.bilibili.com/video/BV1gQ4y1w7dC/?vd_source=7389960e7356c27a5d1849f7ee9ae6f2)

![main.png](docs/_static/images/single_p.png)

![main.png](docs/_static/images/transition.png)

![main.png](docs/_static/images/viterbi.png)

![main.png](docs/_static/images/trace.png)


## 4. 匹配结果可视化

中高频GPS匹配效果:

![main.png](docs/_static/images/m_h_f.gif)

低频GPS匹配效果:

![main.png](docs/_static/images/l_f.gif)
