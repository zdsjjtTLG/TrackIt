
![car_gps.png](docs/_static/images/gotrackit.png)


[![Documentation Status](https://readthedocs.org/projects/gotrackit/badge/?version=latest)](https://gotrackit.readthedocs.io/en/latest/?badge=latest)
![PyPI - Version](https://img.shields.io/pypi/v/gotrackit)
![GitHub License](https://img.shields.io/github/license/zdsjjtTLG/Trackit)
![PyPI - Downloads](https://img.shields.io/pypi/dw/gotrackit)
![PyPI - Downloads](https://img.shields.io/pypi/dm/gotrackit)



作者: 唐铠, 794568794@qq.com, tangkai@zhechengdata.com


**2024.03.28已经更新: v0.1.9**

更新命令：pip install --upgrade  -i https://pypi.org/simple/ gotrackit

- 增加GPS增密功能

- 引入GPS点差分方向向量来修正发射概率

- 地图匹配接口升级，使用更加简单，暴露了更多的[可调参数](https://gotrackit.readthedocs.io/en/latest/%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8.html#id36)

- 构建Net对象限制为：输入的link和node的数据必须为WGS-84 EPSG:4326地理坐标，平面投影坐标系不再需要手动指定，可自动进行6度带的确定

- 增加了路网处理函数：路段划分功能

- 增加了路网处理函数：路段-link_id字段、from_node字段、to_node字段，节点-node_id字段重映射函数

- 修复GPS生成接口中速度为负数的BUG

- 修复Net初始化过程中由于节点ID过大导致内存溢出的问题

- 新增路径缓存开关、ID缓存开关参数

遇到BUG无法解决请进群交流，别忘了给项目一颗star哦~

![car_gps.png](docs/_static/images/wxq.jpg)


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
pip install -i https://test.pypi.org/simple/ gotrackit
```

更新：
``` shell
pip install --upgrade  -i https://test.pypi.org/simple/ gotrackit
```

### 1.2 用户手册

[用户手册](https://gotrackit.readthedocs.io/en/latest/)、
[视频教程1](https://www.bilibili.com/video/BV1gQ4y1w7dC/?vd_source=7389960e7356c27a5d1849f7ee9ae6f2)、
[视频教程2](https://www.bilibili.com/video/BV1nC411z7Vg/?share_source=copy_web&vd_source=9b4518c7de4757ad3b99e18456efbaa6)


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


![car_gps.png](docs/_static/images/l_f.gif)

![car_gps.png](docs/_static/images/m_h_f.gif)

稀疏轨迹增密匹配：

![car_gps.png](docs/_static/images/dense_gps.gif)

![car_gps.png](docs/_static/images/taxi_xishu.gif)

![car_gps.png](docs/_static/images/xa_sample.gif)

