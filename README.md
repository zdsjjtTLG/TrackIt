
![car_gps.png](docs/_static/images/gotrackit.png)


[![Documentation Status](https://readthedocs.org/projects/gotrackit/badge/?version=latest)](https://gotrackit.readthedocs.io/en/latest/?badge=latest)
![PyPI - Version](https://img.shields.io/pypi/v/gotrackit)
![GitHub License](https://img.shields.io/github/license/zdsjjtTLG/Trackit)
![PyPI - Downloads](https://img.shields.io/pypi/dw/gotrackit)
![PyPI - Downloads](https://img.shields.io/pypi/dm/gotrackit)


作者: 唐铠, 794568794@qq.com, tangkai@zhechengdata.com


**04.12已更新: v0.2.0**

更新命令：pip install --upgrade  -i https://pypi.org/simple/ gotrackit

- 匹配过程增加多进程参数，拓扑优化过程增加多进程参数

- GPS候选路段的选择：除开buffer选择外引入了top_k参数，用于指定buffer内最近的top_k个路段作为候选路段

- 增加GPS点停留点识别功能

- 修正匹配结果中坐标不一致的BUG，现统一为EPSG:4326

- 增加依据GPS数据提取带途径点OD的功能

- 增加了路网处理函数：路段、节点重塑

- 修复了部分BUG


遇到BUG无法解决请进群交流，别忘了给项目一颗star哦~

<div align="center">
    <img src="docs/_static/images/l_f.gif" />
</div>

<div align="center">
    <img src="docs/_static/images/m_h_f.gif" />
</div>

稀疏轨迹增密匹配：

<div align="center">
    <img src="docs/_static/images/dense_gps.gif" />
</div>


<div align="center">
    <img src="docs/_static/images/taxi_xishu.gif" />
</div>


<div align="center">
    <img src="docs/_static/images/xa_sample.gif" />
</div>


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

- geopy(2.4.1)
- gdal(3.4.3)
- shapely(2.0.3)
- fiona(1.9.5)
- pyproj(3.6.1)
- geopandas(0.14.3)
- networkx(3.2.1)
- pandas(2.0.3)
- numpy(1.26.2)
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

### 1.2 用户手册与视频教程

[用户手册](https://gotrackit.readthedocs.io/en/latest/)

[基于隐马尔可夫模型(HMM)的地图匹配算法动画版！学不会你来打我！](https://www.bilibili.com/video/BV1gQ4y1w7dC)

[一个python包搞定路网获取+地图匹配！](https://www.bilibili.com/video/BV1nC411z7Vg)

[gotrackit地图匹配包参数详解与问题排查](https://www.bilibili.com/video/BV1qK421Y7hV)

[QGIS路网拓扑显示、底图加载、样式复用、map保存](https://www.bilibili.com/video/BV1Sq421F7QX)


## 2. 地图匹配问题

![car_gps.png](docs/_static/images/car_gps.png)

![where_car.png](docs/_static/images/whereIsCar.png)

__如何依据GPS数据推算车辆的实际路径？__

![main.png](docs/_static/images/single_p.png)

![main.png](docs/_static/images/transition.png)

![main.png](docs/_static/images/viterbi.png)

![main.png](docs/_static/images/trace.png)


