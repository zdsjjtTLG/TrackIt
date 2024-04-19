

<div align="center">
    <img src="docs/_static/images/gotrackit.png" />
</div>

<br>

<div align=center>

[![Documentation Status](https://readthedocs.org/projects/gotrackit/badge/?version=latest)](https://gotrackit.readthedocs.io/en/latest/?badge=latest)
![PyPI - Version](https://img.shields.io/pypi/v/gotrackit)
![GitHub License](https://img.shields.io/github/license/zdsjjtTLG/Trackit)
![PyPI - Downloads](https://img.shields.io/pypi/dw/gotrackit)
![PyPI - Downloads](https://img.shields.io/pypi/dm/gotrackit)

~ 一个包搞定：路网获取、路网优化、宏微观地图匹配、匹配可视化、问题路段快速定位 ~

唐铠, 794568794@qq.com, tangkai@zhechengdata.com
</div>
<br>


**版本状态：04.18已更新: v0.2.1**

更新命令：pip install --upgrade  -i https://pypi.org/simple/ gotrackit

- 地图匹配接口移除geo_res_fldr, geojson文件和html存储在一个目录下，指定html_fldr即可，以及增加其他存储参数(见文档)

- HTML文件中新增警告路段信息，若该次匹配中出现警告, 则在HTML文件中会多一个图层，该图层记录了GPS状态转移出错的路段

- 停留点识别功能改进

- BUG修复


<br>

<div align=center>
~ 稀疏轨迹匹配与路径补全 ~
</div>

<br>


<div align="center">
    <img src="docs/_static/images/极稀疏轨迹匹配.gif" />
</div>


<div align="center">
    <img src="docs/_static/images/匹配动画样例3.gif" />
</div>


<br>

<div align=center>
~ 常规匹配 ~
</div>

<br>

<div align="center">
    <img src="docs/_static/images/匹配动画样例1.gif" />
</div>

<div align="center">
    <img src="docs/_static/images/匹配动画样例2.gif" />
</div>


<div align="center">
    <img src="docs/_static/images/匹配动画样例4.gif" />
</div>

<div align="center">
    <img src="docs/_static/images/geojson_res.jpg" />
</div>

<br>

<div align=center>
~ 用户交流群, 遇到BUG无法解决请进群交流, 别忘了给项目一颗star哦, 您的支持是我迭代的动力 ~
</div>

<br>

<div align="center">
    <img src="docs/_static/images/wxq.jpg" />
</div>


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
