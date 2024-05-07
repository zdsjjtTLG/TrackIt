

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

Developed by Tang Kai, Email: 794568794@qq.com & tangkai@zhechengdata.com
</div>
<br>



**版本状态：05.07已经更新: v0.2.3**

更新命令：pip install --upgrade  -i https://pypi.org/simple/ gotrackit


- 效率优化, 相较于v0.2.2小幅度提升

- crs判断BUG修复、境外路网构建失败BUG修复

- 报错机制优化

- 输出文件目录指定参数html_fldr废除, 改为out_fldr

- 增加环路处理功能
    
- 增加匹配结果即时输出参数, 每匹配完一条轨迹可马上进行结果存储


<br>

<div align=center>
~ v0.2.2效率将大幅度提升, 最高可以提升10倍的性能 !~
</div>

<br>

与上一版本对比:

| 样例数据           | 有效的GPS点数 | top_k(k邻近候选参数) | gps_buffer(临域半径) | 候选路段条数 | 状态转移次数  | v0.2.1版解算时间 | v0.2.2版解算时间 |
|----------------|----------|----------------|------------------|---------|------------|-------------|-----------|
| 1辆车,深圳稀疏轨迹点样例1 | 190      | 60             | 500m             | 10615 | 629788次 | 28秒         | **3.3秒**  |
| 1辆车,深圳稀疏轨迹点样例2 | 400      | 20             | 120m             | 5137 | 82006次  | 7.8秒        | **1.7秒**    |


v0.2.2多核效率对比:

基于上表深圳稀疏轨迹点样例2，我们将他复制150份，进行多核测试，可以看到到6核时, 效率已经不再提升，最快96s解算完150条轨迹，平均每条轨迹0.64s，相较于1.7s再次提升了60%，在车辆数较多时，多核的效率提升很明显。

| 样例数据                                      | 有效的GPS点数 | top_k | gps_buffer | 候选路段条数 | 状态转移次数 | v0.2.2解算时间 |
|-------------------------------------------|----------|----------------|------------------|-----|--------|------------|
| 150辆车的GPS轨迹(单核串行,子图搜索,有构建子图的额外开销,提前预计算路径) | 6w       | 20             | 120m             | 75W | 1200W次 | 300.0秒     | 
| 150辆车的GPS轨迹(3核并行,子图搜索,有构建子图的额外开销,提前预计算路径) | 6w       | 20             | 120m             | 75W | 1200W次 | 139.6秒     |
| 150辆车的GPS轨迹(3核并行,全图搜索,无构建子图的额外开销,提前预计算路径) | 6w       | 20             | 120m             | 75W | 1200W次 | 120.3秒     |
| 150辆车的GPS轨迹(4核并行,全图搜索,无构建子图的额外开销,提前预计算路径) | 6w       | 20             | 120m             | 75W | 1200W次 | 104.9秒     | 
| 150辆车的GPS轨迹(5核并行,全图搜索,无构建子图的额外开销,提前预计算路径) | 6w       | 20             | 120m             | 75W | 1200W次 | 96.4秒      | 
| 150辆车的GPS轨迹(6核并行,全图搜索,无构建子图的额外开销,提前预计算路径) | 6w       | 20             | 120m             | 75W | 1200W次 | 97.5秒      | 


<br>

<div align=center>
~ 稀疏轨迹匹配与路径补全 ~
</div>

<br>

深圳稀疏轨迹点样例1：
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
~ 用户交流群， 遇到BUG无法解决请进群交流，别忘了给项目一颗star哦， 您的支持是我迭代的动力 ~
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
- 对基于HMM匹配的初步路径进行了优化，对于不连通的位置会自动搜路补全，对于实际路网不连通的位置会输出警告信息，方便用户回溯问题。



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
