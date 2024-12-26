---
comments: true
---

[路网生产模块]: ./路网生产.md
[路网优化工具]: ./路网优化.md
[地图匹配]: ./路径匹配.md
[GPS样例数据生产模块]: ./轨迹生产.md
[GPS数据清洗接口]: ./轨迹处理.md
[动画形式]: https://www.bilibili.com/video/BV1gQ4y1w7dC
[FastMapMatching]: 路径匹配.md#accelerate
[多核并行匹配]: ./路径匹配.md#accelerate
[网格参数搜索]: ./路径匹配.md#accelerate
[GPS点匹配结果表]: ./路径匹配.md#match_res
[匹配结果矢量化图层]: ./路径匹配.md#geojson_show
[矢量图层匹配动画(HTML文件)]: ./路径匹配.md#html_show


# GoTrackIt简介
<figure markdown="span">
  ![Image title](../_static/images/gotrackit.svg){ width="300" }
</figure>

GoTrackIt是一个基于改进隐马尔可夫模型实现的**全流程地图匹配**pyhon包， 通过概率图建模的方式将车辆的GPS轨迹数据匹配到道路路网， 获取车辆的标准化时空轨迹， 可以有效支撑出行导航、交通监测、交通治理、碳排核算、交通建模等方向。

!!! note 
    
    这个项目正处在频繁的升级迭代期， 当前版本v0.3.15， 请用户及时更新.

<figure markdown="span">
  ![Image title](../_static/images/MapMatch.PNG)
  <figcaption>地图匹配(路径匹配)示意图</figcaption>
</figure>

<figure markdown="span">
  ![Image title](../_static/images/application.PNG)
  <figcaption>路径匹配的应用场景</figcaption>
</figure>


## GoTrackIt功能特点
利用GoTrackIt可以轻松对GPS数据进行地图匹配，本开源包的特点如下:

### 😻数据无忧
- [x] 提供[路网生产模块]和[路网优化工具]，您不需要准备任何路网和GPS数据即可玩转[地图匹配]；
- [x] 提供[GPS样例数据生产模块]，解决没有GPS数据的难题；
- [x]  提供[GPS数据清洗接口]，包括行程切分、滤波平滑、停留点识别等。


### ☑️文档齐全
- [x] 中文文档，有详细的操作指引；
- [x] 算法原理讲解部分不涉及复杂的公式推导，使用[动画形式]剖析算法原理，简洁明了。

### 🚀匹配算法优化
- [x] 支持基于路径预计算的[FastMapMatching]、支持[多核并行匹配]、支持[网格参数搜索]；
- [x] 对基于HMM匹配的初步路径进行了优化，对于不连通的位置会自动搜路补全，对于实际路网不连通的位置会输出警告信息，方便用户回溯问题。


### 🌈匹配结果支持动画可视化
- [x] 匹配结果提供三种输出形式：[GPS点匹配结果表]、[匹配结果矢量化图层]、[矢量图层匹配动画(HTML文件)]，HTML动画方便用户直观地感受匹配结果，同时可以提高问题排查的效率。


## GoTrackIt功能框架

<figure markdown="span">
  ![Image title](../_static/images/FunctionGraph.png)
  <figcaption>GoTrackIt功能框架</figcaption>
</figure>

## GoTrackIt匹配效果

<figure markdown="span">
  ![Image title](../_static/images/极稀疏轨迹匹配.gif)
  <figcaption>稀疏轨迹匹配</figcaption>
</figure>


<figure markdown="span">
  ![Image title](../_static/images/匹配动画样例1.gif)
  <figcaption>常规轨迹匹配</figcaption>
</figure>

<figure markdown="span">
  ![Image title](../_static/images/匹配动画样例3.gif)
  <figcaption>常规轨迹匹配</figcaption>
</figure>

<figure markdown="span">
  ![Image title](../_static/images/kvs/hk_trip.gif)
  <figcaption>多车匹配可视化</figcaption>
</figure>

<figure markdown="span">
  ![Image title](../_static/images/geojson_res.jpg)
  <figcaption>匹配结果GeoJSON可视化</figcaption>
</figure>


## GoTrackIt视频教程

- [x] [基于隐马尔可夫模型(HMM)的地图匹配算法动画版！学不会你来打我](https://www.bilibili.com/video/BV1gQ4y1w7dC)

- [x] [一个python包搞定路网获取+地图匹配](https://www.bilibili.com/video/BV1nC411z7Vg)

- [x] [gotrackit地图匹配包参数详解与问题排查](https://www.bilibili.com/video/BV1qK421Y7hV)

- [x] [QGIS路网拓扑显示、底图加载、样式复用、map保存](https://www.bilibili.com/video/BV1Sq421F7QX)
