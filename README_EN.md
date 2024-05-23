**Read this in other languages: [Chinese](README.md) or [English](README.md).**

<div align="center">
    <img src="docs/_static/images/gotrackit.png" />
</div>

<br>

<div align=center>

[![Documentation Status](https://readthedocs.org/projects/gotrackit/badge/?version=latest)](https://gotrackit.readthedocs.io/en/latest/?badge=latest)
![PyPI - Version](https://img.shields.io/pypi/v/gotrackit)
![GitHub License](https://img.shields.io/github/license/zdsjjtTLG/Trackit)
[![Downloads](https://static.pepy.tech/badge/gotrackit)](https://pepy.tech/project/gotrackit)
![PyPI - Downloads](https://img.shields.io/pypi/dw/gotrackit)
![PyPI - Downloads](https://img.shields.io/pypi/dm/gotrackit)
![Static Badge](https://img.shields.io/badge/Model-HMM-9EC231)
![Static Badge](https://img.shields.io/badge/Optimization-FastMapMatching-blue)
![Static Badge](https://img.shields.io/badge/Optimization-MultiCoreParallelism-9EC231)
![Github Created At](https://img.shields.io/github/created-at/zdsjjtTLG/Trackit)
![GitHub last commit](https://img.shields.io/github/last-commit/zdsjjtTLG/Trackit)
![GitHub User's stars](https://img.shields.io/github/stars/zdsjjtTLG)
![GitHub forks](https://img.shields.io/github/forks/zdsjjtTLG/Trackit)



~ One package helps you do it: road network acquisition, road network optimization, macro and micro map matching ~

Developed by Tang Kai, Email: 794568794@qq.com & tangkai@zhechengdata.com
</div>
<br>


**version status：05.19 Updated: v0.2.7**

update command：pip install --upgrade  -i https://pypi.org/simple/ gotrackit

- The efficiency of the map matching interface is optimized, which is slightly improved compared to v0.2.4

- Added grid parameter search to help users determine reasonable matching parameters

- BUG fix


**Do not download the code from the GitHub repository to use!!! Just pip install gotrackit as a third-party library and use it**


<div align=center>
~ Sparse trajectory matching and path completion ~
</div>

<br>

Sparse trajectory gps point example：
<div align="center">
    <img src="docs/_static/images/极稀疏轨迹匹配.gif" />
</div>


<div align="center">
    <img src="docs/_static/images/匹配动画样例3.gif" />
</div>


<br>

<div align=center>
~ regular match example ~
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
~ WeChat user communication group. If you encounter a bug that cannot be solved, please join the group to communicate. Don’t forget to give the project a star. Your support is the driving force for my iteration. ~
</div>

<br>

<div align="center">
    <img src="docs/_static/images/wxq.jpg" />
</div>


## 1. Introduction
This map matching package implements probabilistic modeling of continuous GPS points based on Hidden Markov Model (HMM). This package can be used to easily perform map matching on GPS data.


### 1.1. How to install gotrackit

#### __Required pre-dependencies__

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

The version used by the author (based on python3.11) is for reference only.

#### __Install using pip__

install:
``` shell
pip install -i https://pypi.org/simple/ gotrackit
```

renew:
``` shell
pip install --upgrade  -i https://pypi.org/simple/ gotrackit
```

### 1.2 How to use gotrackit

- [User manual](https://gotrackit.readthedocs.io/en/latest/)

- [Animated version of map matching algorithm based on Hidden Markov Model (HMM)!](https://www.bilibili.com/video/BV1gQ4y1w7dC)

- [A python package handles road network acquisition + map matching!](https://www.bilibili.com/video/BV1nC411z7Vg)

- [Detailed explanation and troubleshooting of map matching package parameters](https://www.bilibili.com/video/BV1qK421Y7hV)

- [QGIS road network topology display, base map loading, style reuse, and map saving](https://www.bilibili.com/video/BV1Sq421F7QX)


### 1.3. how to cite gotrackit

If you would like to cite gotrackit in your article, please include the following link：

``` shell
https://github.com/zdsjjtTLG/TrackIt
```

