# TrackIt
## 简介
Map-Match-Algorithm Based on Hidden Markov Model,基于隐马尔可夫模型的离线地图匹配模型

![main.png](DocFiles%2Fimages%2Fmain.png)

想了解算法过程的可以参考B站视频:

## 所需依赖
括号中为作者使用版本(基于python3.11), 仅供参考
- geopandas(0.14.1)
- gdal(3.4.3)
- networkx(3.2.1)
- shapely(2.0.2)
- pandas(2.0.3)
- numpy(1.26.2)
- pyproj(3.6.1)
- keplergl(0.3.2)

geopandas为最新版本, 如果不是最新版本可能会报错(有个函数旧版本没有)

## 所需数据

离线地图匹配需要提供一份拓扑正确的路网以及GPS数据,如果没有GPS数据可以使用本文的GPS生成模块生成样例数据,参见

### 路网

路网有两个文件组成:点层文件和线层文件

#### 线层文件Link

geojson或者shp文件,要求必需字段名称如下:
- link_id: 路段唯一编码, integer
- from_node: 路段拓扑起点节点编号, integer
- to_node: 路段拓扑终点节点编号, integer
- dir: 路段方向, integer, 取值为0或者1, 0代表双向通行,1代表通行方向为路段拓扑正向
- length: 路段长度, float, m
- geometry: 路段几何线型, geometry
- ...(其他属性字段)

#### 点文件Node
geojson或者shp文件,要求必需字段名称如下:
- node_id: 节点唯一编码, integer
- geometry: 节点几何坐标, geometry
- ...(其他属性字段)


#### GPS数据文件
要求必需字段名称如下:
- agent_id: 车辆唯一编码ID, string
- time: 定位时间戳, string, '%Y-%m-%d %H:%M:%S'
- lng: 经度,float
- lat: 纬度,float
- heading: 航向角,float,可有可无
- ...(其他属性字段)

目前的版本,匹配算法还没用到航向角的信息


## 模块构成
由三大块组成：样例GPS数据生成、离线地图匹配、匹配可视化



## 如何使用

### 样例GPS数据生成

#### 引入相关模块
引入路网Net类、路径类Route、车辆类Car、行程信息收集器类RouteInfoCollector、字段常量NetField、GpsField
``` python
import datetime
from src.map.Net import Net
from src.generation.GpsGen import Route
from src.GlobalVal import NetField, GpsField
from src.generation.GpsGen import Car, RouteInfoCollector
```
#### 初始化一个路网对象
指定路网点层和线层的文件路径, 并且指定坐标系
``` python
# 1.新建一个路网对象, 并且指定其地理坐标系(shp源文件的crs)以及要使用的投影坐标系
# 示例为西安的路网, 使用6度带中的32649
# 使用length字段为搜路权重列
plain_crs = 'EPSG:32649'
geo_crs = 'EPSG:4326'
my_net = Net(link_path=r'data/input/net/xian/link.shp',
             node_path=r'data/input/net/xian/node.shp',
             weight_field='length', geo_crs=geo_crs, plane_crs=plain_crs)

```

#### 初始化一个路径
``` python
# 2.新建一个route, 用于车辆car路径导航, 必须传入一个net对象作为车辆行驶的电子地图
# 若不指定ft_seq, 则使用o_node -> d_node进行搜路获取路径, 若没有指定o_node和d_node则使用随机路径
route = Route(net=my_net, o_node=None, d_node=None, ft_seq=None)
```
- 依据起终点节点编号确定路径
``` python
route = Route(net=my_net, o_node=176356, d_node=228133)
```
或者
``` python
route = Route(net=my_net, o_node=None, d_node=None)
route.o_node = 176356
route.d_node = 228133
```

- 直接指定路径节点序列获得路径
``` python
route = Route(net=my_net, ft_seq=[(137226, 42212), (42212, 21174), (21174, 39617)])
```

- 使用随机路径
既不指定o_node和d_node也不指定ft_seq, 将会自动生成随机路径
``` python
route = Route(net=my_net)
```



### 离线地图匹配
``` python
import pandas as pd

```

### 匹配可视化
``` python
import pandas as pd

```



## 作者信息
唐铠, 794568794@qq.com, zhechengdata@tangkai.com
