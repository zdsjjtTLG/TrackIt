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

### 1.样例GPS数据生成

#### 1.1引入相关模块
引入路网Net类、路径类Route、车辆类Car、行程信息收集器类RouteInfoCollector、字段常量NetField、GpsField
``` python
import datetime
from src.map.Net import Net
from src.generation.GpsGen import Route
from src.GlobalVal import NetField, GpsField
from src.generation.GpsGen import Car, RouteInfoCollector
```
#### 1.2初始化一个路网对象
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

#### 1.3初始化一个路径
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

#### 1.4初始化一个行程数据收集器
``` python
# 3.新建一个行程信息收集器对象, 对数据进行统一管理
# 轨迹信息和GPS坐标信息都是平面坐标系, 需要转化为地理坐标系后再进行存储
data_col = RouteInfoCollector(from_crs=plain_crs, to_crs=geo_crs, convert_prj_sys=True)
```

#### 1.5定义车辆仿真参数并且定义一个车辆实体
``` python
# 4.设置仿真参数, 并且初始化一个车辆实体
_time_step = 0.1  # 仿真步长, s
speed_miu = 12.0  # 速度期望值
speed_sigma = 3.6  # 速度标准差
save_gap = 5  # 每多少仿真步保存一次车辆真实位置数据
loc_frequency = 1.0  # 每多少s进行一次GPS定位
loc_error_sigma = 20.0  # 定位误差标准差(m)
loc_error_miu = 0.0  # 定位误差标准期望值(m)

# 新建车对象, 分配一个车辆ID, 配备一个电子地图net, 且设置仿真参数
car = Car(net=my_net, time_step=_time_step, route=route,
          agent_id=car_id, speed_miu=speed_miu, speed_sigma=speed_sigma,
          loc_frequency=loc_frequency, loc_error_sigma=loc_error_sigma, loc_error_miu=loc_error_miu,
          start_time=datetime.datetime(year=2022, month=5, day=12, hour=16, minute=14, second=0),
          save_gap=save_gap)

# 开始行车
car.start_drive()
```

#### 1.6收集数据/存储数据到磁盘
``` python
# 收集真实轨迹数据
data_col.collect_trajectory(car.get_trajectory_info())

# 收集gps数据
data_col.collect_gps(car.get_gps_loc_info())

# 存储数据
data_col.save_trajectory(file_type='geojson', out_fldr=r'./data/output/trajectory/', file_name='rnd_route')
data_col.save_gps_info(file_type='geojson', out_fldr=r'./data/output/gps/', file_name='rnd_route')
data_col.save_mix_info(file_type='geojson', out_fldr=r'./data/output/mix/', file_name='rnd_route')
```

或者直接从内存中拿到结果

``` python
# 收集真实轨迹数据
data_col.collect_trajectory(car.get_trajectory_info())

# 收集gps数据
data_col.collect_gps(car.get_gps_loc_info())

# 存储数据
trajectory_gdf = data_col.save_trajectory()
gps_gdf = data_col.save_gps_info()
mix_gdf = data_col.save_mix_info()
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
