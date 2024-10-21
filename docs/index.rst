Welcome to GoTrackIt's documentation!
=========================================

Read UserManual in ：`English <https://gotrackitdocs.readthedocs.io/en/latest/>`_

作者：唐铠

邮箱：tangkai@zhechengdata.com、794568794@qq.com

项目源码：`项目GitHub主页 <https://github.com/zdsjjtTLG/TrackIt>`_ --- 当前版本v0.3.10

GoTrackIt 是一个基于隐马尔可夫模型实现的地图匹配包, 通过概率图建模的方式将车辆的GPS轨迹数据匹配到道路路网, 获取车辆的标准化时空轨迹, 可以有效支撑出行导航、交通监测、交通治理、碳排核算、交通建模等方向.

.. image:: _static/images/MapMatch.PNG
    :align: center

-----------------------------------------------------


.. image:: _static/images/application.PNG
    :align: center

-----------------------------------------------------------------------------

.. note::

   这个项目正处在频繁的升级迭代期, 当前版本v0.3.11, 请用户及时更新.

.. note::
    由于不同用户的电脑差异，geopandas中crs输出格式难以统一，从v0.3.5开始, 全面移除对地理矢量文件的crs检查，用户自己需要确保输入的几何矢量图层的crs为EPSG:4326


文档内容
--------

.. toctree::

    简介
    如何安装
    模块概述
    快速开始
    数据要求
    路网生产与优化
    轨迹生产
    轨迹处理
    地图匹配
    时空可视化
    常用GIS工具
    报错汇总
    类方法汇总
    迭代记录
    贡献名单
