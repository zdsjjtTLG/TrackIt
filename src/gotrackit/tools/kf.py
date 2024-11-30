# -- coding: utf-8 --
# @Time    : 2024/8/16 10:59
# @Author  : TangKai
# @Team    : ZheChengData


""" Trajectory Kalman Filter"""

import numpy as np
import pandas as pd
import geopandas as gpd
from ..GlobalVal import GpsField
from pykalman import KalmanFilter
from ..tools.time_build import build_time_col


gps_field = GpsField()
agent_field, lng_field, lat_field, time_field = \
    gps_field.AGENT_ID_FIELD, gps_field.LNG_FIELD, gps_field.LAT_FIELD, gps_field.TIME_FIELD
speed_field, x_speed_field, y_speed_field = gps_field.SPEED_FIELD, gps_field.X_SPEED_FIELD, gps_field.Y_SPEED_FIELD


class TrajectoryKalmanFilter(object):
    def __init__(self, trajectory_df: pd.DataFrame or gpd.GeoDataFrame = None):
        self.trajectory_df = trajectory_df

        # state transfer matrix and observation matrix
        self.ts_mat, self.o_mat = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]]), np.array([[1, 0, 0, 0],
                                                                      [0, 1, 0, 0]])

    def init_kf(self, initial_x: float = 1.0, initial_y: float = 1.0,
                p_deviation: list or float = 0.01, o_deviation: list or float = 0.1) -> KalmanFilter:

        initial_state_mean = [initial_x, initial_y, 0, 0]  # initial state, initial speed is 0

        # Observation noise covariance matrix
        if isinstance(o_deviation, list):
            o_cov = np.diag(o_deviation) ** 2
        else:
            o_cov = np.eye(2) * o_deviation ** 2

        # Process noise covariance matrix
        if isinstance(p_deviation, list):
            t_cov = np.diag(p_deviation) ** 2
        else:
            t_cov = np.eye(4) * p_deviation ** 2

        # initial state covariance matrix
        initial_state_covariance = np.eye(4) * 1

        # Initialize Kalman filter
        kf = KalmanFilter(observation_matrices=self.o_mat, transition_matrices=self.ts_mat,
                          initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance,
                          observation_covariance=o_cov, transition_covariance=t_cov)
        return kf

    @staticmethod
    def single_step_process(kf: KalmanFilter = None, dt: float = 1.0, previous_state: list or np.ndarray = None,
                            previous_covariance: np.ndarray = None, now_state: list or np.ndarray = None) -> \
            tuple[np.ndarray, np.ndarray]:
        """

        Args:
            kf: KalmanFilter Object
            dt: time difference, unit: seconds
            previous_state: [x, y, vx, vy]
            previous_covariance: 4 * 4 matrix
            now_state: [x, y, vx, vy]

        Returns:

        """
        # Update the state transition matrix
        kf.transition_matrices = np.array([[1, 0, dt, 0],
                                           [0, 1, 0, dt],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])

        # state estimation based on the current state prediction and observation results
        now_state, now_covariance = kf.filter_update(previous_state, previous_covariance, now_state)
        return now_state, now_covariance


class OffLineTrajectoryKF(TrajectoryKalmanFilter):

    def __init__(self, trajectory_df: pd.DataFrame or gpd.GeoDataFrame = None,
                 x_field: str = 'lng', y_field: str = 'lat'):

        TrajectoryKalmanFilter.__init__(self, trajectory_df)
        self.x_field, self.y_field = x_field, y_field

    def execute(self, p_deviation: list or float = 0.01, o_deviation: list or float = 0.1) -> \
            pd.DataFrame or gpd.GeoDataFrame:

        res_df = self.trajectory_df.groupby(agent_field).apply(lambda df:
                                                               self.smooth(tj_df=df,
                                                                           p_deviation=p_deviation,
                                                                           o_deviation=o_deviation)).reset_index(
            inplace=False, drop=True)
        return res_df

    def smooth(self, tj_df: pd.DataFrame = None,
               p_deviation: list or float = 0.01, o_deviation: list or float = 0.1) -> pd.DataFrame:
        """
        kalman filter to smooth the trajectory, modify x_field, y_field, and add x_speed and y_speed field
        :param tj_df:
        :param p_deviation:
        :param o_deviation:
        :return:
        """
        if len(tj_df) <= 1:
            tj_df[x_speed_field], tj_df[y_speed_field] = 0, 0
            return tj_df
        timestamps = tj_df[time_field]
        observations = tj_df[[self.x_field, self.y_field]].values
        smoothed_states = np.zeros((len(observations), 4))

        # init a new kalman filter
        kf = self.init_kf(initial_x=observations[0, 0], initial_y=observations[0, 1],
                          p_deviation=p_deviation, o_deviation=o_deviation)

        # save initial state
        smoothed_states[0, :] = [observations[0, 0], observations[0, 1], 0, 0]

        # 从第二个状态开始，进行循环迭代
        now_state = kf.initial_state_mean
        now_covariance = kf.initial_state_covariance

        for i in range(1, len(observations)):
            dt = (timestamps.iloc[i] - timestamps.iloc[i - 1]).total_seconds()  # calculate time interval

            now_state, now_covariance = self.single_step_process(kf=kf, dt=dt, previous_state=now_state,
                                                                 previous_covariance=now_covariance,
                                                                 now_state=observations[i])

            # save
            smoothed_states[i, :] = now_state

        tj_df[self.x_field] = smoothed_states[:, 0]
        tj_df[self.y_field] = smoothed_states[:, 1]
        tj_df[x_speed_field] = smoothed_states[:, 2]
        tj_df[y_speed_field] = smoothed_states[:, 3]
        return tj_df


class OnLineTrajectoryKF(TrajectoryKalmanFilter):
    def __init__(self, trajectory_df: pd.DataFrame = None, time_format: str = '%Y-%m-%d %H:%M:%S', time_unit: str = 's',
                 x_field: str = 'lng', y_field: str = 'lat'):
        """实时卡尔曼滤波类

        提供实时平滑功能

        Args:
            trajectory_df: 轨迹数据
            time_format: GPS轨迹数据中时间列的格式化字符串模板
            time_unit: 轨迹数据中时间列的时间单位
            x_field: 轨迹数据表中表示经度的字段名称
            y_field: 轨迹数据表中表示纬度的字段名称
        """

        TrajectoryKalmanFilter.__init__(self, trajectory_df)

        self.kf_group: dict[object, KalmanFilter] = dict()
        self.his_o: dict[object, [np.ndarray, np.ndarray]] = dict()
        self.his_t: dict = dict()
        self.x_field, self.y_field = x_field, y_field
        self.time_format, self.time_unit = time_format, time_unit
        if trajectory_df is not None:
            build_time_col(df=self.trajectory_df, time_format=time_format, time_unit=time_unit)

    def kf_smooth(self, p_deviation: list or float = 0.01, o_deviation: list or float = 0.1,
                  time_gap_threshold: float = 1800.0) -> \
            pd.DataFrame or gpd.GeoDataFrame:
        """平滑
        实时滤波平滑

        Args:
            p_deviation: 过程噪声的标准差
            o_deviation: 观测噪声的标准差, 该值越小, 平滑后的结果就越接近原轨迹
            time_gap_threshold: 时间阈值，如果某agent的当前批GPS数据的最早定位时间和上批GPS数据的最晚定位时间的差值超过该值，则不参考历史概率链进行匹配计算

        Returns:
            平滑后的轨迹表
        """
        res_df = pd.DataFrame()

        for agent_id, tj_df in self.trajectory_df.groupby(agent_field):

            observations = tj_df[[self.x_field, self.y_field]].values
            t = tj_df[time_field]

            start_index = 0
            if agent_id in self.kf_group.keys() and \
                    (t.iloc[0] - self.his_t[agent_id]).total_seconds() <= time_gap_threshold:
                kf = self.kf_group[agent_id]
                smoothed_states = np.zeros((len(observations), 4))
            else:
                kf = self.init_kf(initial_x=observations[0, 0],
                                  initial_y=observations[0, 1],
                                  o_deviation=o_deviation, p_deviation=p_deviation)
                self.kf_group[agent_id] = kf
                self.his_o[agent_id] = [kf.initial_state_mean, kf.initial_state_covariance]
                self.his_t[agent_id] = t.iloc[0]
                smoothed_states = np.zeros((len(observations), 4))

                # save initial state
                smoothed_states[0, :] = [observations[0, 0], observations[0, 1], 0, 0]

                start_index = 1

            for i in range(start_index, len(observations)):
                now_state, now_covariance = self.his_o[agent_id]
                now_t = t.iloc[i]
                dt = (now_t - self.his_t[agent_id]).total_seconds()  # calculate time interval

                now_state, now_covariance = self.single_step_process(kf=kf, dt=dt, previous_state=now_state,
                                                                     previous_covariance=now_covariance,
                                                                     now_state=observations[i])
                smoothed_states[i, :] = now_state
                self.his_o[agent_id] = [now_state, now_covariance]
                self.his_t[agent_id] = now_t

            tj_df[self.x_field] = smoothed_states[:, 0]
            tj_df[self.y_field] = smoothed_states[:, 1]
            tj_df[x_speed_field] = smoothed_states[:, 2]
            tj_df[y_speed_field] = smoothed_states[:, 3]
            res_df = pd.concat([res_df, tj_df])
        res_df.reset_index(inplace=True, drop=True)
        return res_df

    def renew_trajectory(self, trajectory_df: pd.DataFrame = None):
        """更新数据
        更新待处理的轨迹点数据

        Args:
            trajectory_df: 轨迹数据

        Returns:
        """
        self.trajectory_df = trajectory_df
        build_time_col(df=self.trajectory_df, time_format=self.time_format, time_unit=self.time_unit)
