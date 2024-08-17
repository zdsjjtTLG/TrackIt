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
                p_noise_std: list or float = 0.01, o_noise_std: list or float = 0.1) -> KalmanFilter:

        initial_state_mean = [initial_x, initial_y, 0, 0]  # initial state, initial speed is 0

        # Observation noise covariance matrix
        if isinstance(o_noise_std, list):
            observation_covariance = np.diag(o_noise_std) ** 2
        else:
            observation_covariance = np.eye(2) * o_noise_std ** 2

        # Process noise covariance matrix
        if isinstance(p_noise_std, list):
            transition_covariance = np.diag(p_noise_std) ** 2
        else:
            transition_covariance = np.eye(4) * p_noise_std ** 2

        # initial state covariance matrix
        initial_state_covariance = np.eye(4) * 1

        # Initialize Kalman filter
        kf = KalmanFilter(
            transition_matrices=self.ts_mat,
            observation_matrices=self.o_mat,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance)
        return kf

    @staticmethod
    def single_step_process(kf: KalmanFilter = None, dt: float = 1.0, previous_state: list or np.ndarray = None,
                            previous_covariance: np.ndarray = None, now_state: list or np.ndarray = None) -> \
            tuple[np.ndarray, np.ndarray]:
        """

        :param kf: KalmanFilter Object
        :param dt: float, time difference, unit: seconds
        :param previous_state: [x, y, vx, vy]
        :param previous_covariance: 4 * 4 matrix
        :param now_state: [x, y, vx, vy]
        :return:
        """
        # Update the state transition matrix
        kf.transition_matrices = np.array([[1, 0, dt, 0],
                                           [0, 1, 0, dt],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])

        # tate estimation based on the current state prediction and observation results
        now_state, now_covariance = kf.filter_update(previous_state, previous_covariance, now_state)
        return now_state, now_covariance


class OffLineTrajectoryKF(TrajectoryKalmanFilter):

    def __init__(self, trajectory_df: pd.DataFrame or gpd.GeoDataFrame = None):

        TrajectoryKalmanFilter.__init__(self, trajectory_df)

    def execute(self, p_noise_std: list or float = 0.01, o_noise_std: list or float = 0.1) -> \
            pd.DataFrame or gpd.GeoDataFrame:

        res_df = pd.DataFrame()
        for agent_id, tj_df in self.trajectory_df.groupby(agent_field):
            _res = self.smooth(tj_df=tj_df, p_noise_std=p_noise_std, o_noise_std=o_noise_std)
            res_df = pd.concat([res_df, _res])
        res_df.reset_index(inplace=True, drop=True)
        return res_df

    def smooth(self, tj_df: pd.DataFrame = None,
               p_noise_std: list or float = 0.01, o_noise_std: list or float = 0.1) -> pd.DataFrame:
        """
        kalman filter to smooth the trajectory
        :param tj_df:
        :param p_noise_std:
        :param o_noise_std:
        :return:
        """
        timestamps = tj_df[time_field]
        observations = tj_df[[gps_field.PLAIN_X, gps_field.PLAIN_Y]].values
        smoothed_states = np.zeros((len(observations), 4))

        # init a new kalman filter
        kf = self.init_kf(initial_x=observations[0, 0], initial_y=observations[0, 1],
                          p_noise_std=p_noise_std, o_noise_std=o_noise_std)

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

        tj_df[gps_field.PLAIN_X] = smoothed_states[:, 0]
        tj_df[gps_field.PLAIN_Y] = smoothed_states[:, 1]
        return tj_df

class OnLineTrajectoryKF(TrajectoryKalmanFilter):
    def __init__(self, trajectory_df: pd.DataFrame = None):
        TrajectoryKalmanFilter.__init__(self, trajectory_df)

        self.kf_group: dict[object, KalmanFilter] = dict()
        self.his_o: dict[object, [np.ndarray, np.ndarray]] = dict()

    def execute(self, p_noise_std: list or float = 0.01, o_noise_std: list or float = 0.1) -> \
            pd.DataFrame or gpd.GeoDataFrame:
        res_df = pd.DataFrame()

        for agent_id, tj_df in self.trajectory_df.groupby(agent_field):

            observations = tj_df[[gps_field.PLAIN_X, gps_field.PLAIN_Y]].values
            t = tj_df[time_field]

            if agent_id in self.kf_group.keys():
                kf = self.kf_group[agent_id]
            else:
                kf = self.init_kf(initial_x=observations[0, 0],
                                  initial_y=observations[0, 1],
                                  o_noise_std=o_noise_std, p_noise_std=p_noise_std)
                self.kf_group[agent_id] = kf

            smoothed_states = np.zeros((len(observations), 4))
            for i in range(1, len(observations)):
                now_state, now_covariance = kf.initial_state_mean, kf.initial_state_covariance

                dt = (t.iloc[i] - t.iloc[i - 1]).total_seconds()  # calculate time interval

                now_state, now_covariance = self.single_step_process(kf=kf, dt=dt, previous_state=now_state,
                                                                     previous_covariance=now_covariance,
                                                                     now_state=observations[i])
                smoothed_states[i, :] = now_state
            tj_df[gps_field.PLAIN_X] = smoothed_states[:, 0]
            tj_df[gps_field.PLAIN_Y] = smoothed_states[:, 1]
            res_df = pd.concat([res_df, tj_df])

        return res_df


# class KalmanFilter(TrajectoryKalmanFilter):
#     """
#     :param trajectory_df:
#     :return:
#     """
#
#     def __init__(self, trajectory_df: pd.DataFrame = None):
#
#         self.trajectory_df = trajectory_df
#
#     def execute(self, p_noise_std: list or float = 0.01, o_noise_std: list or float = 0.1) -> \
#             pd.DataFrame or gpd.GeoDataFrame:
#         res_df = pd.DataFrame()
#         for agent_id, tj_df in self.trajectory_df.groupby(agent_field):
#             _res = self.smooth(tj_df=tj_df, p_noise_std=p_noise_std, o_noise_std=o_noise_std)
#             res_df = pd.concat([res_df, _res])
#         res_df.reset_index(inplace=True, drop=True)
#         return res_df
#     @staticmethod
#     def smooth(tj_df: pd.DataFrame = None, p_noise_std: list or float = 0.01, o_noise_std: list or float = 0.1):
#         """
#         kalman filter to smooth the trajectory
#         :param tj_df:
#         :param p_noise_std:
#         :param o_noise_std:
#         :return:
#         """
#
#
#         timestamps = tj_df[time_field]
#
#         # State transfer matrix and observation matrix
#         ts_mat, o_mat = np.array([[1, 0, 1, 0],
#                                   [0, 1, 0, 1],
#                                   [0, 0, 1, 0],
#                                   [0, 0, 0, 1]]), np.array([[1, 0, 0, 0],
#                                                             [0, 1, 0, 0]])
#         observations = tj_df[[gps_field.PLAIN_X, gps_field.PLAIN_Y]].values  # (x, y) coords
#         initial_state_mean = [observations[0, 0], observations[0, 1], 0, 0]  # Initial state, initial speed is 0
#
#         # Observation noise covariance matrix
#         if isinstance(o_noise_std, list):
#             observation_covariance = np.diag(o_noise_std) ** 2
#         else:
#             observation_covariance = np.eye(2) * o_noise_std ** 2
#
#         # Process noise covariance matrix
#         if isinstance(p_noise_std, list):
#             transition_covariance = np.diag(p_noise_std) ** 2
#         else:
#             transition_covariance = np.eye(4) * p_noise_std ** 2
#
#         # initial state covariance matrix
#         initial_state_covariance = np.eye(4) * 1
#
#         # Initialize Kalman filter
#         kf = KalmanFilter(
#             transition_matrices=ts_mat,
#             observation_matrices=o_mat,
#             initial_state_mean=initial_state_mean,
#             initial_state_covariance=initial_state_covariance,
#             observation_covariance=observation_covariance,
#             transition_covariance=transition_covariance)
#
#         smoothed_states = np.zeros((len(observations), 4))
#
#         # save smooth state
#         smoothed_states[0, :] = initial_state_mean
#
#         # 从第二个状态开始，进行循环迭代
#         current_state = initial_state_mean
#         current_covariance = initial_state_covariance
#         for i in range(1, len(observations)):
#             dt = (timestamps.iloc[i] - timestamps.iloc[i - 1]).total_seconds()  # calculate time interval
#
#             # Update the state transition matrix
#             kf.transition_matrices = np.array([[1, 0, dt, 0],
#                                                [0, 1, 0, dt],
#                                                [0, 0, 1, 0],
#                                                [0, 0, 0, 1]])
#
#             # tate estimation based on the current state prediction and observation results
#             current_state, current_covariance = kf.filter_update(
#                 current_state, current_covariance, observations[i]
#             )
#
#             # save
#             smoothed_states[i, :] = current_state
#
#         tj_df[gps_field.PLAIN_X] = smoothed_states[:, 0]
#         tj_df[gps_field.PLAIN_Y] = smoothed_states[:, 1]
#
#         return tj_df

#
#
# class KalmanFilter(TrajectoryKalmanFilter):
#     """
#     :param trajectory_df:
#     :return:
#     """
#
#     def __init__(self, trajectory_df: pd.DataFrame = None):
#
#         self.trajectory_df = trajectory_df
#
#     def execute(self, p_noise_std: list or float = 0.01, o_noise_std: list or float = 0.1) -> \
#             pd.DataFrame or gpd.GeoDataFrame:
#         res_df = pd.DataFrame()
#         for agent_id, tj_df in self.trajectory_df.groupby(agent_field):
#             _res = self.smooth(tj_df=tj_df, p_noise_std=p_noise_std, o_noise_std=o_noise_std)
#             res_df = pd.concat([res_df, _res])
#         res_df.reset_index(inplace=True, drop=True)
#         return res_df
#     @staticmethod
#     def smooth(tj_df: pd.DataFrame = None, p_noise_std: list or float = 0.01, o_noise_std: list or float = 0.1):
#         """
#         kalman filter to smooth the trajectory
#         :param tj_df:
#         :param p_noise_std:
#         :param o_noise_std:
#         :return:
#         """
#         timestamps = tj_df[time_field]
#
#         # State transfer matrix and observation matrix
#         ts_mat, o_mat = np.array([[1, 0, 1, 0],
#                                   [0, 1, 0, 1],
#                                   [0, 0, 1, 0],
#                                   [0, 0, 0, 1]]), np.array([[1, 0, 0, 0],
#                                                             [0, 1, 0, 0]])
#         observations = tj_df[[gps_field.PLAIN_X, gps_field.PLAIN_Y]].values  # (x, y) coords
#         initial_state_mean = [observations[0, 0], observations[0, 1], 0, 0]  # Initial state, initial speed is 0
#
#         # Observation noise covariance matrix
#         if isinstance(o_noise_std, list):
#             observation_covariance = np.diag(o_noise_std) ** 2
#         else:
#             observation_covariance = np.eye(2) * o_noise_std ** 2
#
#         # Process noise covariance matrix
#         if isinstance(p_noise_std, list):
#             transition_covariance = np.diag(p_noise_std) ** 2
#         else:
#             transition_covariance = np.eye(4) * p_noise_std ** 2
#
#         # initial state covariance matrix
#         initial_state_covariance = np.eye(4) * 1
#
#         # Initialize Kalman filter
#         kf = KalmanFilter(
#             transition_matrices=ts_mat,
#             observation_matrices=o_mat,
#             initial_state_mean=initial_state_mean,
#             initial_state_covariance=initial_state_covariance,
#             observation_covariance=observation_covariance,
#             transition_covariance=transition_covariance)
#
#         smoothed_states = np.zeros((len(observations), 4))
#
#         # save smooth state
#         smoothed_states[0, :] = initial_state_mean
#
#         # 从第二个状态开始，进行循环迭代
#         current_state = initial_state_mean
#         current_covariance = initial_state_covariance
#         for i in range(1, len(observations)):
#             dt = (timestamps.iloc[i] - timestamps.iloc[i - 1]).total_seconds()  # calculate time interval
#
#             # Update the state transition matrix
#             kf.transition_matrices = np.array([[1, 0, dt, 0],
#                                                [0, 1, 0, dt],
#                                                [0, 0, 1, 0],
#                                                [0, 0, 0, 1]])
#
#             # tate estimation based on the current state prediction and observation results
#             current_state, current_covariance = kf.filter_update(
#                 current_state, current_covariance, observations[i]
#             )
#
#             # save
#             smoothed_states[i, :] = current_state
#
#         tj_df[gps_field.PLAIN_X] = smoothed_states[:, 0]
#         tj_df[gps_field.PLAIN_Y] = smoothed_states[:, 1]
#
#         return tj_df
