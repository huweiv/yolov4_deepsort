# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # 检测结果和跟踪预测结果进行匹配
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        # 匹配成功的tracks更新
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        # 匹配未成功的tracks更新
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 利用未被跟踪匹配的检测框，来创建新的tracker
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # 删除待删除状态的tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        # 更新留下来的tracks的特征集
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    # 检测结果和跟踪预测结果进行匹配
    def _match(self, detections):

        # 计算当前帧每个新检测结果的深度特征与这一层中每个tracks已保存的特征集之间的余弦距离矩阵
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)   # 计算余弦距离
            cost_matrix = linear_assignment.gate_cost_matrix(   # 进行运动信息约束，包括计算马氏距离
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # 将已存在的tracker分为confirmed tracks和unconfirmed tracks
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # 针对之前已经是confirmed tracks，将它们与当前的检测结果进行级联匹配
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age, # 先计算当前帧每个新检测结果的深度特征与这一层中每个tracks已保存的特征集之间的余弦距离矩阵：gated_metric
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        # unconfirmed tracks和上面没有匹配到检测结果的confirmed tracks一起组成iou_track_candidates，
        # 与还没有匹配上的检测结果(unmatched_detections)进行IOU匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]  # 注意这里unmatched_tracks_a中只取了上次匹配成功、这次级联匹配未成功的tracks来进来IOU匹配
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]  # 若unmatched_tracks_a中上次已经未匹配成功的tracks，这次级联匹配又未成功，则没必要再进行IOU匹配了，直接视为未匹配成功的tracks
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,  # 先计算IOU：iou_matching.iou_cost
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    # 利用未被跟踪匹配的检测框，来创建新的tracker
    def _initiate_track(self, detection):
        # 根据初始检测位置初始化新的kalman滤波器的mean和covariance
        mean, covariance = self.kf.initiate(detection.to_xyah())
        # 初始化一个新的tracker
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age, detection.to_class_namess(),  #将与之对应的类别也传过去
            detection.feature))
        # 总的目标id+1
        self._next_id += 1
