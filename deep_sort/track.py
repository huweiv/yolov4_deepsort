# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """
    # 具有状态空间的单个目标轨道（x，y，A，H）和相关联速度，
    # 其中“（x，y）”是bbox的中心，A是高宽比和“H”是高度。
    # mean:初始状态分布的平均向量。
    # covariance:初始状态分布的协方差矩阵
    # track_id: int,唯一的轨迹ID
    # n_init: int,在轨道设置为confirmed之前的连续检测帧数。当一个miss发生时，轨道状态设置为Deleted帧。
    # max_age: int,在侦测状态设置成Deleted前，最大的连续miss数。
    # feature: Optional[ndarray]，特征向量检测的这条轨道的起源。如果为空，则这个特性被添加到'特性'缓存中。
    # trackid：轨迹ID。
    # hits: int，测量更新的总数。
    # age: int，从开始的总帧数
    # time_since_update: int，从上次的测量更新完后，统计的总帧数
    # state: TrackState，当前的侦测状态

    def __init__(self, mean, covariance, track_id, n_init, max_age, class_namess,
                 feature=None):
        # Tracker的构造函数
        self.mean = mean  # 初始的mean
        self.covariance = covariance  # 初始的covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0  # 初始值为0

        self.state = TrackState.Tentative  # 初始为待定状态
        self.features = []
        if feature is not None:
            self.features.append(feature)  # 特征入库

        self._n_init = n_init
        self._max_age = max_age

        self.class_namess = class_namess  #初始化类别

    # 将bbox转换成xywh
    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    # 转换为框的左上角后右下角的坐标点
    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    # 预测，基于kalman filter
    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    # 匹配成功的tracks更新
    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """

        # 更新跟踪框对应的类别
        self.class_namess = detection.to_class_namess()

        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah(), detection.to_class_namess())
        self.features.append(detection.feature)

        self.hits += 1  # 连续命中帧数次数+1
        self.time_since_update = 0  # 是否匹配成功标志位置0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:  # 如果连续命中3帧，则将tracks的状态从Tentative改为Confirmed，视为一个被确认的tracks
            self.state = TrackState.Confirmed

    # 匹配未成功的tracks更新
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        # 待定状态的追踪器直接删除
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        # 已经是confirm状态的追踪器，虽然连续多帧对目标进行了预测，
        # 但中间过程中没有任何一帧能够实现与检测结果的关联，
        # 说明目标可能已经移除了画面，此时直接设置追踪器为待删除状态
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    # 设置三种状态
    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
