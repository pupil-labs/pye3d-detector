"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from collections import deque

import numpy as np

from .constants import _EYE_RADIUS_DEFAULT
from .geometry.primitives import Line
from .geometry.projections import project_line_into_image_plane


class Observation(object):
    def __init__(self, ellipse, circle_3d_pair, timestamp=0.0, focal_length=620.0):

        self.ellipse = ellipse
        self.circle_3d_pair = circle_3d_pair
        self.timestamp = timestamp

        self.gaze_3d_pair = [
            Line(
                circle_3d_pair[i].center,
                circle_3d_pair[i].center + circle_3d_pair[i].normal,
            )
            for i in [0, 1]
        ]
        self.gaze_2d = project_line_into_image_plane(self.gaze_3d_pair[0], focal_length)

        self.aux_2d = np.empty((2, 3))
        v = np.reshape(self.gaze_2d.direction, (2, 1))
        self.aux_2d[:, :2] = np.eye(2) - v @ v.T
        self.aux_2d[:, 2] = (np.eye(2) - v @ v.T) @ self.gaze_2d.origin

        self.aux_3d = np.empty((2, 3, 5))
        for i in range(2):
            Dierkes_line = self.get_Dierkes_line(i)
            v = np.reshape(Dierkes_line.direction, (3, 1))
            self.aux_3d[i, :3, :3] = np.eye(3) - v @ v.T
            self.aux_3d[i, :3, 3] = (np.eye(3) - v @ v.T) @ Dierkes_line.origin
            self.aux_3d[i, :3, 4] = Dierkes_line.origin

    def get_Dierkes_line(self, i):
        origin = (
            self.circle_3d_pair[i].center
            - _EYE_RADIUS_DEFAULT * self.circle_3d_pair[i].normal
        )
        direction = self.circle_3d_pair[i].center
        return Line(origin, direction)

    def __bool__(self):
        return True


class ObservationStorage(object):
    def __init__(self, maxlen=5000):
        self.observations = deque(maxlen=maxlen)
        self._gaze_2d_list = deque(maxlen=maxlen)
        self._aux_2d_list = deque(maxlen=maxlen)
        self._aux_3d_list = deque(maxlen=maxlen)
        self._timestamps = deque(maxlen=maxlen)
        self.counter = 0

    def add_observation(self, observation):
        self.observations.append(observation)
        self._gaze_2d_list.append(
            [*observation.gaze_2d.origin, *observation.gaze_2d.direction]
        )
        self._aux_2d_list.append(observation.aux_2d)
        self._aux_3d_list.append(observation.aux_3d)
        self._timestamps.append(observation.timestamp)
        self.counter += 1

    def count(self):
        return self.counter

    def purge(self, cutoff_time):
        N = np.searchsorted(self.timestamps, cutoff_time)
        for _ in range(N):
            self.observations.popleft()
            self._gaze_2d_list.popleft()
            self._aux_2d_list.popleft()
            self._aux_3d_list.popleft()
            self._timestamps.popleft()

    @property
    def aux_2d(self):
        return np.asarray(self._aux_2d_list)

    @property
    def aux_3d(self):
        return np.asarray(self._aux_3d_list)

    @property
    def gaze_2d(self):
        return np.asarray(self._gaze_2d_list)

    @property
    def timestamps(self):
        return np.asarray(self._timestamps)

    def __getitem__(self, item):
        return self.observations[item]

    def __len__(self):
        return len(self.observations)

    def __bool__(self):
        True
