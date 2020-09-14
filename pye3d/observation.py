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
from itertools import chain

import numpy as np

from .constants import _EYE_RADIUS_DEFAULT
from .geometry.primitives import Line
from .geometry.projections import project_line_into_image_plane
from .geometry.utilities import normalize


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

        self.aux_3d = np.empty((2, 3, 4))
        for i in range(2):
            Dierkes_line = self.get_Dierkes_line(i)
            v = np.reshape(Dierkes_line.direction, (3, 1))
            self.aux_3d[i, :3, :3] = np.eye(3) - v @ v.T
            self.aux_3d[i, :3, 3] = (np.eye(3) - v @ v.T) @ Dierkes_line.origin

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
    BINS = 10

    def __init__(self, maxlen=5000):
        self._observation_bins = [deque(100) for _ in range(bins * bins)]
        # self.observations = deque(maxlen=maxlen)

    def _bin_index(self, x, y):
        return floor(x) * self.BINS + y)

    def add_observation(self, observation):
        direction = normalize(observation.circle_3d_pair[0].normal)
        x, y, _ = direction
        bin_idx = self._bin_index(x, y)
        self._observation_bins[bin_idx].append(observation)

    def purge(self, cutoff_time):
        for obs_bin in self._observation_bins:
            while obs_bin and obs_bin[0].timestamp <= cutoff_time:
                obs_bin.popleft()

    @property
    def observations(self):
        return sorted(
            chain.from_iterable(self._observation_bins), key=lambda obs: obs.timestamp
        )

    @property
    def count(self):
        return sum(len(obs_bin) for obs_bin in self._observation_bins)

    def __bool__(self):
        raise RuntimeError("NEIN NEIN NEIN!")
        True
