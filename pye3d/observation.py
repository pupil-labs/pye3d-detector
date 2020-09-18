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
from math import floor
from abc import abstractmethod, abstractproperty

import numpy as np

from typing import Sequence, Optional, Callable, Tuple

from .constants import _EYE_RADIUS_DEFAULT
from .geometry.primitives import Line, Ellipse
from .geometry.projections import project_line_into_image_plane, unproject_ellipse
from .geometry.utilities import normalize


class UnprojectionError(Exception):
    """Raised when unprojecting ellipse fails."""


class Observation(object):
    def __init__(self, ellipse: Ellipse, timestamp, focal_length):

        self.circle_3d_pair = unproject_ellipse(ellipse, focal_length)
        if not self.circle_3d_pair:
            raise UnprojectionError()

        self.ellipse = ellipse
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


class CameraModel:
    def __init__(self, focal_length: float, resolution: Tuple[float, float]):
        self.focal_length = focal_length
        self.resolution = resolution


class ObservationStorage:
    def __init__(self, *, camera: CameraModel):
        self.camera = camera

    @abstractmethod
    def add(self, ellipse: Ellipse, timestamp: float):
        pass

    def observations(self) -> Sequence[Observation]:
        pass


class BufferedObservationStorage(ObservationStorage):
    def __init__(self, *, buffer_length: int, **kwargs):
        super().__init__(**kwargs)
        self._storage = deque(maxlen=buffer_length)

    def add(self, ellipse: Ellipse, timestamp: float):
        try:
            observation = Observation(
                ellipse,
                timestamp,
                self.camera.focal_length,
            )
        except UnprojectionError:
            return
        self._storage.append(observation)

    def observations(self) -> Sequence[Observation]:
        return list(self._storage)


class BinBufferedObservationStorage(ObservationStorage):
    def __init__(self, *, n_bins_horizontal: int, buffer_length: int, **kwargs):
        super().__init__(**kwargs)
        self.w = n_bins_horizontal
        self.pixels_per_bin = self.camera.resolution[0] / n_bins_horizontal
        self.h = int(round(self.camera.resolution[1] / pixels_per_bin))

        self._storage = [
            [deque(maxlen=buffer_length) for _ in range(self.h)] for _ in range(self.w)
        ]

    def add(self, ellipse: Ellipse, timestamp: float):
        try:
            observation = Observation(
                ellipse,
                timestamp,
                self.camera.focal_length,
            )
        except UnprojectionError:
            return

        x, y = self._get_bin(observation)
        self._storage[x][y].append(observation)

    def observations(self) -> Sequence[Observation]:
        observation_iterator = (
            chain.from_iterable(_bin) for _bin in col for col in self._storage
        )
        return sorted(
            observation_iterator, key=lambda observation: observation.timestamp
        )

    def _get_bin(self, observation: Observation) -> Tuple(int, int):
        return tuple(
            (center + resolution / 2) / self.pixels_per_bin
            for center, resolution in zip(
                observation.ellipse.center, self.camera.resolution
            )
        )
