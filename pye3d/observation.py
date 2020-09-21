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

from typing import Sequence

from .constants import _EYE_RADIUS_DEFAULT
from .geometry.primitives import Line, Ellipse
from .geometry.projections import project_line_into_image_plane, unproject_ellipse
from .camera import CameraModel


class Observation(object):
    def __init__(
        self, ellipse: Ellipse, confidence: float, timestamp: float, focal_length: float
    ):
        self.ellipse = ellipse
        self.confidence = confidence
        self.timestamp = timestamp

        self.circle_3d_pair = None
        self.gaze_3d_pair = None
        self.gaze_2d = None
        self.aux_2d = None
        self.aux_3d = None
        self.invalid = True

        unprojection = unproject_ellipse(ellipse, focal_length)
        if not unprojection:
            # unprojecting ellipse failed, invalid observation!
            return

        self.invalid = False

        self.gaze_3d_pair = [
            Line(
                self.circle_3d_pair[i].center,
                self.circle_3d_pair[i].center + self.circle_3d_pair[i].normal,
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
        # TODO!
        raise RuntimeError("NONONO!")


class ObservationStorage:
    def __init__(self, *, camera: CameraModel, confidence_threshold: float):
        self.camera = camera
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def add(self, observation: Observation):
        pass

    @abstractproperty
    def observations(self) -> Sequence[Observation]:
        pass

    @abstractmethod
    def clear(self):
        pass


class BufferedObservationStorage(ObservationStorage):
    def __init__(self, *, buffer_length: int, **kwargs):
        super().__init__(**kwargs)
        self._storage = deque(maxlen=buffer_length)

    def add(self, observation: Observation):
        if observation.invalid:
            return
        if observation.confidence < self.confidence_threshold:
            return

        self._storage.append(observation)

    @property
    def observations(self) -> Sequence[Observation]:
        return list(self._storage)

    def clear(self):
        self._storage.clear()


class BinBufferedObservationStorage(ObservationStorage):
    def __init__(self, *, n_bins_horizontal: int, bin_buffer_length: int, **kwargs):
        super().__init__(**kwargs)
        self.pixels_per_bin = self.camera.resolution[0] / n_bins_horizontal
        self.w = n_bins_horizontal
        self.h = int(round(self.camera.resolution[1] / self.pixels_per_bin))

        # store 2D bins in 1D list for easier iteration and indexing
        self._storage = [
            deque(maxlen=bin_buffer_length) for _ in range(self.w * self.h)
        ]

    def add(self, observation: Observation):
        if observation.invalid:
            return
        if observation.confidence < self.confidence_threshold:
            return

        idx = self._get_bin(observation)
        self._storage[idx].append(observation)

    @property
    def observations(self) -> Sequence[Observation]:
        observation_iterator = chain.from_iterable(self._storage)
        return sorted(observation_iterator, key=lambda obs: obs.timestamp)

    def clear(self):
        for _bin in self._storage:
            _bin.clear()

    def _get_bin(self, observation: Observation) -> int:
        x, y = (
            floor((ellipse_center + resolution / 2) / self.pixels_per_bin)
            for ellipse_center, resolution in zip(
                observation.ellipse.center, self.camera.resolution
            )
        )
        # convert to 1D bin index
        return x + y * self.h
