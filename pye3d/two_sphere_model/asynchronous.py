"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import abc
import typing as T

import numpy as np

from ..background_helper import mp
from .abstract import (
    AbstractTwoSphereModel,
    Observation,
    ObservationStorage,
    CameraModel,
    Circle,
)
from .blocking import BlockingTwoSphereModel


class AsyncTwoSphereModel(AbstractTwoSphereModel):
    def __init__(
        self,
        camera: CameraModel,
        storage_cls: T.Type[ObservationStorage] = None,
        storage_kwargs: T.Dict = None,
    ):
        raise NotImplementedError

    def add_observation(self, observation: Observation):
        raise NotImplementedError

    @property
    def n_observations(self) -> int:
        raise NotImplementedError

    def set_sphere_center(self, new_sphere_center: np.ndarray):
        raise NotImplementedError

    def estimate_sphere_center(
        self,
        from_2d: T.Optional[np.ndarray] = None,
        prior_3d: T.Optional[np.ndarray] = None,
        prior_strength: float = 0.0,
    ) -> T.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def estimate_sphere_center_2d(self) -> np.ndarray:
        raise NotImplementedError

    def estimate_sphere_center_3d(
        self,
        sphere_center_2d: np.ndarray,
        prior_3d: T.Optional[np.ndarray] = None,
        prior_strength=0.0,
    ) -> np.ndarray:
        raise NotImplementedError

    # GAZE PREDICTION
    def _extract_unproject_disambiguate(self, pupil_datum: T.Dict) -> Circle:
        raise NotImplementedError

    def _disambiguate_circle_3d_pair(
        self, circle_3d_pair: T.Tuple[Circle, Circle]
    ) -> Circle:
        raise NotImplementedError

    def predict_pupil_circle(
        self, observation: Observation, use_unprojection: bool = False
    ) -> Circle:
        raise NotImplementedError

    def apply_refraction_correction(self, pupil_circle: Circle) -> Circle:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class _SyncedTwoSphereModelAbstract(BlockingTwoSphereModel, metaclass=abc.ABCMeta):
    def __init__(
        self,
        synced_sphere_center: mp.Array,  # c_double_Array_3
        synced_corrected_sphere_center: mp.Array,  # c_double_Array_3
        synced_observation_count: mp.Value,  # c_long
        **kwargs,
    ):
        self.__synced_sphere_center = synced_sphere_center
        self.__synced_corrected_sphere_center = synced_corrected_sphere_center
        self.__synced_observation_count = synced_observation_count
        super().__init__(**kwargs)

    @property
    def sphere_center(self):
        return np.asarray(self.__synced_sphere_center)

    @sphere_center.setter
    def sphere_center(self, coordinates: np.array):
        raise NotImplementedError

    @property
    def corrected_sphere_center(self):
        return np.asarray(self.__synced_corrected_sphere_center)

    @corrected_sphere_center.setter
    def corrected_sphere_center(self, coordinates: np.array):
        raise NotImplementedError

    @property
    def n_observations(self) -> int:
        return self.__synced_observation_count.value


class _SyncedTwoSphereModelFrontend(_SyncedTwoSphereModelAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.storage  # There is no storage in the frontend


class _SyncedTwoSphereModelBackend(_SyncedTwoSphereModelAbstract):
    @property
    def sphere_center(self):
        return super().sphere_center

    @sphere_center.setter
    def sphere_center(self, coordinates: np.array):
        with self.__synced_sphere_center:
            self.__synced_sphere_center[:] = coordinates

    @property
    def corrected_sphere_center(self):
        return super().corrected_sphere_center

    @corrected_sphere_center.setter
    def corrected_sphere_center(self, coordinates: np.array):
        with self.__synced_corrected_sphere_center:
            self.__synced_corrected_sphere_center[:] = coordinates

    def add_observation(self, observation: Observation):
        super().add_observation(observation=observation)
        n_observations = super().n_observations
        with self.__synced_observation_count:
            self.__synced_observation_count.value = n_observations
