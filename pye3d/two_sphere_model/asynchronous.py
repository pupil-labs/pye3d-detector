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
import ctypes
import logging
import typing as T

import numpy as np

from .background_helper import mp, BackgroundProcess, Handler
from .abstract import (
    AbstractTwoSphereModel,
    Observation,
    ObservationStorage,
    CameraModel,
    Circle,
)
from .blocking import BlockingTwoSphereModel

logger = logging.getLogger(__name__)


class AsyncTwoSphereModel(AbstractTwoSphereModel):
    def __init__(
        self,
        camera: CameraModel,
        storage_cls: T.Type[ObservationStorage] = None,
        storage_kwargs: T.Dict = None,
        log_handler: T.Optional[Handler] = None,
    ):
        synced_sphere_center = mp.Array(ctypes.c_double, 3)
        synced_corrected_sphere_center = mp.Array(ctypes.c_double, 3)
        synced_observation_count = mp.Value(ctypes.c_long)

        self._frontend = _SyncedTwoSphereModelFrontend(
            synced_sphere_center,
            synced_corrected_sphere_center,
            synced_observation_count,
            camera=camera,
        )

        self._backend_process = BackgroundProcess(
            function=self._relay_commands,
            setup=self._setup_backend,
            setup_args=(
                synced_sphere_center,
                synced_corrected_sphere_center,
                synced_observation_count,
                camera,
                storage_cls,
                storage_kwargs,
            ),
            cleanup=self._cleanup_backend,
            log_handler=log_handler,
        )

    @staticmethod
    def _relay_commands(
        backend: "_SyncedTwoSphereModelBackend", function_name: str, *args, **kwargs
    ):
        logger.debug(f"Relayed: {backend}.{function_name}({args}, {kwargs})")
        function = getattr(backend, function_name)
        result = function(*args, **kwargs)
        logger.debug(f"Result: {result}")
        return result

    @staticmethod
    def _setup_backend(*args, **kwargs) -> "_SyncedTwoSphereModelBackend":
        logger.debug(f"Setting up backend: {args}, {kwargs}")
        return _SyncedTwoSphereModelBackend(*args, **kwargs)

    @staticmethod
    def _cleanup_backend(backend: "_SyncedTwoSphereModelBackend"):
        backend.cleanup()
        logger.debug(f"Backend cleaned")

    def add_observation(self, observation: Observation):
        self._backend_process.send("add_observation", observation)

    @property
    def n_observations(self) -> int:
        return self._frontend.n_observations

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
        return BlockingTwoSphereModel._extract_unproject_disambiguate(self, pupil_datum)

    def _disambiguate_circle_3d_pair(
        self, circle_3d_pair: T.Tuple[Circle, Circle]
    ) -> Circle:
        return BlockingTwoSphereModel._disambiguate_circle_3d_pair(self, circle_3d_pair)

    def predict_pupil_circle(
        self, observation: Observation, use_unprojection: bool = False
    ) -> Circle:
        return BlockingTwoSphereModel.predict_pupil_circle(
            self, observation, use_unprojection
        )

    def apply_refraction_correction(self, pupil_circle: Circle) -> Circle:
        return BlockingTwoSphereModel.apply_refraction_correction(self, pupil_circle)

    def reset(self):
        logger.debug("Cancelling backend process")
        self._backend_process.cancel()


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
