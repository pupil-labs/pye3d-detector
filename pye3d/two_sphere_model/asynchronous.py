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

from ..constants import DEFAULT_SPHERE_CENTER
from .abstract import (
    AbstractTwoSphereModel,
    CameraModel,
    Circle,
    Observation,
    ObservationStorage,
    SphereCenterEstimates,
)
from .background_helper import BackgroundProcess, mp
from .blocking import BlockingTwoSphereModel

logger = logging.getLogger(__name__)


class AsyncTwoSphereModel(AbstractTwoSphereModel):
    def __init__(
        self,
        camera: CameraModel,
        storage_cls: T.Type[ObservationStorage] = None,
        storage_kwargs: T.Dict = None,
    ):
        synced_sphere_center = mp.Array(ctypes.c_double, 3)
        synced_corrected_sphere_center = mp.Array(ctypes.c_double, 3)
        synced_projected_sphere_center = mp.Array(ctypes.c_double, 2)
        synced_observation_count = mp.Value(ctypes.c_long)

        self._frontend = _SyncedTwoSphereModelFrontend(
            synced_sphere_center,
            synced_corrected_sphere_center,
            synced_projected_sphere_center,
            synced_observation_count,
            camera=camera,
        )
        self._backend_process = BackgroundProcess(
            function=self._process_relayed_commands,
            setup=self._setup_backend,
            setup_args=(
                synced_sphere_center,
                synced_corrected_sphere_center,
                synced_projected_sphere_center,
                synced_observation_count,
            ),
            setup_kwargs=dict(
                camera=camera,
                storage_cls=storage_cls,
                storage_kwargs=storage_kwargs,
            ),
            cleanup=self._cleanup_backend,
            log_handlers=logging.getLogger().handlers,
        )

    @property
    def sphere_center(self) -> np.ndarray:
        return self._frontend.sphere_center

    @property
    def corrected_sphere_center(self) -> np.ndarray:
        return self._frontend.corrected_sphere_center

    def relay_command(self, function_name: str, *args, **kwargs):
        self._backend_process.send(function_name, *args, **kwargs)

    @staticmethod
    def _process_relayed_commands(
        backend: "_SyncedTwoSphereModelBackend", function_name: str, *args, **kwargs
    ):
        function = getattr(backend, function_name)
        result = function(*args, **kwargs)
        return result

    @staticmethod
    def _setup_backend(*args, **kwargs) -> "_SyncedTwoSphereModelBackend":
        logger = logging.getLogger(__name__)
        logger.debug(f"Setting up backend: {args}, {kwargs}")
        return _SyncedTwoSphereModelBackend(*args, **kwargs)

    @staticmethod
    def _cleanup_backend(backend: "_SyncedTwoSphereModelBackend"):
        backend.cleanup()
        logger = logging.getLogger(__name__)
        logger.debug(f"Backend cleaned")

    def add_observation(self, observation: Observation):
        self.relay_command("add_observation", observation)

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
    ) -> SphereCenterEstimates:
        self.relay_command("estimate_sphere_center", from_2d, prior_3d, prior_strength)
        projected_sphere_center = self._frontend.projected_sphere_center
        sphere_center = self._frontend.sphere_center
        return SphereCenterEstimates(projected_sphere_center, sphere_center)

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
        return self._frontend._extract_unproject_disambiguate(pupil_datum)

    def _disambiguate_circle_3d_pair(
        self, circle_3d_pair: T.Tuple[Circle, Circle]
    ) -> Circle:
        return self._frontend._disambiguate_circle_3d_pair(circle_3d_pair)

    def predict_pupil_circle(
        self, observation: Observation, use_unprojection: bool = False
    ) -> Circle:
        return self._frontend.predict_pupil_circle(observation, use_unprojection)

    def apply_refraction_correction(self, pupil_circle: Circle) -> Circle:
        return self._frontend.apply_refraction_correction(pupil_circle)

    def cleanup(self):
        logger.debug("Cancelling backend process")
        self._backend_process.cancel()
        self._frontend.cleanup()

    def mean_observation_circularity(self) -> float:
        raise NotImplementedError


class _SyncedTwoSphereModelAbstract(BlockingTwoSphereModel):
    def __init__(
        self,
        synced_sphere_center: mp.Array,  # c_double_Array_3
        synced_corrected_sphere_center: mp.Array,  # c_double_Array_3
        synced_projected_sphere_center: mp.Array,  # c_double_Array_2
        synced_observation_count: mp.Value,  # c_long
        **kwargs,
    ):
        self._synced_sphere_center = synced_sphere_center
        self._synced_corrected_sphere_center = synced_corrected_sphere_center
        self._synced_projected_sphere_center = synced_projected_sphere_center
        self._synced_observation_count = synced_observation_count
        super().__init__(**kwargs)

    @property
    def sphere_center(self):
        return np.asarray(self._synced_sphere_center)

    @sphere_center.setter
    def sphere_center(self, coordinates: np.array):
        raise NotImplementedError

    @property
    def corrected_sphere_center(self):
        return np.asarray(self._synced_corrected_sphere_center)

    @corrected_sphere_center.setter
    def corrected_sphere_center(self, coordinates: np.array):
        raise NotImplementedError

    @property
    def projected_sphere_center(self):
        return np.asarray(self._synced_projected_sphere_center)

    @projected_sphere_center.setter
    def projected_sphere_center(self, coordinates: np.array):
        raise NotImplementedError

    def mean_observation_circularity(self) -> float:
        raise NotImplementedError


class _SyncedTwoSphereModelFrontend(_SyncedTwoSphereModelAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.storage  # There is no storage in the frontend

    def _set_default_model_params(self):
        pass  # (corrected_)sphere_center cannot be set in the frontend
        with self._synced_sphere_center:
            self._synced_sphere_center[:] = DEFAULT_SPHERE_CENTER

        corrected_sphere_center = self.refractionizer.correct_sphere_center(
            np.asarray([[*self.sphere_center]])
        )[0]
        with self._synced_corrected_sphere_center:
            self._synced_corrected_sphere_center[:] = corrected_sphere_center

    @property
    def n_observations(self) -> int:
        return self._synced_observation_count.value


class _SyncedTwoSphereModelBackend(_SyncedTwoSphereModelAbstract):
    @property
    def sphere_center(self):
        return super().sphere_center

    @sphere_center.setter
    def sphere_center(self, coordinates: np.array):
        with self._synced_sphere_center:
            self._synced_sphere_center[:] = coordinates

    @property
    def corrected_sphere_center(self):
        return super().corrected_sphere_center

    @corrected_sphere_center.setter
    def corrected_sphere_center(self, coordinates: np.array):
        with self._synced_corrected_sphere_center:
            self._synced_corrected_sphere_center[:] = coordinates

    @property
    def projected_sphere_center(self):
        return super().projected_sphere_center

    @projected_sphere_center.setter
    def projected_sphere_center(self, coordinates: np.array):
        with self._synced_projected_sphere_center:
            self._synced_projected_sphere_center[:] = coordinates

    def add_observation(self, observation: Observation):
        super().add_observation(observation=observation)
        n_observations = super().n_observations
        with self._synced_observation_count:
            self._synced_observation_count.value = n_observations

    @property
    def n_observations(self) -> int:
        return self._synced_observation_count.value

    def estimate_sphere_center_2d(self) -> np.ndarray:
        estimated: np.ndarray = super().estimate_sphere_center_2d()
        self.projected_sphere_center = estimated
        return estimated
