"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import typing as T

import numpy as np

from .abstract import TwoSphereModelAbstract, SphereCenterEstimates
from ..camera import CameraModel
from ..constants import _EYE_RADIUS_DEFAULT, DEFAULT_SPHERE_CENTER
from ..geometry.intersections import nearest_point_on_sphere_to_line
from ..geometry.primitives import Circle, Line
from ..geometry.projections import (
    project_line_into_image_plane,
    project_point_into_image_plane,
    unproject_ellipse,
)
from ..geometry.utilities import normalize
from ..observation import BasicStorage, Observation, ObservationStorage
from ..refraction import Refractionizer

logger = logging.getLogger(__name__)


class TwoSphereModel(TwoSphereModelAbstract):
    def __init__(
        self,
        camera: CameraModel,
        storage_cls: T.Type[ObservationStorage] = None,
        storage_kwargs: T.Dict = None,
    ):
        if storage_cls:
            kwargs = storage_kwargs if storage_kwargs is not None else {}
            self.storage = storage_cls(**kwargs)
        else:
            self.storage = BasicStorage()
        self.camera = camera

        self.refractionizer = Refractionizer()
        self._set_default_model_params()

    @property
    def sphere_center(self) -> np.ndarray:
        return self._sphere_center

    @sphere_center.setter
    def sphere_center(self, coordinates: np.ndarray):
        self._sphere_center = coordinates

    @property
    def corrected_sphere_center(self) -> np.ndarray:
        return self._corrected_sphere_center

    @corrected_sphere_center.setter
    def corrected_sphere_center(self, coordinates: np.ndarray):
        self._corrected_sphere_center = coordinates

    @property
    def projected_sphere_center(self) -> np.ndarray:
        return self._projected_sphere_center

    @projected_sphere_center.setter
    def projected_sphere_center(self, projected_sphere_center: np.ndarray):
        self._projected_sphere_center = projected_sphere_center

    def _set_default_model_params(self):
        # Overwrite in subclasses that do not allow setting these attributes
        self._sphere_center = np.asarray(DEFAULT_SPHERE_CENTER)
        self._corrected_sphere_center = self.refractionizer.correct_sphere_center(
            np.asarray([[*self.sphere_center]])
        )[0]
        self.rms_residual = np.nan

    def add_observation(self, observation: Observation):
        self.storage.add(observation)

    @property
    def n_observations(self) -> int:
        return self.storage.count()

    def set_sphere_center(self, new_sphere_center):
        self.sphere_center = new_sphere_center
        self.corrected_sphere_center = self.refractionizer.correct_sphere_center(
            np.asarray([[*self.sphere_center]])
        )[0]

    def estimate_sphere_center(self, from_2d=None, prior_3d=None, prior_strength=0.0):
        self.projected_sphere_center = (
            from_2d if from_2d is not None else self.estimate_sphere_center_2d()
        )
        sphere_center, rms_residual = self.estimate_sphere_center_3d(
            self.projected_sphere_center, prior_3d, prior_strength
        )
        self.set_sphere_center(sphere_center)
        self.rms_residual = rms_residual
        return SphereCenterEstimates(self.projected_sphere_center, sphere_center)

    def estimate_sphere_center_2d(self):
        observations = self.storage.observations
        aux_2d = np.array([obs.aux_2d for obs in observations])

        # Estimate projected sphere center by nearest intersection of 2d gaze lines
        sum_aux_2d = aux_2d.sum(axis=0)
        projected_sphere_center = np.linalg.pinv(sum_aux_2d[:2, :2]) @ sum_aux_2d[:2, 2]

        return projected_sphere_center

    def estimate_sphere_center_3d(
        self,
        sphere_center_2d,
        prior_3d=None,
        prior_strength=0.0,
        calculate_rms_residual=True,
    ):
        observations = self.storage.observations
        aux_3d = np.array([obs.aux_3d for obs in observations])
        gaze_2d = np.array(
            [[*obs.gaze_2d.origin, *obs.gaze_2d.direction] for obs in observations]
        )

        # Disambiguate Dierkes lines
        # We want gaze_2d to points towards the sphere center. gaze_2d was collected
        # from Dierkes[0]. If it points into the correct direction, we know that
        # Dierkes[0] is the correct one to use, otherwise we need to use Dierkes[1]. We
        # can check that with the sign of the dot product.
        gaze_2d_origins = gaze_2d[:, :2]
        gaze_2d_directions = gaze_2d[:, 2:]
        gaze_2d_towards_center = gaze_2d_origins - sphere_center_2d

        dot_products = np.sum(gaze_2d_towards_center * gaze_2d_directions, axis=1)
        disambiguation_indices = np.where(dot_products < 0, 1, 0)

        observation_indices = np.arange(len(disambiguation_indices))
        aux_3d_disambiguated = aux_3d[observation_indices, disambiguation_indices, :, :]

        # Estimate sphere center by nearest intersection of Dierkes lines
        sum_aux_3d = aux_3d_disambiguated.sum(axis=0)

        if prior_3d is None:
            sphere_center = np.linalg.pinv(sum_aux_3d[:3, :3]) @ sum_aux_3d[:3, 3]
        else:
            sphere_center = np.linalg.pinv(
                sum_aux_3d[:3, :3] + prior_strength * np.eye(3)
            ) @ (sum_aux_3d[:3, 3] + prior_strength * prior_3d)

        rms_residual = None
        if calculate_rms_residual:
            # Here we use eq. (10) in https://docplayer.net/21072949-Least-squares-intersection-of-lines.html.
            origins_dierkes_lines = np.array(
                [
                    observations[i].get_Dierkes_line(disambiguation_indices[i]).origin
                    for i in observation_indices
                ]
            )
            origins_dierkes_lines.shape = -1, 3, 1
            deltas = origins_dierkes_lines - sphere_center[:, np.newaxis]
            tmp = np.einsum("ijk,ikl->ijl", aux_3d_disambiguated[:, :3, :3], deltas)
            squared_residuals = np.einsum(
                "ikj,ijk->i", np.transpose(deltas, (0, 2, 1)), tmp
            )
            rms_residual = np.clip(squared_residuals, 0.0, None)
            rms_residual = np.mean(np.sqrt(rms_residual))

        return sphere_center, rms_residual

    # GAZE PREDICTION
    def _extract_unproject_disambiguate(self, pupil_datum):
        ellipse = self._extract_ellipse(pupil_datum)
        circle_3d_pair = unproject_ellipse(ellipse, self.camera.focal_length)
        if circle_3d_pair:
            circle_3d = self._disambiguate_circle_3d_pair(circle_3d_pair)
        else:
            circle_3d = Circle([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0)
        return circle_3d

    def _disambiguate_circle_3d_pair(self, circle_3d_pair):
        circle_center_2d = project_point_into_image_plane(
            circle_3d_pair[0].center, self.camera.focal_length
        )
        circle_normal_2d = normalize(
            project_line_into_image_plane(
                Line(circle_3d_pair[0].center, circle_3d_pair[0].normal),
                self.camera.focal_length,
            ).direction
        )
        sphere_center_2d = project_point_into_image_plane(
            self.sphere_center, self.camera.focal_length
        )

        if np.dot(circle_center_2d - sphere_center_2d, circle_normal_2d) >= 0:
            return circle_3d_pair[0]
        else:
            return circle_3d_pair[1]

    def predict_pupil_circle(
        self, observation: Observation, use_unprojection: bool = False
    ) -> Circle:
        if observation.invalid:
            return Circle.null()

        circle_3d = self._disambiguate_circle_3d_pair(observation.circle_3d_pair)
        unprojection_depth = np.linalg.norm(circle_3d.center)
        direction = circle_3d.center / unprojection_depth

        nearest_point_on_sphere = nearest_point_on_sphere_to_line(
            self.sphere_center, _EYE_RADIUS_DEFAULT, [0.0, 0.0, 0.0], direction
        )

        if use_unprojection:
            gaze_vector = circle_3d.normal
        else:
            gaze_vector = normalize(nearest_point_on_sphere - self.sphere_center)

        radius = np.linalg.norm(nearest_point_on_sphere) / unprojection_depth
        pupil_circle = Circle(nearest_point_on_sphere, gaze_vector, radius)
        return pupil_circle

    def apply_refraction_correction(self, pupil_circle):
        input_features = np.asarray(
            [[*self.sphere_center, *pupil_circle.normal, pupil_circle.radius]]
        )
        refraction_corrected_params = self.refractionizer.correct_pupil_circle(
            input_features
        )[0]

        refraction_corrected_gaze_vector = normalize(refraction_corrected_params[:3])
        refraction_corrected_radius = refraction_corrected_params[-1]
        refraction_corrected_pupil_center = (
            self.corrected_sphere_center
            + _EYE_RADIUS_DEFAULT * refraction_corrected_gaze_vector
        )

        refraction_corrected_pupil_circle = Circle(
            refraction_corrected_pupil_center,
            refraction_corrected_gaze_vector,
            refraction_corrected_radius,
        )

        return refraction_corrected_pupil_circle

    def mean_observation_circularity(self):
        observation_circularities = [
            observation.ellipse.circularity()
            for observation in self.storage.observations
        ]
        return np.mean(observation_circularities)

    def cleanup(self):
        pass
