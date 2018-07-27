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

import numpy as np

from .constants import _EYE_RADIUS_DEFAULT
from .geometry.intersections import nearest_point_on_sphere_to_line
from .geometry.primitives import Circle, Ellipse, Line
from .geometry.projections import (
    project_line_into_image_plane,
    project_point_into_image_plane,
    unproject_ellipse,
)
from .geometry.utilities import normalize
from .observation import Observation, ObservationStorage
from .refraction import Refractionizer

logger = logging.getLogger(__name__)


class TwoSphereModel(object):
    def __init__(
        self,
        settings={"focal_length": 283.0, "resolution": (192, 192), "maxlen": 10000},
    ):
        self.settings = settings

        self.refractionizer = Refractionizer()

        self.sphere_center = np.asarray([0.0, 0.0, 35.0])
        self.corrected_sphere_center = self.refractionizer.correct_sphere_center(
            np.asarray([[*self.sphere_center]])
        )[0]

        self.observation_storage = ObservationStorage(maxlen=self.settings["maxlen"])

        self.debug_info = {
            "cost": -1.0,
            "residuals": [],
            "angles": [],
            "Dierkes_lines": [],
        }

    # OBSERVATION HANDLING
    def _extract_ellipse(self, pupil_datum):
        width, height = self.settings["resolution"]
        center = (
            +(pupil_datum["ellipse"]["center"][0] - width / 2),
            -(pupil_datum["ellipse"]["center"][1] - height / 2),
        )
        minor_axis = pupil_datum["ellipse"]["axes"][0] / 2.0
        major_axis = pupil_datum["ellipse"]["axes"][1] / 2.0
        angle = -(pupil_datum["ellipse"]["angle"] + 90.0) * np.pi / 180.0
        ellipse = Ellipse(center, minor_axis, major_axis, angle)
        return ellipse

    def add_to_observation_storage(self, pupil_datum):
        ellipse = self._extract_ellipse(pupil_datum)
        circle_3d_pair = unproject_ellipse(ellipse, self.settings["focal_length"])
        if circle_3d_pair:
            observation = Observation(
                ellipse,
                circle_3d_pair,
                pupil_datum["timestamp"],
                self.settings["focal_length"],
            )
            self.observation_storage.add_observation(observation)
            return observation
        else:
            return False

    def set_sphere_center(self, new_sphere_center):
        self.sphere_center = new_sphere_center
        self.corrected_sphere_center = self.refractionizer.correct_sphere_center(
            np.asarray([[*self.sphere_center]])
        )[0]

    def estimate_sphere_center(self, idxs=None, debug=False):

        if not idxs:
            idxs = range(len(self.observation_storage))
        idxs = np.asarray(idxs)

        aux_3d = self.observation_storage.aux_3d[idxs]
        aux_2d = self.observation_storage.aux_2d[idxs]
        gaze_2d = self.observation_storage.gaze_2d[idxs]

        # Estimate projected sphere center by nearest intersection of 2d gaze lines
        sum_aux_2d = np.sum(aux_2d, axis=0)
        projected_sphere_center = np.linalg.pinv(sum_aux_2d[:2, :2]) @ sum_aux_2d[:2, 2]

        # Use projected sphere center for disambiguating Dierkes lines
        dots = np.einsum(
            "ij,ij->i", gaze_2d[:, :2] - projected_sphere_center, gaze_2d[:, 2:]
        )
        disambiguation_indices = (dots < 0).astype(int)

        # Estimate sphere center by nearest intersection of Dierkes lines
        sum_aux_3d = np.sum(
            [aux_3d[n, idx] for n, idx in enumerate(disambiguation_indices)], axis=0
        )
        sphere_center = np.linalg.pinv(sum_aux_3d[:3, :3]) @ sum_aux_3d[:3, 3]

        if debug:

            # Final Dierkes lines
            Dierkes_lines = [
                self.observation_storage.observations[n].get_Dierkes_line(idx)
                for n, idx in enumerate(disambiguation_indices)
            ]
            self.debug_info["Dierkes_lines"] = [
                [
                    *(line.origin - 20 * line.direction),
                    *(line.origin + 20 * line.direction),
                ]
                for line in Dierkes_lines
            ]

            # Calculate residuals and cost
            a_minus_p = [line.origin - sphere_center for line in Dierkes_lines]
            residuals = [
                a_minus_p[n].T @ aux_3d[n, idx, :3, :3] @ a_minus_p[n]
                for n, idx in enumerate(disambiguation_indices)
            ]
            self.debug_info["residuals"] = residuals
            self.debug_info["cost"] = np.sum(residuals)

            # Calculate gaze angles
            angles = [
                180.0
                / np.pi
                * np.arccos(
                    np.dot(
                        np.asarray([0.0, 0.0, -1.0]),
                        self.observation_storage.observations[n]
                        .circle_3d_pair[idx]
                        .normal,
                    )
                )
                for n, idx in enumerate(disambiguation_indices)
            ]
            self.debug_info["angles"] = angles

        return sphere_center

    @staticmethod
    def deep_sphere_estimate(aux_2d, aux_3d, gaze_2d):

        # Estimate projected sphere center by nearest intersection of 2d gaze lines
        sum_aux_2d = np.sum(aux_2d, axis=0)
        projected_sphere_center = np.linalg.pinv(sum_aux_2d[:2, :2]) @ sum_aux_2d[:2, 2]

        # Use projected sphere center for disambiguating Dierkes lines
        dots = np.einsum(
            "ij,ij->i", gaze_2d[:, :2] - projected_sphere_center, gaze_2d[:, 2:]
        )
        disambiguation_indices = (dots < 0).astype(int)

        # Estimate sphere center by nearest intersection of Dierkes lines
        sum_aux_3d = np.sum(
            [aux_3d[n, idx] for n, idx in enumerate(disambiguation_indices)], axis=0
        )
        sphere_center = np.linalg.pinv(sum_aux_3d[:3, :3]) @ sum_aux_3d[:3, 3]

        yield sphere_center

    # GAZE PREDICTION
    def _extract_unproject_disambiguate(self, pupil_datum):
        ellipse = self._extract_ellipse(pupil_datum)
        circle_3d_pair = unproject_ellipse(ellipse, self.settings["focal_length"])
        if circle_3d_pair:
            circle_3d = self._disambiguate_circle_3d_pair(circle_3d_pair)
        else:
            circle_3d = Circle([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0)
        return circle_3d

    def _disambiguate_circle_3d_pair(self, circle_3d_pair):
        circle_center_2d = project_point_into_image_plane(
            circle_3d_pair[0].center, self.settings["focal_length"]
        )
        circle_normal_2d = normalize(
            project_line_into_image_plane(
                Line(circle_3d_pair[0].center, circle_3d_pair[0].normal),
                self.settings["focal_length"],
            ).direction
        )
        sphere_center_2d = project_point_into_image_plane(
            self.sphere_center, self.settings["focal_length"]
        )

        if np.dot(circle_center_2d - sphere_center_2d, circle_normal_2d) >= 0:
            return circle_3d_pair[0]
        else:
            return circle_3d_pair[1]

    def predict_pupil_circle(
        self, input_, from_given_circle_3d_pair=False, use_unprojection=False
    ):
        if from_given_circle_3d_pair:
            circle_3d = self._disambiguate_circle_3d_pair(input_)
        else:
            circle_3d = self._extract_unproject_disambiguate(input_)
        if circle_3d:
            direction = normalize(circle_3d.center)
            nearest_point_on_sphere = nearest_point_on_sphere_to_line(
                self.sphere_center, _EYE_RADIUS_DEFAULT, [0.0, 0.0, 0.0], direction
            )
            if use_unprojection:
                gaze_vector = circle_3d.normal
            else:
                gaze_vector = normalize(nearest_point_on_sphere - self.sphere_center)
            radius = np.linalg.norm(nearest_point_on_sphere) / np.linalg.norm(
                circle_3d.center
            )
            pupil_circle = Circle(nearest_point_on_sphere, gaze_vector, radius)
        else:
            pupil_circle = Circle([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0)
        return pupil_circle

    def apply_refraction_correction(self, pupil_circle):
        input_features = np.asarray(
            [[*self.sphere_center, *pupil_circle.normal, pupil_circle.radius]]
        )
        refraction_corrected_params = self.refractionizer.correct_pupil_circle(
            input_features
        )[0]

        refraction_corrected_gaze_vector = normalize(refraction_corrected_params)[:3]
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

    # UTILITY FUNCTIONS
    def reset(self):
        self.sphere_center = np.array([0.0, 0.0, 35.0])
        self.observation_storage = ObservationStorage(maxlen=self.settings["maxlen"])
        self.debug_info = {
            "cost": -1.0,
            "residuals": [],
            "angles": [],
            "dierkes_lines": [],
        }
