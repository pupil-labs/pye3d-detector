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
import sys

import numpy as np

from .background_helper import BackgroundProcess
from .constants import _EYE_RADIUS_DEFAULT
from .cpp.pupil_detection_3d import get_edges
from .cpp.pupil_detection_3d import search_on_sphere as search_on_sphere
from .geometry.primitives import Circle, Ellipse, Sphere
from .geometry.projections import (
    project_circle_into_image_plane,
    project_sphere_into_image_plane,
)
from .geometry.utilities import cart2sph, sph2cart
from .kalman import KalmanFilter
from .two_sphere_model import TwoSphereModel

logger = logging.getLogger(__name__)


class Detector3D(object):
    def __init__(
        self,
        settings={
            "focal_length": 283.0,
            "resolution": (192, 192),
            "maximum_integration_time": 30.0,
            "maxlen": 10000,
            "threshold_data_storage": 0.98,
            "threshold_swirski": 0.7,
            "threshold_kalman": 0.98,
        },
    ):
        self.settings = settings

        self.two_sphere_model = TwoSphereModel(settings=self.settings)
        self.currently_optimizing = False
        self.new_observations = False
        self.observation = False

        self.kalman_filter = KalmanFilter()
        self.last_kalman_call = -1

        self.task = BackgroundProcess(TwoSphereModel.deep_sphere_estimate)

        self.debug_result = {}

    def update_and_detect(
        self, pupil_datum, frame, refraction_toggle=True, debug_toggle=False
    ):

        observation = self._estimate_sphere_center(pupil_datum)

        pupil_circle = self._predict_from_two_sphere_model(pupil_datum, observation)

        pupil_circle_kalman = self._predict_from_kalman_filter(
            pupil_datum, pupil_circle
        )

        pupil_circle = self._predict_from_3d_search(
            frame, pupil_datum, pupil_circle, pupil_circle_kalman
        )

        if refraction_toggle:
            pupil_circle = self._apply_refraction_correction(pupil_circle)
            sphere_center = self.two_sphere_model.corrected_sphere_center
        else:
            sphere_center = self.two_sphere_model.sphere_center

        py_result = self._prepare_result(
            sphere_center,
            pupil_datum,
            pupil_circle,
            pupil_circle_kalman,
            flip=-1,
            debug_toggle=debug_toggle,
        )

        return py_result

    def _estimate_sphere_center(self, pupil_datum):
        # CHECK WHETHER NEW SPHERE ESTIMATE IS AVAILABLE
        if self.task.poll():
            result = self.task.recv()
            self._process_sphere_center_estimate(result)

        # SPHERE CENTER UPDATE
        if pupil_datum["confidence"] > self.settings["threshold_data_storage"]:
            observation = self.two_sphere_model.add_to_observation_storage(pupil_datum)
            if observation:
                self.new_observations = True

            if self._sphere_center_should_be_estimated():
                self.currently_optimizing = True
                self.new_observations = False
                self.task.send(
                    self.two_sphere_model.observation_storage.aux_2d,
                    self.two_sphere_model.observation_storage.aux_3d,
                    self.two_sphere_model.observation_storage.gaze_2d,
                )

            return observation

    def _sphere_center_should_be_estimated(self):
        n = self.two_sphere_model.observation_storage.count()
        if (
            ((49 < n and n % 100 == 0) or (20 < n < 50 and n % 10 == 0))
            and self.new_observations
            and not self.currently_optimizing
        ):
            return True
        else:
            False

    def _process_sphere_center_estimate(self, new_sphere_center):
        self.two_sphere_model.set_sphere_center(new_sphere_center)
        self.currently_optimizing = False

    def _predict_from_two_sphere_model(self, pupil_datum, observation=False):
        if pupil_datum["confidence"] > self.settings["threshold_swirski"]:
            if observation:
                pupil_circle = self.two_sphere_model.predict_pupil_circle(
                    observation.circle_3d_pair, from_given_circle_3d_pair=True
                )
            else:
                pupil_circle = self.two_sphere_model.predict_pupil_circle(pupil_datum)
        else:
            pupil_circle = Circle([0, 0, 0], [0, 0, -1], 0)
        return pupil_circle

    def _predict_from_kalman_filter(self, pupil_datum, observed_pupil_circle):
        phi, theta, pupil_radius_kalman = self.kalman_filter.predict(
            pupil_datum["timestamp"]
        )
        gaze_vector_kalman = sph2cart(phi, theta)
        pupil_center_kalman = (
            self.two_sphere_model.sphere_center
            + _EYE_RADIUS_DEFAULT * gaze_vector_kalman
        )
        pupil_circle_kalman = Circle(
            pupil_center_kalman, gaze_vector_kalman, pupil_radius_kalman
        )
        if (
            observed_pupil_circle
            and pupil_datum["confidence"] > self.settings["threshold_kalman"]
        ):
            phi, theta, r = observed_pupil_circle.spherical_representation()
            self.kalman_filter.correct(phi, theta, r)
        return pupil_circle_kalman

    def _predict_from_3d_search(
        self, frame, pupil_datum, pupil_circle, pupil_circle_kalman
    ):
        if (
            pupil_datum["confidence"] <= self.settings["threshold_swirski"]
            and pupil_circle_kalman
        ):

            frame, frame_roi, edge_frame, edges, roi = get_edges(
                frame,
                pupil_circle_kalman.normal,
                pupil_circle_kalman.radius,
                self.two_sphere_model.sphere_center,
                _EYE_RADIUS_DEFAULT,
                self.settings["focal_length"],
                self.settings["resolution"],
                major_axis_factor=2.0,
            )

            if len(edges) > 0:
                (
                    gaze_vector,
                    pupil_radius,
                    final_edges,
                    edges_on_sphere,
                ) = search_on_sphere(
                    edges,
                    pupil_circle_kalman.normal,
                    pupil_circle_kalman.radius,
                    self.two_sphere_model.sphere_center,
                    _EYE_RADIUS_DEFAULT,
                    self.settings["focal_length"],
                    self.settings["resolution"],
                )
                pupil_center = (
                    self.two_sphere_model.sphere_center
                    + _EYE_RADIUS_DEFAULT * gaze_vector
                )
                pupil_circle = Circle(pupil_center, gaze_vector, pupil_radius)

        return pupil_circle

    def _apply_refraction_correction(self, pupil_circle):
        return self.two_sphere_model.apply_refraction_correction(pupil_circle)

    def _prepare_result(
        self,
        sphere_center,
        pupil_datum,
        pupil_circle,
        kalman_prediction=None,
        flip=-1,
        debug_toggle=False,
    ):

        py_result = {
            "topic": "pupil",
            "sphere": {
                "center": (sphere_center[0], flip * sphere_center[1], sphere_center[2]),
                "radius": _EYE_RADIUS_DEFAULT,
            },
        }

        eye_sphere_projected = project_sphere_into_image_plane(
            Sphere(sphere_center, _EYE_RADIUS_DEFAULT),
            transform=True,
            focal_length=self.settings["focal_length"],
            width=self.settings["resolution"][0],
            height=self.settings["resolution"][1],
        )

        py_result["projected_sphere"] = {
            "center": (eye_sphere_projected.center[0], eye_sphere_projected.center[1]),
            "axes": (
                eye_sphere_projected.minor_radius,
                eye_sphere_projected.major_radius,
            ),
            "angle": eye_sphere_projected.angle,
        }

        py_result["circle_3d"] = {
            "center": (
                pupil_circle.center[0],
                flip * pupil_circle.center[1],
                pupil_circle.center[2],
            ),
            "normal": (
                pupil_circle.normal[0],
                flip * pupil_circle.normal[1],
                pupil_circle.normal[2],
            ),
            "radius": pupil_circle.radius,
        }

        py_result["circle_3d_kalman"] = {
            "center": (
                kalman_prediction.center[0],
                flip * kalman_prediction.center[1],
                kalman_prediction.center[2],
            ),
            "normal": (
                kalman_prediction.normal[0],
                flip * kalman_prediction.normal[1],
                kalman_prediction.normal[2],
            ),
            "radius": float(kalman_prediction.radius),
        }

        py_result["confidence"] = pupil_datum["confidence"]
        py_result["timestamp"] = pupil_datum["timestamp"]
        py_result["diameter_3d"] = pupil_circle.radius * 2

        projected_pupil_circle = project_circle_into_image_plane(
            pupil_circle,
            focal_length=self.settings["focal_length"],
            transform=True,
            width=self.settings["resolution"][0],
            height=self.settings["resolution"][1],
        )
        if not projected_pupil_circle:
            projected_pupil_circle = Ellipse(np.asarray([0.0, 0.0]), 0.0, 0.0, 0.0)

        py_result["raw_ellipse"] = pupil_datum["ellipse"]

        py_result["ellipse"] = {
            "center": (
                projected_pupil_circle.center[0],
                projected_pupil_circle.center[1],
            ),
            "axes": (
                projected_pupil_circle.minor_radius,
                projected_pupil_circle.major_radius,
            ),
            "angle": projected_pupil_circle.angle,
        }

        norm_center = (0.0, 0.0)
        py_result["norm_pos"] = norm_center

        py_result["diameter"] = py_result["ellipse"]["axes"][1]

        py_result["model_confidence"] = 1.0
        py_result["model_id"] = 1
        py_result["model_birth_timestamp"] = 0.0

        phi, theta = cart2sph(pupil_circle.normal)
        if not np.any(np.isnan([phi, theta])):
            py_result["theta"] = theta
            py_result["phi"] = phi
        else:
            py_result["theta"] = 0.0
            py_result["phi"] = 0.0

        py_result["method"] = "3d c++"

        if debug_toggle:
            self.debug_result = {
                **py_result,
                "debug_info": self.two_sphere_model.debug_info,
            }

        return py_result

    def reset(self):
        self.two_sphere_model = TwoSphereModel(settings=self.settings)
        self.kalman_filter = KalmanFilter()
        self.last_kalman_call = -1
        self.task.cancel()
        self.currently_optimizing = False
        self.new_observations = False
        self.task = BackgroundProcess(TwoSphereModel.deep_sphere_estimate)
