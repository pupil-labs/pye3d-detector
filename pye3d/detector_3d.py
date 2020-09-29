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
import math
from typing import Dict

import cv2
import numpy as np

from .camera import CameraModel
from .constants import _EYE_RADIUS_DEFAULT
from .cpp.pupil_detection_3d import get_edges
from .cpp.pupil_detection_3d import search_on_sphere as search_on_sphere
from .geometry.primitives import Circle, Ellipse, Sphere
from .geometry.projections import (
    project_circle_into_image_plane,
    project_sphere_into_image_plane,
)
from .geometry.utilities import cart2sph, normalize, sph2cart
from .kalman import KalmanFilter
from .observation import (
    BinBufferedObservationStorage,
    BufferedObservationStorage,
    Observation,
)
from .two_sphere_model import TwoSphereModel

logger = logging.getLogger(__name__)


def ellipse2dict(ellipse: Ellipse) -> Dict:
    return {
        "center": (
            ellipse.center[0],
            ellipse.center[1],
        ),
        "axes": (
            ellipse.minor_radius,
            ellipse.major_radius,
        ),
        "angle": ellipse.angle,
    }


def circle2dict(circle: Circle, flip_y: bool = True) -> Dict:
    flip = -1 if flip_y else 1
    return {
        "center": (
            circle.center[0],
            flip * circle.center[1],
            circle.center[2],
        ),
        "normal": (
            circle.normal[0],
            flip * circle.normal[1],
            circle.normal[2],
        ),
        "radius": float(circle.radius),
    }


class Detector3D(object):
    def __init__(
        self,
        settings={
            "focal_length": 283.0,
            "resolution": (192, 192),
            "maximum_integration_time": 30.0,
            "threshold_data_storage": 0.8,
            "threshold_swirski": 0.7,
            "threshold_kalman": 0.98,
        },
    ):
        self.settings = settings
        self.camera = CameraModel(
            focal_length=settings["focal_length"], resolution=settings["resolution"]
        )

        self.short_term_model = TwoSphereModel(
            camera=self.camera,
            storage=BufferedObservationStorage(
                camera=self.camera,
                confidence_threshold=settings["threshold_data_storage"],
                buffer_length=20,
            ),
        )
        self.long_term_model = TwoSphereModel(
            camera=self.camera,
            storage=BinBufferedObservationStorage(
                camera=self.camera,
                confidence_threshold=0.98,
                n_bins_horizontal=10,
                bin_buffer_length=10,
                forget_min_observations=100,
                forget_min_time=5,
            ),
        )
        self.ultra_long_term_model = TwoSphereModel(
            camera=self.camera,
            storage=BinBufferedObservationStorage(
                camera=self.camera,
                confidence_threshold=0.98,
                n_bins_horizontal=10,
                bin_buffer_length=100,
            ),
        )

        self.kalman_filter = KalmanFilter()

        self.debug_info = None

    def update_and_detect(
        self,
        pupil_datum: Dict,
        frame: np.ndarray,
        apply_refraction_correction: bool = True,
        debug: bool = True,  # TODO: disable again by default
    ):
        # update models
        observation = self._extract_observation(pupil_datum)
        self.update_models(observation)

        # make initial predictions
        pupil_circle = Circle.null()
        if observation.confidence > self.settings["threshold_swirski"]:
            pupil_circle = self.short_term_model.predict_pupil_circle(observation)
        sphere_center = self.long_term_model.sphere_center

        # pupil_circle <-> kalman filter
        # either improve prediction or improve filter
        pupil_circle_kalman = self._predict_from_kalman_filter(pupil_datum["timestamp"])
        if observation.confidence > self.settings["threshold_kalman"]:
            # high confidence: use to correct kalman filter
            self._correct_kalman_filter(pupil_circle)
        elif observation.confidence < self.settings["threshold_swirski"]:
            # low confidence: use kalman result to search for circles in image
            pupil_circle = self._predict_from_3d_search(
                frame, best_guess=pupil_circle_kalman
            )

        # apply refraction correction
        if apply_refraction_correction:
            pupil_circle = self.short_term_model.apply_refraction_correction(
                pupil_circle
            )
            sphere_center = self.long_term_model.corrected_sphere_center

        result = self._prepare_result(
            observation,
            sphere_center,
            pupil_circle,
            pupil_circle_kalman,
        )

        if debug and not observation.invalid:

            def spherical(circle: Circle):
                x, y, z = normalize(circle.normal)
                theta = math.atan2(y, x)
                phi = math.acos(z)
                return [theta, phi]

            incoming = self.long_term_model._disambiguate_circle_3d_pair(
                observation.circle_3d_pair
            )

            projected_short_term = project_sphere_into_image_plane(
                Sphere(self.short_term_model.sphere_center, _EYE_RADIUS_DEFAULT),
                transform=True,
                focal_length=self.camera.focal_length,
                width=self.camera.resolution[0],
                height=self.camera.resolution[1],
            )
            projected_long_term = project_sphere_into_image_plane(
                Sphere(self.long_term_model.sphere_center, _EYE_RADIUS_DEFAULT),
                transform=True,
                focal_length=self.camera.focal_length,
                width=self.camera.resolution[0],
                height=self.camera.resolution[1],
            )
            projected_ultra_long_term = project_sphere_into_image_plane(
                Sphere(self.ultra_long_term_model.sphere_center, _EYE_RADIUS_DEFAULT),
                transform=True,
                focal_length=self.camera.focal_length,
                width=self.camera.resolution[0],
                height=self.camera.resolution[1],
            )

            bin_storage: BinBufferedObservationStorage = self.long_term_model.storage
            bins = bin_storage.get_bin_counts()
            m = np.max(bins)
            if m >= 0:
                bins = bins / m
            bins = np.flip(bins, 0)

            self.debug_info = {
                "incoming": spherical(incoming),
                "predicted": spherical(pupil_circle),
                "short_term_center": list(self.short_term_model.sphere_center),
                "projected_short_term": ellipse2dict(projected_short_term),
                "projected_long_term": ellipse2dict(projected_long_term),
                "projected_ultra_long_term": ellipse2dict(projected_ultra_long_term),
                "Dierkes_lines": [],
            }

            debug_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) / 255

            bin_img_gray = cv2.resize(
                bins, debug_img.shape[:2], interpolation=cv2.INTER_NEAREST
            )
            zeros = np.zeros_like(bin_img_gray)
            b = zeros
            g = zeros
            r = bin_img_gray
            bin_img = cv2.merge([b, g, r])

            debug_img = np.maximum(debug_img, bin_img)

            cv2.imshow("debug", debug_img)
            cv2.waitKey(1)

        if debug:
            result["debug_info"] = self.debug_info

        return result

    def update_models(self, observation: Observation):
        self.short_term_model.add_observation(observation)
        self.long_term_model.add_observation(observation)
        self.ultra_long_term_model.add_observation(observation)

        # TODO: dont trigger every frame? background process maybe?

        if (
            self.short_term_model.n_observations <= 0
            or self.long_term_model.n_observations <= 0
            or self.ultra_long_term_model.n_observations <= 0
        ):
            return

        # update ultra long term model normally
        _, ultra_long_term_3d = self.ultra_long_term_model.estimate_sphere_center()

        # update long term model with ultra long term bias
        long_term_2d, long_term_3d = self.long_term_model.estimate_sphere_center(
            prior_3d=ultra_long_term_3d, prior_strength=0.1
        )

        # update short term model with help of long-term model
        # using 2d center for disambiguation and 3d center as prior bias
        self.short_term_model.estimate_sphere_center(
            from_2d=long_term_2d, prior_3d=long_term_3d, prior_strength=0.1
        )

    def _extract_observation(self, pupil_datum: Dict) -> Observation:
        width, height = self.camera.resolution
        center = (
            +(pupil_datum["ellipse"]["center"][0] - width / 2),
            -(pupil_datum["ellipse"]["center"][1] - height / 2),
        )
        minor_axis = pupil_datum["ellipse"]["axes"][0] / 2.0
        major_axis = pupil_datum["ellipse"]["axes"][1] / 2.0
        angle = -(pupil_datum["ellipse"]["angle"] + 90.0) * np.pi / 180.0
        ellipse = Ellipse(center, minor_axis, major_axis, angle)

        return Observation(
            ellipse,
            pupil_datum["confidence"],
            pupil_datum["timestamp"],
            self.camera.focal_length,
        )

    def _predict_from_kalman_filter(self, timestamp):
        phi, theta, pupil_radius_kalman = self.kalman_filter.predict(timestamp)
        gaze_vector_kalman = sph2cart(phi, theta)
        pupil_center_kalman = (
            self.short_term_model.sphere_center
            + _EYE_RADIUS_DEFAULT * gaze_vector_kalman
        )
        pupil_circle_kalman = Circle(
            pupil_center_kalman, gaze_vector_kalman, pupil_radius_kalman
        )
        return pupil_circle_kalman

    def _correct_kalman_filter(self, observed_pupil_circle: Circle):
        if observed_pupil_circle is not None:
            phi, theta, r = observed_pupil_circle.spherical_representation()
            self.kalman_filter.correct(phi, theta, r)

    def _predict_from_3d_search(self, frame: np.ndarray, best_guess: Circle) -> Circle:
        if best_guess.is_null():
            return best_guess

        frame, frame_roi, edge_frame, edges, roi = get_edges(
            frame,
            best_guess.normal,
            best_guess.radius,
            self.short_term_model.sphere_center,
            _EYE_RADIUS_DEFAULT,
            self.camera.focal_length,
            self.camera.resolution,
            major_axis_factor=2.0,
        )

        if len(edges) <= 0:
            return best_guess

        (gaze_vector, pupil_radius, final_edges, edges_on_sphere) = search_on_sphere(
            edges,
            best_guess.normal,
            best_guess.radius,
            self.short_term_model.sphere_center,
            _EYE_RADIUS_DEFAULT,
            self.camera.focal_length,
            self.camera.resolution,
        )
        pupil_center = (
            self.short_term_model.sphere_center + _EYE_RADIUS_DEFAULT * gaze_vector
        )
        pupil_circle = Circle(pupil_center, gaze_vector, pupil_radius)
        return pupil_circle

    def _prepare_result(
        self,
        observation: Observation,
        sphere_center: np.ndarray,
        pupil_circle: Circle,
        kalman_prediction: Circle,
        flip_y: bool = True,
    ):
        flip = -1 if flip_y else 1

        result = {
            "timestamp": observation.timestamp,
            "sphere": {
                "center": (sphere_center[0], flip * sphere_center[1], sphere_center[2]),
                "radius": _EYE_RADIUS_DEFAULT,
            },
        }

        eye_sphere_projected = project_sphere_into_image_plane(
            Sphere(sphere_center, _EYE_RADIUS_DEFAULT),
            transform=True,
            focal_length=self.camera.focal_length,
            width=self.camera.resolution[0],
            height=self.camera.resolution[1],
        )
        result["projected_sphere"] = ellipse2dict(eye_sphere_projected)

        result["circle_3d"] = circle2dict(pupil_circle, flip_y)
        result["circle_3d_kalman"] = circle2dict(kalman_prediction, flip_y)

        pupil_circle_long_term = self.long_term_model.predict_pupil_circle(observation)
        result["diameter_3d"] = pupil_circle_long_term.radius * 2

        projected_pupil_circle = project_circle_into_image_plane(
            pupil_circle,
            focal_length=self.settings["focal_length"],
            transform=True,
            width=self.settings["resolution"][0],
            height=self.settings["resolution"][1],
        )
        if not projected_pupil_circle:
            projected_pupil_circle = Ellipse(np.asarray([0.0, 0.0]), 0.0, 0.0, 0.0)

        result["ellipse"] = ellipse2dict(projected_pupil_circle)
        result["diameter"] = projected_pupil_circle.major_radius

        # TODO: come up with a confidence measure and probably adjust raw pupil
        # detection confidence?
        result["model_confidence"] = 1.0
        result["confidence"] = observation.confidence

        phi, theta = cart2sph(pupil_circle.normal)
        if not np.any(np.isnan([phi, theta])):
            result["theta"] = theta
            result["phi"] = phi
        else:
            result["theta"] = 0.0
            result["phi"] = 0.0

        return result

    def reset(self):
        self.short_term_model.reset()
        self.long_term_model.reset()
        self.kalman_filter = KalmanFilter()
