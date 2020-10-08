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
from typing import Dict

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
from .geometry.utilities import cart2sph, sph2cart
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
        focal_length=283.0,
        resolution=(192, 192),
        threshold_swirski=0.7,
        threshold_kalman=0.98,
        threshold_short_term=0.8,
        threshold_long_term=0.98,
        long_term_buffer_size=10,
        long_term_forget_time=5,
        long_term_forget_observations=100,
    ):
        self.settings = {
            "focal_length": focal_length,
            "resolution": resolution,
            "threshold_swirski": threshold_swirski,
            "threshold_kalman": threshold_kalman,
            "threshold_short_term": threshold_short_term,
            "threshold_long_term": threshold_long_term,
            "long_term_buffer_size": long_term_buffer_size,
            "long_term_forget_time": long_term_forget_time,
            "long_term_forget_observations": long_term_forget_observations,
        }
        self.camera = CameraModel(focal_length=focal_length, resolution=resolution)

        self.short_term_model = TwoSphereModel(
            camera=self.camera,
            storage=BufferedObservationStorage(
                camera=self.camera,
                confidence_threshold=threshold_short_term,
                buffer_length=10,
            ),
        )
        self.long_term_model = TwoSphereModel(
            camera=self.camera,
            storage=BinBufferedObservationStorage(
                camera=self.camera,
                confidence_threshold=threshold_long_term,
                n_bins_horizontal=10,
                bin_buffer_length=long_term_buffer_size,
                forget_min_observations=long_term_forget_observations,
                forget_min_time=long_term_forget_time,
            ),
        )
        self.ultra_long_term_model = TwoSphereModel(
            camera=self.camera,
            storage=BinBufferedObservationStorage(
                camera=self.camera,
                confidence_threshold=threshold_long_term,
                n_bins_horizontal=10,
                bin_buffer_length=10,
                forget_min_observations=20,
                forget_min_time=60,
            ),
        )

        self.kalman_filter = KalmanFilter()

        # TODO: used for not updating ult-model every frame, will be replaced by background process?
        self.ult_model_update_counter = 0

    def update_and_detect(
        self,
        pupil_datum: Dict,
        frame: np.ndarray,
        apply_refraction_correction: bool = True,
        debug: bool = False,
    ):
        # update models
        observation = self._extract_observation(pupil_datum)
        self.update_models(observation)

        # make initial predictions
        pupil_circle_short_term = Circle.null()
        pupil_circle_long_term = Circle.null()
        if observation.confidence > self.settings["threshold_swirski"]:
            pupil_circle_short_term = self.short_term_model.predict_pupil_circle(
                observation
            )
            pupil_circle_long_term = self.long_term_model.predict_pupil_circle(
                observation
            )
        sphere_center = self.long_term_model.sphere_center
        pupil_circle = Circle(
            pupil_circle_long_term.center,
            pupil_circle_short_term.normal,
            pupil_circle_long_term.radius,
        )

        # pupil_circle <-> kalman filter
        # either improve prediction or improve filter
        pupil_circle_kalman = self._predict_from_kalman_filter(pupil_datum["timestamp"])
        self.used_3dsearch = False
        if observation.confidence > self.settings["threshold_kalman"]:
            # high confidence: use to correct kalman filter
            self._correct_kalman_filter(pupil_circle)
        elif observation.confidence < self.settings["threshold_swirski"]:
            # low confidence: use kalman result to search for circles in image
            pupil_circle = self._predict_from_3d_search(
                frame, best_guess=pupil_circle_kalman
            )
            self.used_3dsearch = True

        # apply refraction correction
        if apply_refraction_correction:
            # TODO: Visualizing this in Pupil is kind of weird, as it does not align
            # well with what the user sees. Maybe we should also always add in the
            # un-corrected data only for visualization?
            pupil_circle = self.long_term_model.apply_refraction_correction(
                pupil_circle
            )
            sphere_center = self.long_term_model.corrected_sphere_center

        result = self._prepare_result(
            observation,
            sphere_center,
            pupil_circle,
        )

        if debug:
            result["debug_info"] = self._collect_debug_info()

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

        try:
            # update ultra long term model normally
            if (
                self.ult_model_update_counter < 1000
                or self.ult_model_update_counter % 500 == 0
            ):
                (
                    _,
                    ultra_long_term_3d,
                ) = self.ultra_long_term_model.estimate_sphere_center()
            else:
                ultra_long_term_3d = self.ultra_long_term_model.sphere_center

            # update long term model with ultra long term bias
            long_term_2d, long_term_3d = self.long_term_model.estimate_sphere_center(
                prior_3d=ultra_long_term_3d, prior_strength=0.1
            )

            # update short term model with help of long-term model
            # using 2d center for disambiguation and 3d center as prior bias
            self.short_term_model.estimate_sphere_center(
                from_2d=long_term_2d, prior_3d=long_term_3d, prior_strength=0.1
            )
        except Exception as e:
            logger.error("Error updating models:")
            logger.error(e)
            raise e
        self.ult_model_update_counter += 1

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
            self.long_term_model.sphere_center,
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
            self.long_term_model.sphere_center,
            _EYE_RADIUS_DEFAULT,
            self.camera.focal_length,
            self.camera.resolution,
        )
        pupil_center = (
            self.long_term_model.sphere_center + _EYE_RADIUS_DEFAULT * gaze_vector
        )
        pupil_circle = Circle(pupil_center, gaze_vector, pupil_radius)
        return pupil_circle

    def _prepare_result(
        self,
        observation: Observation,
        sphere_center: np.ndarray,
        pupil_circle: Circle,
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

        # TODO: remove model confidence? Will require adjustment in Pupil!
        result["model_confidence"] = 1.0
        # TODO: Adjust confidence measure when observation invalid or when having done
        # 3D search from kalman result
        result["confidence"] = observation.confidence

        phi, theta = cart2sph(pupil_circle.normal)
        if not np.any(np.isnan([phi, theta])):
            result["theta"] = theta
            result["phi"] = phi
        else:
            result["theta"] = 0.0
            result["phi"] = 0.0

        return result

    def _collect_debug_info(self):
        debug_info = {}

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
        debug_info["projected_short_term"] = ellipse2dict(projected_short_term)
        debug_info["projected_long_term"] = ellipse2dict(projected_long_term)
        debug_info["projected_ultra_long_term"] = ellipse2dict(
            projected_ultra_long_term
        )

        bin_data = self.long_term_model.storage.get_bin_counts()
        max_bin_level = np.max(bin_data)
        if max_bin_level >= 0:
            bin_data = bin_data / max_bin_level
        bin_data = np.flip(bin_data, axis=0)
        debug_info["bin_data"] = bin_data.tolist()

        # TODO: Pupil visualizer_pye3d.py attempts to draw Dierkes lines. Currently we
        # don't calculate them here, we could probably do that again. Based on which
        # model? Might be hard to do when things run in the background. We might have to
        # remove this from the visualizer_pye3d.py
        debug_info["Dierkes_lines"] = []

        return debug_info

    def reset(self):
        self.short_term_model.reset()
        self.long_term_model.reset()
        self.kalman_filter = KalmanFilter()
