"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import enum
import logging
import traceback
from typing import Dict, NamedTuple, Type

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
from .two_sphere_model import (
    AbstractTwoSphereModel,
    BlockingTwoSphereModel,
    AsyncTwoSphereModel,
)

logger = logging.getLogger(__name__)


class DetectorMode(enum.Enum):
    blocking = BlockingTwoSphereModel
    asynchronous = AsyncTwoSphereModel


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


class Prediction(NamedTuple):
    sphere_center: np.ndarray
    pupil_circle: Circle


def sigmoid(x, baseline=0.1, amplitude=500.0, center=0.99, width=0.02):
    return baseline + amplitude * 1.0 / (1.0 + np.exp(-(x - center) / width))


class Detector3D(object):
    def __init__(
        self,
        camera: CameraModel,
        threshold_swirski=0.7,
        threshold_kalman=0.98,
        threshold_short_term=0.8,
        threshold_long_term=0.98,
        long_term_buffer_size=30,
        long_term_forget_time=5,
        long_term_forget_observations=300,
        long_term_mode: DetectorMode = DetectorMode.blocking,
    ):
        self._camera = camera
        self._long_term_mode = long_term_mode
        # NOTE: changing settings after intialization can lead to inconsistent behavior
        # if .reset() is not called.
        self._settings = {
            "threshold_swirski": threshold_swirski,
            "threshold_kalman": threshold_kalman,
            "threshold_short_term": threshold_short_term,
            "threshold_long_term": threshold_long_term,
            "long_term_buffer_size": long_term_buffer_size,
            "long_term_forget_time": long_term_forget_time,
            "long_term_forget_observations": long_term_forget_observations,
        }
        self.reset()

    @property
    def camera(self) -> CameraModel:
        return self._camera

    def reset_camera(self, camera: CameraModel):
        """Change camera model and reset detector state."""
        self._camera = camera
        self.reset()

    def reset(self):
        self._cleanup_models()
        self._initialize_models(
            long_term_model_cls=self._long_term_mode.value,
            ultra_long_term_model_cls=self._long_term_mode.value,
        )
        self.kalman_filter = KalmanFilter()

    def _initialize_models(
        self,
        short_term_model_cls: Type[AbstractTwoSphereModel] = BlockingTwoSphereModel,
        long_term_model_cls: Type[AbstractTwoSphereModel] = BlockingTwoSphereModel,
        ultra_long_term_model_cls: Type[
            AbstractTwoSphereModel
        ] = BlockingTwoSphereModel,
    ):
        # Recreate all models. This is required in case any of the settings (incl
        # camera) changed in the meantime.
        self.short_term_model = short_term_model_cls(
            camera=self.camera,
            storage_cls=BufferedObservationStorage,
            storage_kwargs=dict(
                confidence_threshold=self._settings["threshold_short_term"],
                buffer_length=10,
            ),
        )
        self.long_term_model = long_term_model_cls(
            camera=self.camera,
            storage_cls=BinBufferedObservationStorage,
            storage_kwargs=dict(
                camera=self.camera,
                confidence_threshold=self._settings["threshold_long_term"],
                n_bins_horizontal=10,
                bin_buffer_length=self._settings["long_term_buffer_size"],
                forget_min_observations=self._settings["long_term_forget_observations"],
                forget_min_time=self._settings["long_term_forget_time"],
            ),
        )
        self.ultra_long_term_model = ultra_long_term_model_cls(
            camera=self.camera,
            storage_cls=BinBufferedObservationStorage,
            storage_kwargs=dict(
                camera=self.camera,
                confidence_threshold=self._settings["threshold_long_term"],
                n_bins_horizontal=10,
                bin_buffer_length=self._settings["long_term_buffer_size"],
                forget_min_observations=(
                    2 * self._settings["long_term_forget_observations"]
                ),
                forget_min_time=60,
            ),
        )
        # TODO: used for not updating ult-model every frame, will be replaced by background process?
        self.ult_counter = 0

    def _cleanup_models(self):
        try:
            self.short_term_model.cleanup()
            self.long_term_model.cleanup()
            self.ultra_long_term_model.cleanup()
        except AttributeError:
            pass  # models have not been initialized yet

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

        # predict target variables
        sphere_center = self.long_term_model.sphere_center
        pupil_circle = self._predict_pupil_circle(observation, frame)
        prediction_uncorrected = Prediction(sphere_center, pupil_circle)

        # apply refraction correction
        if apply_refraction_correction:
            pupil_circle = self.long_term_model.apply_refraction_correction(
                pupil_circle
            )
            sphere_center = self.long_term_model.corrected_sphere_center
        # Falls back to uncorrected version if correction is disabled
        prediction_corrected = Prediction(sphere_center, pupil_circle)

        result = self._prepare_result(
            observation,
            prediction_uncorrected=prediction_uncorrected,
            prediction_corrected=prediction_corrected,
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
            # update ultra long term model without biases
            # After having seen 1000 frames, only update every 500th frame for
            # performance sanity. TODO: cleanup if moving to background process
            if self.ult_counter < 1000 or self.ult_counter % 500 == 0:
                self.ultra_long_term_model.estimate_sphere_center()
            ultra_long_term_3d = self.ultra_long_term_model.sphere_center

            # update long term model with ultra long term bias
            long_term_2d, long_term_3d = self.long_term_model.estimate_sphere_center(
                prior_3d=ultra_long_term_3d,
                prior_strength=0.1,
            )

            # update short term model with help of long-term model
            # using 2d center for disambiguation and 3d center as prior bias
            # prior strength is set as a funcition of circularity of the 2D pupil

            circularity_mean = self.short_term_model.mean_observation_circularity()
            self.short_term_model.estimate_sphere_center(
                from_2d=long_term_2d,
                prior_3d=long_term_3d,
                prior_strength=sigmoid(circularity_mean),
            )
        except Exception as e:
            # Known issues:
            # - Can raise numpy.linalg.LinAlgError: SVD did not converge
            logger.error("Error updating models:")
            logger.debug(traceback.format_exc())

        self.ult_counter += 1

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

    def _predict_pupil_circle(
        self, observation: Observation, frame: np.ndarray
    ) -> Circle:
        # NOTE: General idea: predict pupil circle from long and short term models based
        # on current observation. Filter results with a kalman filter.

        # Kalman filter needs to be queried every timestamp to update it internally.
        pupil_circle_kalman = self._predict_from_kalman_filter(observation.timestamp)

        if observation.confidence > self._settings["threshold_swirski"]:
            # high-confidence observation, use to construct pupil circle from models

            # short-term-model is best for estimating gaze direction (circle normal) and
            # long-term-model ist more stable for positions (center and radius)
            short_term = self.short_term_model.predict_pupil_circle(observation)
            long_term = self.long_term_model.predict_pupil_circle(observation)
            pupil_circle = Circle(
                normal=short_term.normal,
                center=long_term.center,
                radius=long_term.radius,
            )

            if observation.confidence > self._settings["threshold_kalman"]:
                # very-high-confidence: correct kalman filter
                self._correct_kalman_filter(pupil_circle)

        else:
            # low confidence: use kalman prediction to search for circles in image
            pupil_circle = self._predict_from_3d_search(
                frame, best_guess=pupil_circle_kalman
            )
            # TODO: adjust observation.confidence

        return pupil_circle

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
        if observed_pupil_circle.is_null():
            return

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
        prediction_uncorrected: Prediction,
        prediction_corrected: Prediction,
        flip_y: bool = True,
    ) -> Dict:
        """[summary]

        Args:
            observation (Observation): [description]
            prediction_uncorrected (Prediction): Used for 2d projections
            prediction_corrected (Prediction): Used for 3d data
            flip_y (bool, optional): Flips y-axis of 3d data. Defaults to True.

        Returns:
            Dict: pye3d pupil detection result
        """
        flip = -1 if flip_y else 1

        result = {
            "timestamp": observation.timestamp,
            "sphere": {
                "center": (
                    prediction_corrected.sphere_center[0],
                    prediction_corrected.sphere_center[1] * flip,
                    prediction_corrected.sphere_center[2],
                ),
                "radius": _EYE_RADIUS_DEFAULT,
            },
        }

        eye_sphere_projected = project_sphere_into_image_plane(
            Sphere(prediction_uncorrected.sphere_center, _EYE_RADIUS_DEFAULT),
            transform=True,
            focal_length=self.camera.focal_length,
            width=self.camera.resolution[0],
            height=self.camera.resolution[1],
        )
        result["projected_sphere"] = ellipse2dict(eye_sphere_projected)

        result["circle_3d"] = circle2dict(prediction_corrected.pupil_circle, flip_y)

        pupil_circle_long_term = self.long_term_model.predict_pupil_circle(observation)
        result["diameter_3d"] = pupil_circle_long_term.radius * 2

        projected_pupil_circle = project_circle_into_image_plane(
            prediction_uncorrected.pupil_circle,
            focal_length=self.camera.focal_length,
            transform=True,
            width=self.camera.resolution[0],
            height=self.camera.resolution[1],
        )
        if not projected_pupil_circle:
            projected_pupil_circle = Ellipse(np.asarray([0.0, 0.0]), 0.0, 0.0, 0.0)

        result["ellipse"] = ellipse2dict(projected_pupil_circle)
        result["diameter"] = projected_pupil_circle.major_radius

        result["confidence"] = observation.confidence
        result["confidence_2d"] = observation.confidence_2d
        # TODO: model_confidence is currently require in Pupil for visualization
        # (eyeball outline alpha), but we don't yet have a way of estimating the model
        # confidence. Either remove this and cleanup the visualization in Pupil or come
        # up with a measure for model confidence.
        result["model_confidence"] = 1.0

        phi, theta = cart2sph(prediction_corrected.pupil_circle.normal)
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
