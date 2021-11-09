import itertools
from pathlib import Path
from .cpp.refraction_correction import apply_correction_pipeline

import numpy as np
import msgpack

LOAD_DIR = Path(__file__).parent / "refraction_models"
LOAD_VERSION = 1


class ModelDeserializationError(Exception):
    pass


class Refractionizer(object):
    def __init__(self, degree=3, type_="default"):
        self.pipeline_radius_as_list = Refractionizer.load_config_from_msgpack(
            "radius", type_, degree
        )

        self.pipeline_gaze_vector_as_list = Refractionizer.load_config_from_msgpack(
            "gaze_vector", type_, degree
        )

        self.pipeline_sphere_center_as_list = Refractionizer.load_config_from_msgpack(
            "sphere_center", type_, degree
        )

        self.pipeline_pupil_circle_as_list = Refractionizer.load_config_from_msgpack(
            "pupil_circle", type_, degree
        )

    @staticmethod
    def load_config_from_msgpack(feature, type_, degree):
        name = f"{type_}_refraction_model_{feature}_degree_{degree}.msgpack"
        path = LOAD_DIR / name
        with path.open("rb") as file:
            config_model = msgpack.unpack(file)
            Refractionizer._validate_loaded_model_config(config_model)
            try:
                return list(
                    itertools.chain(
                        Refractionizer._polynomial_features_from_config(config_model),
                        Refractionizer._standard_scaler_from_config(config_model),
                        Refractionizer._linear_regression_from_config(config_model),
                    )
                )
            except KeyError as err:
                raise ModelDeserializationError from err

    @staticmethod
    def _validate_loaded_model_config(config_model):
        if not isinstance(config_model, dict) or "version" not in config_model:
            raise ModelDeserializationError("Unrecognized format")
        if config_model["version"] != LOAD_VERSION:
            raise ModelDeserializationError(
                f"Unexpected version `{config_model['version']}` "
                f"(expected `{LOAD_VERSION}``)"
            )

    @staticmethod
    def _polynomial_features_from_config(config_model):
        yield np.array(config_model["steps"]["PolynomialFeatures"]["powers"])

    @staticmethod
    def _standard_scaler_from_config(config_model):
        config_scaler = config_model["steps"]["StandardScaler"]
        yield np.array(config_scaler["mean"])
        yield np.array(config_scaler["var"])

    @staticmethod
    def _linear_regression_from_config(config_model):
        config_lin_reg = config_model["steps"]["LinearRegression"]
        yield np.array(config_lin_reg["coef"])
        yield np.array(config_lin_reg["intercept"])

    @staticmethod
    def _apply_correction_pipeline(X, pipeline_arrays):
        return apply_correction_pipeline(np.asarray(X).T, *pipeline_arrays)

    def correct_radius(self, X):
        return self._apply_correction_pipeline(X, self.pipeline_radius_as_list)

    def correct_gaze_vector(self, X):
        return self._apply_correction_pipeline(X, self.pipeline_gaze_vector_as_list)

    def correct_sphere_center(self, X):
        return self._apply_correction_pipeline(X, self.pipeline_sphere_center_as_list)

    def correct_pupil_circle(self, X):
        return self._apply_correction_pipeline(X, self.pipeline_pupil_circle_as_list)


if __name__ == "__main__":

    refractionizer = Refractionizer()

    print(refractionizer.correct_sphere_center([[0.0, 0.0, 35.0]]))
    print(refractionizer.correct_radius([[0.0, 0.0, 35.0, 0.0, 0.0, -1.0, 2.0]]))
    print(refractionizer.correct_gaze_vector([[0.0, 0.0, 35.0, 0.0, 0.0, -1.0, 2.0]]))
    print(refractionizer.correct_pupil_circle([[0.0, 0.0, 35.0, 0.0, 0.0, -1.0, 2.0]]))
