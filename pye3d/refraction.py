import abc
import os
from pathlib import Path
from .cpp.refraction_correction import apply_correction_pipeline
import numpy as np

import joblib

LOAD_DIR = Path(__file__).parent / "refraction_models"


def pipeline_to_list(pipeline):
    return [pipeline[0].powers_.T.astype(np.float),
            pipeline[1].mean_[np.newaxis, :].T,
            pipeline[1].var_[np.newaxis, :].T,
            pipeline[2].coef_.T,
            pipeline[2].intercept_[:, np.newaxis].T]


class RefractionizerBase(object):
    @abc.abstractmethod
    def __init__(self):
        pass

    @staticmethod
    def _apply_correction_pipeline(X, pipeline_arrays):
        return apply_correction_pipeline(np.asarray(X).T, *pipeline_arrays)

    def correct_radius(self, X, implementation="cpp"):
        if implementation=="cpp":
            y = self._apply_correction_pipeline(X, self.pipeline_radius_as_list)
        else:
            y = self.pipeline_radius.predict(X)
        y.shape = -1, 1
        return y

    def correct_gaze_vector(self, X, implementation="cpp"):
        if implementation=="cpp":
            y = self._apply_correction_pipeline(X, self.pipeline_gaze_vector_as_list)
        else:
            y = self.pipeline_gaze_vector.predict(X)
        y.shape = -1, 3
        return y

    def correct_sphere_center(self, X, implementation="cpp"):
        if implementation=="cpp":
            y = self._apply_correction_pipeline(X, self.pipeline_sphere_center_as_list)
        else:
            y = self.pipeline_sphere_center.predict(X)
        y.shape = -1, 3
        return y

    def correct_pupil_circle(self, X, implementation="cpp"):
        if implementation == "cpp":
            y = self._apply_correction_pipeline(X, self.pipeline_pupil_circle_as_list)
        else:
            y = self.pipeline_pupil_circle.predict(X)
        y.shape = -1, 4
        return y


class Refractionizer(RefractionizerBase):
    def __init__(self, degree=3):

        self.pipeline_radius = joblib.load(
            os.path.join(LOAD_DIR, f"default_refraction_model_radius_degree_{degree}.save")
        )
        self.pipeline_radius_as_list = pipeline_to_list(self.pipeline_radius)

        self.pipeline_gaze_vector = joblib.load(
            os.path.join(LOAD_DIR, f"default_refraction_model_gaze_vector_degree_{degree}.save")
        )
        self.pipeline_gaze_vector_as_list = pipeline_to_list(self.pipeline_gaze_vector)

        self.pipeline_sphere_center = joblib.load(
            os.path.join(
                LOAD_DIR, f"default_refraction_model_sphere_center_degree_{degree}.save"
            )
        )
        self.pipeline_sphere_center_as_list = pipeline_to_list(self.pipeline_sphere_center)

        self.pipeline_pupil_circle = joblib.load(
            os.path.join(
                LOAD_DIR, f"default_refraction_model_pupil_circle_degree_{degree}.save"
            )
        )
        self.pipeline_pupil_circle_as_list = pipeline_to_list(self.pipeline_pupil_circle)


class RefractionizerPhysio(RefractionizerBase):
    def __init__(self, degree=3):

        self.pipeline_radius = joblib.load(
            os.path.join(LOAD_DIR, f"physio_refraction_model_radius_degree_{degree}.save")
        )
        self.pipeline_gaze_vector = joblib.load(
            os.path.join(LOAD_DIR, f"physio_refraction_model_gaze_vector_degree_{degree}.save")
        )
        self.pipeline_sphere_center = joblib.load(
            os.path.join(
                LOAD_DIR, f"physio_refraction_model_sphere_center_degree_{degree}.save"
            )
        )
        self.pipeline_pupil_circle = joblib.load(
            os.path.join(LOAD_DIR, f"physio_refraction_model_pupil_circle_degree_{degree}.save")
        )


if __name__ == "__main__":

    refractionizer = Refractionizer()

    print(refractionizer.correct_sphere_center([[0, 0, 35]]))
    print(refractionizer.correct_radius([[0, 0, 35, 0, 0, -1, 2]]))
    print(refractionizer.correct_gaze_vector([[0, 0, 35, 0, 0, -1, 2]]))
    print(refractionizer.correct_pupil_circle([[0, 0, 35, 0, 0, -1, 2]]))
