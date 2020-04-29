import abc
import os
from pathlib import Path
from .cpp.refraction_correction import correct_gaze_vector_cpp
import numpy as np

import joblib

LOAD_DIR = Path(__file__).parent / "refraction_models"


class RefractionizerBase(object):
    @abc.abstractmethod
    def __init__(self):
        pass

    def correct_radius(self, X):
        return self.pipeline_radius.predict(X)

    def correct_gaze_vector(self, X):
        return self.pipeline_gaze_vector.predict(X)

    def correct_sphere_center(self, X):
        return self.pipeline_sphere_center.predict(X)

    def correct_pupil_circle(self, X):
        return self.pipeline_pupil_circle.predict(X)


class Refractionizer(RefractionizerBase):
    def __init__(self, degree=3):

        self.pipeline_radius = joblib.load(
            os.path.join(LOAD_DIR, f"default_refraction_model_radius_degree_{degree}.save")
        )
        self.pipeline_gaze_vector = joblib.load(
            os.path.join(LOAD_DIR, f"default_refraction_model_gaze_vector_degree_{degree}.save")
        )

        self.pipeline_sphere_center = joblib.load(
            os.path.join(
                LOAD_DIR, f"default_refraction_model_sphere_center_degree_{degree}.save"
            )
        )
        self.pipeline_pupil_circle = joblib.load(
            os.path.join(
                LOAD_DIR, f"default_refraction_model_pupil_circle_degree_{degree}.save"
            )
        )

        self.mean_ = self.pipeline_gaze_vector.steps[1][1].mean_
        self.var_ = self.pipeline_gaze_vector.steps[1][1].var_
        self.powers_ = self.pipeline_gaze_vector.steps[0][1].powers_
        self.coef_ = self.pipeline_gaze_vector.steps[2][1].coef_
        self.intercept_ = self.pipeline_gaze_vector.steps[2][1].intercept_
        self.mean_var_ = np.vstack((self.mean_, np.sqrt(self.var_)))
        self.powers__ = self.powers_.astype(np.float)
        self.intercept__ = self.intercept_[np.newaxis, :]

    def correct_gaze_vector_v2(self, x):
         return correct_gaze_vector_cpp(np.asarray(x).T, self.powers__.T, self.mean_var_.T, self.coef_.T, self.intercept__.T)


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
