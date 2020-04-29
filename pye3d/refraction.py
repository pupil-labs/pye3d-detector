import abc
import os
from pathlib import Path

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
