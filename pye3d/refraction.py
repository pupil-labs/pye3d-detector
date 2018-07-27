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
    def __init__(self):

        self.pipeline_radius = joblib.load(
            os.path.join(LOAD_DIR, "default_refraction_model_radius_degree_3.save")
        )
        self.pipeline_gaze_vector = joblib.load(
            os.path.join(LOAD_DIR, "default_refraction_model_gaze_vector_degree_3.save")
        )
        self.pipeline_sphere_center = joblib.load(
            os.path.join(
                LOAD_DIR, "default_refraction_model_sphere_center_degree_3.save"
            )
        )
        self.pipeline_pupil_circle = joblib.load(
            os.path.join(
                LOAD_DIR, "default_refraction_model_pupil_circle_degree_3.save"
            )
        )


class RefractionizerPhysio(RefractionizerBase):
    def __init__(self):

        self.pipeline_radius = joblib.load(
            os.path.join(LOAD_DIR, "physio_refraction_model_radius_degree_3.save")
        )
        self.pipeline_gaze_vector = joblib.load(
            os.path.join(LOAD_DIR, "physio_refraction_model_gaze_vector_degree_3.save")
        )
        self.pipeline_sphere_center = joblib.load(
            os.path.join(
                LOAD_DIR, "physio_refraction_model_sphere_center_degree_3.save"
            )
        )
        self.pipeline_pupil_circle = joblib.load(
            os.path.join(LOAD_DIR, "physio_refraction_model_pupil_circle_degree_3.save")
        )


if __name__ == "__main__":

    refractionizer = Refractionizer()

    print(refractionizer.correct_sphere_center([[0, 0, 35]]))
    print(refractionizer.correct_radius([[0, 0, 35, 0, 0, -1, 2]]))
    print(refractionizer.correct_gaze_vector([[0, 0, 35, 0, 0, -1, 2]]))
    print(refractionizer.correct_pupil_circle([[0, 0, 35, 0, 0, -1, 2]]))
