import cv2
import numpy as np


class KalmanFilter:
    def __init__(self):
        self.filter = cv2.KalmanFilter(7, 3, 0, cv2.CV_32F)
        self.filter.measurementMatrix = np.asarray(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.filter.processNoiseCov = 1e-4 * np.eye(7, dtype=np.float32)
        self.filter.measurementNoiseCov = 1e-5 * np.eye(3, dtype=np.float32)
        self.filter.measurementNoiseCov[2][2] = 0.1
        self.filter.statePost = np.asarray([0, 0, 0, 0, 0, 0, 2.0], dtype=np.float32)
        self.filter.errorCovPost = np.eye(7, dtype=np.float32)
        self.last_call = -1

    def predict(self, t):
        if self.last_call != -1 and t > self.last_call:
            dt = t - self.last_call
            self.filter.transitionMatrix = np.asarray(
                [
                    [1, 0, dt, 0, 0.5 * dt * dt, 0, 0],
                    [0, 1, 0, dt, 0, 0.5 * dt * dt, 0],
                    [0, 0, 1, 0, dt, 0, 0],
                    [0, 0, 0, 1, 0, dt, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
            prediction = self.filter.predict()
            phi, theta, pupil_radius = (
                prediction[0][0],
                prediction[1][0],
                prediction[6][0],
            )
        else:
            phi, theta, pupil_radius = -np.pi / 2, np.pi / 2, 0

        self.last_call = t

        return phi, theta, pupil_radius

    def correct(self, phi, theta, radius):
        self.filter.correct(np.asarray([phi, theta, radius], dtype=np.float32))
