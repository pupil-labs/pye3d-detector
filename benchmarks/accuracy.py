from pathlib import Path

import cv2
import numpy as np
from pupil_detectors import Detector2D, Detector3D
from pye3d.detector_3d import Detector3D as Pye3D

here = Path(__file__).parent
video_file = here / "data" / "artificial.mp4"
ts_file = here / "data" / "artificial" / "eye0_timestamps.npy"
gaze_file = here / "data" / "artificial" / "gaze_vectors.npy"


def iterate_frame_data(n=-1):
    for file in (video_file, ts_file, gaze_file):
        if not file.exists():
            raise RuntimeWarning(
                f"Could not find artificial dataset! File {str(file)} missing!"
            )

    cap = cv2.VideoCapture(str(video_file))
    timestamps = np.load(ts_file)
    gaze_vectors = np.load(gaze_file)

    frame_n = 0

    while n == -1 or frame_n <= n:
        success, frame = cap.read()
        if not success:
            return

        yield (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            timestamps[frame_n],
            gaze_vectors[frame_n],
        )

        frame_n += 1


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def run():
    d2d = Detector2D(properties={"pupil_size_max": 180, "pupil_size_min": 10,})
    d3d = Detector3D(d2d.get_properties())
    p3d = Pye3D()
    p3d.settings["focal_length"] = 283.0

    angles3d = []
    anglespye3d = []
    for i, (frame, ts, gaze_vec) in enumerate(iterate_frame_data()):

        result3d = d3d.detect(frame, ts)

        result2d = d2d.detect(frame)
        result2d["timestamp"] = ts
        resultpye3d = p3d.update_and_detect(result2d, frame)
        if i < 100:
            continue

        if result3d["confidence"] > 0.9:
            angles3d.append(
                180 / np.pi * angle_between(gaze_vec, result3d["circle_3d"]["normal"])
            )
        if resultpye3d["confidence"] > 0.9:
            anglespye3d.append(
                180
                / np.pi
                * angle_between(gaze_vec, resultpye3d["circle_3d"]["normal"])
            )
    return {
        "median(angle_error(old3d))": np.median(angles3d),
        "median(angle_error(pye3d))": np.median(anglespye3d),
    }


if __name__ == "__main__":
    print(run())
