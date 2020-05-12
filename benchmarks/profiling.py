import time
from pathlib import Path

import numpy as np

import cv2
from pupil_detectors import Detector2D, Detector3D
from pye3d.detector_3d import Detector3D as Pye3D


def get_videos():
    for f in (Path(__file__).parent / "data").iterdir():
        if f.is_file() and f.suffix == ".mp4":
            yield f


def iterate_frames(video: Path):
    cap = cv2.VideoCapture(str(video))
    while True:
        success, frame = cap.read()
        if not success:
            return
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def run():

    d2d = Detector2D()
    d3d = Pye3D()

    times = []

    for vid in get_videos():
        for frame_n, frame in enumerate(iterate_frames(vid)):
            t1 = time.perf_counter()

            result_2d = d2d.detect(frame)
            result_2d["timestamp"] = t1
            result_pye3d = d3d.update_and_detect(result_2d, frame)

            t2 = time.perf_counter()
            times.append(t2 - t1)

    if not times:
        raise RuntimeWarning(
            "Could not find any eye videos in ./data/!"
            "Place videos there to profile them!"
        )

    times = np.array(times)
    return {
        "q0.05": np.quantile(times, 0.05),
        "q0.5": np.quantile(times, 0.5),
        "q0.95": np.quantile(times, 0.95),
    }


if __name__ == "__main__":
    print(run())
