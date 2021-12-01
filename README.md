[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build pye3d](https://github.com/pupil-labs/pye3d-detector/actions/workflows/build-pye3d.yml/badge.svg)](https://github.com/pupil-labs/pye3d-detector/actions/workflows/build-pye3d.yml)
[![PyPI version](https://badge.fury.io/py/pye3d.svg)](https://badge.fury.io/py/pye3d)

# pye3d

`pye3d` implements a published mathematical 3D eye model capturing ocular kinematics and optics (see [Swirski and Dodgson, 2013](https://www.researchgate.net/publication/264658852_A_fully-automatic_temporal_approach_to_single_camera_glint-free_3D_eye_model_fitting "Swirski and Dodgson, 2013"), as well as [Dierkes, Kassner, and Bulling, 2019](https://www.researchgate.net/publication/333490770_A_fast_approach_to_refraction-aware_eye-model_fitting_and_gaze_prediction "Dierkes, Kassner, and Bulling, 2019")).

## Installation
We recommend installing the pre-packaged binary wheels from PyPI:
```
pip install pye3d
```

If you want to install a modified version of the source code, you will have to install
the platform-specific dependencies first. For details, see [`INSTALL_SOURCE.md`](INSTALL_SOURCE.md).


## Usage

Here's a quick example of how to pass 2D pupil detection results to `pye3d` (requires standalone 
[2D pupil detector](https://github.com/pupil-labs/pupil-detectors/edit/master/README.md) installation)

```python
import cv2
from pupil_detectors import Detector2D
from pye3d.detector_3d import CameraModel, DetectorMode, Detector3D

# create 2D detector
detector_2d = Detector2D()

# create pye3D detector
camera = CameraModel(focal_length=561.5, resolution=[400, 400])
detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)

# load eye video
eye_video = cv2.VideoCapture("eye*.mp4")

# read each frame of video and run pupil detectors
while eye_video.isOpened():
    frame_number = eye_video.get(cv2.CAP_PROP_POS_FRAMES)
    fps = eye_video.get(cv2.CAP_PROP_FPS)
    ret, eye_frame = eye_video.read()

    if ret:
        # read video frame as numpy array
        grayscale_array = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

        # run 2D detector on video frame
        result_2d = detector_2d.detect(grayscale_array)
        result_2d["timestamp"] = frame_number / fps

        # pass 2D detection result to 3D detector
        result_3d = detector_3d.update_and_detect(result_2d, grayscale_array)
        ellipse_3d = result_3d["ellipse"]

        # draw 3D detection result on eye frame
        cv2.ellipse(
            eye_frame,
            tuple(int(v) for v in ellipse_3d["center"]),
            tuple(int(v / 2) for v in ellipse_3d["axes"]),
            ellipse_3d["angle"],
            0,
            360,  # start/end angle for drawing
            (0, 255, 0),  # color (BGR): red
        )

        # show frame
        cv2.imshow("eye_frame", eye_frame)

        # press esc to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

eye_video.release()
cv2.destroyAllWindows()

```
