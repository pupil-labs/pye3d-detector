import argparse

import cv2
from pupil_detectors import Detector2D

from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode


def main(eye_video_path):
    # create 2D detector
    detector_2d = Detector2D()
    # create pye3D detector
    camera = CameraModel(focal_length=561.5, resolution=[400, 400])
    detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)
    # load eye video
    eye_video = cv2.VideoCapture(eye_video_path)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eye_video_path")
    args = parser.parse_args()
    main(args.eye_video_path)
