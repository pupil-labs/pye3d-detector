import pytest


def test_import():
    from pye3d.camera import CameraModel
    from pye3d.detector_3d import Detector3D

    camera = CameraModel(1.0, (200, 200))
    d = Detector3D(camera)
