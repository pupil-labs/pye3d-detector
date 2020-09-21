from typing import Tuple


class CameraModel:
    def __init__(self, focal_length: float, resolution: Tuple[float, float]):
        self.focal_length = focal_length
        self.resolution = resolution
