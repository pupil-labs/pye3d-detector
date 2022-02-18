from typing import NamedTuple, Tuple


class CameraModel(NamedTuple):
    focal_length: float
    resolution: Tuple[float, float]
