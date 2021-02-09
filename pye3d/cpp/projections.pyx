import logging
import traceback
import warnings
from functools import wraps

import numpy as np
from libcpp.pair cimport pair

from .common_types cimport Vector3d
from ..geometry.primitives import Circle, Conic, Conicoid


logger = logging.getLogger(__name__)

cdef extern from "unproject_conicoid.h":

    cdef struct Circle3D:
        Vector3d center
        Vector3d normal
        double radius

    cdef pair[Circle3D, Circle3D] unproject_conicoid(
        const double a,
        const double b,
        const double c,
        const double f,
        const double g,
        const double h,
        const double u,
        const double v,
        const double w,
        const double focal_length,
        const double circle_radius
    )

def raise_np_errors(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        old_settings = np.seterr(all="raise")
        result = f(*args, **kwds)
        np.seterr(**old_settings)
        return result
    return wrapper

@raise_np_errors
def unproject_ellipse(ellipse, focal_length, radius=1.0):
    cdef Circle3D c
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            conic = Conic(ellipse)
            pupil_cone = Conicoid(conic, [0, 0, -focal_length])

            circles = unproject_conicoid(
                pupil_cone.A,
                pupil_cone.B,
                pupil_cone.C,
                pupil_cone.F,
                pupil_cone.G,
                pupil_cone.H,
                pupil_cone.U,
                pupil_cone.V,
                pupil_cone.W,
                focal_length,
                radius
            )

            # cannot iterate over C++ std::pair, that's why this looks so ugly
            circle_A = Circle(
                    center=(circles.first.center[0], circles.first.center[1], circles.first.center[2]),
                    normal=(circles.first.normal[0], circles.first.normal[1], circles.first.normal[2]),
                    radius=circles.first.radius
                )
            circle_B = Circle(
                    center=(circles.second.center[0], circles.second.center[1], circles.second.center[2]),
                    normal=(circles.second.normal[0], circles.second.normal[1], circles.second.normal[2]),
                    radius=circles.second.radius
                )
            # cannot iterate over C++ std::pair, that's why this looks so ugly
            if np.isnan([circle_A.radius, *circle_A.center, *circle_A.normal, circle_B.radius, *circle_B.center, *circle_B.normal]).any():
                return False
            else:
                return [circle_A, circle_B]
        except FloatingPointError:
            return False
        except Warning:
            logger.debug(f"Unexpected warning caught in:\n{traceback.format_exc()}")
            return False
