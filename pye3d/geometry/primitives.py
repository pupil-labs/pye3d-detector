"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import abc

import numpy as np

from .utilities import cart2sph, normalize


class Primitive(abc.ABC):
    def __repr__(self):
        klass = "{}.{}".format(self.__class__.__module__, self.__class__.__name__)
        attributes = " ".join(
            "{}={}".format(k, v.__repr__()) for k, v in self.__dict__.items()
        )
        return "<{klass} at {id}: {attributes}>".format(
            klass=klass, id=id(self), attributes=attributes
        )

    def __str__(self):
        def to_str(obj, float_fmt="{:f}") -> str:
            if isinstance(obj, float) or isinstance(obj, int):
                return float_fmt.format(obj)
            if isinstance(obj, np.ndarray):
                if obj.dtype != np.object:
                    return ", ".join(float_fmt.format(x) for x in obj)
            return str(obj)

        klass = self.__class__.__name__
        attributes = " - ".join(
            "{}: {}".format(k, to_str(v)) for k, v in self.__dict__.items()
        )
        return "{klass} -> {attributes}".format(klass=klass, attributes=attributes)


class Line(Primitive):
    def __init__(self, origin, direction):
        self.origin = np.asarray(origin)
        self.direction = normalize(np.asarray(direction))
        self.dim = self.origin.shape[0]


class Circle(Primitive):
    def __init__(self, center=[0.0, 0.0, 0.0], normal=[0.0, 0.0, -1.0], radius=0.0):
        self.center = np.array(center, dtype=np.float)
        self.normal = np.array(normal, dtype=np.float)
        self.radius = radius

    def spherical_representation(self):
        phi, theta = cart2sph(self.normal)
        return phi, theta, self.radius

    @property
    def invalid(self):
        return self.radius <= 0.0

    @staticmethod
    def create_invalid() -> "Circle":
        return Circle(radius=0.0)


class Ellipse(Primitive):
    def __init__(self, center, minor_radius, major_radius, angle):
        self.center = center
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.angle = angle

        if self.minor_radius > self.major_radius:
            current_minor_radius = self.minor_radius
            self.minor_radius = self.major_radius
            self.major_radius = current_minor_radius
            self.angle = self.angle + np.pi / 2

    def area(self):
        return np.pi * self.minor_radius * self.major_radius

    def circularity(self):
        return self.minor_radius / self.major_radius

    def parameters(self):
        return (
            self.center[0],
            self.center[1],
            self.minor_radius,
            self.major_radius,
            self.angle,
        )


class Sphere(Primitive):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __bool__(self):
        return self.radius > 0


class Conicoid(Primitive):
    def __init__(self, conic, vertex):
        alpha = vertex[0]
        beta = vertex[1]
        gamma = vertex[2]
        self.A = (gamma ** 2) * conic.A
        self.B = (gamma ** 2) * conic.C
        self.C = (
            conic.A * (alpha ** 2)
            + conic.B * alpha * beta
            + conic.C * (beta ** 2)
            + conic.D * alpha
            + conic.E * beta
            + conic.F
        )
        self.F = -gamma * (conic.C * beta + conic.B / 2 * alpha + conic.E / 2)
        self.G = -gamma * (conic.B / 2 * beta + conic.A * alpha + conic.D / 2)
        self.H = (gamma ** 2) * conic.B / 2
        self.U = (gamma ** 2) * conic.D / 2
        self.V = (gamma ** 2) * conic.E / 2
        self.W = -gamma * (conic.E / 2 * beta + conic.D / 2 * alpha + conic.F)
        self.D = (gamma ** 2) * conic.F


class Conic(Primitive):
    def __init__(self, *args):
        if len(args) == 1:
            ellipse = args[0]
            ax = np.cos(ellipse.angle)
            ay = np.sin(ellipse.angle)
            a2 = ellipse.major_radius ** 2
            b2 = ellipse.minor_radius ** 2

            self.A = ax * ax / a2 + ay * ay / b2
            self.B = 2.0 * ax * ay / a2 - 2.0 * ax * ay / b2
            self.C = ay * ay / a2 + ax * ax / b2
            self.D = (
                -2 * ax * ay * ellipse.center[1] - 2 * ax * ax * ellipse.center[0]
            ) / a2 + (
                2 * ax * ay * ellipse.center[1] - 2 * ay * ay * ellipse.center[0]
            ) / b2
            self.E = (
                -2 * ax * ay * ellipse.center[0] - 2 * ay * ay * ellipse.center[1]
            ) / a2 + (
                2 * ax * ay * ellipse.center[0] - 2 * ax * ax * ellipse.center[1]
            ) / b2
            self.F = (
                (
                    2 * ax * ay * ellipse.center[0] * ellipse.center[1]
                    + ax * ax * ellipse.center[0] * ellipse.center[0]
                    + ay * ay * ellipse.center[1] * ellipse.center[1]
                )
                / a2
                + (
                    -2 * ax * ay * ellipse.center[0] * ellipse.center[1]
                    + ay * ay * ellipse.center[0] * ellipse.center[0]
                    + ax * ax * ellipse.center[1] * ellipse.center[1]
                )
                / b2
                - 1
            )
        if len(args) == 6:
            self.A, self.B, self.C, self.D, self.E, self.F = args

    def discriminant(self):
        return self.B ** 2 - 4 * self.A * self.C
