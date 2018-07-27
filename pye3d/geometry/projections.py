"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging
import warnings

import numpy as np

from .intersections import intersect_sphere_multiple_lines
from .primitives import Circle, Conic, Conicoid, Ellipse, Line
from .utilities import normalize

logger = logging.getLogger(__name__)


def unproject_ellipse(ellipse, focal_length, radius=1.0):
    with warnings.catch_warnings():
        warnings.filterwarnings("error")

        try:

            conic = Conic(ellipse)
            pupil_cone = Conicoid(conic, [0, 0, -focal_length])

            a = pupil_cone.A
            b = pupil_cone.B
            c = pupil_cone.C
            f = pupil_cone.F
            g = pupil_cone.G
            h = pupil_cone.H
            u = pupil_cone.U
            v = pupil_cone.V
            w = pupil_cone.W

            p = np.zeros(4)
            p[0] = 1
            p[1] = -(a + b + c)
            p[2] = b * c + c * a + a * b - f * f - g * g - h * h
            p[3] = -(a * b * c + 2 * f * g * h - a * f * f - b * g * g - c * h * h)

            lambda_ = np.roots(p)

            n = np.sqrt((lambda_[1] - lambda_[2]) / (lambda_[0] - lambda_[2]))
            m = 0.0
            l = np.sqrt((lambda_[0] - lambda_[1]) / (lambda_[0] - lambda_[2]))

            t1 = (b - lambda_) * g - f * h
            t2 = (a - lambda_) * f - g * h
            t3 = -(a - lambda_) * (t1 / t2) / g - h / g

            mi = 1 / np.sqrt(1 + (t1 / t2) ** 2 + t3 ** 2)
            li = (t1 / t2) * mi
            ni = t3 * mi

            if np.dot(np.cross(li, mi), ni) < 0:
                li = -li
                mi = -mi
                ni = -ni

            T1 = np.asarray([li, mi, ni])

            T2 = -(u * li + v * mi + w * ni) / lambda_

            solution_circles = []

            for l in [l, -l]:

                if l == 0:
                    assert n == 1
                    T3 = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                else:
                    T3 = np.asarray(
                        [[0, -n * np.sign(l), l], [np.sign(l), 0, 0], [0, np.abs(l), n]]
                    )

                A = np.dot(lambda_, T3[:, 0] ** 2)
                B = np.dot(lambda_, T3[:, 0] * T3[:, 2])
                C = np.dot(lambda_, T3[:, 1] * T3[:, 2])
                D = np.dot(lambda_, T3[:, 2] ** 2)

                center_in_Xprime = np.zeros(3)
                center_in_Xprime[2] = A * radius / np.sqrt((B ** 2) + (C ** 2) - A * D)
                center_in_Xprime[0] = -B / A * center_in_Xprime[2]
                center_in_Xprime[1] = -C / A * center_in_Xprime[2]

                T0 = [0, 0, focal_length]

                center = np.dot(T1, np.dot(T3, center_in_Xprime) + T2) + T0
                if center[2] < 0:
                    center_in_Xprime = -center_in_Xprime
                    center = np.dot(T1, np.dot(T3, center_in_Xprime) + T2) + T0

                gaze = np.dot(T1, np.asarray([l, m, n]))
                if np.dot(gaze, center) > 0:
                    gaze = -gaze
                gaze = normalize(gaze)

                solution_circles.append(Circle(center, gaze, radius))

            return solution_circles

        except Warning as e:
            # print(e)
            return False


def unproject_edges_to_sphere(
    edges, focal_length, sphere_center, sphere_radius, width=640, height=480
):

    n_edges = edges.shape[0]

    directions = edges - np.asarray([width / 2, height / 2])
    directions[:, 1] *= -1

    directions = np.hstack((directions, focal_length * np.ones((n_edges, 1))))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=1)

    origins = np.zeros((n_edges, 3))

    edges_on_sphere, idxs = intersect_sphere_multiple_lines(
        sphere_center, sphere_radius, origins, directions
    )

    return edges_on_sphere, idxs


def project_point_into_image_plane(point, focal_length):
    scale = focal_length / point[2]
    point_projected = scale * np.asarray(point)
    return point_projected[:2]


def project_line_into_image_plane(line, focal_length):
    p1 = line.origin
    p2 = line.origin + line.direction

    p1_projected = project_point_into_image_plane(p1, focal_length)
    p2_projected = project_point_into_image_plane(p2, focal_length)

    return Line(p1_projected, p2_projected - p1_projected)


def project_circle_into_image_plane(
    circle, focal_length, transform=True, width=0, height=0
):
    c = circle.center
    n = circle.normal
    r = circle.radius
    f = focal_length

    cn = np.dot(c, n)
    c2r2 = np.dot(c, c) - r ** 2
    ABC = cn ** 2 - 2.0 * cn * (c * n) + c2r2 * (n ** 2)
    F = 2.0 * (c2r2 * n[1] * n[2] - cn * (n[1] * c[2] + n[2] * c[1]))
    G = 2.0 * (c2r2 * n[2] * n[0] - cn * (n[2] * c[0] + n[0] * c[2]))
    H = 2.0 * (c2r2 * n[0] * n[1] - cn * (n[0] * c[1] + n[1] * c[0]))
    conic = Conic(ABC[0], H, ABC[1], G * f, F * f, ABC[2] * f ** 2)

    disc_ = conic.discriminant()

    if disc_ < 0:

        A, B, C, D, E, F = conic.A, conic.B, conic.C, conic.D, conic.E, conic.F
        center_x = (2 * C * D - B * E) / disc_
        center_y = (2 * A * E - B * D) / disc_
        temp_ = 2 * (A * E ** 2 + C * D ** 2 - B * D * E + disc_ * F)
        minor_axis = (
            -np.sqrt(np.abs(temp_ * (A + C - np.sqrt((A - C) ** 2 + B ** 2)))) / disc_
        )  # Todo: Absolute value???
        major_axis = (
            -np.sqrt(np.abs(temp_ * (A + C + np.sqrt((A - C) ** 2 + B ** 2)))) / disc_
        )

        if B == 0 and A < C:
            angle = 0
        elif B == 0 and A >= C:
            angle = np.pi / 2.0
        else:
            angle = np.arctan((C - A - np.sqrt((A - C) ** 2 + B ** 2)) / B)

        # TO BE CONSISTENT WITH PUPIL
        if transform:
            center_x = center_x + width / 2.0
            center_y = height / 2.0 - center_y
            minor_axis, major_axis = 2.0 * minor_axis, 2.0 * major_axis
            angle = -(angle * 180.0 / np.pi - 90.0)

        return Ellipse(np.asarray([center_x, center_y]), minor_axis, major_axis, angle)

    else:

        return False


def project_sphere_into_image_plane(
    sphere, focal_length, transform=True, width=0, height=0
):
    scale = focal_length / sphere.center[2]

    projected_sphere_center = scale * sphere.center
    projected_radius = scale * sphere.radius

    if transform:

        projected_sphere_center[0] += width / 2.0
        projected_sphere_center[1] *= -1.0
        projected_sphere_center[1] += height / 2
        projected_radius *= 2.0

    return Ellipse(projected_sphere_center[:2], projected_radius, projected_radius, 0.0)
