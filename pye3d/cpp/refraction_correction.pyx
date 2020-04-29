"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import cv2
import numpy as np
cimport numpy as np

from .common_types cimport (
    MatrixXd,
    Vector3d,
)


cdef extern from "refraction_correction_cpp.h":

    cdef struct numpy_matrix_view:
        double * data
        unsigned int rows
        unsigned int cols

    Vector3d correct_gaze_vector(numpy_matrix_view &, numpy_matrix_view &, numpy_matrix_view &, numpy_matrix_view &, numpy_matrix_view &)


cdef eigen2np(MatrixXd data):

    d1 = data.rows()
    d2 = data.cols()
    data_np = np.zeros((d1,d2))

    for row in range(d1):
        for column in range(d2):
            data_np[row, column] = data.coeff(row,column)

    return data_np


def correct_gaze_vector_cpp(x, powers_, mean_var_, coefs_, intercept_):

    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    cdef double[:, ::1] x_memview = x
    cdef numpy_matrix_view x_memstruct
    x_memstruct.data = &x_memview[0, 0]
    x_memstruct.rows = x_memview.shape[0]
    x_memstruct.cols = x_memview.shape[1]

    if not powers_.flags["C_CONTIGUOUS"]:
        powers_ = np.ascontiguousarray(powers_)
    cdef double[:, ::1] powers_memview = powers_
    cdef numpy_matrix_view powers_memstruct
    powers_memstruct.data = &powers_memview[0, 0]
    powers_memstruct.rows = powers_memview.shape[0]
    powers_memstruct.cols = powers_memview.shape[1]

    if not mean_var_.flags["C_CONTIGUOUS"]:
        mean_var_ = np.ascontiguousarray(mean_var_)
    cdef double[:, ::1] mean_var_memview = mean_var_
    cdef numpy_matrix_view mean_var_memstruct
    mean_var_memstruct.data = &mean_var_memview[0, 0]
    mean_var_memstruct.rows = mean_var_memview.shape[0]
    mean_var_memstruct.cols = mean_var_memview.shape[1]

    if not coefs_.flags["C_CONTIGUOUS"]:
        coefs_ = np.ascontiguousarray(coefs_)
    cdef double[:, ::1] coefs_memview = coefs_
    cdef numpy_matrix_view coefs_memstruct
    coefs_memstruct.data = &coefs_memview[0, 0]
    coefs_memstruct.rows = coefs_memview.shape[0]
    coefs_memstruct.cols = coefs_memview.shape[1]

    if not intercept_.flags["C_CONTIGUOUS"]:
        intercept_ = np.ascontiguousarray(intercept_)
    cdef double[:, ::1] intercept_memview = intercept_
    cdef numpy_matrix_view intercept_memstruct
    intercept_memstruct.data = &intercept_memview[0, 0]
    intercept_memstruct.rows = intercept_memview.shape[0]
    intercept_memstruct.cols = intercept_memview.shape[1]

    result = correct_gaze_vector(x_memstruct, powers_memstruct, mean_var_memstruct, coefs_memstruct, intercept_memstruct)

    corrected_gaze_vector = np.asarray([result[0],
                                        result[1],
                                        result[2]])

    return corrected_gaze_vector
