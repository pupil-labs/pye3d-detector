# cython: profile=True
"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cpython cimport array
from cython.operator import dereference

import numpy as np
cimport numpy as np
import cv2

from pye3d.geometry.projections import unproject_edges_to_sphere, project_circle_into_image_plane, project_point_into_image_plane
from pye3d.geometry.primitives import Circle, Conic, Conicoid
from pye3d.geometry.utilities import normalize
import warnings

_EYE_RADIUS_DEFAULT: float = 10.392304845413264

cdef extern from '<opencv2/core.hpp>':
  int CV_8UC1
  int CV_8UC3

cdef extern from '<opencv2/core.hpp>' namespace 'cv':
    cdef cppclass Mat :
        Mat() except +
        Mat(int height, int width, int type, void* data) except +
        Mat(int height, int width, int type) except +
        unsigned char* data

cdef extern from '<Eigen/Eigen>' namespace 'Eigen':

    cdef cppclass Vector3d "Eigen::Matrix<double, 3, 1>":
        Matrix31d() except +
        double& operator[](size_t)

    cdef cppclass MatrixXd "Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>":
        MatrixXd() except +
        double * data()
        double& operator[](size_t)
        int rows()
        int cols()
        double coeff(int,int)
        double * resize(int,int)
        

cdef extern from "unproject_conicoid_cpp.h":

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
    


cdef extern from "search_3d_cpp.h":

    cdef struct numpy_matrix_view:
        double * data
        unsigned int rows
        unsigned int cols
    
    cdef struct numpy_vector3d:
        double x
        double y
        double z

    cdef struct search_3d_result:
        MatrixXd inliers
        Vector3d gaze_vector
        double pupil_radius

    search_3d_result search_3d_(numpy_matrix_view &, numpy_vector3d &, double &, numpy_vector3d &, double &, double &)

    #Vector3d correct_gaze_vector(numpy_matrix_view &, numpy_matrix_view &, numpy_matrix_view &, numpy_matrix_view &)

cdef eigen2np(MatrixXd data):

    d1 = data.rows()
    d2 = data.cols()
    data_np = np.zeros((d1,d2))

    for row in range(d1):
        for column in range(d2):
            data_np[row, column] = data.coeff(row,column)

    return data_np


def get_edges(frame,
              predicted_gaze_vector,
              predicted_pupil_radius,
              sphere_center,
              sphere_radius,
              focal_length,
              resolution,
              major_axis_factor=1.5):


    predicted_pupil_center = sphere_center + _EYE_RADIUS_DEFAULT * predicted_gaze_vector
    projected_pupil_center = project_point_into_image_plane(predicted_pupil_center, focal_length)
    major_axis_estimate = predicted_pupil_radius/np.linalg.norm(predicted_pupil_center)*focal_length

    x, y = projected_pupil_center
    x = x + resolution[0]/2
    y = -y + resolution[1]/2
    major_axis =  major_axis_factor * major_axis_estimate
    N,M = frame.shape
    ymin, ymax = max(0,int(y-major_axis)), min(N,int(y+major_axis))
    xmin, xmax = max(0,int(x-major_axis)), min(M,int(x+major_axis))

    if ymin>=ymax or xmin>=xmax:
           return None, None, None, [], [ymin,ymax,xmin,xmax]

    frame_roi = frame[ymin:ymax, xmin:xmax]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    frame_roi = cv2.morphologyEx(frame_roi, cv2.MORPH_OPEN, kernel)
    frame_roi = cv2.medianBlur(frame_roi, 5)
    frame_roi = cv2.normalize(frame_roi, 0, 255, norm_type=cv2.NORM_MINMAX)
    edge_frame = cv2.Canny(frame_roi, 100, 100, 5)

    edges = []
    if cv2.countNonZero(edge_frame)>0:
        edges = np.asarray(cv2.findNonZero(edge_frame))
        edges = edges[:,0,:] + np.asarray([xmin, ymin])

    return frame[ymin:ymax, xmin:xmax].copy(), frame_roi, edge_frame, edges, [ymin,ymax,xmin,xmax]

def search_3d_cpp(edges,
                  predicted_gaze_vector,
                  predicted_pupil_radius,
                  sphere_center,
                  sphere_radius,
                  focal_length,
                  resolution):

    edges_on_sphere, idxs = unproject_edges_to_sphere(
        edges, focal_length, sphere_center, sphere_radius, resolution[0], resolution[1]
    )

    # convert edges numpy array to custom struct for passing to c++
    if edges_on_sphere.shape[0] == 0:
        return np.asarray([0, 0, -1]), 0, [], edges_on_sphere
    # first: make sure 2d data is c-contiguous, otherwise make c-contiguous copies
    if not edges_on_sphere.flags["C_CONTIGUOUS"]:
        edges_on_sphere = np.ascontiguousarray(edges_on_sphere)
    # then create cython memory view, for raw pointer access
    cdef double[:, ::1] edges_on_sphere_memview = edges_on_sphere
    # then copy data information
    # NOTE: This will provide direct access to the underlying data of the numpy array
    #  (given that it is c-contiguous and we did not copy). Therefore watch out to only
    # use this memory view as long as the numpy array exists. Also you shouldn't
    # probably change the underlying data.
    cdef numpy_matrix_view edges_on_sphere_memstruct
    edges_on_sphere_memstruct.data = &edges_on_sphere_memview[0, 0]
    edges_on_sphere_memstruct.rows = edges_on_sphere_memview.shape[0]
    edges_on_sphere_memstruct.cols = edges_on_sphere_memview.shape[1]

    # convert numpy vectors to custom vector representation for passing to c++
    cdef numpy_vector3d predicted_gaze_vector_struct
    predicted_gaze_vector_struct.x = predicted_gaze_vector[0]
    predicted_gaze_vector_struct.y = predicted_gaze_vector[1]
    predicted_gaze_vector_struct.z = predicted_gaze_vector[2]
    cdef numpy_vector3d sphere_center_struct
    sphere_center_struct.x = sphere_center[0]
    sphere_center_struct.y = sphere_center[1]
    sphere_center_struct.z = sphere_center[2]
    
    result = search_3d_(edges_on_sphere_memstruct,
                      predicted_gaze_vector_struct,
                      predicted_pupil_radius,
                      sphere_center_struct,
                      sphere_radius,
                      focal_length)

    gaze_vector = np.asarray([result.gaze_vector[0],
                              result.gaze_vector[1],
                              result.gaze_vector[2]])
    pupil_radius = result.pupil_radius
    inliers = eigen2np(result.inliers).T

    return gaze_vector, pupil_radius, inliers, edges_on_sphere
    

def unproject_ellipse(ellipse, focal_length, radius=1.0):
    cdef Circle3D c
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
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
            return [
                Circle(
                    center=(circles.first.center[0], circles.first.center[1], circles.first.center[2]),
                    normal=(circles.first.normal[0], circles.first.normal[1], circles.first.normal[2]),
                    radius=circles.first.radius
                ),
                Circle(
                    center=(circles.second.center[0], circles.second.center[1], circles.second.center[2]),
                    normal=(circles.second.normal[0], circles.second.normal[1], circles.second.normal[2]),
                    radius=circles.second.radius
                ),
            ]
        except Warning as e:
            return False
