/*
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <Eigen/Dense>
#include <iostream>


struct numpy_matrix_view {
    double * data;
    unsigned int rows;
    unsigned int cols;
};


Eigen::Vector3d correct_gaze_vector(
      const numpy_matrix_view & x_raw,
      const numpy_matrix_view & powers_raw,
      const numpy_matrix_view & mean_var_raw,
      const numpy_matrix_view & coefs_raw,
      const numpy_matrix_view & intercept_raw
    ){

           Eigen::Map<Eigen::MatrixXd> x(x_raw.data, x_raw.cols, x_raw.rows);
           Eigen::Map<Eigen::MatrixXd> powers_(powers_raw.data, powers_raw.cols, powers_raw.rows);
           Eigen::Map<Eigen::MatrixXd> mean_var_(mean_var_raw.data, mean_var_raw.cols, mean_var_raw.rows);
           Eigen::Map<Eigen::MatrixXd> coefs_(coefs_raw.data, coefs_raw.cols, coefs_raw.rows);
           Eigen::Map<Eigen::MatrixXd> intercept_(intercept_raw.data, intercept_raw.cols, intercept_raw.rows);

           Eigen::Matrix<double, 119, 1> features;
           for (int i=0; i<119; i++){
               features(i,0) = 1.0;
               for (int j=0;j<7;j++){
                     features(i,0) *= pow(x(0,j), powers_(i,j));
               }
               features(i,0) -= mean_var_(0, i);
               features(i,0) /= mean_var_(1, i);
           }

           Eigen::Vector3d corrected_gaze_vector;
           corrected_gaze_vector = coefs_ * features + intercept_;

           return corrected_gaze_vector;

     }