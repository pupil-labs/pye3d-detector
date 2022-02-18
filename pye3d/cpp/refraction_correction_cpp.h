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


Eigen::MatrixXd apply_correction_pipeline_cpp(
      const numpy_matrix_view & x_raw,
      const numpy_matrix_view & powers_raw,
      const numpy_matrix_view & mean_raw,
      const numpy_matrix_view & var_raw,
      const numpy_matrix_view & coef_raw,
      const numpy_matrix_view & intercept_raw
    ){

           Eigen::Map<Eigen::MatrixXd> x(x_raw.data, x_raw.cols, x_raw.rows);
           Eigen::Map<Eigen::MatrixXd> powers_(powers_raw.data, powers_raw.cols, powers_raw.rows);
           Eigen::Map<Eigen::MatrixXd> mean_(mean_raw.data, mean_raw.cols, mean_raw.rows);
           Eigen::Map<Eigen::MatrixXd> var_(var_raw.data, var_raw.cols, var_raw.rows);
           Eigen::Map<Eigen::MatrixXd> coef_(coef_raw.data, coef_raw.cols, coef_raw.rows);
           Eigen::Map<Eigen::MatrixXd> intercept_(intercept_raw.data, intercept_raw.cols, intercept_raw.rows);

           Eigen::MatrixXd features;
           features.resize(powers_.rows(),x.rows());
           for (int k=0;k<x.rows();k++){
               for (int i=0; i<powers_.rows(); i++){
                   features(i,k) = 1.0;
                   for (int j=0;j<powers_.cols();j++){
                         for (int l=0; l<powers_(i,j); l++){
                            features(i,k) *= x(k,j);
                         }
                   }
                   features(i,k) -= mean_(0, i);
                   features(i,k) /= sqrt(var_(0, i));
               }
           }

           return ((coef_ * features).colwise() + intercept_.col(0)).transpose();

     }
