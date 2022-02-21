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

struct numpy_vector3d {
    double x, y, z;
};

Eigen::Vector3d sph2cart(double phi, double theta){
    Eigen::Vector3d gaze_vector;
    gaze_vector << sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi);
    return gaze_vector;
}

struct search_3d_result {

    Eigen::Vector3d gaze_vector;
    double pupil_radius;
    Eigen::MatrixXd inliers;

};

search_3d_result find_best_circle(const Eigen::MatrixXd & edges_on_sphere_T,
                     const Eigen::Vector3d & initial_pupil_normal,
                     const double & initial_pupil_radius,
                     const Eigen::Vector3d & sphere_center,
                     const double & sphere_radius,
                     const double & position_variance,
                     const double & stepsize,
                     const double & bandwidth_in_pixels,
                     const double & focal_length){

    Eigen::MatrixXd edges_on_sphere = edges_on_sphere_T.transpose();

    const double h = sphere_radius - sqrt(pow(sphere_radius,2)-pow(initial_pupil_radius,2));
    const double pupil_chord = sqrt(2.0 * sphere_radius * h);
    const double pupil_alpha = 2.0 * asin((pupil_chord / 2) / sphere_radius);
    const double extended_chord = 2.0 * sphere_radius * sin((position_variance + pupil_alpha) / 2.0);

    const Eigen::Vector3d initial_pupil_center = sphere_center+sphere_radius*initial_pupil_normal;
    auto deltas = (edges_on_sphere.colwise() - initial_pupil_center).colwise().squaredNorm();

    Eigen::MatrixXd filtered_edges;
    filtered_edges.resize(edges_on_sphere.rows(), edges_on_sphere.cols());
    int j = 0;
    for (int i=0; i<edges_on_sphere.cols(); i++){
        if (deltas[i]<pow(extended_chord,2)){
            filtered_edges.col(j)=edges_on_sphere.col(i);
            j+=1;
        }
    }
    filtered_edges.conservativeResize(edges_on_sphere.rows(), j);

    double phi_initial = atan2(initial_pupil_normal[2],initial_pupil_normal[0]);
    double theta_initial = acos(initial_pupil_normal[1]);

    int edge_count_max = 0;
    double best_phi = 0;
    double best_theta = 0;

    for (double phi=phi_initial-position_variance; phi<phi_initial+position_variance; phi+=stepsize){
        for (double theta=theta_initial-position_variance; theta<theta_initial+position_variance; theta+=stepsize){

                   const Eigen::Vector3d current_gaze_vector = sph2cart(phi, theta);
                   const Eigen::Vector3d current_pupil_center = sphere_center + sphere_radius * current_gaze_vector;

                   const double bandwidth_in_mm = bandwidth_in_pixels * current_pupil_center[2] / focal_length;

                   auto distances_from_current_pupil_center = (filtered_edges.colwise()-current_pupil_center).colwise().squaredNorm();

                   int edge_count = 0;
                   for (int i=0;i<filtered_edges.cols();i++){
                        if (pow(initial_pupil_radius - bandwidth_in_mm, 2) < distances_from_current_pupil_center[i] &&
                            distances_from_current_pupil_center[i] < pow(initial_pupil_radius + bandwidth_in_mm, 2)){
                            edge_count+=1;
                        }
                   }
                   if (edge_count>edge_count_max){
                        edge_count_max = edge_count;
                        best_phi = phi;
                        best_theta = theta;
                   }
        }
    }

    search_3d_result result;

    if (edge_count_max>0){

         const Eigen::Vector3d gaze_vector = sph2cart(best_phi, best_theta);
         const Eigen::Vector3d pupil_center = sphere_center + sphere_radius * gaze_vector;

         const double bandwidth_in_mm = bandwidth_in_pixels * pupil_center[2] / focal_length;

         auto distances_from_pupil_center = (filtered_edges.colwise()-pupil_center).colwise().squaredNorm();

         result.inliers.resize(filtered_edges.rows(), filtered_edges.cols());
         int edge_count = 0;
         double pupil_radius_acc = 0;
         for (int i=0; i<filtered_edges.cols(); i++){
            if (pow(initial_pupil_radius - bandwidth_in_mm, 2) < distances_from_pupil_center[i] &&
                distances_from_pupil_center[i] < pow(initial_pupil_radius + bandwidth_in_mm, 2)){
                result.inliers.col(edge_count) = filtered_edges.col(i);
                pupil_radius_acc += sqrt(distances_from_pupil_center[i]);
                edge_count += 1;
            }
         }

         result.inliers.conservativeResize(filtered_edges.rows(), edge_count);
         result.gaze_vector = gaze_vector;
         result.pupil_radius = pupil_radius_acc/edge_count;
         return result;

    }
    else{

         result.inliers.resize(filtered_edges.rows(), 0);
         result.gaze_vector << 0, 0, -1;
         result.pupil_radius = 0;
         return result;

    }

 }

search_3d_result search_on_sphere(const numpy_matrix_view & edges_on_sphere_raw,
                  const numpy_vector3d predicted_gaze_vector_raw,
                  const double predicted_pupil_radius,
                  const numpy_vector3d sphere_center_raw,
                  const double sphere_radius,
                  const double focal_length){

        Eigen::Map<Eigen::MatrixXd> edges_on_sphere(edges_on_sphere_raw.data, edges_on_sphere_raw.rows, edges_on_sphere_raw.cols);
        Eigen::Vector3d predicted_gaze_vector(predicted_gaze_vector_raw.x, predicted_gaze_vector_raw.y, predicted_gaze_vector_raw.z);
        Eigen::Vector3d sphere_center(sphere_center_raw.x, sphere_center_raw.y, sphere_center_raw.z);


        search_3d_result result_first_iteration = find_best_circle(edges_on_sphere,
                                                        predicted_gaze_vector,
                                                        predicted_pupil_radius,
                                                        sphere_center,
                                                        sphere_radius,
                                                        0.2,
                                                        0.05,
                                                        6.0,
                                                        focal_length);


        search_3d_result result_second_iteration = find_best_circle(edges_on_sphere,
                                                        result_first_iteration.gaze_vector,
                                                        result_first_iteration.pupil_radius,
                                                        sphere_center,
                                                        sphere_radius,
                                                        0.05,
                                                        0.01,
                                                        2.0,
                                                        focal_length);

        return result_second_iteration;

}
