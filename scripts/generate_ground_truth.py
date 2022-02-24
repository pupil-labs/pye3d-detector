"""
Generates ground truth data for ``../tests/integration/test_synthetic_metrics.py``

- Depends on ``plsd`` - Pupil Labs internal library
- Output should be uploaded to
  https://github.com/pupil-labs/pye3d-detector/wiki/files/pye3d_test_input.npz
"""

import argparse
import re as regex
import typing as T

import numpy as np
import pandas as pd
import skimage.measure as skmeas
from integration.test_synthetic_metrics import pupil_datum_from_raytraced_image
from plsd.camera import PinholeCamera
from plsd.dynamics import Rolling_Eyes
from plsd.eye import LeGrandEye
from plsd.geometry import (
    normalize,
    transform_as_homogeneous_point,
    transform_as_homogeneous_vector,
)
from plsd.primitives import Sphere
from plsd.raytracer import RayTracer
from plsd.scene import Scene
from tqdm import tqdm

from pye3d.geometry.projections import project_sphere_into_image_plane
from pye3d.geometry.utilities import cart2sph


def main(
    focal_length,
    resolution,
    output_path,
    num_samples=1000,
    fps=200,
    visualize=False,
):
    ground_truth_data, images = generate_groundtruth_data_and_images(
        focal_length, resolution, num_samples, fps, visualize
    )
    ground_truth_data = pd.DataFrame(ground_truth_data).to_records(index=False)
    np.savez_compressed(output_path, ground_truth=ground_truth_data, eye_images=images)


def generate_groundtruth_data_and_images(
    focal_length,
    resolution,
    num_samples=1000,
    fps=200,
    visualize=False,
):
    if visualize:
        import cv2

    ground_truth_data = []
    images = []

    setup = _setup_renderer(focal_length, resolution)

    try:
        for j in tqdm(range(num_samples)):
            timestamp = j / fps
            setup.scene.next(flip=False)

            pupil_datum, img = pupil_datum_from_raytraced_image(
                raytracer=setup.raytracer
            )

            if not _check_render_validity(pupil_datum, img):
                continue

            ground_truth = _ground_truth_datum(setup)
            ground_truth["timestamp"] = timestamp

            images.append(img)
            ground_truth_data.append(ground_truth)

            if visualize:
                # optional: visualization
                img_visu = img.copy()
                ellipse = pupil_datum["ellipse"]
                cv2.ellipse(
                    img_visu,
                    (ellipse["center"], ellipse["axes"], ellipse["angle"]),
                    color=[255, 0, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

                cv2.imshow("Visualization", img_visu)
                cv2.waitKey(1)
    finally:
        if visualize:
            cv2.destroyAllWindows()
    return ground_truth_data, images


class RenderSetup(T.NamedTuple):
    scene: Scene
    eye: LeGrandEye
    camera: PinholeCamera
    raytracer: RayTracer


def _setup_renderer(
    focal_length,
    resolution,
    re=12.0,
    ri=6.0,
    rc=7.8,
    n_ref=1.3375,
    IED=63.0,
    mu_r=2.5,
    r_extent=0.0,
) -> RenderSetup:
    # initial eyeball positions (global coordinates)
    E_orig0 = np.asarray([-IED / 2.0, 10.0, 0.0])
    E_orig1 = np.asarray([IED / 2.0, 10.0, 0.0])

    # camera pose (we'll only look at ONE eye = eye0)
    pose = np.eye(4)
    pose[:3, 0] = np.array([1.0, 0.0, 0.0])
    pose[:3, 1] = np.array([0.0, 1.0, 0.0])
    pose[:3, 2] = np.array([0.0, 0.0, -1.0])
    pose[:3, 3] = np.asarray([-50.0, 8.0, 45.0])

    eye_camera0 = PinholeCamera(
        pose=pose,
        focal_length=focal_length,
        width=resolution[0],
        height=resolution[1],
    )
    eye_camera0.look_at(
        eye_camera0.tvec,
        E_orig0,
        np.asarray([0.0, 1.0, 0.0]),
    )
    eye0 = LeGrandEye(
        camera=None,
        n_refraction=n_ref,
        eyeball_radius=re,
        cornea_radius=rc,
        iris_radius=ri,
    )
    eye0.update_from_gaze_vector([0.0, 0.0, 1.0])

    # varying eyeball position
    # optional: eyeball shifts can be added before scene.next() in the gaze angle loop
    d_shift = np.array([-1.5, 2.34, 1.2])
    eye0.move_to_point(E_orig0 + d_shift)

    # the second eye is a dummy eye for the scene object
    eye1 = LeGrandEye(
        camera=None,
        n_refraction=n_ref,
        eyeball_radius=re,
        cornea_radius=rc,
        iris_radius=ri,
    )
    eye1.update_from_gaze_vector([0.0, 0.0, 1.0])
    eye1.move_to_point(E_orig1 + d_shift)

    traject = Rolling_Eyes(
        N=1000,
        r_points=100,
        phi_extent=(-25, 25),
        theta_extent=(-25, 25),
        mu_r0=mu_r,
        r0_extent=r_extent,
        d=500.0,
        eye_left=eye0.tvec,
        eye_right=eye1.tvec,
    )
    raytracer = RayTracer(eye0, eye_camera0)
    scene = Scene(eye_0=eye0, eye_1=eye1, trajectory=traject)
    return RenderSetup(scene, eye0, eye_camera0, raytracer)


def _check_render_validity(
    pupil_datum,
    eye_image,
    threshold_ellipse_axes_length=0.0,
    threshold_ellipse_circularity=0.4,
    pupil_label=10,  # color value in 0th image layer
    threshold_pupil_border_margin=3,
) -> bool:
    if not all(
        ax > threshold_ellipse_axes_length for ax in pupil_datum["ellipse"]["axes"]
    ):
        return False
    ellipse_circularity = (
        pupil_datum["ellipse"]["axes"][0] / pupil_datum["ellipse"]["axes"][1]
    )
    if ellipse_circularity <= threshold_ellipse_circularity:
        return False

    height, width = eye_image.shape
    segmentation_pupil = np.zeros(eye_image.shape).astype(np.uint8)
    segmentation_pupil[eye_image == pupil_label] = pupil_label
    segmentation_pupil, _ = skmeas.label(
        segmentation_pupil, return_num=True, connectivity=1
    )
    properties = skmeas.regionprops(segmentation_pupil)
    if not properties:
        return False  # pupil could not be detected

    bbox = properties[0].bbox
    is_margin_top_sufficient = bbox[0] > threshold_pupil_border_margin
    is_margin_left_sufficient = bbox[1] > threshold_pupil_border_margin
    is_margin_bottom_sufficient = (height - bbox[2]) > threshold_pupil_border_margin
    is_margin_right_sufficient = (width - bbox[3]) > threshold_pupil_border_margin
    return all(
        (
            is_margin_top_sufficient,
            is_margin_left_sufficient,
            is_margin_bottom_sufficient,
            is_margin_right_sufficient,
        )
    )


def _ground_truth_datum(setup: RenderSetup):
    E_gt = transform_as_homogeneous_point(setup.eye.tvec, setup.camera.extrinsics)
    gv_gt = normalize(
        transform_as_homogeneous_vector(setup.eye.gaze_vector, setup.camera.extrinsics)
    )
    phi_gt, theta_gt = cart2sph(gv_gt)
    E_projected = project_sphere_into_image_plane(
        Sphere(E_gt, setup.eye.distance_eyeball_pupil),
        setup.camera.focal_length,
        transform=True,
        width=setup.camera.width,
        height=setup.camera.height,
    )
    pupil_center_gt = normalize(
        transform_as_homogeneous_vector(setup.eye.pupil_center, setup.camera.extrinsics)
    )

    return {
        "sphere_center_x": E_gt[0],
        "sphere_center_y": E_gt[1],
        "sphere_center_z": E_gt[2],
        "sphere_radius": setup.eye.distance_eyeball_pupil,
        "projected_sphere_center_x": E_projected.center[0],
        "projected_sphere_center_y": E_projected.center[1],
        "projected_sphere_minor_axis": E_projected.minor_radius,
        "projected_sphere_major_axis": E_projected.major_radius,
        "projected_sphere_angle": E_projected.angle,
        "circle_3d_center_x": pupil_center_gt[0],
        "circle_3d_center_y": pupil_center_gt[1],
        "circle_3d_center_z": pupil_center_gt[2],
        "circle_3d_normal_x": gv_gt[0],
        "circle_3d_normal_y": gv_gt[1],
        "circle_3d_normal_z": gv_gt[2],
        "circle_3d_radius": setup.eye.pupil_radius,
        "diameter_3d": setup.eye.pupil_radius * 2.0,
        "theta": theta_gt,
        "phi": phi_gt,
    }


def parse_resolution(resolution):
    match = regex.match(r"(?P<width>\d+)(x|,)(?P<height>\d+)", resolution)
    if match is None:
        raise ValueError(
            "Expected resolution as `<width>x<height>` or `<width>,<height>`, "
            f"got {resolution}",
        )
    width = int(match.group("width"))
    height = int(match.group("height"))
    return width, height


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-res",
        "--resolution",
        type=parse_resolution,
        default="400x400",
        help="Format <width>x<height> or <width>,<height>",
    )
    parser.add_argument("-vis", "--visualize", action="store_true")
    parser.add_argument("-fl", "--focal-length", type=float, default=561.5)
    parser.add_argument("-fps", type=float, default=200)
    parser.add_argument("-N", "--num-samples", type=int, default=1000)
    parser.add_argument(
        "-o", "--output", default="../tests/integration/input/pye3d_test_input.npz"
    )
    args = parser.parse_args()

    main(
        focal_length=args.focal_length,
        resolution=args.resolution,
        output_path=args.output,
        num_samples=args.num_samples,
        fps=args.fps,
        visualize=args.visualize,
    )
