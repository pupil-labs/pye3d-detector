import logging
import math

import numpy as np
import pandas as pd
import pytest

from pye3d.detector_3d import CameraModel
from pye3d.detector_3d import Detector3D as Pye3D
from pye3d.detector_3d import DetectorMode

from .utils import abs_diff, input_dir, output_dir, remove_file

# Define all input files
INPUT_PATH = input_dir().joinpath("pye3d_test_input.npz")

# Define all output files
OUTPUT_GENERATED_RESULTS_CSV_PATH = output_dir().joinpath(
    "pye3d_test_generated_results.csv"
)
OUTPUT_PUPIL_RADIUS_PLOT_PATH = output_dir().joinpath(
    "pye3d_test_pupil_radius_plot.png"
)
OUTPUT_PUPIL_RADIUS_ERROR_PLOT_PATH = output_dir().joinpath(
    "pye3d_test_pupil_radius_error_plot.png"
)
OUTPUT_GAZE_ANGLE_PHI_PLOT_PATH = output_dir().joinpath(
    "pye3d_test_gaze_angle_phi_plot.png"
)
OUTPUT_GAZE_ANGLE_THETA_PLOT_PATH = output_dir().joinpath(
    "pye3d_test_gaze_angle_theta_plot.png"
)
OUTPUT_GAZE_ANGLE_VECTOR_PLOT_PATH = output_dir().joinpath(
    "pye3d_test_gaze_angle_vector_plot.png"
)
OUTPUT_GAZE_ANGLE_ERROR_BY_INPUT_ORIENTATION_PLOT_PATH = output_dir().joinpath(
    "pye3d_test_gaze_angle_error_by_input_orientation_plot.png"
)
OUTPUT_EYE_CENTER_3D_PLOT_PATH = output_dir().joinpath(
    "pye3d_test_eye_center_3d_error_plot.png"
)

# Static detector properties
DETECTOR_EXPECTED_CONVERGENCE_MAX_TIME = 2.14
PUPIL_RADIUS_EPS = 0.057
DATUM_COMPONENT_PHI_EPS = 1.010
DATUM_COMPONENT_THETA_EPS = 0.456
EYE_CENTER_3D_EPS = 0.5
GAZE_ANGLE_EPS = 1.022


def test_convergence_time(convergence_time):
    assert (
        convergence_time <= DETECTOR_EXPECTED_CONVERGENCE_MAX_TIME
    )  # TODO: Add description


def test_datum_component_phi(dataset, convergence_time):
    gt_df, gr_df = dataset

    gaze_angle_phi_gt = gt_df["phi"].values * 180.0 / np.pi
    gaze_angle_phi_gr = gr_df["phi"].values * 180.0 / np.pi

    gaze_angle_phi_error = abs_diff(gaze_angle_phi_gt, gaze_angle_phi_gr)

    gaze_angle_phi_gr_mean = gr_df["phi"].mean() * 180.0 / np.pi

    save_plot(
        # Generated Results
        a_label="pye3d",
        ax=gr_df["timestamp"],
        ay=gaze_angle_phi_gr,
        # Ground Truth
        b_label="ground truth",
        bx=gt_df["timestamp"],
        by=gaze_angle_phi_gt,
        # Legend
        figsize=(10, 4),
        title="phi\n",
        xlabel="time [s]",
        ylabel="[°]",
        ylim=(gaze_angle_phi_gr_mean - 55, gaze_angle_phi_gr_mean + 55),
        v_threshold=convergence_time,
        v_threshold_label=f"convergence time = {convergence_time} seconds",
        # Image Path
        path=OUTPUT_GAZE_ANGLE_PHI_PLOT_PATH,
    )

    gaze_angle_phi_error = gaze_angle_phi_error[gr_df["timestamp"] > convergence_time]

    assert np.all(
        gaze_angle_phi_error <= DATUM_COMPONENT_PHI_EPS
    )  # TODO: Add description


def test_datum_component_theta(dataset, convergence_time):
    gt_df, gr_df = dataset

    gaze_angle_theta_gt = gt_df["theta"].values * 180.0 / np.pi
    gaze_angle_theta_gr = gr_df["theta"].values * 180.0 / np.pi

    gaze_angle_theta_error = abs_diff(gaze_angle_theta_gt, gaze_angle_theta_gr)

    gaze_angle_theta_gr_mean = gr_df["theta"].mean() * 180.0 / np.pi

    save_plot(
        # Generated Results
        a_label="pye3d",
        ax=gr_df["timestamp"],
        ay=gaze_angle_theta_gr,
        # Ground Truth
        b_label="ground truth",
        bx=gt_df["timestamp"],
        by=gaze_angle_theta_gt,
        # Legend
        figsize=(10, 4),
        title="theta\n",
        xlabel="time [s]",
        ylabel="[°]",
        ylim=(gaze_angle_theta_gr_mean - 55, gaze_angle_theta_gr_mean + 55),
        v_threshold=convergence_time,
        v_threshold_label=f"convergence time = {convergence_time} seconds",
        # Image Path
        path=OUTPUT_GAZE_ANGLE_THETA_PLOT_PATH,
    )

    gaze_angle_theta_error = gaze_angle_theta_error[
        gr_df["timestamp"] > convergence_time
    ]

    assert np.all(
        gaze_angle_theta_error <= DATUM_COMPONENT_THETA_EPS
    )  # TODO: Add description


def test_pupil_radius(dataset, convergence_time):
    gt_df, gr_df = dataset

    pupil_radius_gt = gt_df["circle_3d_radius"]
    pupil_radius_gr = gr_df["circle_3d_radius"]

    pupil_radius_error = abs_diff(pupil_radius_gt, pupil_radius_gr)

    save_plot(
        # Generated Results
        a_label="pye3d",
        ax=gr_df["timestamp"],
        ay=pupil_radius_gr,
        # Ground Truth
        b_label="ground truth",
        bx=gt_df["timestamp"],
        by=pupil_radius_gt,
        # Legend
        figsize=(10, 4),
        title="pupil radius\n",
        xlabel="time [s]",
        ylabel="[mm]",
        ylim=(0, 5),
        v_threshold=convergence_time,
        v_threshold_label=f"convergence time = {convergence_time} seconds",
        # Image Path
        path=OUTPUT_PUPIL_RADIUS_PLOT_PATH,
    )

    save_plot(
        ax=gr_df["timestamp"],
        ay=pupil_radius_error,
        a_color="r",
        a_label="pupil radius error",
        # Legend
        figsize=(10, 4),
        title="pupil radius error\n",
        xlabel="time [s]",
        ylabel="[mm]",
        ylim=(0, 1),
        h_threshold=PUPIL_RADIUS_EPS,
        h_threshold_label=f"pupil radius eps = {PUPIL_RADIUS_EPS} mm",
        v_threshold=convergence_time,
        v_threshold_label=f"convergence time = {convergence_time} seconds",
        # Image Path
        path=OUTPUT_PUPIL_RADIUS_ERROR_PLOT_PATH,
    )

    pupil_radius_error = pupil_radius_error[gr_df["timestamp"] > convergence_time]

    assert np.all(pupil_radius_error <= PUPIL_RADIUS_EPS)  # TODO: Add description


def test_eye_center_3d(dataset, convergence_time, eye_center_3d_errors):
    gt_df, gr_df = dataset

    save_plot(
        ax=gr_df["timestamp"],
        ay=eye_center_3d_errors,
        a_color="r",
        a_label="eye center 3d errors",
        # Legend
        figsize=(10, 4),
        title="eye center 3d error\n",
        xlabel="time [s]",
        ylabel="[mm]",
        ylim=(0, 5),
        h_threshold=EYE_CENTER_3D_EPS,
        h_threshold_label=f"eye center eps = {EYE_CENTER_3D_EPS} mm",
        v_threshold=convergence_time,
        v_threshold_label=f"convergence time = {convergence_time} seconds",
        # Image Path
        path=OUTPUT_EYE_CENTER_3D_PLOT_PATH,
    )

    eye_center_3d_errors = eye_center_3d_errors[gr_df["timestamp"] > convergence_time]

    assert np.all(eye_center_3d_errors <= EYE_CENTER_3D_EPS)  # TODO: Add description


def test_gaze_angle(dataset, convergence_time):
    gt_df, gr_df = dataset

    gaze_angle_columns = [
        "circle_3d_normal_x",
        "circle_3d_normal_y",
        "circle_3d_normal_z",
    ]

    # cosine distance; no need to calculate norms, as input already has length 1
    dot_product = (gr_df[gaze_angle_columns] * gt_df[gaze_angle_columns]).sum(axis=1)
    gaze_angle_error = np.rad2deg(np.arccos(dot_product))

    save_plot(
        ax=gr_df["timestamp"],
        ay=gaze_angle_error,
        a_color="r",
        # Legend
        figsize=(10, 4),
        title="gaze angle error over time",
        xlabel="time [s]",
        ylabel="[°]",
        ylim=(0, 5),
        h_threshold=GAZE_ANGLE_EPS,
        h_threshold_label=f"gaze angle eps = {GAZE_ANGLE_EPS} deg",
        v_threshold=convergence_time,
        v_threshold_label=f"convergence time = {convergence_time} seconds",
        # Image Path
        path=OUTPUT_GAZE_ANGLE_VECTOR_PLOT_PATH,
    )

    gaze_angle_error = gaze_angle_error[gr_df["timestamp"] > convergence_time]

    input_phi = gt_df.loc[gr_df["timestamp"] > convergence_time, "phi"]
    input_phi = np.rad2deg(input_phi) + 90.0

    input_theta = gt_df.loc[gr_df["timestamp"] > convergence_time, "theta"]
    input_theta = np.rad2deg(input_theta) - 90.0

    save_plot(
        ax=input_phi,
        ay=gaze_angle_error,
        a_label="phi + 90°",
        a_color="C0",
        bx=input_theta,
        by=gaze_angle_error,
        b_label="theta - 90°",
        b_color="C1",
        # Legend
        figsize=(10, 4),
        title="gaze angle error by input orientation (after convergence)",
        xlabel="centered ground truth input orientation [°]",
        ylabel="gaze angle error [°]",
        ylim=(0, 3),
        h_threshold=GAZE_ANGLE_EPS,
        h_threshold_label=f"gaze angle eps = {GAZE_ANGLE_EPS} deg",
        # Image Path
        path=OUTPUT_GAZE_ANGLE_ERROR_BY_INPUT_ORIENTATION_PLOT_PATH,
    )

    assert np.all(gaze_angle_error <= GAZE_ANGLE_EPS)  # TODO: Add description


@pytest.fixture(scope="module")
def convergence_time(dataset, eye_center_3d_errors):
    gt_df, gr_df = dataset

    eye_center_3d_convergence = eye_center_3d_errors > EYE_CENTER_3D_EPS
    convergence_index = -np.argwhere(eye_center_3d_convergence[::-1])[0][0] - 1
    convergence_time = gt_df.timestamp.iloc[convergence_index]
    logging.getLogger().info(f"Calculated convergence time: {convergence_time} seconds")

    return convergence_time


@pytest.fixture(scope="module")
def eye_center_3d_errors(dataset):
    gt_df, gr_df = dataset

    columns = ["sphere_center_x", "sphere_center_y", "sphere_center_z"]
    errors = np.linalg.norm(gr_df[columns].values - gt_df[columns].values, axis=1)

    return errors


@pytest.fixture(scope="module")
def dataset():

    # Check all input files exist
    assert INPUT_PATH.is_file(), f"Missing test input file: {INPUT_PATH}"

    # Cleanup output files from previous runs
    remove_file(OUTPUT_GENERATED_RESULTS_CSV_PATH)
    remove_file(OUTPUT_PUPIL_RADIUS_PLOT_PATH)
    remove_file(OUTPUT_PUPIL_RADIUS_ERROR_PLOT_PATH)
    remove_file(OUTPUT_GAZE_ANGLE_PHI_PLOT_PATH)
    remove_file(OUTPUT_GAZE_ANGLE_THETA_PLOT_PATH)
    remove_file(OUTPUT_EYE_CENTER_3D_PLOT_PATH)

    input_data = np.load(INPUT_PATH)

    image_key = "eye_images"
    error_msg = f"`{image_key}` not in available keys: {list(input_data.keys())}"
    assert image_key in input_data, error_msg
    images = input_data[image_key]
    FPS = 200.0

    detector = create_detector()

    measured_data = []

    for i, img in enumerate(images):

        pupil_datum, _ = pupil_datum_from_raytraced_image(img=img)
        pupil_datum["timestamp"] = i / FPS

        result = detector.update_and_detect(pupil_datum, img, debug=True)

        measured_data.append(
            np.hstack(
                [
                    result["timestamp"],
                    np.array(result["sphere"]["center"]),
                    result["sphere"]["radius"],
                    np.array(result["projected_sphere"]["center"]),
                    np.array(result["projected_sphere"]["axes"]),
                    result["projected_sphere"]["angle"],
                    np.array(result["circle_3d"]["center"]),
                    np.array(result["circle_3d"]["normal"]),
                    result["circle_3d"]["radius"],
                    result["diameter_3d"],
                    pupil_datum["ellipse"]["center"][0],
                    pupil_datum["ellipse"]["center"][1],
                    pupil_datum["ellipse"]["axes"][0],
                    pupil_datum["ellipse"]["axes"][1],
                    pupil_datum["ellipse"]["angle"],
                    pupil_datum["ellipse"]["center"][0],
                    pupil_datum["ellipse"]["center"][1],
                    pupil_datum["ellipse"]["axes"][1],
                    result["confidence"],
                    result["model_confidence"],
                    result["theta"],
                    result["phi"],
                ]
            )
        )

    # Load data frame for Ground Truth (GT) from input dir
    gt_key = "ground_truth"
    error_msg = f"`{gt_key}` not in available keys: {list(input_data.keys())}"
    assert gt_key in input_data, error_msg
    gt_df = input_data[gt_key]
    gt_df = pd.DataFrame.from_records(gt_df)

    # Create data frame for Generated Results (GR) and save to output dir
    gr_df = pd.DataFrame(
        np.concatenate(measured_data, axis=0).reshape(-1, len(COLUMNS_MEASURED)),
        columns=COLUMNS_MEASURED,
    )
    gr_df.to_csv(str(OUTPUT_GENERATED_RESULTS_CSV_PATH))

    # Sanity check: timestamps must match completely

    timestamp_gt = gt_df["timestamp"]
    timestamp_gr = gr_df["timestamp"]

    assert np.all(abs_diff(timestamp_gt, timestamp_gr) == 0.0)

    return gt_df, gr_df


def create_detector():
    camera = CameraModel(focal_length=561.5, resolution=np.array([400, 400]))
    detector = Pye3D(camera=camera, long_term_mode=DetectorMode.blocking)
    detector.reset()
    return detector


COLUMNS_MEASURED = [
    "timestamp",
    "sphere_center_x",
    "sphere_center_y",
    "sphere_center_z",
    "sphere_radius",
    "projected_sphere_center_x",
    "projected_sphere_center_y",
    "projected_sphere_minor_axis",
    "projected_sphere_major_axis",
    "projected_sphere_angle",
    "circle_3d_center_x",
    "circle_3d_center_y",
    "circle_3d_center_z",
    "circle_3d_normal_x",
    "circle_3d_normal_y",
    "circle_3d_normal_z",
    "circle_3d_radius",
    "diameter_3d",
    "ellipse_center_x",
    "ellipse_center_y",
    "ellipse_minor_axis",
    "ellipse_major_axis",
    "ellipse_angle",
    "location_x",
    "location_y",
    "diameter",
    "confidence",
    "model_confidence",
    "theta",
    "phi",
]


def pupil_datum_from_raytraced_image(img=None, raytracer=None, device="cpu"):
    try:
        # At this time, scikit-image==0.18.3 pins numpy==1.19.3 when running Python 3.9
        # which does not build on M1 macOS. Therefore, we allow to the test to be
        # skipped if this dependency is not installed.
        import skimage.measure as skmeas
    except ImportError:
        pytest.skip("scikit-image not installed")

    if img is None:
        img = raytracer.ray_trace_image(device=device)

    if img.ndim == 3:
        img = img[:, :, 0]

    hard_segmentation = img
    pupil_label = 10

    pupil_datum = {}
    pupil_datum["ellipse"] = {}
    pupil_datum["ellipse"]["axes"] = np.array([0.0, 0.0])
    pupil_datum["ellipse"]["angle"] = -90.0
    pupil_datum["ellipse"]["center"] = np.array([0.0, 0.0])
    pupil_datum["confidence"] = 0.0

    segmentation_pupil = np.zeros(hard_segmentation.shape).astype(np.uint8)
    segmentation_pupil[hard_segmentation == pupil_label] = pupil_label

    # ellipse fitting based on pupil region props:

    label_image, n_areas = skmeas.label(
        segmentation_pupil, return_num=True, connectivity=1
    )
    properties = skmeas.regionprops(label_image)
    if len(properties) >= 1:
        properties = properties[0]
    if properties:
        # turn props into dictionary
        minor_axis_length = properties.minor_axis_length
        major_axis_length = properties.major_axis_length
        centroid = properties.centroid
        angle = properties.orientation
        props = {}
        props["minor_axis_length"] = minor_axis_length
        props["major_axis_length"] = major_axis_length
        props["centroid"] = centroid  # centroid = (row, column)
        props["orientation"] = (
            angle - np.pi / 2.0
        )  # -pi/2 due to version change of regionprops

    if props:

        # adapt output of regionprops to construct pupil datum analoguous to
        # Pupil 2D Detector:
        pupil_datum["ellipse"]["axes"] = np.array(
            [props["minor_axis_length"], props["major_axis_length"]]
        )
        angle = props["orientation"] * 180.0 / math.pi
        pupil_datum["ellipse"]["angle"] = 90 - angle
        # see cv2.ellipse:
        # first coord = column/horizontal (from left to right)
        # second coord = row/vertical (from top to bottom)
        pupil_datum["ellipse"]["center"] = np.array(
            [props["centroid"][1], props["centroid"][0]]
        )

        pupil_datum["confidence"] = 1.0

    return pupil_datum, img


def save_plot(
    title,
    path,
    ylim,
    ax,
    ay,
    a_label="",
    bx=None,
    by=None,
    b_label="",
    a_color="b",
    b_color="g",
    h_threshold=None,
    h_threshold_label="",
    v_threshold=None,
    v_threshold_label="",
    figsize=(10, 4),
    xlabel="",
    ylabel="",
):
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=figsize)

    axis.plot(ax, ay, a_color, alpha=1, label=a_label)

    if bx is not None and by is not None:
        axis.plot(bx, by, b_color, alpha=1, label=b_label)

    if h_threshold:
        xlim_lo = min(map(lambda x: x.min(), filter(lambda x: x is not None, [ax, bx])))
        xlim_hi = max(map(lambda x: x.max(), filter(lambda x: x is not None, [ax, bx])))
        xlim = xlim_lo, xlim_hi
        axis.hlines(
            [h_threshold],
            *xlim,
            colors="C2",
            linestyles="dashed",
            label=h_threshold_label,
        )

    if v_threshold:
        axis.vlines(
            [v_threshold],
            *ylim,
            colors="C4",
            linestyles="dashed",
            label=v_threshold_label,
        )

    axis.set_ylim(*ylim)

    axis.legend()

    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

    plt.savefig(str(path))
