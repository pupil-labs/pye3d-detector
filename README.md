[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build pye3d](https://github.com/pupil-labs/pye3d-detector/actions/workflows/build-pye3d.yml/badge.svg)](https://github.com/pupil-labs/pye3d-detector/actions/workflows/build-pye3d.yml)
[![PyPI version](https://badge.fury.io/py/pye3d.svg)](https://badge.fury.io/py/pye3d)

# pye3d

`pye3d` implements a published mathematical 3D eye model capturing ocular kinematics and optics (see [Swirski and Dodgson, 2013](https://www.researchgate.net/publication/264658852_A_fully-automatic_temporal_approach_to_single_camera_glint-free_3D_eye_model_fitting "Swirski and Dodgson, 2013"), as well as [Dierkes, Kassner, and Bulling, 2019](https://www.researchgate.net/publication/333490770_A_fast_approach_to_refraction-aware_eye-model_fitting_and_gaze_prediction "Dierkes, Kassner, and Bulling, 2019")).

## Installation
We recommend installing the pre-packaged binary wheels from PyPI:
```
pip install pye3d
```

If you want to install a modified version of the source code, you will have to install
the platform-specific dependencies first. For details, see [`INSTALL_SOURCE.md`](INSTALL_SOURCE.md).
