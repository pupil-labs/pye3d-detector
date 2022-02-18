import pathlib

import joblib
import msgpack
import numpy as np


def update_refraction_models(model_directory: str):
    print(f"Searching for models in {model_directory}")
    for path in pathlib.Path(model_directory).glob("[!.]*.save"):
        print(f"Converting {path}")
        convert_model_at_path(path)


def convert_model_at_path(path: pathlib.Path):
    pipeline = joblib.load(path)
    _fix_linear_regression_if_necessary(pipeline[2])
    model = {
        "version": 1,
        "steps": {
            "PolynomialFeatures": {
                "params": pipeline[0].get_params(),
                "powers": pipeline[0].powers_.T.astype(float).tolist(),
            },
            "StandardScaler": {
                "params": pipeline[1].get_params(),
                "mean": pipeline[1].mean_[np.newaxis, :].T.tolist(),
                "var": pipeline[1].var_[np.newaxis, :].T.tolist(),
            },
            "LinearRegression": {
                "params": pipeline[2].get_params(),
                "coef": pipeline[2].coef_.T.tolist(),
                "intercept": pipeline[2].intercept_[:, np.newaxis].T.tolist(),
            },
        },
    }
    with path.with_suffix(".msgpack").open("wb") as file:
        msgpack.pack(model, file)


def _fix_linear_regression_if_necessary(lin_reg):
    """Add the missing `positive` attribute if necessary

    If the model was serialized before scikit-learn==0.24, it won't have the `positive`
    attribute which is required for full compatibility with scikit-learn>=0.24.
    """
    try:
        lin_reg.positive
    except AttributeError:
        lin_reg.positive = False  # set to default value


if __name__ == "__main__":
    update_refraction_models("pye3d/refraction_models/")
