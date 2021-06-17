import pathlib

import joblib


def update_refraction_models(model_directory: str):
    print(f"Searching for models in {model_directory}")
    for path in pathlib.Path(model_directory).glob("[!.]*.save"):
        print(f"Updating {path}")
        model = joblib.load(path)
        joblib.dump(model, path)


if __name__ == "__main__":
    update_refraction_models("pye3d/refraction_models/")
