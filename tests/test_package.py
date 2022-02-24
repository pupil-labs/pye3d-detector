import pytest


def test_package_metadata() -> None:
    assert hasattr(pytest, "__version__")


def test_import():
    from pye3d.camera import CameraModel
    from pye3d.detector_3d import Detector3D

    camera = CameraModel(1.0, (200, 200))
    Detector3D(camera)


def test_model_update_schedule():
    from pye3d.detector_3d import _ModelUpdateSchedule

    schedule = _ModelUpdateSchedule(update_interval=5.0, warmup_duration=3.0)
    assert schedule.is_update_due(0.0)  # warmup -> schedule
    assert schedule.is_update_due(2.0)  # warmup -> schedule
    assert schedule.is_update_due(3.0)  # warmup -> schedule
    assert not schedule.is_update_due(5.0)
    assert schedule.is_update_due(8.1)
    assert not schedule.is_update_due(10.1)
    assert not schedule.is_update_due(13.1)
    assert schedule.is_update_due(13.2)
    schedule.pause()  # no updates until resume
    assert not schedule.is_update_due(15.0)
    assert not schedule.is_update_due(20.0)
    schedule.resume()  # resets timer
    assert schedule.is_update_due(20.0)


def test_legacy_sklearn_model_loading(custom_load_dir):
    try:
        import joblib  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        pytest.skip("joblib and/or sklearn not installed.")

    from pye3d.refraction import SklearnRefractionizer

    try:
        SklearnRefractionizer(custom_load_dir=custom_load_dir)
    except FileNotFoundError:
        if custom_load_dir is None:
            pytest.skip(
                "No legacy models found in default location. Set custom location via "
                "`pytest --legacy-model-location <location>`"
            )
        raise
