def pytest_addoption(parser):
    parser.addoption("--legacy-model-location", action="store", default=None)


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.legacy_model_location
    if "custom_load_dir" in metafunc.fixturenames:
        metafunc.parametrize("custom_load_dir", [option_value])
