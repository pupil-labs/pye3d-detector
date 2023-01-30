0.3.2 (2023-01-30)
########################

- Increase physiological ranges to accomodate Neon eye camera positions
  - `phi`: `[-80, 80]deg` -> `[-90,90]deg`
  - `sphere_center.x`: `[-10, 10]mm` -> `[-15,15]mm`
  - `sphere_center.z`: `[20, 75]mm` -> `[15,75]mm`


0.3.1.post1 (2022-10-11)
########################

- Add Python 3.10 and 3.11 wheels

0.3.1 (2022-10-06)
##################

- Correctly calculate direction of ``Observation``'s ``gaze_3d_pair`` lines - #47
- Cleanup build and transition to `Pupil Labs Python module skeleton
  <https://github.com/pupil-labs/python-module-skeleton>`_ structure - #44 and #49
- Add :ref:`examples` - #45
- Remove need to build and link OpenCV - #61

0.3.0 (2021-11-10)
##################

- Change refraction model serialization from pickle to msgpack - #38

0.2.0 (2021-10-12)
##################
- Lower 3d search confidence results by 40% - #36

0.1.1 (2021-07-07)
##################

- Extracted source install instructions into ``INSTALL_SOURCE.md``
- Added ``Codestyle: Black`` badge to ``READEME.md``
- Fixed Github Action badge in ``READEME.md`` indicating status of the build pipeline

0.1.0 (2021-07-07)
##################
- Use long-term-model pupil-circle normal for gaze direction when frozen - #31
- Set ``model_confidence`` to 0.1 if parameter is out of physiological range - #35
- Improve integration tests - #33, #34

0.0.8 (2021-06-17)
##################
- Automated Python wheels for Python 3.6 - 3.9 on Linux, macOS, and Windows

0.0.7 (2021-05-11)
##################
- Simplification of ``Conic`` parameter calculation - #26
- Incremental performance improvements - #27
- Correctly apply corneal-refraction correction to ``diameter_3d`` result - #28

0.0.6 (2021-03-09)
##################

- Expose 3d eye model RMS fitting residuals; disable them by default - #24

0.0.5 (2021-02-09)
##################

- Pin ``scikit-learn==0.24.1`` version and update refraction models to that version -
  Avoids warning when unpickling refraction pipeline
- Replace all occurances of ``np.float`` with ``float`` - ``np.float`` has been
  `deprecated <https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations>`_.
- Stop treating all warnings in `pye3d/cpp/projections.pyx:unproject_ellipse()` as
  errors - The previous behavior was introduced to avoid repeated division-by-zero errors.
  The new implementation continues to handle these as errors but excludes other types of
  warnings, e.g. ``DeprecationWarning``, from this handling.
