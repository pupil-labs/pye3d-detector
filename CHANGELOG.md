# Changelog

## 0.1.0 (2021-07-07)
- Use long-term-model pupil-circle normal for gaze direction when frozen - [#31](https://github.com/pupil-labs/pye3d-detector/pull/31)
- Set `model_confidence` to 0.1 if parameter is out of physiological range - [#35](https://github.com/pupil-labs/pye3d-detector/pull/35)
- Improve integration tests - [#33](https://github.com/pupil-labs/pye3d-detector/pull/33), [#34](https://github.com/pupil-labs/pye3d-detector/pull/34)

## 0.0.8 (2021-06-17)
- Automated Python wheels for Python 3.6 - 3.9 on Linux, macOS, and Windows

## 0.0.7 (2021-05-11)
- Simplification of `Conic` parameter calculation - [#26](https://github.com/pupil-labs/pye3d-detector/pull/26)
- Incremental performance improvements - [#27](https://github.com/pupil-labs/pye3d-detector/pull/27)
- Correctly apply corneal-refraction correction to `diameter_3d` result - [#28](https://github.com/pupil-labs/pye3d-detector/pull/28)

## 0.0.6 (2021-03-09)

- Expose 3d eye model RMS fitting residuals; disable them by default - [#24](https://github.com/pupil-labs/pye3d-detector/pull/24)

## 0.0.5 (2021-02-09)

- Pin `scikit-learn==0.24.1` version and update refraction models to that version -
Avoids warning when unpickling refraction pipeline
- Replace all occurances of `np.float` with `float` - `np.float` has been
[deprecated](https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations).
- Stop treating all warnings in `pye3d/cpp/projections.pyx:unproject_ellipse()` as
errors - The previous behavior was introduced to avoid repeated division-by-zero errors.
The new implementation continues to handle these as errors but excludes other types of
warnings, e.g. `DeprecationWarning`, from this handling.
