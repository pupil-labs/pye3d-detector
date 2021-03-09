# Changelog

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
