# Changelog

## 0.0.5 (2021-02-09)

- Pin `scikit-learn==0.22.2.post1` - Avoids warning when unpickling refraction pipeline
- Replace all occurances of `np.float` with `float` - `np.float` has been
[deprecated](https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations).
- Stop treating all warnings in `pye3d/cpp/projections.pyx:unproject_ellipse()` as
errors - The previous behavior was introduced to avoid repeated division-by-zero errors.
The new implementation continues to handle these as errors but excludes other types of
warnings, e.g. `DeprecationWarning`, from this handling.
