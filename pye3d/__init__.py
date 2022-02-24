try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("pye3d")
except PackageNotFoundError:
    # package is not installed
    pass

__all__ = ["__version__"]
