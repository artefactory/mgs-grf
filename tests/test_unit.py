import importlib
import numpy as np
import pytest

# Helper to locate the MGSGRFOverSampler class in the project
def _get_oversampler_class():
    pkg = pytest.importorskip("mgs_grf")
    # Direct attribute on package
    if hasattr(pkg, "MGSGRFOverSampler"):
        return getattr(pkg, "MGSGRFOverSampler")
    # Try common submodule names
    for sub in ("over_sampling", "oversampling", "sampling", "over", "resampling"):
        try:
            mod = importlib.import_module(f"mgs_grf.{sub}")
        except Exception:
            continue
        if hasattr(mod, "MGSGRFOverSampler"):
            return getattr(mod, "MGSGRFOverSampler")
    pytest.skip("MGSGRFOverSampler not found in mgs_grf package")

def _instantiate(cls, **kwargs):
    try:
        return cls(**kwargs)
    except TypeError:
        # Fallback to no-arg constructor if kwargs not supported
        return cls()

def _fit_resample(estimator, X, y):
    # many samplers follow fit_resample API, test for both fit_resample and fit/transform
    if hasattr(estimator, "fit_resample"):
        return estimator.fit_resample(X, y)
    if hasattr(estimator, "fit") and hasattr(estimator, "resample"):
        estimator.fit(X, y)
        return estimator.resample(X, y)
    raise AttributeError("Estimator does not implement fit_resample or fit+resample")

def _counts(y):
    vals, cnts = np.unique(y, return_counts=True)
    return dict(zip(vals.tolist(), cnts.tolist()))

def test_fit_resample_increases_minority_count():
    MGSGRFOverSampler = _get_oversampler_class()
    rng = np.random.RandomState(0)
    X = rng.randn(100, 4)
    # imbalanced target: class 0 -> 80, class 1 -> 20
    y = np.array([0] * 80 + [1] * 20)
    sampler = _instantiate(MGSGRFOverSampler,K=X.shape[1], random_state=42)
    # Continuous-only fitting should emit a warning; catch it
    with pytest.warns(UserWarning, match="continuous features only"):
        X_res, y_res = _fit_resample(sampler, X, y)
    assert X_res.shape[0] == len(y_res)
    orig_counts = _counts(y)
    new_counts = _counts(y_res)
    # minority class count should increase
    assert new_counts.get(1, 0) > orig_counts.get(1, 0)
    # majority class should not decrease
    assert new_counts.get(0, 0) >= orig_counts.get(0, 0)

def test_reproducible_with_random_state():
    MGSGRFOverSampler = _get_oversampler_class()
    rng = np.random.RandomState(1)
    X = rng.randn(60, 3)
    y = np.array([0] * 45 + [1] * 10 + [2] * 5)  # multiclass imbalanced
    cls = MGSGRFOverSampler
    s1 = _instantiate(cls,K=X.shape[1], random_state=0)
    s2 = _instantiate(cls, K=X.shape[1], random_state=0)
    # Catch warnings for continuous-only fitting
    with pytest.warns(UserWarning, match="continuous features only"):
        X1, y1 = _fit_resample(s1, X, y)
    with pytest.warns(UserWarning, match="continuous features only"):
        X2, y2 = _fit_resample(s2, X, y)
    # Results should be identical when random_state is fixed
    assert X1.shape == X2.shape
    assert y1.shape == y2.shape
    # Compare labels and features exactly
    assert np.array_equal(y1, y2)
    assert np.allclose(X1.astype(float), X2.astype(float))

def test_multiclass_minority_enhanced():
    MGSGRFOverSampler = _get_oversampler_class()
    rng = np.random.RandomState(2)
    X = rng.randn(120, 2)
    # three classes with varying counts
    y = np.array([0] * 70 + [1] * 40 + [2] * 10)
    sampler = _instantiate(MGSGRFOverSampler,K=X.shape[1], random_state=7)
    # Catch warning for continuous-only fitting
    with pytest.warns(UserWarning, match="continuous features only"):
        X_res, y_res = _fit_resample(sampler, X, y)
    new_counts = _counts(y_res)
    # smallest class (2) should increase
    assert new_counts.get(2, 0) > 10
    # no class should disappear
    for cls_label in (0, 1, 2):
        assert new_counts.get(cls_label, 0) > 0

def test_invalid_input_raises_for_single_class():
    MGSGRFOverSampler = _get_oversampler_class()
    X = np.zeros((10, 2))
    y = np.array([0] * 10)  # single class
    sampler = _instantiate(MGSGRFOverSampler ,K=X.shape[1], random_state=0)
    with pytest.raises(Exception):
        _fit_resample(sampler, X, y)