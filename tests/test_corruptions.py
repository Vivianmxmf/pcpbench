"""Tests for pcpbench.datasets.corruptions -- the pure-Python Hendrycks-Dietterich corruption
suite."""

from __future__ import annotations

import numpy as np
import pytest

from pcpbench.datasets import corruptions as c

_MONOTONIC_CORRUPTIONS = tuple(
    name for name in c.CORRUPTIONS if name not in c.NON_MONOTONIC
)


@pytest.fixture
def images() -> np.ndarray:
    """4x32x32x3 random uint8 -- tiny, synthetic, no download."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)


@pytest.fixture
def float_images() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.random((4, 32, 32, 3)).astype(np.float32)


@pytest.fixture
def large_images() -> np.ndarray:
    """2x224x224x3 random uint8 -- ImageNet-C-scale fixture, so the non-vacuity floor test
    catches a near-no-op table at ImageNet resolution too, not just at CIFAR (32x32) scale."""
    rng = np.random.default_rng(2)
    return rng.integers(0, 256, size=(2, 224, 224, 3), dtype=np.uint8)


_MIN_MEAN_ABS_DIFF = 0.01


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


class TestRegistries:
    def test_corruptions_is_nonempty_tuple_of_str(self):
        assert isinstance(c.CORRUPTIONS, tuple)
        assert len(c.CORRUPTIONS) > 0
        assert all(isinstance(name, str) for name in c.CORRUPTIONS)

    def test_omitted_documents_the_wand_dependent_corruptions(self):
        expected = {"glass_blur", "motion_blur", "zoom_blur", "snow", "frost"}
        assert expected.issubset(set(c.OMITTED))
        for name, reason in c.OMITTED.items():
            assert isinstance(reason, str) and reason, name

    def test_corruptions_and_omitted_are_disjoint(self):
        assert set(c.CORRUPTIONS).isdisjoint(set(c.OMITTED))

    def test_gaussian_noise_is_implemented(self):
        assert "gaussian_noise" in c.CORRUPTIONS


class TestShapeAndRange:
    @pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
    def test_uint8_shape_and_range(self, images, severity):
        for name in c.CORRUPTIONS:
            out = c.apply_corruption(images, name, severity, seed=0)
            assert out.shape == images.shape, name
            assert out.dtype == images.dtype, name
            assert np.isfinite(out.astype(np.float64)).all(), name
            assert out.min() >= 0 and out.max() <= 255, name

    @pytest.mark.parametrize("severity", [1, 3, 5])
    def test_float_shape_dtype_and_range(self, float_images, severity):
        for name in c.CORRUPTIONS:
            out = c.apply_corruption(float_images, name, severity, seed=0)
            assert out.shape == float_images.shape, name
            assert out.dtype == float_images.dtype, name
            assert np.isfinite(out).all(), name
            assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6, name


class TestDeterminism:
    @pytest.mark.parametrize("severity", [1, 3, 5])
    def test_same_seed_gives_identical_output(self, images, severity):
        for name in c.CORRUPTIONS:
            a = c.apply_corruption(images, name, severity, seed=42)
            b = c.apply_corruption(images, name, severity, seed=42)
            assert np.array_equal(a, b), name

    def test_different_seed_generally_gives_different_output(self, images):
        rng_dependent = (
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "speckle_noise",
            "fog",
            "elastic_transform",
            "spatter",
        )
        for name in rng_dependent:
            if name not in c.CORRUPTIONS:
                continue
            a = c.apply_corruption(images, name, 3, seed=1)
            b = c.apply_corruption(images, name, 3, seed=2)
            assert not np.array_equal(a, b), name


class TestNonVacuous:
    @pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
    def test_corruption_actually_changes_the_image(self, images, severity):
        for name in c.CORRUPTIONS:
            out = c.apply_corruption(images, name, severity, seed=3)
            assert (
                _mean_abs_diff(out, images) > 0.0
            ), f"{name} severity={severity} was a no-op"

    @pytest.mark.parametrize("fixture_name", ["images", "large_images"])
    def test_minimum_effect_size_at_severity_one(self, request, fixture_name):
        fixture = request.getfixturevalue(fixture_name)
        for name in c.CORRUPTIONS:
            out = c.apply_corruption(fixture, name, 1, seed=5)
            diff = _mean_abs_diff(out, fixture)
            assert diff >= _MIN_MEAN_ABS_DIFF, (
                f"{name} severity=1 on {fixture_name} mean|diff|={diff:.5f} is below the "
                f"minimum {_MIN_MEAN_ABS_DIFF} floor -- suspiciously close to a no-op"
            )


class TestSeverityMonotonicity:
    """Higher severity -> strictly larger mean|corrupted - clean| for the noise/blur/digital/
    weather families where that is a meaningful property of the Hendrycks-Dietterich parameter
    tables."""

    def test_monotonic_families_are_covered(self):
        assert set(_MONOTONIC_CORRUPTIONS) | set(c.NON_MONOTONIC) == set(c.CORRUPTIONS)

    def test_mean_abs_diff_is_strictly_increasing_in_severity(self, images):
        eps = 1e-6
        for name in _MONOTONIC_CORRUPTIONS:
            diffs = [
                _mean_abs_diff(c.apply_corruption(images, name, sev, seed=7), images)
                for sev in (1, 2, 3, 4, 5)
            ]
            assert all(
                diffs[i] < diffs[i + 1] - eps for i in range(len(diffs) - 1)
            ), f"{name} mean|diff| not strictly increasing in severity: {diffs}"

    def test_non_monotonic_families_documented_and_still_change_image(self, images):
        for name in c.NON_MONOTONIC:
            if name not in c.CORRUPTIONS:
                continue
            for sev in (1, 3, 5):
                out = c.apply_corruption(images, name, sev, seed=7)
                assert _mean_abs_diff(out, images) > 0.0, (name, sev)


class TestValidation:
    def test_unknown_corruption_raises(self, images):
        with pytest.raises(ValueError):
            c.apply_corruption(images, "not_a_real_corruption", 3, seed=0)

    def test_omitted_corruption_raises_with_reason(self, images):
        with pytest.raises(ValueError, match="ImageMagick"):
            c.apply_corruption(images, "motion_blur", 3, seed=0)

    @pytest.mark.parametrize("severity", [0, 6, -1])
    def test_out_of_range_severity_raises(self, images, severity):
        with pytest.raises(ValueError):
            c.apply_corruption(images, "gaussian_noise", severity, seed=0)

    def test_wrong_shape_raises(self):
        bad = np.zeros((4, 32, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            c.apply_corruption(bad, "gaussian_noise", 3, seed=0)


class TestApplyCorruptionToImages:
    def test_handles_varying_size_pil_images(self):
        pytest.importorskip("PIL")
        from PIL import Image

        rng = np.random.default_rng(0)
        sizes = [(16, 16), (24, 20), (32, 32)]
        images = [
            Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))
            for (h, w) in sizes
        ]
        out = c.apply_corruption_to_images(images, "gaussian_noise", 3, seed=0)
        assert len(out) == len(images)
        for src, dst in zip(images, out):
            assert dst.size == src.size
            assert np.asarray(dst).shape == np.asarray(src).shape

    def test_deterministic_given_seed(self):
        pytest.importorskip("PIL")
        from PIL import Image

        rng = np.random.default_rng(0)
        images = [
            Image.fromarray(rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8))
            for _ in range(3)
        ]
        out1 = c.apply_corruption_to_images(images, "fog", 4, seed=11)
        out2 = c.apply_corruption_to_images(images, "fog", 4, seed=11)
        for a, b in zip(out1, out2):
            assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_images_do_not_share_identical_noise_pattern(self):
        pytest.importorskip("PIL")
        from PIL import Image

        base = np.random.default_rng(0).integers(
            0, 256, size=(16, 16, 3), dtype=np.uint8
        )
        images = [Image.fromarray(base.copy()) for _ in range(3)]
        out = c.apply_corruption_to_images(images, "gaussian_noise", 3, seed=0)
        arrs = [np.asarray(im) for im in out]
        assert not (
            np.array_equal(arrs[0], arrs[1]) and np.array_equal(arrs[1], arrs[2])
        )
