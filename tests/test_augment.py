import cv2
import numpy as np
import pytest

from nebius_physical_ai.augment import MAX_DRIFT_PX, Drift, measure_drift

H, W = 480, 640


def scene(seed: int = 42, n: int = 24) -> np.ndarray:
    """A grey mat with coloured blocks and bars: enough edges in most tiles to register against."""
    rng = np.random.default_rng(seed)
    image = np.full((H, W, 3), 150, np.uint8)
    for _ in range(n):
        x, y = int(rng.integers(0, W - 90)), int(rng.integers(0, H - 90))
        w, h = int(rng.integers(30, 90)), int(rng.integers(30, 90))
        cv2.rectangle(image, (x, y), (x + w, y + h), tuple(int(c) for c in rng.integers(20, 235, 3)), -1)
    return image


def shifted(image: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    return np.roll(image, (dy, dx), axis=(0, 1))


def restyled(image: np.ndarray, seed: int = 0) -> np.ndarray:
    """Same geometry, different look: other hues, brighter, lower contrast, a little sensor noise."""
    rng = np.random.default_rng(seed)
    out = image[..., ::-1].astype(np.float32) * 0.7 + 60
    out += rng.normal(0, 4, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


class TestMeasureDrift:
    def test_identical_frames_have_no_drift(self) -> None:
        frame = scene()
        drift = measure_drift([frame], [frame])
        assert drift.median_px == pytest.approx(0, abs=0.2) and drift.unmatched == 0 and drift.acceptable()

    @pytest.mark.parametrize("pixels", [3, 6, 12])
    def test_a_shift_is_measured_in_pixels(self, pixels: int) -> None:
        frame = scene()
        drift = measure_drift([frame], [shifted(frame, pixels)])
        assert drift.median_px == pytest.approx(pixels, abs=1.0)

    def test_a_diagonal_shift_is_the_length_of_the_move(self) -> None:
        frame = scene()
        assert measure_drift([frame], [shifted(frame, 6, 8)]).median_px == pytest.approx(10.0, abs=1.5)

    def test_appearance_changes_are_not_drift(self) -> None:
        frame = scene()
        drift = measure_drift([frame], [restyled(frame)])
        assert drift.median_px < 1.0 and drift.acceptable()

    def test_a_restyled_and_shifted_video_is_still_caught(self) -> None:
        frame = scene()
        drift = measure_drift([frame], [restyled(shifted(frame, 10))])
        assert drift.median_px == pytest.approx(10, abs=1.5) and not drift.acceptable()

    def test_a_different_scene_does_not_line_up(self) -> None:
        drift = measure_drift([scene(1)], [scene(2)])
        assert not drift.acceptable()

    def test_the_threshold_is_a_few_pixels(self) -> None:
        frame = scene()
        assert measure_drift([frame], [shifted(frame, 2)]).acceptable()
        assert not measure_drift([frame], [shifted(frame, 9)]).acceptable()
        assert 2 <= MAX_DRIFT_PX <= 6

    def test_frames_of_another_resolution_are_compared_at_the_originals(self) -> None:
        frame = scene()
        small = cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        assert measure_drift([frame], [small]).acceptable()

    def test_every_frame_pair_counts(self) -> None:
        a, b = scene(1), scene(2)
        mixed = measure_drift([a, b], [a, shifted(b, 12)])  # one clean frame, one moved
        assert mixed.tiles > measure_drift([a], [a]).tiles  # both frame pairs contribute tiles
        assert mixed.max_px == pytest.approx(12, abs=1.5)  # the moved frame is seen, not averaged away
        assert measure_drift([a], [a]).max_px < 1.0

    def test_mismatched_frame_counts_are_an_error(self) -> None:
        with pytest.raises(ValueError, match="1 original frames but 2 augmented"):
            measure_drift([scene()], [scene(), scene()])

    def test_a_drift_with_no_tiles_is_never_acceptable(self) -> None:
        assert not Drift(0.0, 0.0, 0.0, 0).acceptable()
