import numpy as np
import pytest

from nebius_physical_ai.capture.lift import HeightLifter, MatView
from nebius_physical_ai.capture.mat import DetectedObject, MatSpec

SPEC = MatSpec(350.0, 227.0)
BOX = (170.0, 110.0, 60.0, 40.0, 40.0, 0.0)  # centre x, y (mat mm), length, width, height (mm), yaw (deg)
SIZE = (240, 320)
VIEWS = [(175, 380, -330), (-60, 200, -250), (420, 30, -280)]  # board frame: z negative is above the table


def view_from(position: tuple[float, float, float], target: tuple[float, float, float] = (175, 113, 0)) -> MatView:
    """Camera at `position` looking at `target`, image down = mat down."""
    pos = np.array(position, float)
    z = np.subtract(target, pos) / np.linalg.norm(np.subtract(target, pos))
    y = np.array([0.0, 1.0, 0.0]) - z[1] * z
    y /= np.linalg.norm(y)
    R = np.stack([np.cross(y, z), y, z])
    f = SIZE[1] / 2 / np.tan(np.radians(30))
    K = np.array([[f, 0, SIZE[1] / 2], [0, f, SIZE[0] / 2], [0, 0, 1]])
    return MatView(K, R, -R @ pos, SIZE)


def ray_cast(view: MatView, boxes: list[tuple[float, ...]]) -> np.ndarray:
    """Camera-frame point map (mm) of the mat with boxes standing on it: slab test in each box's own frame."""
    _, hit = view.plane_hits()
    origin = -view.R.T @ view.t
    dirs = hit - origin  # scaled so t = 1 reaches the plane
    t = np.ones(SIZE)
    for cx, cy, length, width, height, yaw in boxes:
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # box frame -> board
        o, d = (origin - [cx, cy, 0]) @ rot, dirs @ rot
        half = np.array([length / 2, width / 2])
        lo, hi = np.array([*-half, -height]), np.array([*half, 0.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (lo - o) / d, (hi - o) / d
        near, far = np.minimum(ta, tb).max(-1), np.maximum(ta, tb).min(-1)
        t = np.where((near <= far) & (near > 0), np.minimum(near, t), t)
    return (origin + dirs * t[..., None]) @ view.R.T + view.t


def lift(objects: list[DetectedObject], boxes: list[tuple[float, ...]], cameras: list[tuple[float, ...]] = VIEWS):
    views = [view_from(p) for p in cameras]
    maps = [(v, ray_cast(v, boxes) * 1.25, np.ones(SIZE, bool)) for v in views]  # predictor 25% off in scale
    return HeightLifter().lift(objects, maps, SPEC)


def perceived(x: float = 172.0, y: float = 105.0) -> DetectedObject:
    """What the mat-plane detector reports: centre and footprint inflated by parallax, no height."""
    return DetectedObject("red", x, y, 3400.0, length_mm=70.0, width_mm=55.0)


@pytest.mark.parametrize("cameras", [VIEWS[:1], VIEWS], ids=["one view", "three views"])
@pytest.mark.parametrize("yaw", [0.0, 30.0, 89.0])
def test_recovers_height_footprint_and_yaw_despite_scale_error(cameras: list[tuple[float, ...]], yaw: float) -> None:
    (obj,) = lift([perceived()], [(*BOX[:5], yaw)], cameras)
    assert obj.height_mm == pytest.approx(40.0, abs=1.5)
    assert (obj.x_mm, obj.y_mm) == pytest.approx((170.0, 110.0), abs=2.0)
    assert (obj.length_mm, obj.width_mm) == pytest.approx((60.0, 40.0), abs=3.0)
    assert abs((obj.yaw_deg - yaw + 90) % 180 - 90) < 3  # yaw is defined modulo 180
    assert obj.shape == "box"


def test_a_taller_neighbour_does_not_leak_in() -> None:
    tall = (240.0, 110.0, 40.0, 40.0, 80.0, 0.0)  # 30 mm gap from the 40 mm box
    short, high = lift([perceived(), perceived(242.0, 108.0)], [BOX, tall])
    assert short.height_mm == pytest.approx(40.0, abs=1.5)
    assert high.height_mm == pytest.approx(80.0, abs=1.5)


def test_object_nothing_stands_on_is_unchanged() -> None:
    (obj,) = lift([perceived()], [], VIEWS[:1])
    assert obj == perceived()


def test_plane_hits_lie_on_the_mat() -> None:
    view = view_from(VIEWS[0])
    _, hit = view.plane_hits()
    assert view.camera_height_mm == pytest.approx(330.0)
    assert view_from((175, 113, -300)).tilt_deg == pytest.approx(0.0, abs=1e-6)  # straight down
    assert view.tilt_deg == pytest.approx(np.degrees(np.arctan2(267, 330)), abs=0.1)
    assert np.abs(hit[..., 2]).max() < 1e-6  # every ray lands on z = 0


@pytest.mark.parametrize(("shape", "expected"), [("disc", np.pi / 4), ("square", 1.0)])
def test_top_fill_tells_a_lid_from_a_disc(shape: str, expected: float) -> None:
    xy = np.random.default_rng(0).uniform(-40, 40, (5000, 2))
    if shape == "disc":
        xy = xy[np.hypot(*xy.T) < 40]
    assert HeightLifter.fill(xy) == pytest.approx(expected, abs=0.03)
