import cv2
from PIL import Image, ImageDraw

from nebius_physical_ai.capture.mat import ARUCO_DICT, CORNER_NAMES, MatSpec

A4_MM = (210, 297)
DPI = 300
MARGIN_MM = 20
LABEL_GAP_MM = 25


def mm_to_px(mm: float) -> int:
    return round(mm / 25.4 * DPI)


def render_marker_sheet(spec: MatSpec) -> Image.Image:
    """One A4 page with the four corner markers at true size, laid out 2x2."""
    cell_w = (A4_MM[0] - 2 * MARGIN_MM) / 2
    cell_h = spec.marker_mm + LABEL_GAP_MM
    if spec.marker_mm > cell_w or 2 * cell_h + 2 * MARGIN_MM > A4_MM[1]:
        raise ValueError(f"{spec.marker_mm} mm markers do not fit four to an A4 page")
    page = Image.new("L", tuple(mm_to_px(v) for v in A4_MM), 255)
    draw = ImageDraw.Draw(page)
    side = mm_to_px(spec.marker_mm)
    for i, corner in CORNER_NAMES.items():
        x = mm_to_px(MARGIN_MM + (i % 2) * cell_w)
        y = mm_to_px(MARGIN_MM + (i // 2) * cell_h)
        page.paste(Image.fromarray(cv2.aruco.generateImageMarker(ARUCO_DICT, i, side)), (x, y))
        draw.text((x, y + side + mm_to_px(4)), f"marker {i}: {corner}", fill=0)
    draw.text(
        (mm_to_px(MARGIN_MM), mm_to_px(A4_MM[1] - MARGIN_MM)),
        f"Place marker centres on the corners of a {spec.width_mm:.0f} x {spec.height_mm:.0f} mm "
        f"rectangle, upright. Check each marker prints at {spec.marker_mm:.0f} mm.",
        fill=0,
    )
    return page
