"""
Calibration Engine for PDF Coordinate Mapping to Clinical KM Coordinates.

Handles transformations between PDF points (origin at top-left, inverted Y)
and clinical survival coordinates: time t in [t_min, t_max] and S(t) in [0.0, 1.0].
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import fitz
import numpy as np


@dataclass
class Calibration:
    """
    Holds coordinate calibration bounding parameters.

    In PDF space:
      - x increases from left to right.
      - y increases downwards (0 is top of page).
      - y_min_pdf corresponds to S = s_max (top of Y axis, typically 1.0 or 100%).
      - y_max_pdf corresponds to S = s_min (bottom of Y axis, typically 0.0).
      - x_min_pdf corresponds to t = t_min (typically 0).
      - x_max_pdf corresponds to t = t_max (e.g., 60 months).
    """
    x_min_pdf: float
    x_max_pdf: float
    y_min_pdf: float
    y_max_pdf: float
    t_min: float = 0.0
    t_max: float = 60.0
    s_min: float = 0.0
    s_max: float = 1.0

    def pdf_to_clinical(self, x: float, y: float) -> Tuple[float, float]:
        """Convert a PDF coordinate (x, y) to clinical (t, S(t))."""
        x_span = self.x_max_pdf - self.x_min_pdf
        y_span = self.y_max_pdf - self.y_min_pdf

        if abs(x_span) < 1e-6:
            t = self.t_min
        else:
            t = self.t_min + ((x - self.x_min_pdf) / x_span) * (self.t_max - self.t_min)

        if abs(y_span) < 1e-6:
            s = self.s_max
        else:
            # Note: y_min_pdf is top (s_max), y_max_pdf is bottom (s_min)
            s = self.s_min + ((self.y_max_pdf - y) / y_span) * (self.s_max - self.s_min)

        t = max(self.t_min, min(self.t_max, float(t)))
        s = max(self.s_min, min(self.s_max, float(s)))
        return (t, s)

    def clinical_to_pdf(self, t: float, s: float) -> Tuple[float, float]:
        """Convert clinical (t, S(t)) to PDF coordinate (x, y)."""
        t_span = self.t_max - self.t_min
        s_span = self.s_max - self.s_min

        if abs(t_span) < 1e-6:
            x = self.x_min_pdf
        else:
            x = self.x_min_pdf + ((t - self.t_min) / t_span) * (self.x_max_pdf - self.x_min_pdf)

        if abs(s_span) < 1e-6:
            y = self.y_min_pdf
        else:
            y = self.y_max_pdf - ((s - self.s_min) / s_span) * (self.y_max_pdf - self.y_min_pdf)

        return (float(x), float(y))

    def calibrate_points(
        self, raw_pts: List[Tuple[float, float]], enforce_monotonic: bool = True
    ) -> List[Tuple[float, float]]:
        """
        Calibrate a list of raw (x, y) PDF points into (t, S(t)).
        Ensures sorting by time and cleans vertical drops into monotonic step progression.
        """
        if not raw_pts:
            return []

        # Convert all points
        clin_pts = [self.pdf_to_clinical(x, y) for x, y in raw_pts]
        # Sort by t ascending, and S descending (start of step before end of step drop)
        clin_pts.sort(key=lambda p: (p[0], -p[1]))

        if not enforce_monotonic:
            return clin_pts

        # Clean step drops and enforce non-increasing S(t)
        cleaned: List[Tuple[float, float]] = []
        running_min_s = self.s_max

        # Ensure curve starts at t=t_min with S=s_max
        if not clin_pts or clin_pts[0][0] > self.t_min + 1e-3:
            cleaned.append((self.t_min, self.s_max))

        for t, s in clin_pts:
            # KM curves cannot go up
            s = min(running_min_s, s)
            running_min_s = s

            if not cleaned:
                cleaned.append((t, s))
            else:
                prev_t, prev_s = cleaned[-1]
                if abs(t - prev_t) < 1e-4:
                    # Vertical drop at exact same time: keep lowest S
                    cleaned[-1] = (t, min(prev_s, s))
                else:
                    cleaned.append((t, s))

        return cleaned


def detect_axis_lines(page: fitz.Page, min_h_len: float = 80.0, min_v_len: float = 50.0) -> Dict[str, Any]:
    """
    Detect major axis line drawings on a page.
    Identifies horizontal axis lines (X-axes) and vertical axis lines (Y-axes).
    """
    drawings = page.get_drawings()
    h_lines = []
    v_lines = []

    for d in drawings:
        for item in d.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                dx = abs(p2.x - p1.x)
                dy = abs(p2.y - p1.y)

                # Horizontal line
                if dy < 1.0 and dx >= min_h_len:
                    h_lines.append({
                        "x0": min(p1.x, p2.x),
                        "x1": max(p1.x, p2.x),
                        "y": (p1.y + p2.y) / 2.0,
                        "color": d.get("color"),
                        "width": d.get("width")
                    })
                # Vertical line
                elif dx < 1.0 and dy >= min_v_len:
                    v_lines.append({
                        "x": (p1.x + p2.x) / 2.0,
                        "y0": min(p1.y, p2.y),
                        "y1": max(p1.y, p2.y),
                        "color": d.get("color"),
                        "width": d.get("width")
                    })

    # Sort lines by y
    h_lines.sort(key=lambda l: l["y"])
    v_lines.sort(key=lambda l: (l["y0"], l["x"]))

    return {"h_lines": h_lines, "v_lines": v_lines}
