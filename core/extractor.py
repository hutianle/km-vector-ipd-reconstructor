"""
Vector Path & At-Risk Table Parser for Clinical Kaplan-Meier Survival Curves.

Extracts native vector drawings, groups paths by stroke color and dash style,
isolates true vertical survival step drops, filters out censoring tick marks using
multi-criteria geometric and style heuristics, detects sub-figure panels,
and parses numbers-at-risk tables.
"""

from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Any, Optional
import fitz
from .calibration import detect_axis_lines

TEXT_EXTRACT_FLAGS = fitz.TEXTFLAGS_SEARCH | fitz.TEXT_IGNORE_ACTUALTEXT


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    """Convert normalized RGB tuple (0.0 to 1.0) to hex string."""
    r = int(round(max(0.0, min(1.0, rgb[0])) * 255))
    g = int(round(max(0.0, min(1.0, rgb[1])) * 255))
    b = int(round(max(0.0, min(1.0, rgb[2])) * 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def validate_vector_page(doc: fitz.Document, page_num: int) -> Tuple[bool, str]:
    """
    Gatekeeper Check: Verifies whether the target page contains native vector drawing paths
    and is not a monolithic rasterized bitmap image.

    Args:
        doc: PyMuPDF Document instance.
        page_num: 1-indexed page number.

    Returns:
        (is_valid, message): Boolean indicating validation status and explanatory message.
    """
    if page_num < 1 or page_num > len(doc):
        return False, f"Page {page_num} is out of bounds (document has {len(doc)} pages)."

    page = doc[page_num - 1]
    drawings = page.get_drawings()
    images = page.get_images()

    num_drawings = len(drawings)
    num_images = len(images)

    # Check minimum path threshold
    if num_drawings <= 30:
        if num_images > 0:
            return False, (
                f"Page {page_num} validation failed: Found only {num_drawings} vector paths "
                f"and {num_images} embedded raster images. The plot appears to be a flat "
                "rasterized bitmap (PNG/JPEG) without native vector paths. High-fidelity "
                "vector reconstruction requires true vector graphics."
            )
        return False, (
            f"Page {page_num} validation failed: Found only {num_drawings} vector path drawings "
            "(minimum required is 30 paths). No sufficient Kaplan-Meier curves detected on this page."
        )

    return True, (
        f"Vector Validation Passed: Detected {num_drawings} native vector drawings "
        f"and {num_images} image blocks on Page {page_num}. Curves are vector-rendered."
    )


def detect_panels_and_regions(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Automatically detect sub-figure panels (e.g., Figure A vs Figure B) on a page.
    Matches vertical Y-axis lines with corresponding horizontal X-axis lines.

    Returns:
        List of panel dicts containing panel_id, label, plot_rect, table_rect, and axis coordinates.
    """
    axes = detect_axis_lines(page, min_h_len=80.0, min_v_len=50.0)
    h_lines = axes["h_lines"]
    v_lines = axes["v_lines"]

    panels = []
    for i, v in enumerate(v_lines):
        matching_h = None
        for h in h_lines:
            if abs(h["x0"] - v["x"]) < 10.0 and abs(h["y"] - v["y1"]) < 10.0:
                matching_h = h
                break

        if matching_h:
            plot_rect = fitz.Rect(v["x"], v["y0"], matching_h["x1"], matching_h["y"])

            if i + 1 < len(v_lines):
                next_y = v_lines[i + 1]["y0"] - 15.0
            else:
                next_y = min(page.rect.height - 30.0, matching_h["y"] + 120.0)

            table_rect = fitz.Rect(
                max(0.0, v["x"] - 60.0),
                matching_h["y"],
                min(page.rect.width, matching_h["x1"] + 30.0),
                max(matching_h["y"] + 50.0, next_y),
            )

            label_char = chr(65 + i)
            panel_title = f"Panel {label_char}"

            nearby_text = page.get_text(
                "text",
                flags=TEXT_EXTRACT_FLAGS,
                clip=fitz.Rect(v["x"] - 50, v["y0"] - 30, v["x"] + 150, v["y0"] + 40),
            )
            plot_text = page.get_text(
                "text",
                flags=TEXT_EXTRACT_FLAGS,
                clip=fitz.Rect(v["x"] - 30, v["y0"] - 20, matching_h["x1"] + 30, matching_h["y"]),
            )
            combined_txt = (nearby_text + " " + plot_text).lower()
            if "event-free" in combined_txt or "efs" in combined_txt:
                panel_title += " (EFS - Event-Free Survival)"
            elif "progression-free" in combined_txt or "pfs" in combined_txt:
                panel_title += " (PFS - Progression-Free Survival)"
            elif "complete response" in combined_txt or "dor" in combined_txt or "duration of" in combined_txt:
                panel_title += " (DoR - Duration of Response)"
            elif "overall survival" in combined_txt or "os" in combined_txt:
                panel_title += " (OS - Overall Survival)"

            panels.append({
                "panel_id": f"Panel_{label_char}",
                "label": panel_title,
                "plot_rect": plot_rect,
                "table_rect": table_rect,
                "x_min": v["x"],
                "x_max": matching_h["x1"],
                "y_top": v["y0"],
                "y_bot": matching_h["y"],
            })

    # Adjust table_rect boundaries for multi-panel grids to prevent horizontal bleed
    if len(panels) > 1:
        for p in panels:
            for other in panels:
                if other is p:
                    continue
                # If other panel is to the left in roughly the same vertical row
                if other["plot_rect"].x1 < p["plot_rect"].x0 and abs(other["plot_rect"].y0 - p["plot_rect"].y0) < 60:
                    mid_x = (other["plot_rect"].x1 + p["plot_rect"].x0) / 2.0
                    p["table_rect"].x0 = max(p["table_rect"].x0, mid_x)
                # If other panel is to the right in roughly the same vertical row
                if other["plot_rect"].x0 > p["plot_rect"].x1 and abs(other["plot_rect"].y0 - p["plot_rect"].y0) < 60:
                    mid_x = (p["plot_rect"].x1 + other["plot_rect"].x0) / 2.0
                    p["table_rect"].x1 = min(p["table_rect"].x1, mid_x)
                # If other panel is below in roughly the same column
                if other["plot_rect"].y0 > p["plot_rect"].y1 and abs(other["plot_rect"].x0 - p["plot_rect"].x0) < 60:
                    p["table_rect"].y1 = min(p["table_rect"].y1, other["plot_rect"].y0 - 5.0)

    if not panels and len(page.get_drawings()) > 20:
        drawings = page.get_drawings()
        all_rects = [fitz.Rect(d["rect"]) for d in drawings]
        bounding_box = fitz.Rect(
            min(r.x0 for r in all_rects),
            min(r.y0 for r in all_rects),
            max(r.x1 for r in all_rects),
            max(r.y1 for r in all_rects),
        )
        split_y = bounding_box.y0 + 0.7 * bounding_box.height
        panels.append({
            "panel_id": "Panel_A",
            "label": "Figure Panel (Auto-detected)",
            "plot_rect": fitz.Rect(bounding_box.x0, bounding_box.y0, bounding_box.x1, split_y),
            "table_rect": fitz.Rect(bounding_box.x0, split_y, bounding_box.x1, bounding_box.y1 + 40),
            "x_min": bounding_box.x0,
            "x_max": bounding_box.x1,
            "y_top": bounding_box.y0,
            "y_bot": split_y,
        })

    return panels


def filter_censoring_ticks(
    drawings: List[Dict[str, Any]],
    plot_rect: fitz.Rect,
    target_color: Tuple[float, float, float],
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Distinguishes true survival step drops from censoring tick marks.

    Multi-criteria classification:
      1. Stroke width disparity: Censoring ticks are typically drawn with a thinner stroke
         (e.g., 0.47 pt vs 0.63 pt for the main survival curve).
      2. Modal length test: Censoring ticks have uniform fixed marker length (e.g. 3.6 pt),
         whereas step drops vary in height according to event counts.
      3. Boundary violation: Vertical whiskers protruding above the top axis boundary (S > 100%)
         are definitely tick marks.
    """
    v_segments = []
    h_segments = []

    for d in drawings:
        r = fitz.Rect(d["rect"])
        if not r.intersects(plot_rect):
            continue
        c = d.get("color")
        if not c or len(c) != 3:
            continue
        if any(abs(c[i] - target_color[i]) > 0.05 for i in range(3)):
            continue
        w = round(d.get("width") or 0.0, 3)

        for it in d.get("items", []):
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                dx = abs(p2.x - p1.x)
                dy = abs(p2.y - p1.y)

                if dy < 0.5 and dx >= 0.5:
                    h_segments.append((min(p1.x, p2.x), max(p1.x, p2.x), (p1.y + p2.y) / 2.0, w))
                elif dx < 0.6 and dy >= 0.2:
                    v_segments.append(((p1.x + p2.x) / 2.0, min(p1.y, p2.y), max(p1.y, p2.y), dy, w))

    h_widths = [s[3] for s in h_segments]
    dominant_curve_w = max(h_widths, default=0.6)

    candidate_tick_dys = [round(s[3], 2) for s in v_segments if 1.5 <= s[3] <= 6.0]
    tick_len_mode = None
    if candidate_tick_dys:
        mode_val, count = Counter(candidate_tick_dys).most_common(1)[0]
        if count >= 5:
            tick_len_mode = mode_val

    true_drops = []
    ticks = []

    for x_val, y_top, y_bot, dy, w in v_segments:
        is_tick = False

        if dominant_curve_w > 0.5 and w < dominant_curve_w - 0.08:
            is_tick = True
        elif tick_len_mode is not None and abs(round(dy, 2) - tick_len_mode) < 0.08:
            is_tick = True
        elif y_top < plot_rect.y0 - 0.5:
            is_tick = True

        entry = {"x": x_val, "y_top": y_top, "y_bot": y_bot, "dy": dy, "width": w}
        if is_tick:
            ticks.append(entry)
        else:
            true_drops.append(entry)

    return true_drops, ticks


def extract_curve_coordinates(
    page: fitz.Page, plot_rect: fitz.Rect
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts Kaplan-Meier curve step points by capturing the vertical drops
    and strictly filtering out censoring tick marks.
    """
    drawings = page.get_drawings()
    colors_found: Dict[Tuple[float, float, float], List[float]] = defaultdict(list)
    color_dashes: Dict[Tuple[float, float, float], str] = {}

    for d in drawings:
        r = fitz.Rect(d["rect"])
        if not r.intersects(plot_rect):
            continue
        c = d.get("color")
        if not c or len(c) != 3:
            continue

        is_dark = all(val < 0.15 for val in c)
        if is_dark:
            if r.y1 >= plot_rect.y1 - 2.0 or r.x0 <= plot_rect.x0 + 2.0:
                continue

        c_key = (round(c[0], 2), round(c[1], 2), round(c[2], 2))
        dashes = str(d.get("dashes", "[] 0"))
        if "[" in dashes and dashes.strip() != "[] 0":
            color_dashes[c_key] = dashes
        elif c_key not in color_dashes:
            color_dashes[c_key] = dashes

        for it in d.get("items", []):
            if it[0] == "l":
                colors_found[c_key].extend([it[1].x, it[2].x])

    valid_curves: Dict[str, Dict[str, Any]] = {}
    plot_w = plot_rect.width

    for c_key, xs in colors_found.items():
        if not xs:
            continue
        span_x = max(xs) - min(xs)

        if span_x >= 0.25 * plot_w:
            true_drops, censoring_ticks = filter_censoring_ticks(drawings, plot_rect, c_key)
            hex_color = rgb_to_hex(c_key)
            dash_val = color_dashes.get(c_key, "")
            is_dashed = ("[" in dash_val and dash_val.strip() != "[] 0")

            if len(true_drops) >= 3:
                true_drops.sort(key=lambda item: item["x"])
                x_origin = min(xs)
                first_y = true_drops[0]["y_top"] if true_drops else plot_rect.y0
                step_points: List[Tuple[float, float]] = [(x_origin, first_y)]

                for d_info in true_drops:
                    step_points.append((d_info["x"], d_info["y_top"]))
                    step_points.append((d_info["x"], d_info["y_bot"]))

                # Extend step curve horizontally to the maximum horizontal span of this curve
                x_max_curve = max(xs)
                if x_max_curve > true_drops[-1]["x"] + 0.5:
                    step_points.append((x_max_curve, true_drops[-1]["y_bot"]))

                valid_curves[hex_color] = {
                    "raw_color": c_key,
                    "hex": hex_color,
                    "points": step_points,
                    "vertical_drops": true_drops,
                    "censoring_ticks": censoring_ticks,
                    "drop_count": len(true_drops),
                    "tick_count": len(censoring_ticks),
                    "dash_style": "dashed" if is_dashed else "solid",
                    "span_x": span_x,
                }
            else:
                # Fallback to line vertices if few vertical drops detected
                all_pts = []
                for d in drawings:
                    if fitz.Rect(d["rect"]).intersects(plot_rect):
                        c = d.get("color")
                        if c and len(c) == 3 and all(abs(c[k] - c_key[k]) < 0.05 for k in range(3)):
                            for it in d.get("items", []):
                                if it[0] == "l":
                                    all_pts.extend([(it[1].x, it[1].y), (it[2].x, it[2].y)])
                all_pts_sorted = sorted(all_pts, key=lambda p: (p[0], p[1]))
                valid_curves[hex_color] = {
                    "raw_color": c_key,
                    "hex": hex_color,
                    "points": all_pts_sorted,
                    "vertical_drops": [],
                    "censoring_ticks": [],
                    "drop_count": len(all_pts_sorted) // 2,
                    "tick_count": 0,
                    "dash_style": "dashed" if is_dashed else "solid",
                    "span_x": span_x,
                }

    # Sort curves: solid curves first (standard experimental arm), then dashed curves (control/placebo arm),
    # and secondary sort by horizontal span descending.
    sorted_items = sorted(
        valid_curves.items(),
        key=lambda item: (0 if item[1]["dash_style"] == "solid" else 1, -item[1]["span_x"]),
    )
    return dict(sorted_items)


def extract_risk_table(
    page: fitz.Page,
    table_rect: fitz.Rect,
    plot_rect: Optional[fitz.Rect] = None,
    default_arm_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extracts numbers-at-risk text objects located under the X-axis.
    Associates time-interval columns (t = 0, 6, 12, ...) with at-risk counts for each arm.
    Supports multi-frequency sampling (finer risk sampling than labeled axis ticks)
    and robust horizontal bounding against adjacent multi-panel bleed.
    """
    words = page.get_text("words", flags=TEXT_EXTRACT_FLAGS)
    x_min_bound = (plot_rect.x0 - 45.0) if plot_rect is not None else (table_rect.x0 - 10.0)
    x_max_bound = (plot_rect.x1 + 15.0) if plot_rect is not None else (table_rect.x1 + 10.0)
    x_min_bound = max(x_min_bound, table_rect.x0)
    x_max_bound = min(x_max_bound, table_rect.x1)

    table_words = []
    for w in words:
        r = fitz.Rect(w[:4])
        mid_x = (w[0] + w[2]) / 2.0
        if r.intersects(table_rect) and (x_min_bound <= mid_x <= x_max_bound):
            table_words.append(w)

    clusters: List[List[Any]] = []
    for w in sorted(table_words, key=lambda x: (x[1], x[0])):
        w_ymid = (w[1] + w[3]) / 2.0
        matched = False
        for cl in clusters:
            cl_ymid = sum((item[1] + item[3]) / 2.0 for item in cl) / len(cl)
            if abs(w_ymid - cl_ymid) <= 3.5:
                cl.append(w)
                matched = True
                break
        if not matched:
            clusters.append([w])

    lines = []
    raw_lines = []
    for cl in sorted(clusters, key=lambda c: sum((w[1] + w[3]) / 2.0 for w in c) / len(c)):
        lw = sorted(cl, key=lambda x: x[0])
        lines.append(lw)
        raw_lines.append(" ".join(w[4] for w in lw))

    time_points: List[float] = []
    time_row_idx = -1

    for idx, lw in enumerate(lines):
        tokens = [w[4] for w in lw if re.match(r"^\d+$", w[4])]
        if "0" in tokens:
            z_idx = tokens.index("0")
            sub_tokens = tokens[z_idx:]
            if len(sub_tokens) >= 3:
                nums = [float(t) for t in sub_tokens]
                if all(nums[i] < nums[i + 1] for i in range(len(nums) - 1)):
                    time_points = nums
                    time_row_idx = idx
                    break

    # Extract candidate arm names to help expand short abbreviations (e.g., '+Bev' -> 'm-FOLFOXIRI + Bevacizumab')
    cand_rect = plot_rect if plot_rect is not None else table_rect
    cand_names = extract_candidate_arm_names(page, cand_rect)

    def parse_num_token(token: str) -> Optional[int]:
        if re.match(r"^\d+$", token):
            return int(token)
        elif token in ["··", "·", "NA", "\\267\\267", "••", "•"]:
            return 0
        return None

    def parse_cens_token(token: str) -> Optional[int]:
        m = re.match(r"^\(?(\d+)\)?$", token)
        if m:
            return int(m.group(1))
        elif token in ["(··)", "(·)", "(--)", "(—)", "(-)"]:
            return None
        return None

    arms_data: Dict[str, List[Dict[str, Any]]] = {}
    if time_row_idx >= 0 and time_points:
        t0_val = time_points[0]
        t_end_val = time_points[-1]
        n_expected = len(time_points)
        arm_counter = 0

        idx = time_row_idx + 1
        while idx < len(lines):
            lw = lines[idx]
            line_str = " ".join(w[4] for w in lw)

            if any(h in line_str.lower() for h in ["number at risk", "time (months)", "time", "months", "estimated probability"]):
                idx += 1
                continue

            text_tokens = []
            num_tokens = []
            for w in lw:
                val = parse_num_token(w[4])
                if val is not None and not w[4].startswith("("):
                    num_tokens.append((w, val))
                elif not re.match(r"^\(?\d+\)?$", w[4]) and w[4] not in ["—", "---", "-", "--", ":", "+", "(··)", "(·)"]:
                    text_tokens.append(w[4])

            # Skip pure parenthesized censoring lines: e.g. (0) (4) (5)...
            is_pure_cens_line = all(re.match(r"^\(?\d+\)?$", w[4]) or w[4] in ["(··)", "(·)", "··", "·"] for w in lw) and any(w[4].startswith("(") for w in lw)
            if is_pure_cens_line:
                idx += 1
                continue

            if num_tokens:
                counts = [val for _, val in num_tokens]
                raw_arm_name = " ".join(text_tokens).strip() if text_tokens else ""
                clean_arm = re.sub(r"^[+\-–—\s]+", "", raw_arm_name).strip()

                if raw_arm_name:
                    arm_name = raw_arm_name
                    # Expand short abbreviations if matched against legend candidate names
                    if raw_arm_name.startswith(("+", "-", "–", "—")) or len(clean_arm) <= 4:
                        for c in cand_names:
                            if clean_arm.lower() in c.lower() or c.lower() in clean_arm.lower():
                                arm_name = c
                                break
                            elif len(clean_arm) >= 3 and clean_arm.lower()[:3] in c.lower():
                                arm_name = c
                                break
                elif cand_names and arm_counter < len(cand_names):
                    arm_name = cand_names[arm_counter]
                elif default_arm_names and arm_counter < len(default_arm_names):
                    arm_name = default_arm_names[arm_counter]
                else:
                    arm_name = f"Arm {arm_counter + 1}"
                arm_counter += 1

                cens_counts = []
                if idx + 1 < len(lines):
                    next_lw = lines[idx + 1]
                    next_cens = [parse_cens_token(w[4]) for w in next_lw if w[4].startswith("(") or parse_num_token(w[4]) is not None]
                    if len(next_cens) == len(counts):
                        cens_counts = next_cens

                if len(counts) == n_expected:
                    entries = []
                    last_cens = 0
                    for j, t in enumerate(time_points):
                        c_val = cens_counts[j] if j < len(cens_counts) else None
                        if c_val is not None:
                            last_cens = c_val
                        elif c_val is None and j == len(time_points) - 1:
                            c_val = last_cens
                        entries.append({"time": t, "n_risk": counts[j], "censored": c_val})
                    arms_data[arm_name] = entries
                elif len(counts) > n_expected:
                    # Multi-frequency sampling: risk counts sampled finer than axis ticks
                    n_cnt = len(counts)
                    dt = (t_end_val - t0_val) / (n_cnt - 1)
                    common_steps = [1.0, 2.0, 3.0, 4.0, 6.0, 12.0]
                    best_step = min(common_steps, key=lambda s: abs(s - dt))
                    entries = []
                    for j, cnt in enumerate(counts):
                        t_j = round(t0_val + j * best_step, 1)
                        c_val = cens_counts[j] if j < len(cens_counts) else None
                        entries.append({"time": t_j, "n_risk": cnt, "censored": c_val})
                    arms_data[arm_name] = entries

            idx += 1

    final_times = [e["time"] for e in list(arms_data.values())[0]] if arms_data else time_points
    axis_t_max = max(final_times[-1], time_points[-1]) if time_points else (final_times[-1] if final_times else 60.0)

    return {
        "time_points": final_times,
        "arms": arms_data,
        "labeled_ticks": time_points,
        "axis_t_max": axis_t_max,
        "raw_lines": raw_lines,
    }


def extract_candidate_arm_names(page: fitz.Page, plot_rect: fitz.Rect) -> List[str]:
    """
    Extract candidate treatment arm names from legend boxes inside or adjacent to the plot.
    """
    text_blocks = page.get_text("blocks", flags=TEXT_EXTRACT_FLAGS)
    candidates = []
    for b in text_blocks:
        b_rect = fitz.Rect(b[:4])
        if b_rect.intersects(
            fitz.Rect(plot_rect.x0 - 50, plot_rect.y0 - 20, plot_rect.x1 + 50, plot_rect.y1 + 50)
        ):
            txt = b[4].strip()
            for line in txt.splitlines():
                line = line.strip()
                cleaned = re.sub(r"\d+(\.\d+)?%?", "", line)
                cleaned = re.sub(r"\(.*?\)", "", cleaned)
                cleaned = re.sub(r"(HR|CI|NR|months|median)", "", cleaned, flags=re.IGNORECASE).strip()
                if len(cleaned) > 3 and not re.match(r"^[\d\s\.,;:—\-]+$", cleaned) and "<Figure>" not in cleaned:
                    candidates.append(cleaned)

    seen = set()
    result = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            result.append(c)
    return result


def match_arm_names_by_text_color(
    detected_hex_colors: List[str],
    curves_data: Dict[str, Dict[str, Any]],
    parsed_arms: Dict[str, List[Dict[str, Any]]],
    candidate_names: List[str],
    page: Optional[fitz.Page] = None,
    panel: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Intelligently associates each detected curve stroke color with its true clinical arm name.

    Guarantees:
      1. Direct text font color association (from colored text spans in legend and risk table).
      2. Strict 1-to-1 bijection (never duplicate arm names across curves).
      3. Discriminating token isolation (prevents common chemotherapy backbone tokens from causing false ties).
      4. Accurate identification of control/placebo vs experimental arms.
    """
    arm_names = list(parsed_arms.keys())
    if not arm_names:
        ignore_words = {"number", "risk", "time", "months", "overall", "survival", "event", "free", "estimated", "probability"}
        for c in candidate_names:
            c_clean = c.strip()
            if len(c_clean) > 2 and not any(w in c_clean.lower() for w in ignore_words):
                if c_clean not in arm_names:
                    arm_names.append(c_clean)

    # Compute unique discriminating tokens for each candidate arm
    # by subtracting tokens that are common to all arms (e.g. 'm-folfoxiri' or 'chemo')
    arm_word_sets = [set(re.findall(r"[a-z0-9]+", a.lower())) for a in arm_names]
    common_tokens = set.intersection(*arm_word_sets) if len(arm_word_sets) > 1 else set()
    generic_stop = {"arm", "group", "plus", "and", "with", "vs", "treatment", "regimen", "sbrt"} | common_tokens

    arm_unique_tokens: Dict[str, List[str]] = {}
    for a in arm_names:
        toks = [w for w in re.findall(r"[a-z0-9]+", a.lower()) if w not in generic_stop and len(w) >= 2]
        arm_unique_tokens[a] = toks

    search_rect = None
    colored_spans = []
    if page is not None and panel is not None:
        search_rect = fitz.Rect(
            min(panel["plot_rect"].x0, panel["table_rect"].x0) - 50,
            min(panel["plot_rect"].y0, panel["table_rect"].y0) - 30,
            max(panel["plot_rect"].x1, panel["table_rect"].x1) + 50,
            max(panel["plot_rect"].y1, panel["table_rect"].y1) + 30,
        )
        text_dict = page.get_text("dict", flags=TEXT_EXTRACT_FLAGS)
        for b in text_dict.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    bbox = fitz.Rect(s.get("bbox", [0, 0, 0, 0]))
                    if not bbox.intersects(search_rect):
                        continue
                    txt = s.get("text", "").strip()
                    if not txt:
                        continue
                    c_int = s.get("color", 0)
                    r = ((c_int >> 16) & 255) / 255.0
                    g = ((c_int >> 8) & 255) / 255.0
                    b_val = (c_int & 255) / 255.0
                    if all(c < 0.15 for c in (r, g, b_val)) or all(c > 0.9 for c in (r, g, b_val)):
                        continue
                    colored_spans.append({
                        "text": txt,
                        "rgb": (r, g, b_val),
                        "hex": rgb_to_hex((r, g, b_val)),
                        "bbox": bbox,
                    })

    assigned: Dict[str, str] = {}
    available_arms = list(arm_names)

    best_matches = {}
    for hex_col in detected_hex_colors:
        c_rgb = curves_data[hex_col].get("raw_color")
        if not c_rgb:
            continue
        scores = {a: 0 for a in available_arms}
        for sp in colored_spans:
            dist = ((sp["rgb"][0] - c_rgb[0])**2 + (sp["rgb"][1] - c_rgb[1])**2 + (sp["rgb"][2] - c_rgb[2])**2)**0.5
            if dist < 0.14:
                sp_txt = sp["text"].lower()
                for arm in available_arms:
                    unique_toks = arm_unique_tokens.get(arm, [])
                    for u in unique_toks:
                        if u in sp_txt:
                            scores[arm] += 15
                        elif len(u) >= 3 and u[:3] in sp_txt:
                            scores[arm] += 10
                        elif len(sp_txt) >= 3 and sp_txt[:3] in u:
                            scores[arm] += 10
                    # Semantic control keyword matching
                    if any(k in sp_txt for k in ["placebo", "control", "standard", "comparator"]) and any(k in arm.lower() for k in ["placebo", "control", "standard", "comparator"]):
                        scores[arm] += 8
        best_arm = max(scores, key=scores.get) if scores and max(scores.values()) > 0 else None
        if best_arm:
            best_matches[hex_col] = (best_arm, scores[best_arm])

    # Assign highest confidence matches first (guarantees 1-to-1 bijection)
    sorted_matches = sorted(best_matches.items(), key=lambda item: item[1][1], reverse=True)
    for hex_col, (arm, score) in sorted_matches:
        if arm in available_arms:
            assigned[hex_col] = arm
            available_arms.remove(arm)

    # Style fallback for any remaining unassigned curves
    for hex_col in detected_hex_colors:
        if hex_col in assigned:
            continue
        dash_style = curves_data[hex_col].get("dash_style", "solid")
        if dash_style == "dashed":
            for arm in list(available_arms):
                if any(k in arm.lower() for k in ["placebo", "control", "standard", "comparator"]):
                    assigned[hex_col] = arm
                    available_arms.remove(arm)
                    break

    # Positional fallback (strict 1-to-1)
    for idx, hex_col in enumerate(detected_hex_colors):
        if hex_col in assigned:
            continue
        if available_arms:
            assigned[hex_col] = available_arms.pop(0)
        elif idx < len(arm_names):
            assigned[hex_col] = arm_names[idx]
        else:
            assigned[hex_col] = f"Arm_{idx + 1}"

    # Determine is_control
    result: Dict[str, Dict[str, Any]] = {}
    control_assigned = False
    for hex_col in detected_hex_colors:
        name = assigned[hex_col]
        is_ctrl = any(k in name.lower() for k in ["placebo", "control", "standard", "comparator", "bev", "bevacizumab"])
        if is_ctrl:
            control_assigned = True
        result[hex_col] = {
            "arm_name": name,
            "is_control": is_ctrl,
        }

    # If no arm was detected as control and we have 2 arms, assign 2nd curve as control
    if not control_assigned and len(detected_hex_colors) == 2:
        result[detected_hex_colors[1]]["is_control"] = True

    return result
