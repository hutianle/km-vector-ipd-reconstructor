"""
KM Vector IPD Reconstructor Core Modules.
"""

from .calibration import Calibration, detect_axis_lines
from .extractor import (
    rgb_to_hex,
    validate_vector_page,
    detect_panels_and_regions,
    extract_curve_coordinates,
    extract_risk_table,
    extract_candidate_arm_names,
)
from .guyot_reconstructor import (
    reconstruct_ipd_guyot,
    reconstruct_all_arms,
    compute_survival_summary,
    compute_cox_hazard_ratio,
)

__all__ = [
    "Calibration",
    "detect_axis_lines",
    "rgb_to_hex",
    "validate_vector_page",
    "detect_panels_and_regions",
    "extract_curve_coordinates",
    "extract_risk_table",
    "extract_candidate_arm_names",
    "reconstruct_ipd_guyot",
    "reconstruct_all_arms",
    "compute_survival_summary",
    "compute_cox_hazard_ratio",
]
