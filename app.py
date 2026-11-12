"""
KM Vector IPD Reconstructor - Streamlit Web Application

Reverse engineers published Kaplan-Meier survival curves from clinical trial PDFs
into pseudo-Individual Patient Data (IPD) using the Guyot et al. (2012) methodology.
Optimized to extract discrete vertical step drops and strictly isolate/ignore censoring tick marks.
"""

import io
import os
import re
import fitz
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.calibration import Calibration
from core.extractor import (
    validate_vector_page,
    detect_panels_and_regions,
    extract_curve_coordinates,
    extract_risk_table,
    extract_candidate_arm_names,
    match_arm_names_by_text_color,
)
from core.guyot_reconstructor import (
    reconstruct_ipd_guyot,
    reconstruct_all_arms,
    compute_survival_summary,
    compute_cox_hazard_ratio,
)

# Page configuration
st.set_page_config(
    page_title="KM Vector IPD Reconstructor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical styling and color swatches
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fb;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .swatch-box {
        display: inline-block;
        width: 22px;
        height: 22px;
        border-radius: 4px;
        vertical-align: middle;
        margin-right: 8px;
        border: 1px solid rgba(0,0,0,0.2);
    }
    .badge-vector {
        background-color: #e6fffa;
        color: #234e52;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #b2f5ea;
    }
    .badge-step {
        background-color: #ebf8ff;
        color: #2b6cb0;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-right: 4px;
        margin-top: 4px;
    }
    .badge-tick {
        background-color: #fffaf0;
        color: #9c4221;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-top: 4px;
        border: 1px solid #feebc8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "reconstruction_submitted" not in st.session_state:
    st.session_state.reconstruction_submitted = False
if "current_file_bytes" not in st.session_state:
    st.session_state.current_file_bytes = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = 5
if "selected_panel_id" not in st.session_state:
    st.session_state.selected_panel_id = "Panel_A"


def reset_reconstruction():
    st.session_state.reconstruction_submitted = False
    for k in list(st.session_state.keys()):
        if k.startswith("arm_name_") or k.startswith("is_ctrl_") or k in ("arm_form_data", "edited_risk_df"):
            del st.session_state[k]


# Application Header
col_header, col_badge = st.columns([4, 1])
with col_header:
    st.title("📈 KM Vector IPD Reconstructor")
    st.caption(
        "Direct vector-path step-drop extraction & mathematical IPD reconstruction "
        "via Guyot et al. (2012) for clinical trial publications (JCO, NEJM, Lancet)."
    )
with col_badge:
    st.markdown("<br><span class='badge-vector'>⚡ Native Step-Drop Engine</span>", unsafe_allow_html=True)

st.divider()

# Locate bundled example PDFs if available
example_pdf_1 = os.path.join(
    "KM example",
    "jabbour-et-al-2026-keynote-867-a-phase-iii-randomized-double-blind-study-of-stereotactic-body-radiotherapy-with.pdf",
)
example_pdf_2 = os.path.join(
    "KM example",
    "sunakawa-et-al-2026-long-term-survival-outcomes-of-modified-folfoxiri-plus-cetuximab-versus-bevacizumab-in-ras-braf.pdf",
)
example_pdf_3 = os.path.join(
    "KM example",
    "borchmann-et-al-2026-zamtocabtagene-autoleucel-daly-2-eu-lancet-oncology.pdf",
)
has_example_1 = os.path.exists(example_pdf_1)
has_example_2 = os.path.exists(example_pdf_2)
has_example_3 = os.path.exists(example_pdf_3)


# Sidebar: File Uploader & Document Configuration
with st.sidebar:
    st.header("📂 1. PDF Document Input")

    uploaded_file = st.file_uploader("Upload published trial PDF", type=["pdf"], key="pdf_uploader")

    st.markdown("**Bundled Clinical Trials:**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if has_example_1 and st.button("KEYNOTE-867\n(Pg 5)", use_container_width=True, key="sidebar_load_ex1"):
            with open(example_pdf_1, "rb") as f:
                st.session_state.current_file_bytes = f.read()
            st.session_state.current_file_name = os.path.basename(example_pdf_1)
            st.session_state.selected_page = 5
            reset_reconstruction()
            st.rerun()
    with col_ex2:
        if has_example_2 and st.button("DEEPER\n(Pg 3)", use_container_width=True, key="sidebar_load_ex2"):
            with open(example_pdf_2, "rb") as f:
                st.session_state.current_file_bytes = f.read()
            st.session_state.current_file_name = os.path.basename(example_pdf_2)
            st.session_state.selected_page = 3
            reset_reconstruction()
            st.rerun()
    with col_ex3:
        if has_example_3 and st.button("DALY 2-EU\n(Pg 8)", use_container_width=True, key="sidebar_load_ex3"):
            with open(example_pdf_3, "rb") as f:
                st.session_state.current_file_bytes = f.read()
            st.session_state.current_file_name = os.path.basename(example_pdf_3)
            st.session_state.selected_page = 8
            reset_reconstruction()
            st.rerun()

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        if (
            st.session_state.current_file_bytes is None
            or st.session_state.current_file_name != uploaded_file.name
        ):
            st.session_state.current_file_bytes = file_bytes
            st.session_state.current_file_name = uploaded_file.name
            reset_reconstruction()
            # Auto-detect first vector page in the uploaded document
            try:
                temp_doc = fitz.open(stream=file_bytes, filetype="pdf")
                first_vector_page = 1
                for p_idx in range(len(temp_doc)):
                    is_v, _ = validate_vector_page(temp_doc, p_idx + 1)
                    if is_v:
                        first_vector_page = p_idx + 1
                        break
                st.session_state.selected_page = first_vector_page
            except Exception:
                st.session_state.selected_page = 1
            st.rerun()

# Main page prompt if no PDF is currently loaded
if st.session_state.current_file_bytes is None:
    st.info("👋 Welcome! Please upload a clinical trial vector PDF in the sidebar, or load one of the published phase III trials below to explore:")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        if has_example_1 and st.button("🚀 Load KEYNOTE-867\n(Jabbour et al. 2026 - Page 5)", type="primary", use_container_width=True, key="main_load_ex1"):
            with open(example_pdf_1, "rb") as f:
                st.session_state.current_file_bytes = f.read()
            st.session_state.current_file_name = os.path.basename(example_pdf_1)
            st.session_state.selected_page = 5
            reset_reconstruction()
            st.rerun()
    with col_m2:
        if has_example_2 and st.button("🚀 Load DEEPER Trial\n(Sunakawa et al. 2026 - Page 3)", type="secondary", use_container_width=True, key="main_load_ex2"):
            with open(example_pdf_2, "rb") as f:
                st.session_state.current_file_bytes = f.read()
            st.session_state.current_file_name = os.path.basename(example_pdf_2)
            st.session_state.selected_page = 3
            reset_reconstruction()
            st.rerun()
    with col_m3:
        if has_example_3 and st.button("🚀 Load DALY 2-EU Trial\n(Borchmann et al. 2026 - Page 8)", type="secondary", use_container_width=True, key="main_load_ex3"):
            with open(example_pdf_3, "rb") as f:
                st.session_state.current_file_bytes = f.read()
            st.session_state.current_file_name = os.path.basename(example_pdf_3)
            st.session_state.selected_page = 8
            reset_reconstruction()
            st.rerun()
    st.stop()

# Document is loaded
try:
    doc = fitz.open(stream=st.session_state.current_file_bytes, filetype="pdf")
except Exception as e:
    st.error(f"Failed to read PDF document: {e}")
    st.stop()

total_pages = len(doc)

with st.sidebar:
    st.success(f"Loaded: `{st.session_state.current_file_name}`")
    page_num = st.number_input(
        f"Select Figure Page (1 to {total_pages})",
        min_value=1,
        max_value=total_pages,
        value=min(st.session_state.selected_page, total_pages),
        step=1,
        on_change=reset_reconstruction,
    )
    st.session_state.selected_page = page_num


# =========================================================================
# Step 2: Gatekeeper Check (Vector Validation)
# =========================================================================
page = doc[page_num - 1]
is_vector, validation_msg = validate_vector_page(doc, page_num)

if not is_vector:
    st.error(f"⛔ **Gatekeeper Halt: Rasterized Figure Detected**\n\n{validation_msg}")
    valid_pages = [p_i + 1 for p_i in range(len(doc)) if validate_vector_page(doc, p_i + 1)[0]]
    if valid_pages:
        st.info(f"💡 **Native vector KM curves were detected on page(s): {valid_pages}**. You can select one of these pages in the sidebar.")
    st.warning(
        "**Why this check is mandatory:**\n"
        "Rasterized bitmap images (PNG/JPEG scans or flattened plots) do not contain underlying vector drawing commands. "
        "The Guyot vector reconstruction algorithm requires native stroke paths to extract exact coordinate vertices without pixel blurring or raster distortion."
    )
    st.stop()
else:
    st.success(f"✅ {validation_msg}")

# Detect Panels / Sub-figures
panels = detect_panels_and_regions(page)
if not panels:
    st.error("No coordinate plot regions or axes detected on this page. Please select a page containing a KM figure.")
    st.stop()

# Sidebar: Sub-figure Selector
with st.sidebar:
    st.header("🎯 2. Sub-Figure Selection")
    panel_options = {p["panel_id"]: p["label"] for p in panels}
    panel_keys = list(panel_options.keys())

    selected_panel_id = st.selectbox(
        "Target Sub-Figure Panel",
        options=panel_keys,
        format_func=lambda k: panel_options[k],
        index=0 if st.session_state.selected_panel_id not in panel_keys else panel_keys.index(st.session_state.selected_panel_id),
        on_change=reset_reconstruction,
    )
    st.session_state.selected_panel_id = selected_panel_id

# Retrieve chosen panel
chosen_panel = next(p for p in panels if p["panel_id"] == selected_panel_id)
plot_rect = chosen_panel["plot_rect"]
table_rect = chosen_panel["table_rect"]

# Extract vector step drops and risk table (strictly filtering censoring ticks)
known_arm_names = []
for p in panels:
    p_info = extract_risk_table(page, p["table_rect"], p["plot_rect"])
    if p_info.get("arms"):
        known_arm_names = list(p_info["arms"].keys())
        break

raw_curves = extract_curve_coordinates(page, plot_rect)
risk_table_info = extract_risk_table(page, table_rect, plot_rect, default_arm_names=known_arm_names)
candidate_names = extract_candidate_arm_names(page, plot_rect)
if not candidate_names and known_arm_names:
    candidate_names = known_arm_names

if not raw_curves:
    st.warning("No distinct vector curves detected inside the selected plot rectangle. Check panel bounds or select another page.")
    st.stop()

parsed_time_points = risk_table_info.get("time_points", [])
parsed_arms = risk_table_info.get("arms", {})
axis_t_max = risk_table_info.get("axis_t_max", (parsed_time_points[-1] if parsed_time_points else 60.0))

if not parsed_time_points:
    parsed_time_points = [0.0, 12.0, 24.0, 36.0, 48.0, 60.0]


# =========================================================================
# Step 3: Interactive Arm Confirmation & Calibration Form
# =========================================================================
st.subheader("⚙️ 3. Treatment Arm Confirmation & Calibration")
st.write(
    "Verify the extracted stroke colors, confirm treatment arm assignments, "
    "and review the step-drop transition points (censoring ticks automatically ignored) before initiating reconstruction."
)

with st.form("arm_confirmation_form"):
    tab_arms, tab_calib, tab_risk = st.tabs(["🏷️ Treatment Arms", "📐 Axis Calibration", "📋 Numbers at Risk Table"])

    detected_hex_colors = list(raw_curves.keys())
    matched_arms_map = match_arm_names_by_text_color(
        detected_hex_colors=detected_hex_colors,
        curves_data=raw_curves,
        parsed_arms=parsed_arms,
        candidate_names=candidate_names,
        page=page,
        panel=chosen_panel,
    )
    arm_form_data = {}

    with tab_arms:
        st.write("Assign arm names and reference status to each detected stroke color:")
        cols_arms = st.columns(max(1, len(detected_hex_colors)))

        for i, hex_col in enumerate(detected_hex_colors):
            cinfo = raw_curves[hex_col]
            num_drops = cinfo.get("drop_count", len(cinfo.get("points", [])) // 2)
            num_ticks = cinfo.get("tick_count", 0)
            dash_style_val = cinfo.get("dash_style", "solid")
            span_x_val = cinfo.get("span_x", 0.0)

            with cols_arms[i]:
                st.markdown(
                    f"<div><span class='swatch-box' style='background-color:{hex_col};'></span>"
                    f"<strong>Color: {hex_col}</strong> ({dash_style_val})</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div><span class='badge-step'>📉 {num_drops} step drops</span> "
                    f"<span class='badge-tick'>✂️ {num_ticks} ticks ignored</span></div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Horizontal span: {span_x_val:.1f} pt")

                matched_arm_info = matched_arms_map.get(hex_col, {})
                default_name = matched_arm_info.get("arm_name", f"Arm_{i+1}")
                is_control_default = matched_arm_info.get("is_control", False)

                arm_name_val = st.text_input(
                    f"Arm Name for {hex_col}",
                    value=default_name,
                    key=f"arm_name_{selected_panel_id}_{hex_col}",
                )

                is_control = st.checkbox(
                    "Set as Reference / Control",
                    value=is_control_default,
                    key=f"is_ctrl_{selected_panel_id}_{hex_col}",
                )

                arm_form_data[hex_col] = {
                    "arm_name": arm_name_val,
                    "is_control": is_control,
                    "color_hex": hex_col,
                    "points": cinfo.get("points", []),
                    "dash_style": dash_style_val,
                    "drop_count": num_drops,
                    "tick_count": num_ticks,
                    "censoring_ticks": cinfo.get("censoring_ticks", []),
                }

    with tab_calib:
        st.write("Set clinical axis limits and fine-tune PDF bounding coordinates:")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            t_min_val = st.number_input("Time Min (t_min)", value=float(parsed_time_points[0]), step=1.0)
        with col_c2:
            t_max_val = st.number_input("Time Max (t_max)", value=float(axis_t_max), step=6.0)
        with col_c3:
            s_min_val = st.number_input("Survival Min S(t)", value=0.0, step=0.1, min_value=0.0, max_value=1.0)
        with col_c4:
            s_max_val = st.number_input("Survival Max S(t)", value=1.0, step=0.1, min_value=0.0, max_value=1.0)

        with st.expander("🛠️ Advanced PDF Space Coordinates (Auto-detected from vector axes)", expanded=False):
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                pdf_x_min = st.number_input("PDF X_min (Axis Origin)", value=float(chosen_panel["x_min"]), format="%.2f")
            with col_p2:
                pdf_x_max = st.number_input("PDF X_max (Axis End)", value=float(chosen_panel["x_max"]), format="%.2f")
            with col_p3:
                pdf_y_top = st.number_input("PDF Y_top (S = 100%)", value=float(chosen_panel["y_top"]), format="%.2f")
            with col_p4:
                pdf_y_bot = st.number_input("PDF Y_bot (S = 0%)", value=float(chosen_panel["y_bot"]), format="%.2f")

    with tab_risk:
        st.write("Numbers at risk table (auto-parsed below X-axis; edit values if necessary):")

        risk_table_dict = {"Time": [float(t) for t in parsed_time_points]}
        assigned_table_arms = set()
        for i, hex_col in enumerate(detected_hex_colors):
            arm_name_key = arm_form_data[hex_col]["arm_name"]
            counts = []
            if parsed_arms:
                matched_arm = None
                for a_name in parsed_arms:
                    if a_name not in assigned_table_arms:
                        n1, n2 = a_name.lower(), arm_name_key.lower()
                        toks1 = [w for w in re.findall(r"[a-z0-9]+", n1) if len(w) >= 3 and w not in ["arm", "folfoxiri", "plus"]]
                        toks2 = [w for w in re.findall(r"[a-z0-9]+", n2) if len(w) >= 3 and w not in ["arm", "folfoxiri", "plus"]]
                        if n1 in n2 or n2 in n1 or any(t1 in t2 or t2 in t1 or t1[:3] == t2[:3] for t1 in toks1 for t2 in toks2):
                            matched_arm = a_name
                            break
                if not matched_arm:
                    remaining_parsed = [a for a in parsed_arms if a not in assigned_table_arms]
                    if remaining_parsed:
                        matched_arm = remaining_parsed[0]

                if matched_arm:
                    assigned_table_arms.add(matched_arm)
                    counts = [e["n_risk"] for e in parsed_arms[matched_arm]]

            while len(counts) < len(parsed_time_points):
                counts.append(100 if len(counts) == 0 else max(0, counts[-1] - 15))

            col_label = arm_name_key
            if col_label in risk_table_dict:
                col_label = f"{col_label} ({hex_col})"
            risk_table_dict[col_label] = counts[: len(parsed_time_points)]

        df_risk_editor = pd.DataFrame(risk_table_dict)
        edited_risk_df = st.data_editor(df_risk_editor, num_rows="dynamic", use_container_width=True)

    submit_button = st.form_submit_button(
        "🚀 Submit Confirmation & Run Guyot Reconstruction",
        type="primary",
        use_container_width=True,
    )

    if submit_button:
        st.session_state.reconstruction_submitted = True
        st.session_state.calibration_obj = Calibration(
            x_min_pdf=pdf_x_min,
            x_max_pdf=pdf_x_max,
            y_min_pdf=pdf_y_top,
            y_max_pdf=pdf_y_bot,
            t_min=t_min_val,
            t_max=t_max_val,
            s_min=s_min_val,
            s_max=s_max_val,
        )
        st.session_state.arm_form_data = arm_form_data
        st.session_state.edited_risk_df = edited_risk_df


if not st.session_state.reconstruction_submitted:
    st.info("👆 Please confirm the treatment arm assignments and click **Submit Confirmation & Run Guyot Reconstruction**.")
    st.stop()


# =========================================================================
# Step 4: Guyot Reconstruction Execution
# =========================================================================
cal: Calibration = st.session_state.calibration_obj
arm_data = st.session_state.arm_form_data
df_risk_final: pd.DataFrame = st.session_state.edited_risk_df

st.divider()
st.subheader("📊 4. Reconstruction Results & Quality Assurance")

t_risk_final = [float(t) for t in df_risk_final["Time"]]
arms_reconstructed = {}
reference_arm_name = None

with st.spinner("Executing Guyot numerical inversion and solving risk set progressions..."):
    for hex_col, info in arm_data.items():
        arm_name = info["arm_name"]
        if info.get("is_control", False) and reference_arm_name is None:
            reference_arm_name = arm_name

        calibrated_curve = cal.calibrate_points(info.get("points", []), enforce_monotonic=True)

        matching_cols = [c for c in df_risk_final.columns if c != "Time" and (arm_name.lower() in c.lower() or c.lower() in arm_name.lower() or hex_col in c)]
        if matching_cols:
            n_risk_final = [int(v) for v in df_risk_final[matching_cols[0]]]
        else:
            col_idx = list(arm_data.keys()).index(hex_col) + 1
            n_risk_final = [int(v) for v in df_risk_final.iloc[:, min(col_idx, len(df_risk_final.columns) - 1)]]

        df_arm_ipd = reconstruct_ipd_guyot(
            t_risk=t_risk_final,
            n_risk=n_risk_final,
            curve_pts=calibrated_curve,
            arm_name=arm_name,
            t_max=cal.t_max,
        )

        arms_reconstructed[arm_name] = {
            "df": df_arm_ipd,
            "calibrated_pts": calibrated_curve,
            "hex": hex_col,
            "dash_style": info.get("dash_style", "solid"),
            "n_risk": n_risk_final,
            "drop_count": info.get("drop_count", len(calibrated_curve)),
            "tick_count": info.get("tick_count", 0),
            "censoring_ticks": info.get("censoring_ticks", []),
        }

    all_ipd_dfs = [v["df"] for v in arms_reconstructed.values()]
    df_ipd_combined = pd.concat(all_ipd_dfs, ignore_index=True)


# =========================================================================
# Metrics & Survival Summary
# =========================================================================
summary_metrics = compute_survival_summary(df_ipd_combined)
cox_results = compute_cox_hazard_ratio(df_ipd_combined, ref_arm=reference_arm_name)

st.markdown("#### Clinical Metrics Summary")
cols_metric = st.columns(len(arms_reconstructed) + (1 if "hazard_ratio" in cox_results else 0))

for idx, (arm_name, arm_sum) in enumerate(summary_metrics.items()):
    with cols_metric[idx]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div style='color:#718096; font-size:0.85rem; font-weight:600;'>ARM: {arm_name.upper()}</div>
                <div style='font-size:1.5rem; font-weight:700; color:#2d3748;'>{arm_sum['median_survival']} <span style='font-size:0.9rem; font-weight:400;'>months</span></div>
                <div style='font-size:0.85rem; color:#4a5568; margin-top:4px;'>
                    <strong>Total Patients (N):</strong> {arm_sum['total_n']}<br>
                    <strong>Events:</strong> {arm_sum['events']} ({arm_sum['event_rate']})<br>
                    <strong>Censored:</strong> {arm_sum['censored']}<br>
                    <strong>Captured Drops:</strong> {arms_reconstructed[arm_name].get('drop_count', 0)}<br>
                    <strong>Filtered Ticks:</strong> {arms_reconstructed[arm_name].get('tick_count', 0)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if "hazard_ratio" in cox_results:
    with cols_metric[-1]:
        st.markdown(
            f"""
            <div class='metric-card' style='border-left: 4px solid #3182ce;'>
                <div style='color:#3182ce; font-size:0.85rem; font-weight:600;'>COX PROPORTIONAL HAZARDS</div>
                <div style='font-size:1.5rem; font-weight:700; color:#2b6cb0;'>HR {cox_results['hazard_ratio']:.2f}</div>
                <div style='font-size:0.85rem; color:#4a5568; margin-top:4px;'>
                    <strong>95% CI:</strong> {cox_results['ci_lower']:.2f} – {cox_results['ci_upper']:.2f}<br>
                    <strong>Wald p-value:</strong> {cox_results['p_str']}<br>
                    <strong>Log-rank p:</strong> {cox_results['logrank_p']:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================================
# Plotly Overlay Comparison: Vector Steps vs Reconstructed KM
# =========================================================================
st.markdown("#### 📈 Interactive Overlay: Extracted Vector Drops vs Reconstructed Pseudo-IPD")
st.caption(
    "Dotted/faint step lines represent original extracted PDF vector step drops (censoring ticks excluded). "
    "Solid step lines represent the empirical Kaplan-Meier curve estimated from the reconstructed pseudo-IPD."
)

fig = go.Figure()

for arm_name, arm_obj in arms_reconstructed.items():
    cal_pts = arm_obj["calibrated_pts"]
    hex_color = arm_obj["hex"]
    df_arm = arm_obj["df"]

    # 1. Extracted vector path step line
    pts_x = [p[0] for p in cal_pts]
    pts_y = [p[1] * 100.0 for p in cal_pts]

    fig.add_trace(
        go.Scatter(
            x=pts_x,
            y=pts_y,
            mode="lines",
            line_shape="hv",
            line=dict(color=hex_color, width=2, dash="dot"),
            name=f"{arm_name} (Extracted Vector Steps)",
            hovertemplate=f"<b>{arm_name} (Vector)</b><br>Time: %{{x:.1f}} mo<br>Survival: %{{y:.1f}}%<extra></extra>",
        )
    )

    # 2. Reconstructed pseudo-IPD Kaplan-Meier curve
    kmf = summary_metrics[arm_name]["kmf"]
    km_times = kmf.survival_function_.index.tolist()
    km_surv = (kmf.survival_function_[arm_name] * 100.0).tolist()

    fig.add_trace(
        go.Scatter(
            x=km_times,
            y=km_surv,
            mode="lines",
            line_shape="hv",
            line=dict(color=hex_color, width=3.5),
            name=f"{arm_name} (Reconstructed IPD)",
            hovertemplate=f"<b>{arm_name} (Reconstructed)</b><br>Time: %{{x:.1f}} mo<br>Survival: %{{y:.1f}}%<extra></extra>",
        )
    )

    # 3. Reconstructed IPD Censoring marks
    cens_df = df_arm[df_arm["event"] == 0]
    if not cens_df.empty:
        cens_times = cens_df["time"].tolist()
        cens_survs = [float(np.interp(ct, km_times, km_surv)) for ct in cens_times]

        fig.add_trace(
            go.Scatter(
                x=cens_times,
                y=cens_survs,
                mode="markers",
                marker=dict(symbol="cross", size=6, color=hex_color),
                name=f"{arm_name} (Reconstructed Censored)",
                showlegend=False,
                hovertemplate=f"<b>{arm_name} (Censored)</b><br>Time: %{{x:.1f}} mo<br>Survival: %{{y:.1f}}%<extra></extra>",
            )
        )

    # 4. Filtered Original Vector Ticks (Toggleable trace)
    raw_ticks = arm_obj.get("censoring_ticks", [])
    if raw_ticks:
        tick_times = []
        tick_survs = []
        for tk in raw_ticks:
            t_clin, _ = cal.pdf_to_clinical(tk["x"], tk["y_bot"])
            _, s_clin = cal.pdf_to_clinical(tk["x"], tk["y_bot"])
            tick_times.append(t_clin)
            tick_survs.append(s_clin * 100.0)

        fig.add_trace(
            go.Scatter(
                x=tick_times,
                y=tick_survs,
                mode="markers",
                marker=dict(symbol="line-ns-open", size=8, color=hex_color, line=dict(width=1.5)),
                name=f"{arm_name} (Original Filtered Ticks: {len(raw_ticks)})",
                visible="legendonly",
                hovertemplate=f"<b>{arm_name} (Ignored Tick)</b><br>Time: %{{x:.1f}} mo<br>Survival: %{{y:.1f}}%<extra></extra>",
            )
        )

fig.update_layout(
    xaxis=dict(
        title="Time (months)",
        range=[cal.t_min, cal.t_max * 1.02],
        gridcolor="#edf2f7",
        zeroline=False,
    ),
    yaxis=dict(
        title="Survival Probability (%)",
        range=[-2, 105],
        gridcolor="#edf2f7",
        zeroline=False,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    plot_bgcolor="white",
    height=550,
    margin=dict(l=50, r=30, t=50, b=50),
    hovermode="closest",
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Reconstructed IPD Table & Export
# =========================================================================
st.markdown("#### 💾 Reconstructed Pseudo-IPD Dataset")

col_tbl_head, col_dl = st.columns([3, 1])
with col_tbl_head:
    st.caption(
        f"Generated {len(df_ipd_combined)} tidy individual patient records. "
        "Columns: `patient_id`, `arm`, `time` (event or censoring time), `event` (1 = event, 0 = censored)."
    )
with col_dl:
    csv_bytes = df_ipd_combined.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Pseudo-IPD CSV",
        data=csv_bytes,
        file_name="pseudo_ipd_reconstructed.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

arm_filter = st.multiselect("Filter by Arm", options=list(arms_reconstructed.keys()), default=list(arms_reconstructed.keys()))
df_filtered = df_ipd_combined[df_ipd_combined["arm"].isin(arm_filter)]

st.dataframe(
    df_filtered,
    use_container_width=True,
    height=300,
    column_config={
        "patient_id": st.column_config.TextColumn("Patient ID"),
        "arm": st.column_config.TextColumn("Treatment Arm"),
        "time": st.column_config.NumberColumn("Event / Censoring Time (mo)", format="%.3f"),
        "event": st.column_config.NumberColumn("Event Status (1=Event, 0=Censored)"),
    },
)

# QA Diagnostics Expander
with st.expander("🔍 Guyot Quality Assurance Diagnostics & Risk Set Progression", expanded=False):
    st.write("Interval-by-interval patient accounting and risk reduction progression:")
    for arm_name, arm_obj in arms_reconstructed.items():
        st.markdown(f"**Treatment Arm: {arm_name}**")
        df_arm = arm_obj["df"]
        n_risk_arr = arm_obj["n_risk"]
        diag_rows = []

        for k in range(len(t_risk_final) - 1):
            t1, t2 = t_risk_final[k], t_risk_final[k + 1]
            n_start, n_end = n_risk_arr[k], n_risk_arr[k + 1]
            sub = df_arm[(df_arm["time"] >= t1) & (df_arm["time"] < t2)]
            ev_k = sub["event"].sum()
            cen_k = (sub["event"] == 0).sum()
            diag_rows.append({
                "Interval": f"[{t1:.0f}, {t2:.0f})",
                "N_start": n_start,
                "N_end": n_end,
                "Delta N": n_start - n_end,
                "Reconstructed Events": ev_k,
                "Reconstructed Censored": cen_k,
                "Sum (Events + Censors)": ev_k + cen_k,
            })
        st.table(pd.DataFrame(diag_rows))
