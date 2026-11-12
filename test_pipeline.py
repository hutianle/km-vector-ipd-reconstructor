"""
Comprehensive Test Pipeline for KM Vector IPD Reconstructor.

Tests:
1. Calibration transforms & inverted PDF coordinate systems.
2. Mock vector coordinates and Guyot reconstruction.
3. Real clinical PDF validation, panel detection, step drop extraction,
   and censoring tick mark filtering.
4. Cox Proportional Hazards and log-rank statistics.
5. Edge cases: empty curves, monotonic enforcement, zero risk sets.
"""

import os
import fitz
import numpy as np
import pandas as pd
from core.calibration import Calibration, detect_axis_lines
from core.extractor import (
    validate_vector_page,
    detect_panels_and_regions,
    extract_curve_coordinates,
    extract_risk_table,
    extract_candidate_arm_names,
    filter_censoring_ticks,
    match_arm_names_by_text_color,
)
from core.guyot_reconstructor import (
    reconstruct_ipd_guyot,
    reconstruct_all_arms,
    compute_survival_summary,
    compute_cox_hazard_ratio,
)


def test_calibration():
    print("--> Running test_calibration...")
    cal = Calibration(
        x_min_pdf=100.0,
        x_max_pdf=500.0,
        y_min_pdf=200.0,
        y_max_pdf=400.0,
        t_min=0.0,
        t_max=60.0,
        s_min=0.0,
        s_max=1.0,
    )

    t, s = cal.pdf_to_clinical(100.0, 200.0)
    assert abs(t - 0.0) < 1e-4 and abs(s - 1.0) < 1e-4, f"Top-left failed: {t}, {s}"

    t, s = cal.pdf_to_clinical(500.0, 400.0)
    assert abs(t - 60.0) < 1e-4 and abs(s - 0.0) < 1e-4, f"Bottom-right failed: {t}, {s}"

    t, s = cal.pdf_to_clinical(300.0, 300.0)
    assert abs(t - 30.0) < 1e-4 and abs(s - 0.5) < 1e-4, f"Midpoint failed: {t}, {s}"

    x, y = cal.clinical_to_pdf(30.0, 0.5)
    assert abs(x - 300.0) < 1e-4 and abs(y - 300.0) < 1e-4, f"Inverse failed: {x}, {y}"

    raw_pts = [(100, 200), (200, 250), (250, 240), (300, 300)]
    cal_pts = cal.calibrate_points(raw_pts, enforce_monotonic=True)
    for i in range(len(cal_pts) - 1):
        assert cal_pts[i][0] <= cal_pts[i + 1][0], "Times must be ascending"
        assert cal_pts[i][1] >= cal_pts[i + 1][1], "Survival must be non-increasing"

    print("    [PASS] Calibration tests passed.")


def test_guyot_mock():
    print("--> Running test_guyot_mock...")
    t_risk = [0.0, 12.0, 24.0, 36.0]
    n_risk_ctrl = [100, 70, 45, 20]
    n_risk_trt = [100, 85, 65, 40]

    times = np.linspace(0, 36, 37)
    curve_ctrl = [(t, float(np.exp(-0.04 * t))) for t in times]
    curve_trt = [(t, float(np.exp(-0.02 * t))) for t in times]

    df_ctrl = reconstruct_ipd_guyot(t_risk, n_risk_ctrl, curve_ctrl, arm_name="Control")
    df_trt = reconstruct_ipd_guyot(t_risk, n_risk_trt, curve_trt, arm_name="Treatment")

    assert len(df_ctrl) == 100, f"Expected 100 control patients, got {len(df_ctrl)}"
    assert len(df_trt) == 100, f"Expected 100 treatment patients, got {len(df_trt)}"
    assert set(df_ctrl.columns) == {"patient_id", "arm", "time", "event"}, "Tidy columns check"

    df_comb = pd.concat([df_trt, df_ctrl], ignore_index=True)
    summary = compute_survival_summary(df_comb)
    assert "Treatment" in summary and "Control" in summary

    cox_res = compute_cox_hazard_ratio(df_comb, ref_arm="Control")
    assert "hazard_ratio" in cox_res
    hr = cox_res["hazard_ratio"]
    print(f"    Mock trial Cox HR (Treatment vs Control): {hr:.2f} (95% CI: {cox_res['ci_lower']:.2f}-{cox_res['ci_upper']:.2f})")
    assert hr < 1.0, f"Treatment should have lower hazard than control, got HR={hr}"
    print("    [PASS] Guyot mock tests passed.")


def test_real_pdf_extraction():
    pdf_path = os.path.join(
        "KM example",
        "jabbour-et-al-2026-keynote-867-a-phase-iii-randomized-double-blind-study-of-stereotactic-body-radiotherapy-with.pdf",
    )
    if not os.path.exists(pdf_path):
        print(f"    [SKIP] Example PDF not found at {pdf_path}")
        return

    print("--> Running test_real_pdf_extraction on Page 5...")
    doc = fitz.open(pdf_path)

    # 1. Gatekeeper vector validation
    is_valid, msg = validate_vector_page(doc, 5)
    assert is_valid, f"Page 5 should pass vector validation: {msg}"
    print(f"    Vector Gatekeeper: {msg}")

    is_valid_p9, _ = validate_vector_page(doc, 9)
    assert not is_valid_p9, "Page 9 should fail vector validation (0 drawings)"

    page = doc[4]

    # 2. Panel detection
    panels = detect_panels_and_regions(page)
    assert len(panels) >= 2, f"Expected at least 2 panels on Page 5, got {len(panels)}"
    print(f"    Detected {len(panels)} sub-figure panels: {[p['label'] for p in panels]}")

    # 3. Test Panel A (EFS) curve extraction & Censoring Tick Filter
    panel_a = panels[0]
    curves_a = extract_curve_coordinates(page, panel_a["plot_rect"])
    assert len(curves_a) >= 2, f"Expected at least 2 curves in Panel A, got {len(curves_a)}"

    for col_hex, cinfo in curves_a.items():
        drops = cinfo["drop_count"]
        ticks = cinfo["tick_count"]
        print(f"    Panel A Curve {col_hex}: {drops} step drops captured | {ticks} censoring ticks filtered")
        assert drops > 50, f"Expected >50 true step drops, got {drops}"
        assert ticks > 50, f"Expected >50 censoring ticks identified, got {ticks}"

    # 4. Test Panel A Risk table parsing & Arm Name Matching
    risk_table_a = extract_risk_table(page, panel_a["table_rect"])
    assert len(risk_table_a["time_points"]) >= 5, "Expected time points in risk table"
    assert len(risk_table_a["arms"]) >= 2, f"Expected 2 arms in risk table, got {list(risk_table_a['arms'].keys())}"
    print(f"    Risk Table Times: {risk_table_a['time_points']}")
    for arm_name, entries in risk_table_a["arms"].items():
        print(f"    Risk Table Arm '{arm_name}': {[e['n_risk'] for e in entries]}")

    cands_a = extract_candidate_arm_names(page, panel_a["plot_rect"])
    matched_a = match_arm_names_by_text_color(
        detected_hex_colors=list(curves_a.keys()),
        curves_data=curves_a,
        parsed_arms=risk_table_a["arms"],
        candidate_names=cands_a,
        page=page,
        panel=panel_a,
    )
    # Verify no duplicate arm names across curves
    assigned_names_a = [v["arm_name"] for v in matched_a.values()]
    assert len(assigned_names_a) == len(set(assigned_names_a)), f"Duplicate arm names found in Panel A: {assigned_names_a}"
    for hex_col, info in matched_a.items():
        if "87" in hex_col or "7d" in hex_col or "7c" in hex_col:
            assert info["arm_name"] == "Pembro", f"Teal curve should match Pembro, got {info['arm_name']}"
            assert not info["is_control"], "Pembro should not be control"
        else:
            assert info["arm_name"] == "Placebo", f"Maroon curve should match Placebo, got {info['arm_name']}"
            assert info["is_control"], "Placebo should be control"
    print(f"    [PASS] Panel A arm name matching: {[(k, v['arm_name'], v['is_control']) for k, v in matched_a.items()]}")

    # Also test Panel B Arm Name Matching
    panel_b = panels[1]
    curves_b = extract_curve_coordinates(page, panel_b["plot_rect"])
    risk_table_b = extract_risk_table(page, panel_b["table_rect"])
    cands_b = extract_candidate_arm_names(page, panel_b["plot_rect"])
    matched_b = match_arm_names_by_text_color(
        detected_hex_colors=list(curves_b.keys()),
        curves_data=curves_b,
        parsed_arms=risk_table_b["arms"],
        candidate_names=cands_b,
        page=page,
        panel=panel_b,
    )
    assigned_names_b = [v["arm_name"] for v in matched_b.values()]
    assert len(assigned_names_b) == len(set(assigned_names_b)), f"Duplicate arm names found in Panel B: {assigned_names_b}"
    for hex_col, info in matched_b.items():
        if "87" in hex_col or "7d" in hex_col or "7c" in hex_col:
            assert info["arm_name"] == "Pembro", f"Teal curve in Panel B should match Pembro, got {info['arm_name']}"
            assert not info["is_control"], "Pembro in Panel B should not be control"
        else:
            assert info["arm_name"] == "Placebo", f"Maroon curve in Panel B should match Placebo, got {info['arm_name']}"
            assert info["is_control"], "Placebo in Panel B should be control"
    print(f"    [PASS] Panel B arm name matching: {[(k, v['arm_name'], v['is_control']) for k, v in matched_b.items()]}")

    # 5. Full End-to-End Reconstruction on Panel A
    cal_a = Calibration(
        x_min_pdf=panel_a["x_min"],
        x_max_pdf=panel_a["x_max"],
        y_min_pdf=panel_a["y_top"],
        y_max_pdf=panel_a["y_bot"],
        t_min=risk_table_a["time_points"][0],
        t_max=risk_table_a["time_points"][-1],
        s_min=0.0,
        s_max=1.0,
    )

    colors = list(curves_a.keys())
    arm_names = list(risk_table_a["arms"].keys())

    # Pembro is teal (#00877d), Placebo is maroon (#630036)
    teal_col = [c for c in colors if "87" in c or "7d" in c or "7c" in c][0]
    maroon_col = [c for c in colors if c != teal_col][0]

    curve_pembro_pts = cal_a.calibrate_points(curves_a[teal_col]["points"])
    curve_placebo_pts = cal_a.calibrate_points(curves_a[maroon_col]["points"])

    t_risk = risk_table_a["time_points"]
    n_pembro = [e["n_risk"] for e in risk_table_a["arms"]["Pembro"]]
    n_placebo = [e["n_risk"] for e in risk_table_a["arms"]["Placebo"]]

    df_pembro = reconstruct_ipd_guyot(t_risk, n_pembro, curve_pembro_pts, arm_name="Pembro")
    df_placebo = reconstruct_ipd_guyot(t_risk, n_placebo, curve_placebo_pts, arm_name="Placebo")

    assert len(df_pembro) == n_pembro[0], f"Pembro N mismatch: {len(df_pembro)} vs {n_pembro[0]}"
    assert len(df_placebo) == n_placebo[0], f"Placebo N mismatch: {len(df_placebo)} vs {n_placebo[0]}"

    df_ipd = pd.concat([df_pembro, df_placebo], ignore_index=True)
    summary = compute_survival_summary(df_ipd)
    print(f"    Reconstructed Median EFS Pembro: {summary['Pembro']['median_survival']} months (Published: 31.2)")
    print(f"    Reconstructed Median EFS Placebo: {summary['Placebo']['median_survival']} months (Published: 28.3)")

    cox_res = compute_cox_hazard_ratio(df_ipd, ref_arm="Placebo")
    print(f"    Reconstructed Cox HR: {cox_res['hazard_ratio']:.2f} (Published: 0.92 [95% CI, 0.69 to 1.24])")

    pembro_med = summary["Pembro"]["median_survival_val"]
    placebo_med = summary["Placebo"]["median_survival_val"]
    assert abs(pembro_med - 31.2) < 2.0, f"Pembro median deviated too much: {pembro_med}"
    assert abs(placebo_med - 28.3) < 2.0, f"Placebo median deviated too much: {placebo_med}"
    assert abs(cox_res["hazard_ratio"] - 0.92) < 0.15, f"Cox HR deviated too much: {cox_res['hazard_ratio']}"

    # Verify tail drop capturing: Pembro arm must have event drop around 41-42 months
    pembro_tail_events = df_pembro[(df_pembro["time"] >= 41.0) & (df_pembro["time"] <= 43.0) & (df_pembro["event"] == 1)]
    assert len(pembro_tail_events) > 0, "Expected event drop in Pembro at ~42 months"
    print(f"    Verified Pembro tail event drop captured at t = {pembro_tail_events['time'].iloc[0]:.2f} months")

    # Verify curve is not artificially flat between 37 and 47 months
    kmf_p = summary["Pembro"]["kmf"]
    s_37 = float(kmf_p.predict(37.0))
    s_43 = float(kmf_p.predict(43.0))
    assert s_37 > s_43, f"Pembro curve should drop between 37 and 43 months: S(37)={s_37:.4f}, S(43)={s_43:.4f}"
    print(f"    Verified Pembro drops between 37 and 43 months: S(37)={s_37:.4f} -> S(43)={s_43:.4f}")

    print("    [PASS] Real PDF extraction, tick filtering, and reconstruction tests passed!")


def test_deeper_trial_pdf_extraction():
    pdf_path = os.path.join(
        "KM example",
        "sunakawa-et-al-2026-long-term-survival-outcomes-of-modified-folfoxiri-plus-cetuximab-versus-bevacizumab-in-ras-braf.pdf",
    )
    if not os.path.exists(pdf_path):
        print(f"    [SKIP] DEEPER trial PDF not found at {pdf_path}")
        return

    print("\n--> Running test_deeper_trial_pdf_extraction on Sunakawa et al. (2026)...")
    doc = fitz.open(pdf_path)

    # 1. Gatekeeper vector validation
    # Page 3 (Figure 2 OS panel) has native vector drawings
    is_valid_p3, msg_p3 = validate_vector_page(doc, 3)
    assert is_valid_p3, f"Page 3 should pass vector validation: {msg_p3}"
    print(f"    Page 3 Vector Gatekeeper: {msg_p3}")

    # Page 11 (Figure A2 raster images) should be rejected
    is_valid_p11, msg_p11 = validate_vector_page(doc, 11)
    assert not is_valid_p11, f"Page 11 (raster plot) should fail vector validation: {msg_p11}"
    print(f"    Page 11 Raster Gatekeeper correctly rejected: {msg_p11[:65]}...")

    page = doc[2]  # 0-indexed Page 3

    # 2. Panel detection
    panels = detect_panels_and_regions(page)
    assert len(panels) >= 1, f"Expected detected OS panel on Page 3, got {len(panels)}"
    panel_os = panels[0]
    print(f"    Detected panel: {panel_os['label']}")

    # 3. Curve extraction & Censoring tick filtering
    curves = extract_curve_coordinates(page, panel_os["plot_rect"])
    assert len(curves) == 2, f"Expected 2 curves in OS panel, got {len(curves)}"
    assert "#9c408f" in curves and "#f28500" in curves, f"Expected purple and orange curves, got {list(curves.keys())}"

    for col_hex, cinfo in curves.items():
        drops = cinfo["drop_count"]
        ticks = cinfo["tick_count"]
        print(f"    Curve {col_hex}: {drops} step drops captured | {ticks} censoring ticks filtered")
        assert drops > 40, f"Expected >40 true step drops, got {drops}"

    # 4. Multi-frequency risk table parsing & Discriminative arm matching
    risk_table = extract_risk_table(page, panel_os["table_rect"], panel_os["plot_rect"])
    assert len(risk_table["time_points"]) == 17, f"Expected 17 time points (every 6 mo), got {len(risk_table['time_points'])}"
    assert risk_table["time_points"][-1] == 96.0, f"Expected final risk time 96.0, got {risk_table['time_points'][-1]}"
    assert risk_table["axis_t_max"] == 108.0, f"Expected axis t_max 108.0, got {risk_table['axis_t_max']}"

    cands = extract_candidate_arm_names(page, panel_os["plot_rect"])
    matched = match_arm_names_by_text_color(
        detected_hex_colors=list(curves.keys()),
        curves_data=curves,
        parsed_arms=risk_table["arms"],
        candidate_names=cands,
        page=page,
        panel=panel_os,
    )

    # Validate correct arm assignment: Orange (#f28500) = Cetuximab, Purple (#9c408f) = Bevacizumab
    orange_arm = matched["#f28500"]["arm_name"]
    purple_arm = matched["#9c408f"]["arm_name"]
    print(f"    Orange curve assigned arm: '{orange_arm}' (is_control={matched['#f28500']['is_control']})")
    print(f"    Purple curve assigned arm: '{purple_arm}' (is_control={matched['#9c408f']['is_control']})")
    assert "cet" in orange_arm.lower(), f"Orange should be Cetuximab, got {orange_arm}"
    assert "bev" in purple_arm.lower(), f"Purple should be Bevacizumab, got {purple_arm}"
    assert matched["#9c408f"]["is_control"], "Bevacizumab should be marked as control/standard arm"

    # 5. Full End-to-End Guyot Reconstruction
    cal = Calibration(
        x_min_pdf=panel_os["x_min"],
        x_max_pdf=panel_os["x_max"],
        y_min_pdf=panel_os["y_top"],
        y_max_pdf=panel_os["y_bot"],
        t_min=0.0,
        t_max=risk_table["axis_t_max"],
        s_min=0.0,
        s_max=1.0,
    )

    t_risk = risk_table["time_points"]
    pts_orange = cal.calibrate_points(curves["#f28500"]["points"])
    pts_purple = cal.calibrate_points(curves["#9c408f"]["points"])

    n_cet = [e["n_risk"] for e in risk_table["arms"][orange_arm]]
    n_bev = [e["n_risk"] for e in risk_table["arms"][purple_arm]]

    df_cet = reconstruct_ipd_guyot(t_risk, n_cet, pts_orange, arm_name="Cetuximab", t_max=cal.t_max)
    df_bev = reconstruct_ipd_guyot(t_risk, n_bev, pts_purple, arm_name="Bevacizumab", t_max=cal.t_max)

    # Validate patient conservation: N_cet == 86, N_bev == 92
    assert len(df_cet) == 86, f"Expected 86 Cetuximab patients, got {len(df_cet)}"
    assert len(df_bev) == 92, f"Expected 92 Bevacizumab patients, got {len(df_bev)}"

    df_combined = pd.concat([df_cet, df_bev], ignore_index=True)
    summary = compute_survival_summary(df_combined)
    cox = compute_cox_hazard_ratio(df_combined, ref_arm="Bevacizumab")

    med_cet = summary["Cetuximab"]["median_survival_val"]
    med_bev = summary["Bevacizumab"]["median_survival_val"]
    hr_val = cox["hazard_ratio"]

    print(f"    Reconstructed Median OS Cetuximab: {med_cet:.1f} months (Published: 50.2 [95% CI, 39.9 to 56.0])")
    print(f"    Reconstructed Median OS Bevacizumab: {med_bev:.1f} months (Published: 40.2 [95% CI, 33.5 to 48.8])")
    print(f"    Reconstructed Cox HR: {hr_val:.2f} (Published: 0.74 [95% CI, 0.53 to 1.05])")

    assert abs(med_cet - 50.2) < 2.0, f"Cetuximab median deviated too much: {med_cet}"
    assert abs(med_bev - 40.2) < 2.5, f"Bevacizumab median deviated too much: {med_bev}"
    assert abs(hr_val - 0.74) < 0.10, f"Cox HR deviated too much: {hr_val}"

    print("    [PASS] DEEPER trial extraction, multi-frequency table, and reconstruction passed!")


def test_daly_2_eu_lancet_extraction():
    print("\n--> Running test_daly_2_eu_lancet_extraction on Borchmann et al. (Lancet Oncol 2026)...")
    pdf_path = os.path.join(
        "KM example",
        "borchmann-et-al-2026-zamtocabtagene-autoleucel-daly-2-eu-lancet-oncology.pdf",
    )
    doc = fitz.open(pdf_path)

    # 1. Gatekeeper test on Page 8 (Figure 2)
    is_valid, msg = validate_vector_page(doc, 8)
    assert is_valid, f"Page 8 vector gatekeeper failed: {msg}"
    print(f"    Page 8 Vector Gatekeeper: {msg}")

    # 2. Panel detection
    page = doc[7]  # 0-indexed Page 8
    panels = detect_panels_and_regions(page)
    assert len(panels) == 4, f"Expected 4 panels, found {len(panels)}"
    panel_labels = [p["label"] for p in panels]
    print(f"    Detected {len(panels)} sub-figure panels: {panel_labels}")
    assert "EFS" in panels[0]["label"], f"Panel A label missing EFS: {panels[0]['label']}"
    assert "PFS" in panels[1]["label"], f"Panel B label missing PFS: {panels[1]['label']}"
    assert "OS" in panels[3]["label"], f"Panel D label missing OS: {panels[3]['label']}"

    # 3. Curve coordinate extraction for Panel A (EFS)
    panel_a = panels[0]
    curves = extract_curve_coordinates(page, panel_a["plot_rect"])
    assert len(curves) == 2, f"Expected 2 curves, got {len(curves)}"
    assert "#00549e" in curves and "#ed3024" in curves

    # 4. Risk table extraction for Panel A
    risk_table_a = extract_risk_table(page, panel_a["table_rect"], panel_a["plot_rect"])
    assert len(risk_table_a["time_points"]) == 11, f"Expected 11 time points, got {len(risk_table_a['time_points'])}"
    assert risk_table_a["time_points"] == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]
    assert "Zamto-cel" in risk_table_a["arms"]
    assert "R-GemOx" in risk_table_a["arms"]
    zamto_risk = [e["n_risk"] for e in risk_table_a["arms"]["Zamto-cel"]]
    gemox_risk = [e["n_risk"] for e in risk_table_a["arms"]["R-GemOx"]]
    assert zamto_risk == [82, 59, 41, 31, 21, 19, 12, 8, 2, 0, 0]
    assert gemox_risk == [78, 32, 14, 5, 5, 5, 2, 1, 0, 0, 0]

    # 5. Arm matching bijection
    matched = match_arm_names_by_text_color(
        list(curves.keys()),
        curves,
        risk_table_a["arms"],
        ["Zamto-cel", "R-GemOx"],
        page=page,
        panel=panel_a,
    )
    assert matched["#00549e"]["arm_name"] == "Zamto-cel" and not matched["#00549e"]["is_control"]
    assert matched["#ed3024"]["arm_name"] == "R-GemOx" and matched["#ed3024"]["is_control"]
    print("    [PASS] Arm name matching: #00549e -> Zamto-cel, #ed3024 -> R-GemOx")

    # 6. Guyot IPD Reconstruction
    cal = Calibration(
        x_min_pdf=panel_a["plot_rect"].x0,
        x_max_pdf=panel_a["plot_rect"].x1,
        y_min_pdf=panel_a["plot_rect"].y0,
        y_max_pdf=panel_a["plot_rect"].y1,
        t_min=0.0,
        t_max=30.0,
        s_min=0.0,
        s_max=1.0,
    )
    zamto_pts = cal.calibrate_points(curves["#00549e"]["points"], enforce_monotonic=True)
    gemox_pts = cal.calibrate_points(curves["#ed3024"]["points"], enforce_monotonic=True)

    df_zamto = reconstruct_ipd_guyot(risk_table_a["time_points"], zamto_risk, zamto_pts, arm_name="Zamto-cel", t_max=30.0)
    df_gemox = reconstruct_ipd_guyot(risk_table_a["time_points"], gemox_risk, gemox_pts, arm_name="R-GemOx", t_max=30.0)

    assert len(df_zamto) == 82, f"Zamto-cel N mismatch: {len(df_zamto)} vs 82"
    assert len(df_gemox) == 78, f"R-GemOx N mismatch: {len(df_gemox)} vs 78"

    df_ipd = pd.concat([df_zamto, df_gemox], ignore_index=True)
    summary = compute_survival_summary(df_ipd)
    cox = compute_cox_hazard_ratio(df_ipd, ref_arm="R-GemOx")

    med_zamto = summary["Zamto-cel"]["median_survival_val"]
    med_gemox = summary["R-GemOx"]["median_survival_val"]
    hr = cox["hazard_ratio"]

    print(f"    Reconstructed Median EFS Zamto-cel: {med_zamto:.2f} months (Published: 6.21 [95% CI 3.84–13.77])")
    print(f"    Reconstructed Median EFS R-GemOx: {med_gemox:.2f} months (Published: 2.53 [95% CI 1.97–3.35])")
    print(f"    Reconstructed Cox HR: {hr:.2f} (Published: 0.39 [95% CI 0.27–0.58])")

    assert abs(med_zamto - 6.21) < 0.20, f"Zamto-cel median deviated: {med_zamto}"
    assert abs(med_gemox - 2.53) < 0.20, f"R-GemOx median deviated: {med_gemox}"
    assert abs(hr - 0.39) < 0.05, f"Cox HR deviated: {hr}"
    print("    [PASS] Borchmann et al. DALY 2-EU trial reconstruction verified!")


if __name__ == "__main__":
    print("========================================")
    print("Starting Test Pipeline...")
    print("========================================")
    test_calibration()
    test_guyot_mock()
    test_real_pdf_extraction()
    test_deeper_trial_pdf_extraction()
    test_daly_2_eu_lancet_extraction()
    print("========================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("========================================")
