"""
Guyot et al. (2012) Numerical Kaplan-Meier Inversion Algorithm Engine.

Reconstructs pseudo-Individual Patient Data (IPD) from digitized coordinates (t, S(t))
and discrete numbers-at-risk tables (T_k, N_k).
Reference: Guyot et al. BMC Medical Research Methodology 2012, 12:9.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


def reconstruct_ipd_guyot(
    t_risk: List[float],
    n_risk: List[int],
    curve_pts: List[Tuple[float, float]],
    arm_name: str = "Arm_1",
    t_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Implements the Guyot et al. (2012) inversion algorithm to generate pseudo-IPD.
    Guarantees strict patient conservation (sum(events) + sum(censored) == N_1)
    and distributes events and censors within each risk interval [T_k, T_{k+1}]
    to match the digitized survival curve drops and number-at-risk progression.

    Args:
        t_risk: Discrete time points from the numbers-at-risk table, e.g. [0, 6, 12, ...].
        n_risk: Number of patients at risk at each time point, e.g. [227, 192, 130, ...].
        curve_pts: Calibrated survival coordinates [(t_0, S_0), (t_1, S_1), ...] with S in [0.0, 1.0].
        arm_name: Name of the treatment arm.
        t_max: Maximum follow-up time (defaults to max of t_risk and curve_pts).

    Returns:
        Pandas DataFrame with tidy columns: ['patient_id', 'arm', 'time', 'event'].
    """
    if len(t_risk) != len(n_risk):
        raise ValueError(f"Lengths of t_risk ({len(t_risk)}) and n_risk ({len(n_risk)}) must match.")

    if not curve_pts:
        raise ValueError("curve_pts cannot be empty.")

    # Clean and sort curve points
    curve_pts = sorted(curve_pts, key=lambda p: (p[0], -p[1]))
    pts_times = np.array([p[0] for p in curve_pts], dtype=float)
    pts_survs = np.array([p[1] for p in curve_pts], dtype=float)

    def get_surv_at(t_val: float) -> float:
        """
        Step-function survival probability lookup at given time point.
        Kaplan-Meier survival is a right-continuous step function that stays constant
        between event drops, NEVER linearly interpolated.
        """
        if t_val < pts_times[0]:
            return 1.0
        idx = np.searchsorted(pts_times, t_val, side="right") - 1
        if idx < 0:
            return 1.0
        return float(pts_survs[idx])

    # Determine maximum time boundary
    t_end_curve = float(pts_times[-1])
    if t_max is None:
        t_max = t_end_curve
    else:
        t_max = min(t_max, t_end_curve)

    events: List[Tuple[float, int]] = []
    censored: List[Tuple[float, int]] = []
    cur_s_recon: float = 1.0

    K = len(t_risk)

    # Process each risk interval [T_k, T_{k+1}]
    for k in range(K - 1):
        t1, t2 = float(t_risk[k]), float(t_risk[k + 1])
        n1, n2 = int(n_risk[k]), int(n_risk[k + 1])

        delta_n = n1 - n2
        if delta_n <= 0:
            continue

        s_t1 = get_surv_at(t1)
        s_t2 = get_surv_at(t2)

        # Points strictly inside (t1, t2] representing step drops in this interval
        pts_in_k = [p for p in curve_pts if t1 < p[0] <= t2]

        # Closed-loop target ratio tracking: targets the true absolute survival curve level S(t2)
        # from the current reconstructed survival state cur_s_recon. This prevents discretization
        # and patient-capping errors from compounding across subsequent intervals.
        if cur_s_recon > 1e-6:
            target_ratio = min(1.0, max(0.0, s_t2 / cur_s_recon))
        else:
            target_ratio = s_t2 / s_t1 if s_t1 > 0 else 0.0

        if not pts_in_k or (s_t1 - s_t2) < 1e-5:
            # Curve is flat across this interval: s_t1 == s_t2
            # Zero events occurred! All delta_n patients are censored.
            mid_t = min(t_max, (t1 + t2) / 2.0) if t1 < t_max else t_max
            censored.extend([(mid_t, 0)] * delta_n)
            continue

        m = len(pts_in_k)
        drop_times = [p[0] for p in pts_in_k]
        drop_amounts = []
        cur_s = s_t1
        for p in pts_in_k:
            amt = max(0.0, cur_s - p[1])
            drop_amounts.append(amt)
            cur_s = p[1]

        tot_amt = sum(drop_amounts)
        if tot_amt < 1e-5:
            mid_t = (t1 + t2) / 2.0
            censored.extend([(mid_t, 0)] * delta_n)
            continue

        w_d = [da / tot_amt for da in drop_amounts]

        best_diff = float("inf")
        best_d: List[int] = []
        best_c: List[int] = []
        best_km_ratio: float = 1.0

        # Optimization search over total events cand_D in [0, delta_n]
        # strictly conserving total patients: cand_D + cand_C == delta_n
        for cand_D in range(0, delta_n + 1):
            cand_C = delta_n - cand_D

            # Discretize events across sub-steps
            d_ints: List[int] = []
            rem_d = cand_D
            for j in range(m):
                cnt = rem_d if j == m - 1 else min(rem_d, int(round(cand_D * w_d[j])))
                d_ints.append(cnt)
                rem_d -= cnt

            # Discretize censors across sub-steps
            c_ints: List[int] = []
            rem_c = cand_C
            for j in range(m):
                cnt = rem_c if j == m - 1 else min(rem_c, int(round(cand_C / m)))
                c_ints.append(cnt)
                rem_c -= cnt

            # Simulate KM ratio: prod(1 - d_j / n_j)
            cur_n = n1
            km_ratio = 1.0
            feasible = True
            for j in range(m):
                dj, cj = d_ints[j], c_ints[j]
                if dj + cj > cur_n or cur_n <= 0:
                    feasible = False
                    break
                km_ratio *= (1.0 - dj / cur_n)
                cur_n -= (dj + cj)

            if not feasible:
                continue

            diff = abs(km_ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_d = d_ints
                best_c = c_ints
                best_km_ratio = km_ratio
                if diff < 1e-4:
                    break

        # Fallback if no feasible step distribution was found
        if not best_d:
            cand_D = int(round(delta_n * (1.0 - target_ratio)))
            cand_D = max(0, min(delta_n, cand_D))
            cand_C = delta_n - cand_D
            best_d = [0] * m
            best_c = [0] * m
            best_d[-1] = cand_D
            best_c[-1] = cand_C
            best_km_ratio = 1.0 - cand_D / n1 if n1 > 0 else 1.0

        cur_s_recon *= best_km_ratio

        # Append generated events and censors.
        # In the Guyot survival recursion, events occur at step_time on the active risk set,
        # and censoring occurs concurrently or after the event (between step_time and next_time),
        # NEVER before the event. Placing censors before the event artificially shrinks the risk set
        # and causes an exaggerated, plunging survival drop at late follow-up steps.
        for j in range(m):
            step_time = drop_times[j]
            next_time = drop_times[j + 1] if j + 1 < m else t2
            cens_time = (step_time + next_time) / 2.0
            if best_d[j] > 0:
                events.extend([(step_time, 1)] * best_d[j])
            if best_c[j] > 0:
                censored.extend([(cens_time, 0)] * best_c[j])

    # Final interval: patients remaining at t_risk[-1]
    n_final = int(n_risk[-1])
    t_last = float(t_risk[-1])

    if n_final > 0:
        pts_after = [p for p in curve_pts if p[0] > t_last]
        n_rem = n_final

        if pts_after:
            s_prev = get_surv_at(t_last)
            for t_p, s_p in pts_after:
                if n_rem <= 0:
                    break
                ratio = max(0.0, (s_prev - s_p) / s_prev) if s_prev > 0 else 0.0
                d_p = min(int(round(n_rem * ratio)), n_rem)
                if d_p > 0:
                    events.extend([(t_p, 1)] * d_p)
                    n_rem -= d_p
                s_prev = s_p

        # Any remaining patients at maximum observation time are censored
        if n_rem > 0:
            censored.extend([(t_max, 0)] * n_rem)

    # Assemble tidy DataFrame
    all_records = events + censored
    if not all_records:
        return pd.DataFrame(columns=["patient_id", "arm", "time", "event"])

    df = pd.DataFrame(all_records, columns=["time", "event"])
    # Sort by time, events before censors
    df = df.sort_values(by=["time", "event"], ascending=[True, False]).reset_index(drop=True)
    df["arm"] = arm_name
    df["patient_id"] = [f"{arm_name}_{i+1:04d}" for i in range(len(df))]

    return df[["patient_id", "arm", "time", "event"]]


def reconstruct_all_arms(arms_config: Dict[str, Dict[str, Any]], t_max: Optional[float] = None) -> pd.DataFrame:
    """
    Reconstructs pseudo-IPD across multiple treatment arms and combines into a single DataFrame.

    Args:
        arms_config: Dict mapping arm_name to dict with:
          - 't_risk': List of times
          - 'n_risk': List of risk counts
          - 'curve_pts': List of (t, S(t)) calibrated coordinates
        t_max: Global maximum follow-up time.

    Returns:
        Combined tidy DataFrame with all arms.
    """
    dfs = []
    for arm_name, cfg in arms_config.items():
        df_arm = reconstruct_ipd_guyot(
            t_risk=cfg["t_risk"],
            n_risk=cfg["n_risk"],
            curve_pts=cfg["curve_pts"],
            arm_name=arm_name,
            t_max=t_max,
        )
        dfs.append(df_arm)

    if not dfs:
        return pd.DataFrame(columns=["patient_id", "arm", "time", "event"])

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def compute_survival_summary(df_ipd: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes Kaplan-Meier survival metrics (median survival, 95% CI, event and censoring counts)
    for each arm in the reconstructed IPD.
    """
    summary: Dict[str, Any] = {}
    arms = df_ipd["arm"].unique()

    for arm in arms:
        df_arm = df_ipd[df_ipd["arm"] == arm]
        kmf = KaplanMeierFitter()
        kmf.fit(df_arm["time"], df_arm["event"], label=arm)

        med_surv = float(kmf.median_survival_time_)
        if np.isinf(med_surv):
            med_surv_str = "Not Reached (NR)"
        else:
            med_surv_str = f"{med_surv:.1f}"

        total_patients = len(df_arm)
        total_events = int(df_arm["event"].sum())
        total_censored = total_patients - total_events

        summary[arm] = {
            "total_n": total_patients,
            "events": total_events,
            "censored": total_censored,
            "event_rate": f"{(total_events / total_patients * 100):.1f}%" if total_patients > 0 else "0%",
            "median_survival": med_surv_str,
            "median_survival_val": med_surv,
            "kmf": kmf,
        }

    return summary


def compute_cox_hazard_ratio(
    df_ipd: pd.DataFrame, ref_arm: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fits a Cox Proportional Hazards regression model to estimate the Hazard Ratio (HR),
    95% Confidence Interval, and Wald/Log-rank p-values.
    """
    arms = list(df_ipd["arm"].unique())
    if len(arms) < 2:
        return {"error": "Cox regression requires at least two treatment arms."}

    if ref_arm is None or ref_arm not in arms:
        ref_arm = arms[1]

    exp_arm = [a for a in arms if a != ref_arm][0]

    df_model = df_ipd.copy()
    df_model["is_experimental"] = (df_model["arm"] == exp_arm).astype(int)

    cph = CoxPHFitter()
    try:
        cph.fit(df_model[["time", "event", "is_experimental"]], duration_col="time", event_col="event")

        hr = float(np.exp(cph.params_["is_experimental"]))
        ci_lower = float(np.exp(cph.confidence_intervals_.loc["is_experimental", "95% lower-bound"]))
        ci_upper = float(np.exp(cph.confidence_intervals_.loc["is_experimental", "95% upper-bound"]))
        p_val = float(cph.summary.loc["is_experimental", "p"])

        df_exp = df_ipd[df_ipd["arm"] == exp_arm]
        df_ref = df_ipd[df_ipd["arm"] == ref_arm]
        lr_res = logrank_test(df_exp["time"], df_ref["time"], df_exp["event"], df_ref["event"])

        return {
            "experimental_arm": exp_arm,
            "reference_arm": ref_arm,
            "hazard_ratio": hr,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_val,
            "logrank_p": float(lr_res.p_value),
            "hr_str": f"{hr:.2f} (95% CI: {ci_lower:.2f} to {ci_upper:.2f})",
            "p_str": f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001",
        }
    except Exception as e:
        return {"error": f"Cox model fitting failed: {str(e)}"}
