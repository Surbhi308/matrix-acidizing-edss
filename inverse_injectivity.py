"""
=======================================================================
  Inverse Injectivity Diagnostic Plot for Matrix Acidizing
  SPE-28548-PA  |  Hill & Zhu (1996)
=======================================================================

KEY EQUATIONS  (SPE-28548, oilfield units)
------------------------------------------
Eq. 2  : p_wf = p_ti + ΔpPE(t) − ΔpF(t)
            ΔpPE = variable-density hydrostatic  =  Σ_i 0.052·ρ_i·Δh_i
            ΔpF  = tubing friction (Darcy-Weisbach + Churchill ff, per segment)

Eq. 7  : m′ = 162.6·B·μ/(k·h) × 1440     [psi/(bbl/min)/log-cycle]
              (×1440: standard derivation uses q in bbl/day)

Eq. 8  : b′ = m′·[log10(k/φμct·rw²) − 3.2275 + 0.86859·s]

Eq. 9  : Δtsup = Σ_{j=1}^{N} (qj−q_{j-1})/qN · log10(t_N−t_{j-1})

Eq. 6  : Δp/qN  =  m′·Δtsup + b′     (constant skin s)

VARIABLE-DENSITY FLUID COLUMN MODEL
-------------------------------------
At time t the cumulative volume pumped is V(t) = ∫q dt.
The tubing (depth 0 → H) contains the slice of the pump schedule
spanning cumulative volumes  [V(t) − V_tub, V(t)]  (clamped to ≥ 0).
Each stage that overlaps this window contributes a depth segment with
its own density ρ_i and viscosity μ_i, giving an exact column.

Hydrostatic:   ΔpPE = Σ_i 0.052·ρ_i[ppg]·Δh_i[ft]     →  psi
Friction:      ΔpF  = Σ_i (Darcy-Weisbach with Churchill ff per segment)

EXCEL WORKBOOK LAYOUT
---------------------
Sheet "field data":
    time(hr)  |  qi(bbl/min)  |  pwf(psi)   ← use if BHP is measured directly
    time(hr)  |  qi(bbl/min)  |  pti(psi)   ← use if only surface pressure known

Sheet "Reservoir_Data"  (Parameter | Value):
    k(md), h(ft), phi, mu(cp), B, ct(psi-1), rw(ft), pi(psi)
    H(ft)    — well TVD to mid-perfs        [only needed when using pti]
    d_in(in) — tubing inner diameter        [only needed for friction]

Sheet "fluid_schedule"  (only needed when computing pwf from pti):
    Stage | fluid_name | volume(bbl) | density(ppg) | viscosity(cp)
    List stages in PUMP ORDER (first row = first fluid pumped).
    Volumes should be surface volumes.

USAGE
-----
    python inverse_injectivity.py                          # Field_Example_1.xlsx
    python inverse_injectivity.py my_well.xlsx
    python inverse_injectivity.py my_well.xlsx --skin 0,10,20,40
    python inverse_injectivity.py my_well.xlsx --no-friction
    python inverse_injectivity.py my_well.xlsx --out result.png   # also save PNG
"""

import sys, os, argparse, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
#  1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_excel(filepath: str):
    xl = pd.ExcelFile(filepath)

    # ── field data ─────────────────────────────────────────────────────
    fd = xl.parse("field data")
    fd.columns = [str(c).strip() for c in fd.columns]
    for need in ("time(hr)", "qi(bbl/min)"):
        if need not in fd.columns:
            raise ValueError(f"'field data' sheet missing column: {need}")
    has_pwf = "pwf(psi)" in fd.columns
    has_pti = "pti(psi)" in fd.columns
    if not has_pwf and not has_pti:
        raise ValueError("'field data' needs either 'pwf(psi)' or 'pti(psi)'.")
    drop = {"time(hr)", "qi(bbl/min)",
            "pwf(psi)" if has_pwf else "pti(psi)"}
    fd = fd.dropna(subset=list(drop)).reset_index(drop=True)

    # ── reservoir / well data ──────────────────────────────────────────
    rd = xl.parse("Reservoir_Data")
    rd.columns = [str(c).strip() for c in rd.columns]
    res = {}
    for _, row in rd.iterrows():
        try:
            res[str(row["Parameter"]).strip()] = float(row["Value"])
        except Exception:
            pass
    for need in ("k","h","phi","mu","B","ct","rw","pi"):
        if need not in res:
            raise ValueError(f"Reservoir_Data missing: {need}")

    # ── fluid schedule (optional) ──────────────────────────────────────
    fluid_sched = None
    if "fluid_schedule" in xl.sheet_names:
        fs = xl.parse("fluid_schedule")
        fs.columns = [str(c).strip() for c in fs.columns]
        fluid_sched = fs

    return fd, res, has_pwf, has_pti, fluid_sched


# ═══════════════════════════════════════════════════════════════════════
#  2.  VARIABLE-DENSITY FLUID COLUMN  (corrected piston-displacement model)
# ═══════════════════════════════════════════════════════════════════════

def tubing_volume_bbl(d_in: float, H: float) -> float:
    """Tubing capacity [bbl] for inner diameter d_in [in] and depth H [ft]."""
    r_ft = (d_in / 2.0) / 12.0
    return math.pi * r_ft**2 * H / 5.61458


def build_fluid_column(cum_vol_bbl: float,
                        fluid_stages: list,
                        tubing_vol_bbl: float,
                        H: float) -> list:
    """
    Determine which fluid slugs occupy the tubing at a given instant.

    Physical model
    --------------
    Fluids are pumped in order from the surface, displacing each other
    downward like pistons (no mixing).  After pumping V bbl total, the
    tubing contains the slice of the pump schedule that spans cumulative
    volumes  [ max(0, V − V_tub),  V ].

    The BOTTOM of the tubing (perf depth H) always contains the
    earliest-pumped fluid that has not yet reached the formation.
    The TOP (surface) always contains the most recently pumped fluid.

    Parameters
    ----------
    cum_vol_bbl   : total volume pumped so far [bbl]
    fluid_stages  : list of dicts {name, vol_bbl, density_ppg, viscosity_cp}
                    in pump order (index 0 = first fluid pumped)
    tubing_vol_bbl: total tubing capacity [bbl]
    H             : well TVD [ft]

    Returns
    -------
    column : list of dicts ordered from BOTTOM (deepest) to TOP,
             each with keys: name, density_ppg, viscosity_cp,
                             depth_top_ft, depth_bot_ft
    """
    ft_per_bbl = H / tubing_vol_bbl

    # Volume window currently occupying the tubing
    v_bot = max(0.0, cum_vol_bbl - tubing_vol_bbl)   # earliest vol in tubing
    v_top = cum_vol_bbl                               # latest vol in tubing
    if v_top <= 0.0:
        return []    # nothing pumped yet

    # Walk through stages (pump order = surface-to-bottom inside tubing as time→∞)
    v_stage_start = 0.0
    segments = []
    for stg in fluid_stages:
        v_stage_end = v_stage_start + stg["vol_bbl"]

        # Overlap of this stage with the tubing window [v_bot, v_top]
        ov_lo = max(v_stage_start, v_bot)
        ov_hi = min(v_stage_end,   v_top)

        if ov_hi > ov_lo:
            # Convert cumulative volumes to depths.
            # v_bot ↔ depth H (bottom of tubing / perforations)
            # v_top ↔ depth 0 (surface)
            depth_bot = H - (ov_lo - v_bot) * ft_per_bbl   # deeper
            depth_top = H - (ov_hi - v_bot) * ft_per_bbl   # shallower

            segments.append({
                "name"        : stg["name"],
                "density_ppg" : stg["density_ppg"],
                "viscosity_cp": stg["viscosity_cp"],
                "depth_top_ft": max(0.0, depth_top),
                "depth_bot_ft": min(H,   depth_bot),
            })

        v_stage_start = v_stage_end

    return segments          # already bottom-to-top (earliest stage = deepest)


# ═══════════════════════════════════════════════════════════════════════
#  3.  HYDROSTATIC PRESSURE FROM COLUMN
# ═══════════════════════════════════════════════════════════════════════

def hydrostatic_dp(column: list, H: float) -> float:
    """
    ΔpPE = Σ_i  0.052 · ρ_i [ppg] · Δh_i [ft]    [psi]

    If the column doesn't fill the whole tubing (early time, few bbl pumped)
    the unfilled section above is treated as the first fluid (or zero).

    0.052 is the oilfield hydrostatic gradient constant:
        psi = ppg × ft × 0.052
    """
    if not column:
        return 0.0

    total_fluid_height = sum(seg["depth_bot_ft"] - seg["depth_top_ft"]
                              for seg in column)
    unfilled_height = H - total_fluid_height  # above the topmost fluid

    dp = 0.0
    # unfilled section: treat as the TOPMOST fluid density
    # (fluid is being injected continuously so there is always fluid here)
    if unfilled_height > 0 and column:
        top_seg_rho = column[-1]["density_ppg"]   # topmost segment
        dp += 0.052 * top_seg_rho * unfilled_height

    for seg in column:
        dh = seg["depth_bot_ft"] - seg["depth_top_ft"]
        dp += 0.052 * seg["density_ppg"] * dh

    return dp


# ═══════════════════════════════════════════════════════════════════════
#  4.  TUBING FRICTION  (Darcy-Weisbach + Churchill 1977)
# ═══════════════════════════════════════════════════════════════════════

def churchill_ff(Re: float, eps_over_d: float = 0.0) -> float:
    """
    Churchill (1977) Fanning friction factor – single equation valid for all
    Reynolds numbers and any relative roughness.
    """
    if Re <= 0:
        return 0.0
    if Re < 2100:
        return 16.0 / Re
    A = (-2.457 * math.log((7.0/Re)**0.9 + 0.27*eps_over_d))**16
    B = (37530.0 / Re)**16
    return 2.0 * ((8.0/Re)**12 + (A + B)**(-1.5))**(1.0/12.0)


def friction_dp_segment(q_bbl_min: float,
                         rho_ppg: float, mu_cp: float,
                         d_in: float, L_ft: float,
                         roughness: float = 0.0) -> float:
    """
    Frictional pressure drop [psi] for one fluid segment flowing through a
    pipe of inner diameter d_in [in] and length L_ft [ft].
    """
    if d_in <= 0 or L_ft <= 0 or q_bbl_min <= 0:
        return 0.0
    rho_lbft3 = rho_ppg  * 7.4805            # ppg → lb/ft³
    q_ft3s    = q_bbl_min * 5.61458 / 60.0   # bbl/min → ft³/s
    d_ft      = d_in / 12.0
    area      = math.pi * d_ft**2 / 4.0
    v_fps     = q_ft3s / area
    mu_lbfts  = mu_cp * 6.7197e-4             # cp → lb/(ft·s)
    Re        = rho_lbft3 * v_fps * d_ft / mu_lbfts
    f         = churchill_ff(Re, roughness)
    dp_lbft2  = 4.0 * f * (L_ft/d_ft) * rho_lbft3 * v_fps**2 / 2.0
    return dp_lbft2 / 144.0                   # lbf/ft² → psi


def total_friction_dp(q_bbl_min: float, column: list,
                       d_in: float, H: float) -> float:
    """
    Total tubing friction [psi] summed over all fluid segments.
    Each segment uses its own density and viscosity.
    """
    if d_in <= 0 or not column:
        return 0.0
    dp = 0.0
    for seg in column:
        L = seg["depth_bot_ft"] - seg["depth_top_ft"]
        dp += friction_dp_segment(
            q_bbl_min,
            seg["density_ppg"],
            seg["viscosity_cp"],
            d_in, L
        )
    return dp


# ═══════════════════════════════════════════════════════════════════════
#  5.  PWF FROM PTI  (Eq. 2)
# ═══════════════════════════════════════════════════════════════════════

def parse_fluid_schedule(fluid_sched_df, res: dict) -> list:
    """
    Convert the fluid_schedule DataFrame into a list of stage dicts.
    Falls back to a single-fluid column if no schedule is provided.
    """
    if fluid_sched_df is None:
        rho = float(res.get("rho", 8.33))
        mu  = float(res.get("mu",  1.0))
        print("  No fluid_schedule sheet — single fluid assumed: "
              f"ρ={rho} ppg, μ={mu} cp.")
        return [{"name": "fluid", "vol_bbl": 1e9,
                 "density_ppg": rho, "viscosity_cp": mu}]

    cols = {c.strip().lower(): c for c in fluid_sched_df.columns}
    stages = []
    for _, row in fluid_sched_df.iterrows():
        def get(keys, default=0.0):
            for k in keys:
                if k in cols:
                    return float(row[cols[k]])
            return default
        name = str(row.get(list(cols.values())[
            list(cols.keys()).index(
                next(k for k in ("fluid_name","fluid","name") if k in cols)
            )], "fluid"))
        stages.append({
            "name"        : name,
            "vol_bbl"     : get(["volume(bbl)","volume_bbl","vol(bbl)","vol_bbl"]),
            "density_ppg" : get(["density(ppg)","density_ppg","rho(ppg)","rho_ppg"], 8.33),
            "viscosity_cp": get(["viscosity(cp)","viscosity_cp","mu(cp)","mu_cp"],   1.0),
        })
    return stages


def calc_pwf_array(times: np.ndarray, rates: np.ndarray, pti_arr: np.ndarray,
                   res: dict, fluid_sched_df,
                   use_friction: bool = True) -> tuple:
    """
    Compute p_wf at each time step.

        p_wf(t) = p_ti(t)  +  ΔpPE(t)  −  ΔpF(t)       [Eq. 2]

    ΔpPE and ΔpF are both computed from the variable-density fluid column
    that occupies the tubing at time t, derived from cumulative pumped volume.

    Returns
    -------
    pwf_arr       : np.ndarray
    col_snapshots : list of column lists (one per time step, for reporting)
    dp_PE_arr     : np.ndarray  [psi]
    dp_F_arr      : np.ndarray  [psi]
    """
    H    = float(res.get("H",    0.0))
    d_in = float(res.get("d_in", 0.0))

    if H <= 0:
        print("  WARNING: H not in Reservoir_Data — p_wf = p_ti (no correction).")
        n = len(times)
        return pti_arr.copy(), [], np.zeros(n), np.zeros(n)

    fluid_stages  = parse_fluid_schedule(fluid_sched_df, res)
    tubing_vol    = tubing_volume_bbl(d_in, H) if d_in > 0 else 1e9

    print(f"\n  Tubing capacity = {tubing_vol:.1f} bbl  "
          f"(d_in={d_in} in, H={H} ft)")
    print(f"  Fluid stages (pump order):")
    for s in fluid_stages:
        print(f"    {s['name']:30s}  {s['vol_bbl']:7.1f} bbl  "
              f"ρ={s['density_ppg']:.3f} ppg  μ={s['viscosity_cp']:.2f} cp")

    # Cumulative volume pumped at each time step (trapezoidal integration)
    cum_vol = np.zeros(len(times))
    for i in range(1, len(times)):
        dt_hr = times[i] - times[i-1]
        q_avg = (rates[i] + rates[i-1]) / 2.0
        cum_vol[i] = cum_vol[i-1] + q_avg * dt_hr * 60.0   # bbl/min × hr × 60 = bbl

    pwf_arr       = np.empty(len(times))
    col_snapshots = []
    dp_PE_arr     = np.empty(len(times))
    dp_F_arr      = np.empty(len(times))

    for i in range(len(times)):
        col = build_fluid_column(cum_vol[i], fluid_stages, tubing_vol, H)
        col_snapshots.append(col)

        dp_PE = hydrostatic_dp(col, H)
        dp_F  = (total_friction_dp(rates[i], col, d_in, H)
                 if use_friction else 0.0)

        pwf_arr[i]   = pti_arr[i] + dp_PE - dp_F
        dp_PE_arr[i] = dp_PE
        dp_F_arr[i]  = dp_F

    return pwf_arr, col_snapshots, dp_PE_arr, dp_F_arr


# ═══════════════════════════════════════════════════════════════════════
#  6.  SPE-28548 TRANSIENT EQUATIONS
# ═══════════════════════════════════════════════════════════════════════

def slope_m(B, mu, k, h):
    """Eq. 7 (unit-corrected for q in bbl/min): m′ = 162.6·B·μ/(k·h)·1440"""
    return 162.6 * B * mu / (k * h) * 1440.0


def log_karg(k, phi, mu, ct, rw):
    return np.log10(k / (phi * mu * ct * rw**2))


def intercept_b(m_val, lkarg, s):
    """Eq. 8: b′ = m′·[log10(k/φμct·rw²) − 3.2275 + 0.86859·s]"""
    return m_val * (lkarg - 3.2275 + 0.86859 * s)


def superposition_time(times, rates, idx):
    """Eq. 9: Δtsup = Σ (qj−q_{j-1})/qN · log10(t_N−t_{j-1})"""
    t_N, q_N = times[idx], rates[idx]
    if q_N <= 0:
        return np.nan
    result = 0.0
    for j in range(idx + 1):
        dq    = rates[j] - (rates[j-1] if j > 0 else 0.0)
        t_jm1 = times[j-1] if j > 0 else 0.0
        dt    = t_N - t_jm1
        if dt <= 0:
            continue
        result += dq / q_N * np.log10(dt)
    return result


def compute_arrays(times, rates, pwf, pi):
    """Return (Δtsup, Δp/q).  Sign: Δp = p_wf − pi > 0 for injection."""
    tsup = np.array([superposition_time(times, rates, i)
                     for i in range(len(times))])
    inv  = np.where(rates > 0, (pwf - pi) / rates, np.nan)
    return tsup, inv


def back_calc_skin(inv, tsup, m_val, lkarg):
    """Invert Eq. 6+8 to get instantaneous skin at each data point."""
    b = inv - m_val * tsup
    return (b / m_val - lkarg + 3.2275) / 0.86859


def guide_line_y(tsup_range, m_val, lkarg, s):
    """Eq. 6 guide line for constant skin s."""
    return m_val * tsup_range + intercept_b(m_val, lkarg, s)


# ═══════════════════════════════════════════════════════════════════════
#  7.  PLOT
# ═══════════════════════════════════════════════════════════════════════

def plot_diagnostic(field_df, res, has_pwf, has_pti,
                    fluid_sched_df=None,
                    skin_values=None,
                    use_friction=True,
                    title="Inverse Injectivity Diagnostic Plot",
                    output_path=None):

    k, h    = float(res["k"]),   float(res["h"])
    phi, mu = float(res["phi"]), float(res["mu"])
    B,  ct  = float(res["B"]),   float(res["ct"])
    rw, pi  = float(res["rw"]),  float(res["pi"])

    times = field_df["time(hr)"].values.astype(float)
    rates = field_df["qi(bbl/min)"].values.astype(float)

    # ── bottomhole pressure ────────────────────────────────────────────
    dp_PE_arr = dp_F_arr = col_snapshots = None
    if has_pwf:
        pwf          = field_df["pwf(psi)"].values.astype(float)
        pressure_note = "p_wf  direct from data"
    else:
        pti = field_df["pti(psi)"].values.astype(float)
        pwf, col_snapshots, dp_PE_arr, dp_F_arr = calc_pwf_array(
            times, rates, pti, res, fluid_sched_df, use_friction)
        H    = float(res.get("H", 0))
        d_in = float(res.get("d_in", 0))
        pressure_note = (
            f"p_wf from p_ti  (Eq.2)  |  H={H} ft"
            + (f", d_in={d_in} in  friction={'ON' if use_friction else 'OFF'}"
               if d_in > 0 else "  friction=OFF (no d_in)"))

    # ── SPE-28548 arrays ───────────────────────────────────────────────
    m_val         = slope_m(B, mu, k, h)
    lkarg         = log_karg(k, phi, mu, ct, rw)
    tsup_arr, inv_arr = compute_arrays(times, rates, pwf, pi)
    s_arr         = back_calc_skin(inv_arr, tsup_arr, m_val, lkarg)

    # ── console report ─────────────────────────────────────────────────
    print("\n" + "═"*72)
    print("  RESERVOIR PARAMETERS")
    print("═"*72)
    print(f"  k={k} md | h={h} ft | φ={phi} | μ={mu} cp | B={B}")
    print(f"  ct={ct:.2e} psi⁻¹ | rw={rw} ft | pi={pi} psi")
    print(f"  m′={m_val:.4f} psi·min/bbl | log(k/φμct·rw²)={lkarg:.4f}")
    print(f"  {pressure_note}")

    print("\n" + "═"*72)
    print("  POINT-BY-POINT RESULTS")
    print("═"*72)
    hdr = f"  {'#':>3}  {'t(hr)':>7}  {'q(bbl/m)':>9}  {'p_wf':>8}"
    if has_pti:
        hdr += f"  {'p_ti':>8}  {'dp_PE':>7}  {'dp_F':>6}"
    hdr += f"  {'Δtsup':>8}  {'Δp/q':>9}  {'skin':>6}"
    print(hdr)

    for i in range(len(times)):
        row = (f"  {i+1:>3}  {times[i]:>7.3f}  {rates[i]:>9.3f}  {pwf[i]:>8.1f}")
        if has_pti:
            pti_v = field_df["pti(psi)"].values[i]
            row  += (f"  {pti_v:>8.1f}  "
                     f"{dp_PE_arr[i]:>7.1f}  {dp_F_arr[i]:>6.2f}")
        row += (f"  {tsup_arr[i]:>8.4f}  {inv_arr[i]:>9.2f}  {s_arr[i]:>6.1f}")
        print(row)

    print(f"\n  Initial skin ≈ {s_arr[0]:.1f}  →  Final skin ≈ {s_arr[-1]:.1f}"
          f"  (ΔSkin ≈ {s_arr[0]-s_arr[-1]:.1f})")

    # ── skin guide lines ───────────────────────────────────────────────
    valid   = ~(np.isnan(tsup_arr) | np.isnan(inv_arr))
    s_valid = s_arr[valid]

    if skin_values is None:
        s_lo  = max(0.0, np.floor(s_valid.min()/5)*5)
        s_hi  =          np.ceil (s_valid.max()/5)*5
        step  = max(5.0, float(int((s_hi - s_lo)/7/5)*5))
        skin_values = list(np.arange(s_lo, s_hi + step, step))
        print(f"  Auto skin guide lines: {[int(s) for s in skin_values]}")

    t_valid = tsup_arr[valid]
    span    = t_valid.max() - t_valid.min() if len(t_valid) > 1 else 1.0
    pad     = max(0.35, 0.5*span)
    t_range = np.linspace(t_valid.min()-pad, t_valid.max()+pad, 500)

    # ── figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    n_s  = max(len(skin_values), 2)
    cmap = plt.cm.plasma_r
    cols = [cmap(0.05 + 0.85*i/(n_s-1)) for i in range(len(skin_values))]

    inv_v     = inv_arr[valid]
    y_lo      = inv_v.min();  y_hi = inv_v.max()
    y_span    = y_hi - y_lo
    win_lo    = y_lo - 0.15*y_span
    win_hi    = y_hi + 0.60*y_span   # headroom for line labels

    for s_val, col in zip(skin_values, cols):
        yline = guide_line_y(t_range, m_val, lkarg, s_val)
        ax.plot(t_range, yline, "--", color=col, linewidth=1.5, alpha=0.80, zorder=2)
        in_win = (yline >= win_lo) & (yline <= win_hi)
        xi, yi = ((t_range[in_win][-1], yline[in_win][-1])
                  if in_win.any() else (t_range[-1], yline[-1]))
        ax.text(xi+0.04, yi, f"s={int(s_val)}",
                fontsize=9, color=col, va="center", fontweight="bold", zorder=10)

    ax.plot(tsup_arr[valid], inv_arr[valid],
            color="black", linewidth=1.8, zorder=5)
    ax.scatter(tsup_arr[valid], inv_arr[valid],
               facecolors="white", edgecolors="black",
               s=60, linewidths=1.8, zorder=6)
    for vi in np.where(valid)[0]:
        ax.annotate(str(vi+1), (tsup_arr[vi], inv_arr[vi]),
                    textcoords="offset points", xytext=(5,5),
                    fontsize=8, color="black", zorder=7)

    ax.set_xlabel("Superposition Time Function,  $\\Delta t_{\\rm sup}$",
                  fontsize=13, labelpad=8)
    ax.set_ylabel("Inverse Injectivity,  $\\Delta p / q_i$   (psi / bbl·min$^{-1}$)",
                  fontsize=13, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(bottom=max(0, win_lo))
    ax.grid(True, which="major", linestyle=":", alpha=0.5, color="grey")
    ax.grid(True, which="minor", linestyle=":", alpha=0.2, color="grey")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    note = (f"Initial skin  ≈  {s_arr[0]:.0f}\n"
            f"Final skin    ≈  {s_arr[-1]:.0f}\n"
            f"Skin removed  ≈  {s_arr[0]-s_arr[-1]:.0f}\n\n"
            f"m′ = {m_val:.4f} psi/(bbl/min)")
    if has_pti:
        note += "\n(p_wf: variable-ρ column, Eq.2)"
    ax.text(0.02, 0.98, note,
            transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow",
                      ec="gray", alpha=0.90), zorder=11)

    ax.legend(handles=[Line2D([0],[0], marker="o", color="black", linewidth=1.8,
                               markerfacecolor="white", markersize=7,
                               label="Field data")],
              loc="upper right", fontsize=10, framealpha=0.85)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"\n  ✓  Plot saved  →  {output_path}")

    plt.show()   # always show on screen

    return fig, ax, s_arr


# ═══════════════════════════════════════════════════════════════════════
#  8.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Inverse injectivity plot (SPE-28548)")
    p.add_argument("excel",       nargs="?", default="Field_Example_1.xlsx")
    p.add_argument("--skin",      default=None,
                   help="Comma-separated skin values, e.g. 0,10,20,40")
    p.add_argument("--no-friction", action="store_true",
                   help="Ignore tubing friction when converting p_ti → p_wf")
    p.add_argument("--out",       default=None,
                   help="Also save PNG to this path")
    p.add_argument("--title",     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.excel):
        print(f"ERROR: File not found → {args.excel}"); sys.exit(1)

    print(f"\nLoading: {args.excel}")
    fd, res, has_pwf, has_pti, fluid_sched = load_excel(args.excel)
    print(f"Loaded {len(fd)} data points.")

    skin_values = None
    if args.skin:
        try:
            skin_values = [float(s.strip()) for s in args.skin.split(",")]
        except ValueError:
            print("WARNING: could not parse --skin; using auto-detect.")

    title = args.title or (
        f"Inverse Injectivity Diagnostic Plot\n"
        f"{os.path.splitext(os.path.basename(args.excel))[0]}  (SPE-28548)"
    )

    plot_diagnostic(
        fd, res, has_pwf, has_pti,
        fluid_sched_df = fluid_sched,
        skin_values    = skin_values,
        use_friction   = not args.no_friction,
        title          = title,
        output_path    = args.out,
    )


if __name__ == "__main__":
    main()
