"""
Hill & Zhu Inverse Injectivity Diagnostic — Streamlit App
SPE-28548-PA | Real-time matrix acidizing skin monitoring
Run: streamlit run app.py
"""

import io, math, textwrap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Hill & Zhu — Inverse Injectivity",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# Palette — drawn from petroleum engineering visual culture:
#   deep formation blue-black as base, not generic dark mode
#   amber/amber-gold as the single accent (resembles crude, warning panels)
#   slate-grey supporting tones, no teal/green clichés
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── tokens ─────────────────────────────────────────────────────── */
:root {
  --bg:        #0B0F14;
  --surface:   #131920;
  --border:    #1E2730;
  --border-hi: #2C3844;
  --amber:     #D4891A;
  --amber-lt:  #E8A93A;
  --amber-dim: #7A4E0D;
  --slate:     #8B99AA;
  --text:      #D6DFE8;
  --text-dim:  #6B7A8A;
  --success:   #3D9970;
  --danger:    #C0392B;
  --mono:      'JetBrains Mono', monospace;
  --sans:      'Space Grotesk', sans-serif;
}

/* ── reset ──────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
  background: var(--bg) !important;
  font-family: var(--sans);
  color: var(--text);
}
.main .block-container { padding-top: 0 !important; max-width: 100%; }

/* ── sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
  font-family: var(--mono);
  font-size: 0.65rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--amber) !important;
  margin: 1.4rem 0 0.5rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--border);
}

/* ── header bar ─────────────────────────────────────────────────── */
.app-bar {
  background: var(--surface);
  border-bottom: 2px solid var(--amber);
  padding: 1rem 2rem 0.9rem;
  margin-bottom: 0;
  display: flex;
  align-items: center;
  gap: 1.2rem;
}
.app-bar-icon { font-size: 1.8rem; line-height: 1; }
.app-bar-title {
  font-family: var(--mono);
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--amber-lt);
  letter-spacing: -0.01em;
  margin: 0;
}
.app-bar-sub {
  font-size: 0.72rem;
  color: var(--text-dim);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-top: 0.15rem;
}

/* ── section eyebrow ────────────────────────────────────────────── */
.eyebrow {
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--amber);
  margin: 1.8rem 0 0.7rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.eyebrow::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── KPI strip ──────────────────────────────────────────────────── */
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 1.6rem;
}
.kpi-cell {
  background: var(--surface);
  padding: 1.1rem 1.3rem 0.9rem;
  position: relative;
}
.kpi-cell::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--amber);
}
.kpi-cell.success::before { background: var(--success); }
.kpi-cell.danger::before  { background: var(--danger); }
.kpi-cell.slate::before   { background: var(--border-hi); }
.kpi-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 0.4rem;
}
.kpi-num {
  font-family: var(--mono);
  font-size: 2rem;
  font-weight: 600;
  color: var(--text);
  line-height: 1;
}
.kpi-unit {
  font-size: 0.65rem;
  color: var(--text-dim);
  margin-top: 0.3rem;
}

/* ── info / warn / err banners ──────────────────────────────────── */
.banner {
  padding: 0.7rem 1rem;
  border-radius: 5px;
  font-size: 0.82rem;
  margin: 0.6rem 0 1rem;
  border-left: 3px solid;
  background: var(--surface);
}
.banner.info  { border-color: #3D9970; color: #8EC9AF; }
.banner.warn  { border-color: var(--amber); color: var(--amber-lt); }
.banner.error { border-color: var(--danger); color: #E87C7C; }

/* ── upload drop zone ───────────────────────────────────────────── */
[data-testid="stFileUploader"] > div {
  border: 1.5px dashed var(--border-hi) !important;
  border-radius: 7px !important;
  background: var(--bg) !important;
  transition: border-color 0.2s;
}
[data-testid="stFileUploader"] > div:hover {
  border-color: var(--amber) !important;
}

/* ── buttons ────────────────────────────────────────────────────── */
.stButton > button {
  background: var(--amber) !important;
  color: #0B0F14 !important;
  font-family: var(--sans) !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 5px !important;
  padding: 0.5rem 1.4rem !important;
  letter-spacing: 0.02em;
}
.stButton > button:hover { background: var(--amber-lt) !important; }
.stDownloadButton > button {
  background: transparent !important;
  color: var(--amber) !important;
  border: 1px solid var(--amber) !important;
  border-radius: 5px !important;
  font-weight: 500 !important;
}
.stDownloadButton > button:hover {
  background: var(--amber-dim) !important;
}

/* ── number inputs / selects / sliders ──────────────────────────── */
input, select, textarea {
  background: var(--bg) !important;
  color: var(--text) !important;
  border-color: var(--border-hi) !important;
  font-family: var(--mono) !important;
}
label { color: var(--slate) !important; font-size: 0.8rem !important; }

/* ── tabs ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  gap: 0;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-dim) !important;
  padding: 0.65rem 1.3rem;
  border-radius: 0;
  background: transparent;
}
.stTabs [aria-selected="true"] {
  color: var(--amber) !important;
  border-bottom: 2px solid var(--amber);
  background: transparent;
}

/* ── dataframe ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 6px;
  overflow: hidden;
}

/* ── hide Streamlit chrome ──────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-bar">
  <div class="app-bar-icon">⬡</div>
  <div>
    <div class="app-bar-title">Hill &amp; Zhu — Inverse Injectivity Diagnostic</div>
    <div class="app-bar-sub">SPE-28548-PA &nbsp;·&nbsp; Real-time matrix acidizing skin monitoring &nbsp;·&nbsp; Variable-density BHP model</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PHYSICS ENGINE  (all functions from the CLI script, verbatim)
# ══════════════════════════════════════════════════════════════════════

def tubing_volume_bbl(d_in, H):
    r_ft = (d_in / 2.0) / 12.0
    return math.pi * r_ft**2 * H / 5.61458

def build_fluid_column(cum_vol_bbl, fluid_stages, tubing_vol_bbl, H):
    if tubing_vol_bbl <= 0:
        return []
    ft_per_bbl = H / tubing_vol_bbl
    v_bot = max(0.0, cum_vol_bbl - tubing_vol_bbl)
    v_top = cum_vol_bbl
    if v_top <= 0.0:
        return []
    v_stage_start = 0.0
    segments = []
    for stg in fluid_stages:
        v_stage_end = v_stage_start + stg["vol_bbl"]
        ov_lo = max(v_stage_start, v_bot)
        ov_hi = min(v_stage_end, v_top)
        if ov_hi > ov_lo:
            depth_bot = H - (ov_lo - v_bot) * ft_per_bbl
            depth_top = H - (ov_hi - v_bot) * ft_per_bbl
            segments.append({
                "name":         stg["name"],
                "density_ppg":  stg["density_ppg"],
                "viscosity_cp": stg["viscosity_cp"],
                "depth_top_ft": max(0.0, depth_top),
                "depth_bot_ft": min(H, depth_bot),
            })
        v_stage_start = v_stage_end
    return segments

def hydrostatic_dp(column, H):
    if not column:
        return 0.0
    total_h = sum(s["depth_bot_ft"] - s["depth_top_ft"] for s in column)
    unfilled = H - total_h
    dp = 0.0
    if unfilled > 0 and column:
        dp += 0.052 * column[-1]["density_ppg"] * unfilled
    for seg in column:
        dp += 0.052 * seg["density_ppg"] * (seg["depth_bot_ft"] - seg["depth_top_ft"])
    return dp

def churchill_ff(Re, eps_over_d=0.0):
    if Re <= 0:   return 0.0
    if Re < 2100: return 16.0 / Re
    A = (-2.457 * math.log((7.0 / Re)**0.9 + 0.27 * eps_over_d))**16
    B = (37530.0 / Re)**16
    return 2.0 * ((8.0 / Re)**12 + (A + B)**(-1.5))**(1.0 / 12.0)

def friction_dp_segment(q, rho_ppg, mu_cp, d_in, L_ft):
    if d_in <= 0 or L_ft <= 0 or q <= 0:
        return 0.0
    rho   = rho_ppg * 7.4805
    q_f   = q * 5.61458 / 60.0
    d_ft  = d_in / 12.0
    v     = q_f / (math.pi * d_ft**2 / 4.0)
    Re    = rho * v * d_ft / (mu_cp * 6.7197e-4)
    f     = churchill_ff(Re)
    return 4.0 * f * (L_ft / d_ft) * rho * v**2 / 2.0 / 144.0

def total_friction_dp(q, column, d_in, H):
    if d_in <= 0 or not column:
        return 0.0
    return sum(friction_dp_segment(q, s["density_ppg"], s["viscosity_cp"],
                                    d_in, s["depth_bot_ft"] - s["depth_top_ft"])
               for s in column)

def parse_fluid_schedule(df, res):
    if df is None:
        rho = float(res.get("rho", 8.33))
        mu  = float(res.get("mu", 1.0))
        return [{"name": "fluid", "vol_bbl": 1e9, "density_ppg": rho, "viscosity_cp": mu}]
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    stages = []
    for _, row in df.iterrows():
        def g(keys, default=0.0):
            for k in keys:
                if k in df.columns:
                    return float(row[k])
            return default
        name_col = next((c for c in df.columns if any(x in c for x in ("name","fluid"))), df.columns[1])
        stages.append({
            "name":         str(row[name_col]),
            "vol_bbl":      g(["volume(bbl)", "volume_bbl", "vol(bbl)", "vol_bbl"]),
            "density_ppg":  g(["density(ppg)", "density_ppg", "rho(ppg)"], 8.33),
            "viscosity_cp": g(["viscosity(cp)", "viscosity_cp", "mu(cp)"], 1.0),
        })
    return stages

def calc_pwf_array(times, rates, pti, res, fluid_sched_df, use_friction):
    H    = float(res.get("H",    0.0))
    d_in = float(res.get("d_in", 0.0))
    n = len(times)
    if H <= 0:
        return pti.copy(), [], np.zeros(n), np.zeros(n), 0.0
    stages  = parse_fluid_schedule(fluid_sched_df, res)
    tub_vol = tubing_volume_bbl(d_in, H) if d_in > 0 else 1e9
    cum_vol = np.zeros(n)
    for i in range(1, n):
        cum_vol[i] = cum_vol[i-1] + (rates[i] + rates[i-1]) / 2.0 * (times[i] - times[i-1]) * 60.0
    pwf_a, pe_a, pf_a, cols = [], [], [], []
    for i in range(n):
        col  = build_fluid_column(cum_vol[i], stages, tub_vol, H)
        dp_PE = hydrostatic_dp(col, H)
        dp_F  = total_friction_dp(rates[i], col, d_in, H) if use_friction else 0.0
        pwf_a.append(pti[i] + dp_PE - dp_F)
        pe_a.append(dp_PE); pf_a.append(dp_F); cols.append(col)
    return np.array(pwf_a), cols, np.array(pe_a), np.array(pf_a), tub_vol

def slope_m(B, mu, k, h):
    return 162.6 * B * mu / (k * h) * 1440.0

def log_karg(k, phi, mu, ct, rw):
    return np.log10(k / (phi * mu * ct * rw**2))

def intercept_b(m, lk, s):
    return m * (lk - 3.2275 + 0.86859 * s)

def superposition_time(times, rates, idx):
    t_N, q_N = times[idx], rates[idx]
    if q_N <= 0:
        return np.nan
    result = 0.0
    for j in range(idx + 1):
        dq    = rates[j] - (rates[j - 1] if j > 0 else 0.0)
        t_jm1 = times[j - 1] if j > 0 else 0.0
        dt    = t_N - t_jm1
        if dt > 0:
            result += dq / q_N * np.log10(dt)
    return result

def run_analysis(field_df, res, has_pwf, has_pti, fluid_sched_df, use_friction):
    k, h    = float(res["k"]),   float(res["h"])
    phi, mu = float(res["phi"]), float(res["mu"])
    B,  ct  = float(res["B"]),   float(res["ct"])
    rw, pi  = float(res["rw"]),  float(res["pi"])

    times = field_df["time(hr)"].values.astype(float)
    rates = field_df["qi(bbl/min)"].values.astype(float)

    col_snaps, dp_PE_arr, dp_F_arr = [], None, None
    tub_vol = 0.0
    if has_pwf:
        pwf = field_df["pwf(psi)"].values.astype(float)
    else:
        pti = field_df["pti(psi)"].values.astype(float)
        pwf, col_snaps, dp_PE_arr, dp_F_arr, tub_vol = calc_pwf_array(
            times, rates, pti, res, fluid_sched_df, use_friction)

    m_val = slope_m(B, mu, k, h)
    lkarg = log_karg(k, phi, mu, ct, rw)
    tsup  = np.array([superposition_time(times, rates, i) for i in range(len(times))])
    inv   = np.where(rates > 0, (pwf - pi) / rates, np.nan)
    b_arr = inv - m_val * tsup
    skin  = (b_arr / m_val - lkarg + 3.2275) / 0.86859

    return {
        "times": times, "rates": rates, "pwf": pwf,
        "tsup": tsup, "inv": inv, "skin": skin,
        "m_val": m_val, "lkarg": lkarg,
        "dp_PE": dp_PE_arr, "dp_F": dp_F_arr,
        "col_snaps": col_snaps, "tub_vol": tub_vol,
        "has_pti": has_pti,
        "pti": field_df["pti(psi)"].values.astype(float) if has_pti else None,
        "res": res,
    }


# ══════════════════════════════════════════════════════════════════════
# PLOTLY FIGURE
# ══════════════════════════════════════════════════════════════════════

# Colour sequence for skin lines — amber gradient into cooler slates
SKIN_COLORS = [
    "#D4891A", "#C0711A", "#A85A1A",
    "#5B7FA6", "#3D6B96", "#2A5580",
    "#8B99AA", "#6B7A8A",
]

def build_figure(r, skin_vals, plot_title):
    valid  = ~(np.isnan(r["tsup"]) | np.isnan(r["inv"]))
    tv     = r["tsup"][valid]
    iv     = r["inv"][valid]

    span   = (tv.max() - tv.min()) if len(tv) > 1 else 1.0
    pad    = max(0.4, 0.55 * span)
    t_rng  = np.linspace(tv.min() - pad, tv.max() + pad, 600)

    y_lo   = iv.min(); y_hi = iv.max()
    y_sp   = max(y_hi - y_lo, 1.0)
    win_lo = max(0.0, y_lo - 0.12 * y_sp)
    win_hi = y_hi + 0.70 * y_sp

    fig = go.Figure()

    # ── skin reference lines ──────────────────────────────────────────
    for i, s_val in enumerate(skin_vals):
        col   = SKIN_COLORS[i % len(SKIN_COLORS)]
        yline = r["m_val"] * t_rng + intercept_b(r["m_val"], r["lkarg"], s_val)
        fig.add_trace(go.Scatter(
            x=t_rng, y=yline, mode="lines",
            line=dict(dash="dash", color=col, width=1.5),
            name=f"s = {int(s_val)}",
            hovertemplate=f"<b>s = {int(s_val)}</b><br>Δtsup = %{{x:.3f}}<br>Δp/q = %{{y:.1f}} psi/(bbl/min)<extra></extra>",
        ))
        # label at rightmost visible point
        in_win = (yline >= win_lo) & (yline <= win_hi)
        if in_win.any():
            xi, yi = t_rng[in_win][-1], yline[in_win][-1]
        else:
            xi, yi = t_rng[-1], yline[-1]
        fig.add_annotation(
            x=xi, y=yi, text=f"<b>s = {int(s_val)}</b>",
            showarrow=False, xanchor="left", xshift=8,
            font=dict(color=col, size=11, family="JetBrains Mono"),
        )

    # ── field data trajectory ─────────────────────────────────────────
    vi_idx = np.where(valid)[0]
    hover  = []
    for vi in vi_idx:
        h = (f"<b>Point {vi + 1}</b><br>"
             f"t = {r['times'][vi]:.3f} hr<br>"
             f"q = {r['rates'][vi]:.3f} bbl/min<br>"
             f"p_wf = {r['pwf'][vi]:.1f} psi<br>"
             f"Δtsup = {r['tsup'][vi]:.4f}<br>"
             f"Δp/q = {r['inv'][vi]:.2f}<br>"
             f"<b>Skin ≈ {r['skin'][vi]:.1f}</b>")
        if r["has_pti"] and r["pti"] is not None and r["dp_PE"] is not None:
            h += (f"<br>─────<br>p_ti = {r['pti'][vi]:.1f} psi<br>"
                  f"ΔpPE = {r['dp_PE'][vi]:.1f} psi<br>"
                  f"ΔpF = {r['dp_F'][vi]:.2f} psi")
        hover.append(h)

    fig.add_trace(go.Scatter(
        x=r["tsup"][valid], y=r["inv"][valid],
        mode="lines+markers+text",
        line=dict(color="#D6DFE8", width=2.0),
        marker=dict(size=10, color="#131920",
                    line=dict(color="#D6DFE8", width=2)),
        text=[str(vi + 1) for vi in vi_idx],
        textposition="top right",
        textfont=dict(size=9, color="#8B99AA", family="JetBrains Mono"),
        name="Field data",
        hovertext=hover, hoverinfo="text",
    ))

    # ── layout ────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor="#0B0F14",
        plot_bgcolor="#0E1318",
        font=dict(family="JetBrains Mono, monospace", color="#D6DFE8", size=12),
        title=dict(
            text=(f"<b style='color:#D4891A'>{plot_title}</b><br>"
                  f"<span style='font-size:11px;color:#6B7A8A;font-family:Space Grotesk'>"
                  f"SPE-28548 · Hill & Zhu (1996) · Inverse Injectivity vs Superposition Time</span>"),
            x=0.03, xanchor="left", font=dict(size=14),
        ),
        xaxis=dict(
            title=dict(text="Superposition Time Function  Δt<sub>sup</sub>",
                       font=dict(size=13, color="#8B99AA")),
            gridcolor="#1A2030", gridwidth=1,
            zerolinecolor="#2C3844", zerolinewidth=1,
            tickfont=dict(size=11), linecolor="#2C3844",
        ),
        yaxis=dict(
            title=dict(text="Inverse Injectivity  Δp/q<sub>i</sub>  (psi · min / bbl)",
                       font=dict(size=13, color="#8B99AA")),
            gridcolor="#1A2030", gridwidth=1,
            zerolinecolor="#2C3844", zerolinewidth=1,
            tickfont=dict(size=11), linecolor="#2C3844",
            range=[win_lo, win_hi],
        ),
        legend=dict(
            bgcolor="#131920", bordercolor="#1E2730", borderwidth=1,
            font=dict(size=11), x=1.01, y=1.0, xanchor="left",
        ),
        margin=dict(l=80, r=180, t=100, b=70),
        hoverlabel=dict(
            bgcolor="#131920", bordercolor="#2C3844",
            font=dict(family="JetBrains Mono", size=12),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Upload")
    uploaded = st.file_uploader(
        "Excel workbook (.xlsx)",
        type=["xlsx", "xls"],
        help="Needs sheets: 'field data' and 'Reservoir_Data'. "
             "Add 'fluid_schedule' for multi-fluid BHP conversion.",
        label_visibility="collapsed",
    )

    st.markdown("### Plot settings")
    plot_title = st.text_input(
        "Plot title", value="Inverse Injectivity Diagnostic Plot",
        label_visibility="visible",
    )

    skin_mode = st.radio("Skin guide lines", ["Auto-detect", "Manual"], horizontal=True)
    skin_input_str = ""
    if skin_mode == "Manual":
        skin_input_str = st.text_input(
            "Skin values (comma-separated)", "0,5,10,20,30,40"
        )

    st.markdown("### BHP options")
    use_friction = st.toggle("Include tubing friction (Churchill 1977)", value=True)

    st.markdown("### About")
    st.markdown(
        "<span style='font-size:0.75rem;color:#6B7A8A;line-height:1.8'>"
        "<b style='color:#D4891A'>field data</b> sheet:<br>"
        "&nbsp;time(hr) · qi(bbl/min)<br>"
        "&nbsp;+ pwf(psi) <i>or</i> pti(psi)<br><br>"
        "<b style='color:#D4891A'>Reservoir_Data</b> sheet:<br>"
        "&nbsp;Parameter · Value<br>"
        "&nbsp;k · h · phi · mu · B · ct · rw · pi<br>"
        "&nbsp;H · d_in &nbsp;(if using pti)<br><br>"
        "<b style='color:#D4891A'>fluid_schedule</b> (optional):<br>"
        "&nbsp;Stage · fluid_name<br>"
        "&nbsp;volume(bbl) · density(ppg)<br>"
        "&nbsp;viscosity(cp)"
        "</span>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# LANDING STATE  (no file yet)
# ══════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown('<div class="eyebrow">Getting started</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
**Upload your Excel workbook** in the sidebar to begin.

The app generates the inverse injectivity diagnostic plot from
Hill & Zhu (1996), SPE-28548-PA. The plot shows **Δp/q vs. Δt_sup**
overlaid on constant-skin reference lines — the field data trajectory
reads directly against these lines to track skin factor evolution
throughout the acid treatment.

**If your data has surface injection pressure (p_ti)** rather than
direct BHP, the app converts it automatically using a
variable-density piston-displacement column model (Eq. 2):

> p_wf = p_ti + ΔpPE − ΔpF

where ΔpPE integrates each fluid slug's density along its actual
depth interval, and ΔpF uses the Churchill (1977) friction factor
valid for all flow regimes.
""")
    with c2:
        st.markdown("**Key equations implemented:**")
        for eq, desc in [
            ("Eq. 2",  "p_wf from p_ti via variable-density hydrostatic + friction"),
            ("Eq. 6",  "Δp/qN = m′·Δtsup + b′ — constant-skin reference lines"),
            ("Eq. 7",  "m′ = 162.6·B·μ/(k·h)·1440  [q in bbl/min]"),
            ("Eq. 8",  "b′ = m′·[log(k/φμct·rw²) − 3.2275 + 0.86859·s]"),
            ("Eq. 9",  "Δtsup = Σ (qj−q_{j−1})/qN·log10(t_N−t_{j−1})"),
        ]:
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
                f"padding:0.4rem 0.8rem;margin:0.25rem 0;background:#131920;"
                f"border-left:2px solid #D4891A;border-radius:0 4px 4px 0'>"
                f"<span style='color:#D4891A'>{eq}</span>"
                f"<span style='color:#6B7A8A'> — {desc}</span></div>",
                unsafe_allow_html=True,
            )
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Reading workbook…")
def load_workbook(file_bytes, file_name):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))

    fd = xl.parse("field data")
    fd.columns = [str(c).strip() for c in fd.columns]
    for need in ("time(hr)", "qi(bbl/min)"):
        if need not in fd.columns:
            raise ValueError(f"'field data' sheet is missing column: '{need}'")
    has_pwf = "pwf(psi)" in fd.columns
    has_pti = "pti(psi)" in fd.columns
    if not has_pwf and not has_pti:
        raise ValueError("'field data' needs a 'pwf(psi)' or 'pti(psi)' column.")
    key_cols = {"time(hr)", "qi(bbl/min)", "pwf(psi)" if has_pwf else "pti(psi)"}
    fd = fd.dropna(subset=list(key_cols)).reset_index(drop=True)

    rd = xl.parse("Reservoir_Data")
    rd.columns = [str(c).strip() for c in rd.columns]
    res = {}
    for _, row in rd.iterrows():
        try:
            res[str(row["Parameter"]).strip()] = float(row["Value"])
        except Exception:
            pass
    missing = [p for p in ("k", "h", "phi", "mu", "B", "ct", "rw", "pi") if p not in res]
    if missing:
        raise ValueError(f"Reservoir_Data sheet is missing: {missing}")

    fs_df = None
    if "fluid_schedule" in xl.sheet_names:
        fs_df = xl.parse("fluid_schedule")
        fs_df.columns = [str(c).strip() for c in fs_df.columns]

    return fd, res, has_pwf, has_pti, fs_df


try:
    file_bytes = uploaded.getvalue()
    fd, res, has_pwf, has_pti, fs_df = load_workbook(file_bytes, uploaded.name)
except Exception as e:
    st.markdown(f'<div class="banner error">⚠ Load error: {e}</div>',
                unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# RUN ANALYSIS
# ══════════════════════════════════════════════════════════════════════
try:
    r = run_analysis(fd, res, has_pwf, has_pti, fs_df, use_friction)
except Exception as e:
    st.markdown(f'<div class="banner error">⚠ Computation error: {e}</div>',
                unsafe_allow_html=True)
    st.stop()

# parse skin values
skin_vals = None
if skin_mode == "Manual" and skin_input_str.strip():
    try:
        skin_vals = [float(s.strip()) for s in skin_input_str.split(",") if s.strip()]
    except ValueError:
        st.markdown('<div class="banner warn">Could not parse skin values — using auto-detect.</div>',
                    unsafe_allow_html=True)

if skin_vals is None:
    valid_s = r["skin"][~np.isnan(r["skin"])]
    s_lo  = max(0.0, np.floor(valid_s.min() / 5) * 5)
    s_hi  =          np.ceil(valid_s.max()  / 5) * 5
    step  = max(5.0, float(int((s_hi - s_lo) / 7 / 5) * 5))
    skin_vals = list(np.arange(s_lo, s_hi + step, step))


# ══════════════════════════════════════════════════════════════════════
# BHP SOURCE BANNER
# ══════════════════════════════════════════════════════════════════════
if has_pti:
    H    = res.get("H", 0)
    d_in = res.get("d_in", 0)
    fric = "ON" if (use_friction and d_in > 0) else "OFF"
    tub  = f"{r['tub_vol']:.1f} bbl" if r["tub_vol"] > 0 else "n/a"
    st.markdown(
        f'<div class="banner warn">🔧 BHP computed from surface p_ti via Eq. 2 &nbsp;|&nbsp; '
        f'H = {H:.0f} ft &nbsp;|&nbsp; Tubing capacity ≈ {tub} &nbsp;|&nbsp; '
        f'Variable-density column model &nbsp;|&nbsp; Friction {fric}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="banner info">✓ Bottomhole pressure p_wf read directly from data — no conversion needed.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════
valid = ~(np.isnan(r["tsup"]) | np.isnan(r["skin"]))
s0  = r["skin"][valid][0]
sf  = r["skin"][valid][-1]
ds  = s0 - sf
mv  = r["m_val"]
pts = int(valid.sum())

st.markdown('<div class="eyebrow">Treatment summary</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-cell">
    <div class="kpi-label">Initial Skin</div>
    <div class="kpi-num">{s0:.0f}</div>
    <div class="kpi-unit">before acidizing</div>
  </div>
  <div class="kpi-cell success">
    <div class="kpi-label">Final Skin</div>
    <div class="kpi-num">{sf:.0f}</div>
    <div class="kpi-unit">end of treatment</div>
  </div>
  <div class="kpi-cell {'danger' if ds < 0 else 'slate'}">
    <div class="kpi-label">Skin Removed</div>
    <div class="kpi-num">{ds:.0f}</div>
    <div class="kpi-unit">Δs = s_i − s_f</div>
  </div>
  <div class="kpi-cell slate">
    <div class="kpi-label">Slope m′</div>
    <div class="kpi-num" style="font-size:1.35rem">{mv:.4f}</div>
    <div class="kpi-unit">psi·min/bbl per log-cycle &nbsp;|&nbsp; {pts} data points</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN PLOT
# ══════════════════════════════════════════════════════════════════════
st.markdown('<div class="eyebrow">Diagnostic plot</div>', unsafe_allow_html=True)

fig = build_figure(r, skin_vals, plot_title)
st.plotly_chart(fig, use_container_width=True, config={
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "width": 1600, "height": 900, "scale": 2},
})

# PNG download via kaleido
try:
    import plotly.io as pio
    png_bytes = pio.to_image(fig, format="png", width=1600, height=900, scale=2)
    st.download_button(
        "⬇  Download plot as PNG",
        data=png_bytes,
        file_name=f"{uploaded.name.replace('.xlsx','')}_diagnostic.png",
        mime="image/png",
    )
except Exception:
    st.caption("Install kaleido (`pip install kaleido`) to enable PNG export.")


# ══════════════════════════════════════════════════════════════════════
# TABBED OUTPUT
# ══════════════════════════════════════════════════════════════════════
st.markdown('<div class="eyebrow">Detailed results</div>', unsafe_allow_html=True)

tab_res, tab_res2, tab_fluid, tab_param = st.tabs([
    "Point-by-point results",
    "Rate & pressure profile",
    "Fluid column snapshots",
    "Reservoir parameters",
])

# ── Tab 1: numerical table ────────────────────────────────────────────
with tab_res:
    rows = []
    for i in range(len(r["times"])):
        row = {
            "Pt":           i + 1,
            "time (hr)":    round(r["times"][i], 4),
            "q (bbl/min)":  round(r["rates"][i], 4),
            "p_wf (psi)":   round(r["pwf"][i],   1),
        }
        if r["has_pti"] and r["pti"] is not None:
            row["p_ti (psi)"] = round(r["pti"][i], 1)
            if r["dp_PE"] is not None:
                row["ΔpPE (psi)"] = round(r["dp_PE"][i], 1)
                row["ΔpF (psi)"]  = round(r["dp_F"][i],  2)
        row["Δtsup"]   = round(r["tsup"][i], 5)
        row["Δp/q"]    = round(r["inv"][i],  2)
        row["Skin"]    = round(r["skin"][i], 1)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    st.dataframe(df_out, use_container_width=True, hide_index=True,
                 height=min(520, 42 + 35 * len(rows)))

    csv = df_out.to_csv(index=False).encode()
    st.download_button("⬇  Download as CSV", csv,
                       file_name="results.csv", mime="text/csv")

# ── Tab 2: rate + pressure vs time ───────────────────────────────────
with tab_res2:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Injection Rate", "Bottomhole Pressure"),
        vertical_spacing=0.1,
    )
    fig2.add_trace(go.Scatter(
        x=r["times"], y=r["rates"],
        mode="lines+markers",
        line=dict(color="#D4891A", width=2),
        marker=dict(size=6, color="#D4891A"),
        name="q (bbl/min)",
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=r["times"], y=r["pwf"],
        mode="lines+markers",
        line=dict(color="#5B7FA6", width=2),
        marker=dict(size=6, color="#5B7FA6"),
        name="p_wf (psi)",
    ), row=2, col=1)
    if r["has_pti"] and r["pti"] is not None:
        fig2.add_trace(go.Scatter(
            x=r["times"], y=r["pti"],
            mode="lines",
            line=dict(color="#8B99AA", width=1.5, dash="dot"),
            name="p_ti (psi)",
        ), row=2, col=1)
    fig2.update_layout(
        paper_bgcolor="#0B0F14", plot_bgcolor="#0E1318",
        font=dict(family="JetBrains Mono", color="#D6DFE8", size=11),
        height=480, margin=dict(l=70, r=40, t=50, b=50),
        legend=dict(bgcolor="#131920", bordercolor="#1E2730"),
    )
    for row_n in [1, 2]:
        fig2.update_xaxes(gridcolor="#1A2030", linecolor="#2C3844", row=row_n, col=1)
        fig2.update_yaxes(gridcolor="#1A2030", linecolor="#2C3844", row=row_n, col=1)
    fig2.update_xaxes(title_text="Time (hr)", row=2, col=1,
                      title_font=dict(color="#8B99AA"))
    fig2.update_yaxes(title_text="q (bbl/min)", row=1, col=1,
                      title_font=dict(color="#8B99AA"))
    fig2.update_yaxes(title_text="Pressure (psi)", row=2, col=1,
                      title_font=dict(color="#8B99AA"))
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: fluid column snapshots ─────────────────────────────────────
with tab_fluid:
    if not r["col_snaps"]:
        st.markdown('<div class="banner info">p_wf was read directly — no BHP conversion performed.</div>',
                    unsafe_allow_html=True)
    else:
        tub_vol = r["tub_vol"]
        st.markdown(
            f'<div class="banner info">Tubing capacity: <b>{tub_vol:.1f} bbl</b> &nbsp;|&nbsp; '
            f'Piston-displacement model — no mixing assumed</div>',
            unsafe_allow_html=True,
        )
        snap_rows = []
        for i, col in enumerate(r["col_snaps"]):
            desc = "  |  ".join(
                f"{s['name']} ({s['depth_top_ft']:.0f}–{s['depth_bot_ft']:.0f} ft, "
                f"{s['density_ppg']:.3f} ppg)"
                for s in col
            ) or "— tubing empty —"
            snap_rows.append({
                "Pt":           i + 1,
                "t (hr)":       round(r["times"][i], 3),
                "q (bbl/min)":  round(r["rates"][i], 3),
                "ΔpPE (psi)":   round(r["dp_PE"][i], 1),
                "ΔpF (psi)":    round(r["dp_F"][i],  2),
                "Column — bottom to top": desc,
            })
        st.dataframe(pd.DataFrame(snap_rows), use_container_width=True,
                     hide_index=True)

# ── Tab 4: reservoir parameters ───────────────────────────────────────
with tab_param:
    m_v  = r["m_val"]
    lk   = r["lkarg"]
    params = {
        "k (md)":      res["k"],
        "h (ft)":      res["h"],
        "φ (fraction)": res["phi"],
        "μ (cp)":      res["mu"],
        "B":           res["B"],
        "ct (psi⁻¹)":  f"{res['ct']:.3e}",
        "rw (ft)":     res["rw"],
        "pi (psi)":    res["pi"],
    }
    if "H"    in res: params["H (ft)"]    = res["H"]
    if "d_in" in res: params["d_in (in)"] = res["d_in"]
    params["— derived —"] = ""
    params["m′ (psi·min/bbl)"] = f"{m_v:.6f}"
    params["log(k/φμct·rw²)"]  = f"{lk:.4f}"

    p_df = pd.DataFrame({"Parameter": list(params.keys()),
                          "Value":     list(params.values())})
    st.dataframe(p_df, use_container_width=True, hide_index=True, height=380)
