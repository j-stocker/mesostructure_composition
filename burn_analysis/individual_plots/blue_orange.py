#!/usr/bin/env python3
"""
Particle Characterization: Burn Rate & Morphology Analysis
Full updated version: D_w and S_m vs Burn Rate, linear and log-log
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# DATA
# =========================

samples = np.array([
    "A","B","C","D","E","F","I","J","K","R","S","T"
])

groups = {
    "Porous": ["A","B","C","D","E","F"],
    "Solid":  ["I","J","K","R","S","T"],
}

burn_groups = ["Porous", "Solid"]

dw = np.array([
    7.7313, 4.5596, 5.3438, 4.3559, 8.1854, 8.6591,
    3.0309, 3.7919, 3.6889, 14.8659, 7.8058, 2.8298
])

Sm = np.array([
    597.22, 773.63, 658.95, 855.68, 455.32, 402.26,
    702.22, 561.1,  577.16, 142.36, 272.92, 752.37
])

model_1 = np.array([
    2.83, 3.04, 3.25, 2.65, 3.53, 3.70,
    3.75, 3.25, 3.25, 1.62, 2.29, 3.86
])

exp_1 = np.array([
    3.8, 3.2, 3.6, 3.2, 3.5, 3.0,
    3.4, 3.2, 3.2, 2.2, 2.6, 3.5
])

model_7 = np.array([
    9.52, 9.79, 9.85, 9.47, 9.83, 10.19,
    5.63, 6.00, 5.83, 5.22, 5.63, 5.66
])

exp_7 = np.array([
    10.1, 7.1, 7.6, 7.2, 7.8, 6.4,
    7.2, 6.8, 6.9, 5.3, 6.0, 7.2
])

# Colors
POROUS_COLOR = "#FF6200"
SOLID_COLOR  = "#1565C0"

color_map = {s: POROUS_COLOR for s in groups["Porous"]}
color_map.update({s: SOLID_COLOR for s in groups["Solid"]})

# =========================
# HELPERS
# =========================

def get_marker(sample):
    return 'o' if sample in groups["Porous"] else '*'

def get_marker_size(sample):
    return 90 if sample in groups["Porous"] else 110

def fit_line(x, y):
    return np.poly1d(np.polyfit(x, y, 1))

def fit_power_law(x, y):
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    logx = np.log10(x)
    logy = np.log10(y)

    b, loga = np.polyfit(logx, logy, 1)
    a = 10**loga

    def func(xv):
        return a * xv**b

    return func, a, b

def compute_r2(x, y, fit_fn):
    y_pred = fit_fn(x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot

def subset(group_name, x, y_m, y_e):
    mask = np.array([s in groups[group_name] for s in samples])
    return x[mask], y_m[mask], y_e[mask], samples[mask]

# =========================
# DRAW FUNCTION
# =========================

def draw_plot(ax, x, y_m, y_e, labels, title, xlabel, log=False):

    # Scatter points (model only)
    for i, s in enumerate(labels):
        ax.scatter(x[i], y_m[i],
                   marker=get_marker(s),
                   color=color_map[s],
                   s=get_marker_size(s))

    for group_name in burn_groups:
        mask = np.array([s in groups[group_name] for s in labels])
        if mask.sum() < 2:
            continue

        x_g = x[mask]
        color = POROUS_COLOR if group_name=="Porous" else SOLID_COLOR
        x_fit = np.linspace(min(x_g), max(x_g), 100)

        if log:
            # MODEL
            f_m, a_m, b_m = fit_power_law(x_g, y_m[mask])
            r2_m = compute_r2(x_g, y_m[mask], f_m)
            ax.plot(x_fit, f_m(x_fit), '-', lw=3, color=color)

            # EXPERIMENTAL
            f_e, a_e, b_e = fit_power_law(x_g, y_e[mask])
            r2_e = compute_r2(x_g, y_e[mask], f_e)
            ax.plot(x_fit, f_e(x_fit), '--', lw=3, color=color, alpha=0.6)

            # PRINT POWER-LAW EQUATIONS
            print(f"\n--- {title} ({group_name}) ---")
            print(f"Model:        y = {a_m:.3f} x^{b_m:.3f}   (R² = {r2_m:.3f})")
            print(f"Experimental: y = {a_e:.3f} x^{b_e:.3f}   (R² = {r2_e:.3f})")
        else:
            # Linear fits
            ax.plot(x_fit, fit_line(x_g, y_m[mask])(x_fit),
                    '-', lw=3, color=color)
            ax.plot(x_fit, fit_line(x_g, y_e[mask])(x_fit),
                    '--', lw=3, color=color, alpha=0.6)

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(which='both', linestyle='--', alpha=0.4)
    else:
        ax.grid(True, linestyle='--', alpha=0.4)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Burn Rate (mm/s)", fontsize=18)
    ax.set_title(title, fontsize=20)  # not bold
    ax.tick_params(labelsize=14)

# =========================
# SAVE FUNCTION
# =========================

script_dir = os.path.dirname(os.path.abspath(__file__))
ind_dir = os.path.join(script_dir, "individual_plots")
os.makedirs(ind_dir, exist_ok=True)

def save_plot(filename, x, y_m, y_e, labels, title, xlabel, log=False):
    fig, ax = plt.subplots(figsize=(9,7))
    fig.patch.set_facecolor('#f5f5f5')

    draw_plot(ax, x, y_m, y_e, labels, title, xlabel, log)

    fig.tight_layout()
    path = os.path.join(ind_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {path}")

# =========================
# GENERATE ALL PLOTS
# =========================

for pressure, y_m, y_e in [
    ("1 MPa", model_1, exp_1),
    ("7 MPa", model_7, exp_7)
]:

    # --- D_w plots ---
    save_plot(f"Dw_vs_BurnRate_{pressure}_linear.png",
              dw, y_m, y_e, samples,
              f"$D_w$ vs Burn Rate ({pressure})",
              "$D_w$ (μm)", log=False)

    save_plot(f"Dw_vs_BurnRate_{pressure}_loglog.png",
              dw, y_m, y_e, samples,
              f"$D_w$ vs Burn Rate (log-log, {pressure})",
              "$D_w$ (μm)", log=True)

    # Porous D_w
    x_p, ymp, yep, sp = subset("Porous", dw, y_m, y_e)
    save_plot(f"Dw_vs_BurnRate_{pressure}_Porous_linear.png",
              x_p, ymp, yep, sp,
              f"$D_w$ vs Burn Rate – Porous ({pressure})",
              "$D_w$ (μm)", log=False)

    save_plot(f"Dw_vs_BurnRate_{pressure}_Porous_loglog.png",
              x_p, ymp, yep, sp,
              f"$D_w$ vs Burn Rate – Porous (log-log, {pressure})",
              "$D_w$ (μm)", log=True)

    # Solid D_w
    x_s, yms, yes, ss = subset("Solid", dw, y_m, y_e)
    save_plot(f"Dw_vs_BurnRate_{pressure}_Solid_linear.png",
              x_s, yms, yes, ss,
              f"$D_w$ vs Burn Rate – Solid ({pressure})",
              "$D_w$ (μm)", log=False)

    save_plot(f"Dw_vs_BurnRate_{pressure}_Solid_loglog.png",
              x_s, yms, yes, ss,
              f"$D_w$ vs Burn Rate – Solid (log-log, {pressure})",
              "$D_w$ (μm)", log=True)

    # --- S_m plots ---
    save_plot(f"Sm_vs_BurnRate_{pressure}_linear.png",
              Sm, y_m, y_e, samples,
              f"$S_m$ vs Burn Rate ({pressure})",
              "$S_m$ (m²/kg)", log=False)

    save_plot(f"Sm_vs_BurnRate_{pressure}_loglog.png",
              Sm, y_m, y_e, samples,
              f"$S_m$ vs Burn Rate (log-log, {pressure})",
              "$S_m$ (m²/kg)", log=True)

    # Porous S_m
    x_p, ymp, yep, sp = subset("Porous", Sm, y_m, y_e)
    save_plot(f"Sm_vs_BurnRate_{pressure}_Porous_linear.png",
              x_p, ymp, yep, sp,
              f"$S_m$ vs Burn Rate – Porous ({pressure})",
              "$S_m$ (m²/kg)", log=False)

    save_plot(f"Sm_vs_BurnRate_{pressure}_Porous_loglog.png",
              x_p, ymp, yep, sp,
              f"$S_m$ vs Burn Rate – Porous (log-log, {pressure})",
              "$S_m$ (m²/kg)", log=True)

    # Solid S_m
    x_s, yms, yes, ss = subset("Solid", Sm, y_m, y_e)
    save_plot(f"Sm_vs_BurnRate_{pressure}_Solid_linear.png",
              x_s, yms, yes, ss,
              f"$S_m$ vs Burn Rate – Solid ({pressure})",
              "$S_m$ (m²/kg)", log=False)

    save_plot(f"Sm_vs_BurnRate_{pressure}_Solid_loglog.png",
              x_s, yms, yes, ss,
              f"$S_m$ vs Burn Rate – Solid (log-log, {pressure})",
              "$S_m$ (m²/kg)", log=True)

# =========================
# EXTERNAL LEGEND
# =========================

fig_legend = plt.figure(figsize=(7,2.5))
handles = [
    Line2D([0],[0], marker='o', color=POROUS_COLOR, linestyle='None', markersize=10, label='Porous'),
    Line2D([0],[0], marker='*', color=SOLID_COLOR, linestyle='None', markersize=10, label='Solid'),
    Line2D([0],[0], color=POROUS_COLOR, lw=3, linestyle='-', label='Porous – Model'),
    Line2D([0],[0], color=POROUS_COLOR, lw=3, linestyle='--', alpha=0.6, label='Porous – Experimental'),
    Line2D([0],[0], color=SOLID_COLOR, lw=3, linestyle='-', label='Solid – Model'),
    Line2D([0],[0], color=SOLID_COLOR, lw=3, linestyle='--', alpha=0.6, label='Solid – Experimental')
]

fig_legend.legend(handles=handles, loc='center', ncol=3, fontsize=14)
legend_path = os.path.join(ind_dir, "external_legend.png")
fig_legend.savefig(legend_path, dpi=150, bbox_inches='tight')
plt.close(fig_legend)

print(f"\nAll plots saved → {ind_dir}/")