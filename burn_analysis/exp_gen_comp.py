#!/usr/bin/env python3
"""
Particle Characterization: Burn Rate & Morphology Analysis
Run with:  python particle_analysis.py
Output:    particle_analysis.png  (saved next to this script, then opened)
"""

import os
import sys
import subprocess
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# =========================
# DATA
# =========================

samples = np.array([
    "A","B","C","D","E","F","G","H","I","J","K","R","S","T"
])

groups = {
    "Porous": ["A","B","C","D","E","F"],
    "Hollow": ["G","H"],
    "Solid":  ["I","J","K","R","S","T"],
}

burn_groups = ["Porous", "Solid"]
combined_mwd_sm_samples = groups["Porous"] + groups["Hollow"] + groups["Solid"]

mwd = np.array([
    7.7313, 4.5596, 5.3438, 4.3559, 8.1854, 8.6591,
    3.4167, 3.7836,
    3.0309, 3.7919, 3.6889, 14.8659, 7.8058, 2.8298
])

Sm = np.array([
    597.22, 773.63, 658.95, 855.68, 455.32, 402.26,
    1110.4, 980.05,
    702.22, 561.1,  577.16, 142.36, 272.92, 752.37
])

color_map = {
    "A": (112/255,  66/255,  20/255),
    "B": (128/255,   0/255,   0/255),
    "C": (255/255,   0/255,   0/255),
    "D": (255/255, 140/255,   0/255),
    "E": (218/255, 165/255,  32/255),
    "F": (173/255, 255/255,  47/255),
    "G": ( 50/255, 205/255,  50/255),
    "H": (107/255, 142/255,  35/255),
    "I": (  0/255, 176/255, 240/255),
    "J": (  0/255,   0/255, 128/255),
    "K": (  0/255,   0/255, 255/255),
    "R": (120/255,  81/255, 169/255),
    "S": (199/255,  21/255, 133/255),
    "T": (230/255, 230/255, 250/255),
}

alamo_1 = np.array([2.83, 3.04, 3.25, 2.65, 3.53, 3.70, 4.38, 4.59, 3.75, 3.25, 3.25, 1.62, 2.29, 3.86])
kohga_1 = np.array([3.8,  3.2,  3.6,  3.2,  3.5,  3.0,  4.6,  5.1,  3.4,  3.2,  3.2,  2.2,  2.6,  3.5 ])
alamo_7 = np.array([9.52, 9.79, 9.85, 9.47, 9.83,10.19, 7.59, 7.44, 5.63, 6.00, 5.83, 5.22, 5.63, 5.66])
kohga_7 = np.array([10.1, 7.1,  7.6,  7.2,  7.8,  6.4, 13.1, 12.0,  7.2,  6.8,  6.9,  5.3,  6.0,  7.2 ])

# Per-group fit line colors: solid line = Alamo (ours), dashed = Kohga
FIT_COLORS = {
    "Porous": {"alamo": "#8B4513", "kohga": "#D2691E"},  # dark/light brown
    "Solid":  {"alamo": "#1E3A8A", "kohga": "#4169E1"},  # dark/light blue
}

# =========================
# HELPERS
# =========================

def get_marker(sample):
    if sample in groups["Porous"]:  return 'o'
    elif sample in groups["Hollow"]: return 's'
    return '*'

def fit_line(x, y):
    return np.poly1d(np.polyfit(x, y, 1))

def fit_line_loglog(x, y):
    coeffs = np.polyfit(np.log10(x), np.log10(y), 1)
    def power_law(xv):
        return 10 ** np.poly1d(coeffs)(np.log10(xv))
    return power_law

def get_xy_data(sample_list, x, y):
    idx = [i for i, s in enumerate(samples) if s in sample_list]
    x_g, y_g, s_g = x[idx], y[idx], samples[idx]
    valid = ~np.isnan(x_g) & ~np.isnan(y_g)
    return x_g[valid], y_g[valid], s_g[valid]

# =========================
# DRAW HELPERS
# =========================

LEGEND_ELEMENTS = None  # set after FIT_COLORS is defined, used by standalone saves

def _add_legend(ax, fontsize=4.5, ncol=2):
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l,h)).values(), dict(zip(l,h)).keys(),
              fontsize=fontsize, ncol=ncol, loc='best', framealpha=0.6)

def draw_combined_burn_plot(ax, x_all, y_alamo_all, y_kohga_all, labels_all,
                             title, xlabel, log=False):
    """Scatter all samples; draw separate Porous & Solid fit lines for both datasets."""
    for i, s in enumerate(labels_all):
        ax.scatter(x_all[i], y_alamo_all[i],
                   marker=get_marker(s), color=color_map[s], s=60, zorder=3)

    fit_fn = fit_line_loglog if log else fit_line

    for group_name in burn_groups:
        mask = np.array([s in groups[group_name] for s in labels_all])
        if mask.sum() < 2:
            continue
        x_g = x_all[mask]
        fc  = FIT_COLORS[group_name]
        x_fit = np.linspace(min(x_g), max(x_g), 100)

        ax.plot(x_fit, fit_fn(x_g, y_alamo_all[mask])(x_fit),
                '-',  lw=1.8, color=fc["alamo"], label=f"{group_name} – Alamo (Ours)")
        ax.plot(x_fit, fit_fn(x_g, y_kohga_all[mask])(x_fit),
                '--', lw=1.8, color=fc["kohga"], label=f"{group_name} – Kohga")

    if log:
        ax.set_xscale('log'); ax.set_yscale('log')

    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("Burn Rate (mm/s)", fontsize=7)
    ax.set_title(title, fontsize=7.5, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=6)
    _add_legend(ax, fontsize=4.5, ncol=2)


def draw_mwd_sm_plot(ax, sample_list, title, log=False):
    x_g, y_g, s_g = get_xy_data(sample_list, mwd, Sm)
    for i, s in enumerate(s_g):
        ax.scatter(x_g[i], y_g[i],
                   marker=get_marker(s), color=color_map[s], s=60, zorder=3, label=s)
        ax.annotate(s, (x_g[i], y_g[i]),
                    textcoords="offset points", xytext=(4, 3), fontsize=5.5)

    fit_fn = fit_line_loglog if log else fit_line

    for group_name in burn_groups:
        xg, yg, _ = get_xy_data(groups[group_name], mwd, Sm)
        if len(xg) < 2:
            continue
        x_fit = np.linspace(min(xg), max(xg), 100)
        ax.plot(x_fit, fit_fn(xg, yg)(x_fit), '-', lw=1.8,
                color=FIT_COLORS[group_name]["alamo"], alpha=0.85,
                label=f"{group_name} Fit")

    if log:
        ax.set_xscale('log'); ax.set_yscale('log')

    ax.set_xlabel("MWD (μm)", fontsize=7)
    ax.set_ylabel("S/m (m²/kg)", fontsize=7)
    ax.set_title(title, fontsize=7.5, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=6)
    _add_legend(ax, fontsize=5, ncol=2)

# =========================
# BUILD COMBINED BURN ARRAYS (Porous + Solid)
# =========================

burn_sample_list = groups["Porous"] + groups["Solid"]
burn_idx = np.array([i for i, s in enumerate(samples) if s in burn_sample_list])

x_mwd_all = mwd[burn_idx]
x_sm_all  = Sm[burn_idx]
y_a1      = alamo_1[burn_idx]
y_k1      = kohga_1[burn_idx]
y_a7      = alamo_7[burn_idx]
y_k7      = kohga_7[burn_idx]
s_all     = samples[burn_idx]

# =========================
# LAYOUT  (3 rows × 5 cols)
# Row 0 – MWD vs BR  (linear, linear, log-log, log-log, empty)
# Row 1 – S/m vs BR  (linear, linear, log-log, log-log, empty)
# Row 2 – MWD vs S/m combined (empty, linear, empty, log-log, empty)
# =========================

NCOLS = 5
NROWS = 3

fig = plt.figure(figsize=(22, 13))
fig.patch.set_facecolor('#f5f5f5')
fig.suptitle("Particle Characterization: Burn Rate & Morphology Analysis",
             fontsize=14, fontweight='bold', y=0.995)

gs = gridspec.GridSpec(NROWS, NCOLS, figure=fig,
                       hspace=0.52, wspace=0.35,
                       left=0.05, right=0.98,
                       top=0.97, bottom=0.08)

# Row 0: MWD vs Burn Rate
draw_combined_burn_plot(fig.add_subplot(gs[0,0]), x_mwd_all, y_a1, y_k1, s_all, "MWD vs BR (1 MPa)",          "MWD (μm)")
draw_combined_burn_plot(fig.add_subplot(gs[0,1]), x_mwd_all, y_a7, y_k7, s_all, "MWD vs BR (7 MPa)",          "MWD (μm)")
draw_combined_burn_plot(fig.add_subplot(gs[0,2]), x_mwd_all, y_a1, y_k1, s_all, "MWD vs BR log-log (1 MPa)",  "MWD (μm)", log=True)
draw_combined_burn_plot(fig.add_subplot(gs[0,3]), x_mwd_all, y_a7, y_k7, s_all, "MWD vs BR log-log (7 MPa)",  "MWD (μm)", log=True)
fig.add_subplot(gs[0,4]).set_visible(False)

# Row 1: S/m vs Burn Rate
draw_combined_burn_plot(fig.add_subplot(gs[1,0]), x_sm_all, y_a1, y_k1, s_all, "S/m vs BR (1 MPa)",          "S/m (m²/kg)")
draw_combined_burn_plot(fig.add_subplot(gs[1,1]), x_sm_all, y_a7, y_k7, s_all, "S/m vs BR (7 MPa)",          "S/m (m²/kg)")
draw_combined_burn_plot(fig.add_subplot(gs[1,2]), x_sm_all, y_a1, y_k1, s_all, "S/m vs BR log-log (1 MPa)",  "S/m (m²/kg)", log=True)
draw_combined_burn_plot(fig.add_subplot(gs[1,3]), x_sm_all, y_a7, y_k7, s_all, "S/m vs BR log-log (7 MPa)",  "S/m (m²/kg)", log=True)
fig.add_subplot(gs[1,4]).set_visible(False)

# Row 2: combined MWD vs S/m
draw_mwd_sm_plot(fig.add_subplot(gs[2,1]), combined_mwd_sm_samples,
                 "Porous + Hollow + Solid: MWD vs S/m", log=False)
draw_mwd_sm_plot(fig.add_subplot(gs[2,3]), combined_mwd_sm_samples,
                 "Porous + Hollow + Solid: MWD vs S/m (log-log)", log=True)
for col in (0, 2, 4):
    fig.add_subplot(gs[2,col]).set_visible(False)

# =========================
# GLOBAL LEGEND ELEMENTS (reused for individual plots)
# =========================
LEGEND_ELEMENTS = [
    Line2D([0],[0], marker='o', color='gray', linestyle='None', markersize=7, label='Porous'),
    Line2D([0],[0], marker='*', color='gray', linestyle='None', markersize=9, label='Solid'),
    Line2D([0],[0], marker='s', color='gray', linestyle='None', markersize=7, label='Hollow (S/m plots only)'),
    Line2D([0],[0], color=FIT_COLORS["Porous"]["alamo"], lw=2, linestyle='-',  label='Porous – Alamo (Ours)'),
    Line2D([0],[0], color=FIT_COLORS["Porous"]["kohga"], lw=2, linestyle='--', label='Porous – Kohga'),
    Line2D([0],[0], color=FIT_COLORS["Solid"]["alamo"],  lw=2, linestyle='-',  label='Solid – Alamo (Ours)'),
    Line2D([0],[0], color=FIT_COLORS["Solid"]["kohga"],  lw=2, linestyle='--', label='Solid – Kohga'),
]

fig.legend(handles=LEGEND_ELEMENTS, loc='lower center',
           ncol=7, fontsize=7.5, framealpha=0.85,
           bbox_to_anchor=(0.5, 0.0), borderpad=0.6)

# =========================
# SAVE COMBINED & OPEN
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, "particle_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
plt.close(fig)

# =========================
# SAVE INDIVIDUAL PLOTS
# =========================

# Helper to make a standalone figure with one plot and a shared legend at bottom
def save_individual(filename, draw_fn, *draw_args, figsize=(8, 6), **draw_kwargs):
    fig_i, ax_i = plt.subplots(figsize=figsize)
    fig_i.patch.set_facecolor('#f5f5f5')
    draw_fn(ax_i, *draw_args, **draw_kwargs)
    fig_i.legend(handles=LEGEND_ELEMENTS, loc='lower center',
                 ncol=4, fontsize=7, framealpha=0.85,
                 bbox_to_anchor=(0.5, 0.0), borderpad=0.5)
    fig_i.tight_layout(rect=[0, 0.13, 1, 1])
    p = os.path.join(script_dir, filename)
    fig_i.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig_i.get_facecolor())
    print(f"Saved → {p}")
    plt.close(fig_i)

# Individual plots directory
ind_dir = os.path.join(script_dir, "individual_plots")
os.makedirs(ind_dir, exist_ok=True)

# Burn-rate plots — MWD x-axis
for pressure, y_a, y_k in [("1MPa", y_a1, y_k1), ("7MPa", y_a7, y_k7)]:
    save_individual(
        os.path.join(ind_dir, f"MWD_vs_BR_{pressure}.png"),
        draw_combined_burn_plot,
        x_mwd_all, y_a, y_k, s_all, f"MWD vs BR ({pressure.replace('MPa',' MPa')})", "MWD (μm)"
    )
    save_individual(
        os.path.join(ind_dir, f"MWD_vs_BR_{pressure}_loglog.png"),
        draw_combined_burn_plot,
        x_mwd_all, y_a, y_k, s_all, f"MWD vs BR log-log ({pressure.replace('MPa',' MPa')})", "MWD (μm)",
        log=True
    )

# Burn-rate plots — S/m x-axis
for pressure, y_a, y_k in [("1MPa", y_a1, y_k1), ("7MPa", y_a7, y_k7)]:
    save_individual(
        os.path.join(ind_dir, f"Sm_vs_BR_{pressure}.png"),
        draw_combined_burn_plot,
        x_sm_all, y_a, y_k, s_all, f"S/m vs BR ({pressure.replace('MPa',' MPa')})", "S/m (m²/kg)"
    )
    save_individual(
        os.path.join(ind_dir, f"Sm_vs_BR_{pressure}_loglog.png"),
        draw_combined_burn_plot,
        x_sm_all, y_a, y_k, s_all, f"S/m vs BR log-log ({pressure.replace('MPa',' MPa')})", "S/m (m²/kg)",
        log=True
    )

# MWD vs S/m plots
save_individual(
    os.path.join(ind_dir, "MWD_vs_Sm.png"),
    draw_mwd_sm_plot,
    combined_mwd_sm_samples, "Porous + Hollow + Solid: MWD vs S/m"
)
save_individual(
    os.path.join(ind_dir, "MWD_vs_Sm_loglog.png"),
    draw_mwd_sm_plot,
    combined_mwd_sm_samples, "Porous + Hollow + Solid: MWD vs S/m (log-log)",
    log=True
)

print(f"\nAll individual plots saved to → {ind_dir}/")

if sys.platform.startswith("darwin"):
    subprocess.Popen(["open", out_path])
elif sys.platform.startswith("win"):
    os.startfile(out_path)
else:
    subprocess.Popen(["xdg-open", out_path])