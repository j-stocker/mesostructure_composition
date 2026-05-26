#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# ── CONFIG ─────────────────────────────────────────────

AP_PATH   = "test_files/nonvoid_ap55_vf00_00_AP.xyzr"
VOID_PATH = "test_files/nonvoid_ap55_vf00_00_void.xyzr"
OUTPUT_IMAGE = "test_files/images/nonvoid_ap55_vf00.png"

PHYSICAL_SIZE = 50e-6
IMG_SIZE      = 50.0

SLICE_AXIS = "z"
SLICE_POS  = 20 #um

MODE = "overlay"  # "clipped", "overlay", or "htpb_clipped"


AP_COLOR   = "#FF6B6B"
VOID_COLOR = "#000000"
BG_COLOR   = "#4A90D9"

TITLE = "No Pores"

# ── LOAD ───────────────────────────────────────────────

def read_xyzr(path, physical_size, img_size):
    pts = []
    if not os.path.exists(path):
        return pts
    scale = img_size / physical_size
    with open(path) as f:
        for line in f:
            vals = line.split()
            if len(vals) >= 4:
                x, y, z, r = map(float, vals[:4])
                pts.append((x*scale, y*scale, z*scale, r*scale))
    return pts

# ── SLICE ──────────────────────────────────────────────

def slice_circle(x0, y0, z0, r, axis, pos):
    if axis == "x":
        d = abs(pos - x0); cx, cy = y0, z0
    elif axis == "y":
        d = abs(pos - y0); cx, cy = x0, z0
    else:
        d = abs(pos - z0); cx, cy = x0, y0

    if d >= r:
        return None

    return cx, cy, np.sqrt(r*r - d*d)

def circles_intersect(cx1, cy1, cr1, cx2, cy2, cr2):
    return np.hypot(cx1 - cx2, cy1 - cy2) < (cr1 + cr2)

# ── DRAW ───────────────────────────────────────────────

def draw(ax, ap, voids):
    ax.set_aspect("equal")
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(0, IMG_SIZE)

    # background
    ax.add_patch(Rectangle((0, 0), IMG_SIZE, IMG_SIZE,
                           color=BG_COLOR, zorder=0))

    # ── slice AP + voids ─────────────────────────────
    ap_sliced = []
    void_sliced = []

    for x0, y0, z0, r in ap:
        res = slice_circle(x0, y0, z0, r, SLICE_AXIS, SLICE_POS)
        if res:
            ap_sliced.append(res)

    for x0, y0, z0, r in voids:
        res = slice_circle(x0, y0, z0, r, SLICE_AXIS, SLICE_POS)
        if res:
            void_sliced.append(res)

    # ── LAYER 1: AP ──────────────────────────────────
    for x, y, r in ap_sliced:
        ax.add_patch(Circle((x, y), r,
                            facecolor=AP_COLOR,
                            edgecolor="none",
                            zorder=1))

    # ── LAYER 2: VOIDS (MODE SWITCH) ────────────────
    if MODE == "clipped":
        for vx, vy, vr in void_sliced:
            for axc, ayc, ar in ap_sliced:
                if circles_intersect(vx, vy, vr, axc, ayc, ar):
                    clip = Circle((axc, ayc), ar, transform=ax.transData)
                    p = Circle((vx, vy), vr,
                               facecolor=VOID_COLOR,
                               edgecolor="none",
                               zorder=2)
                    ax.add_patch(p)
                    p.set_clip_path(clip)

    elif MODE == "overlay":
        for x, y, r in void_sliced:
            ax.add_patch(Circle((x, y), r,
                                facecolor=VOID_COLOR,
                                edgecolor="none",
                                zorder=2))

    # ── LAYER 2: VOIDS (MODE SWITCH) ────────────────



    elif MODE == "htpb_clipped":
        # 🔥 CORRECT binder-only behavior:
        # remove voids that intersect ANY AP grain

        filtered_voids = []

        for vx, vy, vr in void_sliced:
            intersects_ap = False
            for axc, ayc, ar in ap_sliced:
                if circles_intersect(vx, vy, vr, axc, ayc, ar):
                    intersects_ap = True
                    break

            if not intersects_ap:
                filtered_voids.append((vx, vy, vr))

        # draw remaining binder voids
        for x, y, r in filtered_voids:
            ax.add_patch(Circle((x, y), r,
                                facecolor=VOID_COLOR,
                                edgecolor="none",
                                zorder=2))

# ── MAIN ───────────────────────────────────────────────

def main():
    ap = read_xyzr(AP_PATH, PHYSICAL_SIZE, IMG_SIZE)
    voids = read_xyzr(VOID_PATH, PHYSICAL_SIZE, IMG_SIZE)

    fig, ax = plt.subplots(figsize=(6, 6))

    draw(ax, ap, voids)

    ax.set_title(TITLE, fontsize=20)

    # ── µm AXES (no geometry change) ─────────────────
    ticks = np.linspace(0, IMG_SIZE, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels([f"{int(x)}" for x in np.linspace(0, 50, 6)], fontsize=14)
    ax.set_yticklabels([f"{int(y)}" for y in np.linspace(0, 50, 6)], fontsize=14)

    ax.set_xlabel("X (µm)", fontsize=16)
    ax.set_ylabel("Y (µm)", fontsize=16)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()