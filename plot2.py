#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches

# ── CONFIG ────────────────────────────────────────────────────────────────────

AP_PATH   = "3D_xyzrs/test3_00_AP.xyzr"
VOID_PATH = "3D_xyzrs/test3_00_void.xyzr"

PHYSICAL_SIZE = 20e-6
IMG_SIZE      = 1.0

SLICE_AXIS  = "z"
SLICE_POS   = 0.4

SPHERE_RESOLUTION = 5
ELEV   = 25
AZIM   = 45

AP_COLOR    = "#FF6B6B"   # red
VOID_COLOR  = "#939494"   # grey
PLANE_COLOR = "#FFD700"
BG_COLOR    = "#4A90D9"  # red background for layered plot
AP_FILL     = "#FF6B6B"   # blue for AP solid region

AP_ALPHA_3D = 1.0

# ── helpers ───────────────────────────────────────────────────────────────────

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

def make_sphere_mesh_half(x0, y0, z0, r, resolution, domain,
                          axis, pos, keep):
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)

    x = x0 + r*np.cos(u)*np.sin(v)
    y = y0 + r*np.sin(u)*np.sin(v)
    z = z0 + r*np.cos(v)

    lo, hi = domain
    mask = ((x < lo) | (x > hi) |
            (y < lo) | (y > hi) |
            (z < lo) | (z > hi))

    coord = {"x": x, "y": y, "z": z}[axis]
    if keep == "below":
        mask |= (coord > pos)
    else:
        mask |= (coord < pos)

    x[mask] = np.nan
    y[mask] = np.nan
    z[mask] = np.nan
    return x, y, z

def draw_plane(ax, axis, pos, img_size):
    lo, hi = 0, img_size
    if axis == "x":
        verts = [(pos, lo, lo), (pos, hi, lo), (pos, hi, hi), (pos, lo, hi)]
    elif axis == "y":
        verts = [(lo, pos, lo), (hi, pos, lo), (hi, pos, hi), (lo, pos, hi)]
    else:
        verts = [(lo, lo, pos), (hi, lo, pos), (hi, hi, pos), (lo, hi, pos)]

    ax.add_collection3d(Poly3DCollection(
        [verts], facecolor=PLANE_COLOR, alpha=0.6, edgecolor="none"
    ))

def style_3d(ax):
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(0, IMG_SIZE)
    ax.set_zlim(0, IMG_SIZE)
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=ELEV, azim=AZIM)
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

def slice_circle(x0, y0, z0, r, axis, pos):
    if axis == "x":
        d = abs(pos - x0); cx, cy = y0, z0
    elif axis == "y":
        d = abs(pos - y0); cx, cy = x0, z0
    else:
        d = abs(pos - z0); cx, cy = x0, y0

    if d >= r:
        return None
    return cx, cy, np.sqrt(r**2 - d**2)

def circles_intersect(cx1, cy1, cr1, cx2, cy2, cr2):
    """Check if two 2D circles overlap."""
    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    return dist < (cr1 + cr2)

# ── 3D DRAW ───────────────────────────────────────────────────────────────────

def draw_3d(ax, circles):
    domain = (0, IMG_SIZE)

    for (x0, y0, z0, r) in circles:
        x, y, z = make_sphere_mesh_half(
            x0, y0, z0, r, SPHERE_RESOLUTION, domain,
            SLICE_AXIS, SLICE_POS, "below"
        )
        ax.plot_surface(x, y, z, color=AP_COLOR, alpha=AP_ALPHA_3D, linewidth=0)

    draw_plane(ax, SLICE_AXIS, SLICE_POS, IMG_SIZE)

    for (x0, y0, z0, r) in circles:
        x, y, z = make_sphere_mesh_half(
            x0, y0, z0, r, SPHERE_RESOLUTION, domain,
            SLICE_AXIS, SLICE_POS, "above"
        )
        ax.plot_surface(x, y, z, color=AP_COLOR, alpha=AP_ALPHA_3D, linewidth=0)

# ── 2D SLICE: original ────────────────────────────────────────────────────────

def draw_2d(ax, circles, voids):
    ax.set_aspect("equal")
    ax.set_xlim(IMG_SIZE, 0)
    ax.set_ylim(IMG_SIZE, 0)

    for (x0, y0, z0, r) in circles:
        res = slice_circle(x0, y0, z0, r, "z", SLICE_POS)
        if res:
            cx, cy, cr = res
            ax.add_patch(Circle((cx, cy), cr, color=AP_COLOR))

    for (x0, y0, z0, r) in voids:
        res = slice_circle(x0, y0, z0, r, "z", SLICE_POS)
        if res:
            cx, cy, cr = res
            ax.add_patch(Circle((cx, cy), cr, color=VOID_COLOR))

# ── 2D SLICE: layered ─────────────────────────────────────────────────────────

def draw_2d_layered(ax, circles, voids):
    """
    Layer 1: solid red background
    Layer 2: blue filled AP circles
    Layer 3: grey void circles clipped to only show inside AP circles
    """
    ax.set_aspect("equal")
    ax.set_xlim(IMG_SIZE, 0)
    ax.set_ylim(IMG_SIZE, 0)

    # Layer 1: red background
    ax.add_patch(Rectangle((0, 0), IMG_SIZE, IMG_SIZE,
                            color=BG_COLOR, zorder=0))

    # Slice all AP and void circles at this z-plane
    ap_sliced = []
    for (x0, y0, z0, r) in circles:
        res = slice_circle(x0, y0, z0, r, "z", SLICE_POS)
        if res:
            ap_sliced.append(res)  # (cx, cy, cr)

    void_sliced = []
    for (x0, y0, z0, r) in voids:
        res = slice_circle(x0, y0, z0, r, "z", SLICE_POS)
        if res:
            void_sliced.append(res)  # (cx, cy, cr)

    # Layer 2: red AP circles
    for (cx, cy, cr) in ap_sliced:
        ax.add_patch(Circle((cx, cy), cr, color=AP_FILL, zorder=1, linewidth=0))

    # Layer 3: grey voids — only where they overlap an AP circle
    # We use clipping: draw each void circle clipped to each AP circle it intersects
    for (vcx, vcy, vcr) in void_sliced:
        for (acx, acy, acr) in ap_sliced:
            if circles_intersect(vcx, vcy, vcr, acx, acy, acr):
                # Draw the void circle with a clip path set to the AP circle
                void_patch = Circle((vcx, vcy), vcr, color=VOID_COLOR, zorder=2)
                clip_patch = Circle((acx, acy), acr, transform=ax.transData)
                ax.add_patch(void_patch)
                void_patch.set_clip_path(clip_patch)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    circles = read_xyzr(AP_PATH, PHYSICAL_SIZE, IMG_SIZE)
    voids   = read_xyzr(VOID_PATH, PHYSICAL_SIZE, IMG_SIZE)

    fig = plt.figure(figsize=(20, 6))

    # 3D plot
    ax3d = fig.add_subplot(131, projection='3d')
    draw_3d(ax3d, circles)
    style_3d(ax3d)
    ax3d.set_title("3D view")

    # Original 2D slice
    ax2d = fig.add_subplot(132)
    draw_2d(ax2d, circles, voids)
    ax2d.set_title("2D slice (original)")

    # Layered 2D slice
    ax2d_l = fig.add_subplot(133)
    draw_2d_layered(ax2d_l, circles, voids)
    ax2d_l.set_title("2D slice (layered)")

    plt.tight_layout()
    plt.savefig("3D_xyzrs/example.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()