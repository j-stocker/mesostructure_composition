#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── CONFIG ────────────────────────────────────────────────────────────────────

AP_PATH   = "3D_xyzrs/example_00_AP.xyzr"
VOID_PATH = "3D_xyzrs/example_00_void.xyzr"

PHYSICAL_SIZE = 100e-6
IMG_SIZE      = 1.0

SLICE_AXIS  = "z"
SLICE_POS   = 0.7

SPHERE_RESOLUTION = 25
ELEV   = 25
AZIM   = 45

AP_COLOR    = "#FF6B6B"
VOID_COLOR  = "#939494"
PLANE_COLOR = "#FFD700"

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
        verts = [(pos, lo, lo), (pos, hi, lo),
                 (pos, hi, hi), (pos, lo, hi)]
    elif axis == "y":
        verts = [(lo, pos, lo), (hi, pos, lo),
                 (hi, pos, hi), (lo, pos, hi)]
    else:
        verts = [(lo, lo, pos), (hi, lo, pos),
                 (hi, hi, pos), (lo, hi, pos)]

    ax.add_collection3d(Poly3DCollection(
        [verts], facecolor=PLANE_COLOR, alpha=0.6, edgecolor="none"
    ))

def style_3d(ax):
    # Original plotting
    ax.set_xlim(0, IMG_SIZE)
    ax.set_ylim(0, IMG_SIZE)
    ax.set_zlim(0, IMG_SIZE)
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=ELEV, azim=AZIM)

    # Ticks 0, 0.2, ..., 1
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

# ── 3D DRAW (AP ONLY) ─────────────────────────────────────────────────────────

def draw_3d(ax, circles):
    domain = (0, IMG_SIZE)

    # Back half
    for (x0,y0,z0,r) in circles:
        x,y,z = make_sphere_mesh_half(
            x0,y0,z0,r,SPHERE_RESOLUTION,domain,
            SLICE_AXIS,SLICE_POS,"below"
        )
        ax.plot_surface(x,y,z,color=AP_COLOR,alpha=AP_ALPHA_3D,linewidth=0)

    # Plane
    draw_plane(ax, SLICE_AXIS, SLICE_POS, IMG_SIZE)

    # Front half
    for (x0,y0,z0,r) in circles:
        x,y,z = make_sphere_mesh_half(
            x0,y0,z0,r,SPHERE_RESOLUTION,domain,
            SLICE_AXIS,SLICE_POS,"above"
        )
        ax.plot_surface(x,y,z,color=AP_COLOR,alpha=AP_ALPHA_3D,linewidth=0)

# ── 2D SLICE (AP + VOIDS) ─────────────────────────────────────────────────────
# ── 2D SLICE (AP + VOIDS) ─────────────────────────────────────────────────────
def draw_2d(ax, circles, voids):
    ax.set_aspect("equal")
    # Flip both axes: 1 → 0
    ax.set_xlim(IMG_SIZE, 0)
    ax.set_ylim(IMG_SIZE, 0)

    # Draw AP particles
    for (x0, y0, z0, r) in circles:
        res = slice_circle(x0, y0, z0, r, "z", SLICE_POS)
        if res:
            cx, cy, cr = res
            ax.add_patch(Circle((cx, cy), cr, color=AP_COLOR))

    # Draw voids
    for (x0, y0, z0, r) in voids:
        res = slice_circle(x0, y0, z0, r, "z", SLICE_POS)
        if res:
            cx, cy, cr = res
            ax.add_patch(Circle((cx, cy), cr, color=VOID_COLOR))

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    circles = read_xyzr(AP_PATH, PHYSICAL_SIZE, IMG_SIZE)
    voids   = read_xyzr(VOID_PATH, PHYSICAL_SIZE, IMG_SIZE)

    fig = plt.figure(figsize=(14,6))

    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)

    draw_3d(ax3d, circles)
    style_3d(ax3d)

    draw_2d(ax2d, circles, voids)

    plt.tight_layout()
    plt.show()
    plt.savefig("3D_xyzrs/example.png", dpi=300)

if __name__ == "__main__":
    main()