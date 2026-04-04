#!/usr/bin/env python3

import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── CONFIG ────────────────────────────────────────────────────────────────────

AP_PATH   = "2D_xyzrs/I_00_AP.xyzr"
VOID_PATH = None #"2D_xyzrs/A_00_void.xyzr"

X_MIN = 73.5e-6
Y_MIN = 73.5e-6
X_MAX = 146.5e-6
Y_MAX = 126.5e-6

AP_COLOR   = "#FF6B6B"
VOID_COLOR = "#FFFFFF"
BG_COLOR   = "#4A90D9"

UM = 1e-6

# ── helpers ───────────────────────────────────────────────────────────────────

def read_xyzr(path):
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path) as f:
        for line in f:
            vals = line.split()
            if len(vals) >= 4:
                x, y, z, r = map(float, vals[:4])
                pts.append((x, y, z, r))
    return pts

def void_inside_any_ap(vx, vy, vr, circles):
    for (x0, y0, z0, r) in circles:
        dist = math.sqrt((vx - x0)**2 + (vy - y0)**2)
        if dist + vr <= r:
            return True
    return False

def draw_2d(ax, circles, voids, x_min, y_min, x_max, y_max):
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.set_xlim(0, (x_max - x_min) / UM)
    ax.set_ylim(0, (y_max - y_min) / UM)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

    # Include APs whose circles overlap the crop window (not just centred in it)
    in_crop = [(x0, y0, z0, r) for (x0, y0, z0, r) in circles
               if x0 + r >= x_min and x0 - r <= x_max
               and y0 + r >= y_min and y0 - r <= y_max]

    for (x0, y0, z0, r) in in_crop:
        ax.add_patch(Circle(((x0 - x_min) / UM, (y0 - y_min) / UM), r / UM,
                            color=AP_COLOR))

    for (vx, vy, vz, vr) in voids:
        if x_min <= vx <= x_max and y_min <= vy <= y_max:
            if void_inside_any_ap(vx, vy, vr, in_crop):
                ax.add_patch(Circle(((vx - x_min) / UM, (vy - y_min) / UM), vr / UM,
                                    color=VOID_COLOR))

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    circles = read_xyzr(AP_PATH)
    voids   = read_xyzr(VOID_PATH) if VOID_PATH is not None else []

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_2d(ax, circles, voids, X_MIN, Y_MIN, X_MAX, Y_MAX)

    plt.tight_layout()
    plt.savefig("2D_xyzrs/I_crop.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()