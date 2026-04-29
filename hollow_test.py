#!/usr/bin/env python3
"""
Standalone hollow void placement.

Reads an existing AP .xyzr file (grains already packed), selects grains
that fall within a target radius range, and places a single hollow void
at the centre of each selected grain.  Void radius is scaled so that the
cumulative void volume matches void_fraction * domain_volume.

Accounting mirrors sphere_packing v9:
  - grain volume tracked with clipped_sphere_volume / fully_inside
  - void  volume tracked as full sphere  4/3 * pi * rv**3
  - rv derived from actual candidate volume (not a passed-in parameter)
  - radii processed largest-first
"""

import math
import numpy as np
import os


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_xyzr_m(filepath):
    """Read .xyzr file, return (N,4) array in metres."""
    rows = []
    with open(filepath) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                rows.append([float(p) for p in parts[:4]])
            except ValueError:
                continue
    return np.array(rows, dtype=float) if rows else np.zeros((0, 4))


def write_xyzr_m(filepath, xyzr_m):
    """Write (N,4) array in metres to .xyzr file."""
    lines = [f"{x:.8e} {y:.8e} {z:.8e} {r:.8e}\n"
             for x, y, z, r in xyzr_m]
    with open(filepath, "w") as fh:
        fh.writelines(lines)


# ---------------------------------------------------------------------------
# Geometry helpers  (work in normalised [0, img_size] space)
# ---------------------------------------------------------------------------

def _clipped_sphere_volume(x, y, z, r, lo=0.0, hi=1.0):
    def cap(h):
        if h <= 0:   return 0.0
        if h >= 2*r: return 4/3 * math.pi * r**3
        return math.pi * h**2 * (3*r - h) / 3
    V  = 4/3 * math.pi * r**3
    V -= cap(r - (x - lo));  V -= cap(r - (hi - x))
    V -= cap(r - (y - lo));  V -= cap(r - (hi - y))
    V -= cap(r - (z - lo));  V -= cap(r - (hi - z))
    return max(V, 0.0)


def _fully_inside(x, y, z, r, img_size=1.0):
    return (x > r and x < img_size - r and
            y > r and y < img_size - r and
            z > r and z < img_size - r)


def _grain_vol(x, y, z, r, img_size=1.0):
    if _fully_inside(x, y, z, r, img_size):
        return 4/3 * math.pi * r**3
    return _clipped_sphere_volume(x, y, z, r, lo=0.0, hi=img_size)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def place_hollow_voids(
    ap_xyzr_path: str,
    void_xyzr_out: str,
    physical_size: float,
    void_fraction: float,
    mean_rad_hollow: float,
    rad_dev: float = 0.4,
    img_size: float = 1.0,
    min_grain_r_factor: float = 0.5,
    max_grain_r_factor: float = 2.0,
) -> float:
    """
    Place hollow voids into grains read from *ap_xyzr_path*.

    The rv/r ratio is derived entirely from the actual volume of the selected
    candidate grains — no vol_percent_hollow parameter needed.

    Parameters
    ----------
    ap_xyzr_path      : path to existing AP .xyzr file (metres)
    void_xyzr_out     : output path for void .xyzr file (metres)
    physical_size     : domain side length in metres
    void_fraction     : target void volume / domain volume  (e.g. 0.14)
    mean_rad_hollow   : mean grain radius used during AP generation (metres)
                        — used to define the eligible radius window
    rad_dev           : lognormal sigma parameter (default 0.4, matches packer)
    img_size          : normalised domain size (default 1.0, matches packer)
    min_grain_r_factor: grains with r < mean_r_norm * factor are skipped
    max_grain_r_factor: grains with r > mean_r_norm * factor are skipped

    Returns
    -------
    achieved void fraction (float)
    """

    # ---- load AP grains --------------------------------------------------
    ap_m = read_xyzr_m(ap_xyzr_path)
    if len(ap_m) == 0:
        raise ValueError(f"No grains found in {ap_xyzr_path}")

    scale            = img_size / physical_size
    ap_norm          = ap_m * scale
    total_domain_vol = img_size ** 3

    # ---- derive eligible radius window -----------------------------------
    mean_r_norm = mean_rad_hollow * scale
    r_min       = mean_r_norm * min_grain_r_factor
    r_max       = mean_r_norm * max_grain_r_factor

    # ---- select & sort candidates largest-first --------------------------
    candidates = []
    for row in ap_norm:
        x, y, z, r = row
        if r_min <= r <= r_max:
            candidates.append((x, y, z, r))
    candidates.sort(key=lambda c: c[3], reverse=True)

    if len(candidates) == 0:
        raise ValueError(
            f"No grains found in radius window [{r_min:.4f}, {r_max:.4f}] "
            f"(normalised). Check mean_rad_hollow and min/max_grain_r_factor."
        )

    # ---- compute actual hollow fraction from candidates upfront ----------
    total_candidate_vol = sum(
        _grain_vol(x, y, z, r, img_size) for x, y, z, r in candidates
    )
    actual_hollow_frac = total_candidate_vol / total_domain_vol
    rv_r_ratio         = (void_fraction / actual_hollow_frac) ** (1/3)

    print(f"\n{'='*56}")
    print(f"  Hollow void placement")
    print(f"  AP grains loaded  : {len(ap_norm)}")
    print(f"  Eligible grains   : {len(candidates)}  "
          f"(r in [{r_min:.4f}, {r_max:.4f}] norm units)")
    print(f"  Actual hollow frac: {actual_hollow_frac:.4f}")
    print(f"  rv / r ratio      : {rv_r_ratio:.4f}")
    print(f"  Target void frac  : {void_fraction:.4f}")
    print(f"{'='*56}")

    # ---- place one void per candidate ------------------------------------
    voids_norm       = []
    current_void_vol = 0.0

    for x, y, z, r in candidates:
        rv    = r * rv_r_ratio
        v_vol = 4/3 * math.pi * rv**3          # full sphere (matches packer)
        current_void_vol += v_vol
        voids_norm.append((x, y, z, rv))

    achieved = current_void_vol / total_domain_vol

    print(f"  Voids placed      : {len(voids_norm)}")
    print(f"  Void frac achieved: {achieved:.4f}")
    print(f"  Error             : {abs(achieved - void_fraction):.2e}")

    # ---- write output in metres ------------------------------------------
    inv_scale = physical_size / img_size
    voids_m   = [(x*inv_scale, y*inv_scale, z*inv_scale, rv*inv_scale)
                 for x, y, z, rv in voids_norm]
    write_xyzr_m(void_xyzr_out, voids_m)
    print(f"  Written to        : {void_xyzr_out}")
    print(f"{'='*56}\n")

    return achieved


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    achieved = place_hollow_voids(
        ap_xyzr_path  = "3D_xyzrs/G_00_AP.xyzr",
        void_xyzr_out = "3D_xyzrs/G_00_void.xyzr",
        physical_size = 50e-6,
        void_fraction = 0.14,
        mean_rad_hollow = 1.9e-6 / (1.2 * math.exp(math.sqrt(math.log(1 + 0.4**2))**2)),
        rad_dev       = 0.4,
    )