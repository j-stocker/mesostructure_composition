#!/usr/bin/env python3

import os
import numpy as np
from tabulate import tabulate
import math

OUTPUT_FILE = "void_statistics_3D.txt"

DENSITY_KG_M3 = 1.95 * 1000  # 1950 kg/m³

# ── VOID MODE ─────────────────────────────────────────────────────────────────
# "unclipped" : use raw void sphere volumes/areas (original behaviour)
# "clipped"   : only count the portion of each void that is inside an AP grain
#               (void ∩ AP intersection volume/area); void material outside all
#               AP grains is ignored entirely.
VOID_MODE = "unclipped"
# ─────────────────────────────────────────────────────────────────────────────

MANUAL_VALUES = {
    "A_vf_11": 2100,
}

DOMAIN_VOLUME_UM3 = 50 * 50 * 50
DOMAIN_SIZE_UM    = 50


def read_xyzr(filename, lo=0.0, hi=DOMAIN_SIZE_UM):
    rows = []
    with open(filename, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z, r = [float(p) * 1e6 for p in parts[:4]]  # m → µm
                if (x + r > lo and x - r < hi and
                    y + r > lo and y - r < hi and
                    z + r > lo and z - r < hi):
                    rows.append((x, y, z, r))
            except ValueError:
                continue
    return np.array(rows, dtype=float) if rows else np.zeros((0, 4))


def cap_vol(h, r):
    """Volume of a spherical cap of height h on a sphere of radius r."""
    h = np.asarray(h, dtype=float)
    return np.where(h <= 0, 0.0,
           np.where(h >= 2*r, 4/3 * np.pi * r**3,
                    np.pi * h**2 * (3*r - h) / 3))


def clipped_sphere_volumes(xyzr, lo=0.0, hi=None):
    """
    Clipped volume for each sphere against the domain box [lo, hi]^3.
    """
    if hi is None:
        hi = DOMAIN_SIZE_UM
    x, y, z, r = xyzr[:, 0], xyzr[:, 1], xyzr[:, 2], xyzr[:, 3]
    V = 4/3 * np.pi * r**3
    V -= cap_vol(r - (x - lo), r)
    V -= cap_vol(r - (hi - x), r)
    V -= cap_vol(r - (y - lo), r)
    V -= cap_vol(r - (hi - y), r)
    V -= cap_vol(r - (z - lo), r)
    V -= cap_vol(r - (hi - z), r)
    return np.maximum(V, 0.0)


def clipped_surface_areas(xyzr, lo=0.0, hi=None):
    """
    Exposed surface area for each sphere clipped to [lo, hi]^3.
    """
    if hi is None:
        hi = DOMAIN_SIZE_UM
    x, y, z, r = xyzr[:, 0], xyzr[:, 1], xyzr[:, 2], xyzr[:, 3]
    A = 4 * np.pi * r**2

    def buried_cap_area(h, r):
        h = np.asarray(h, dtype=float)
        return np.where(h <= 0, 0.0,
               np.where(h >= 2*r, 4 * np.pi * r**2,
                        2 * np.pi * r * h))

    A -= buried_cap_area(r - (x - lo), r)
    A -= buried_cap_area(r - (hi - x), r)
    A -= buried_cap_area(r - (y - lo), r)
    A -= buried_cap_area(r - (hi - y), r)
    A -= buried_cap_area(r - (z - lo), r)
    A -= buried_cap_area(r - (hi - z), r)
    return np.maximum(A, 0.0)


# ── Sphere–sphere intersection geometry ──────────────────────────────────────

def sphere_sphere_intersection_volume(d, r1, r2):
    """
    Volume of the intersection (lens) of two spheres.
    d  = distance between centers
    r1, r2 = radii
    All inputs are scalars or broadcastable arrays.
    """
    d  = np.asarray(d,  dtype=float)
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    fully_contained = d <= np.abs(r1 - r2)
    no_overlap      = d >= r1 + r2

    d_safe = np.where(d > 0, d, 1.0)
    h1     = (r1**2 - r2**2 + d**2) / (2.0 * d_safe)
    h1_cap = r1 - h1
    h2_cap = r2 - (d - h1)

    def cap_v(R, h):
        h = np.maximum(h, 0.0)
        return np.pi * h**2 * (3*R - h) / 3.0

    lens_vol = cap_v(r1, h1_cap) + cap_v(r2, h2_cap)

    small_r  = np.minimum(r1, r2)
    full_vol = 4/3 * np.pi * small_r**3

    vol = np.where(no_overlap, 0.0,
          np.where(fully_contained, full_vol,
                   lens_vol))
    return vol


def sphere_sphere_intersection_surface(d, r_void, r_ap):
    """
    Surface area of the void sphere that lies INSIDE the AP sphere.
    """
    d      = np.asarray(d,      dtype=float)
    r_void = np.asarray(r_void, dtype=float)
    r_ap   = np.asarray(r_ap,   dtype=float)

    no_overlap     = d >= r_void + r_ap
    void_inside_ap = d + r_void <= r_ap

    d_safe   = np.where(d > 0, d, 1.0)
    h1       = (r_void**2 - r_ap**2 + d**2) / (2.0 * d_safe)
    h_cap    = np.maximum(r_void - h1, 0.0)
    cap_area = 2 * np.pi * r_void * h_cap
    full_area = 4 * np.pi * r_void**2

    area = np.where(no_overlap, 0.0,
           np.where(void_inside_ap, full_area,
                    cap_area))
    return area


# ── Aggregate void metrics with AP-intersection clipping ─────────────────────

def void_metrics_clipped(void_xyzr, AP_xyzr, chunk_size=500):
    """
    For each void sphere, sum its intersection volume and interior surface area
    across ALL overlapping AP grains, then cap each void's contribution at its
    full sphere volume/area to avoid double-counting at grain boundaries.

    Processed in chunks of `chunk_size` voids to keep memory bounded.
    Returns (total_volume_um3, total_surface_area_um2).
    """
    if len(void_xyzr) == 0 or len(AP_xyzr) == 0:
        return 0.0, 0.0

    vx, vy, vz, vr = void_xyzr[:, 0], void_xyzr[:, 1], void_xyzr[:, 2], void_xyzr[:, 3]
    ax, ay, az, ar = AP_xyzr[:, 0],   AP_xyzr[:, 1],   AP_xyzr[:, 2],   AP_xyzr[:, 3]

    full_vol  = 4/3 * np.pi * vr**3
    full_area = 4   * np.pi * vr**2

    total_vol  = 0.0
    total_area = 0.0

    n_voids = len(void_xyzr)
    for start in range(0, n_voids, chunk_size):
        end = min(start + chunk_size, n_voids)

        # (chunk, N_grains)
        d = np.sqrt(
            (vx[start:end, None] - ax[None, :])**2 +
            (vy[start:end, None] - ay[None, :])**2 +
            (vz[start:end, None] - az[None, :])**2
        )

        vol_matrix  = sphere_sphere_intersection_volume(d, vr[start:end, None], ar[None, :])
        area_matrix = sphere_sphere_intersection_surface(d, vr[start:end, None], ar[None, :])

        # Sum over grains, cap at full sphere
        total_vol  += np.sum(np.minimum(vol_matrix.sum(axis=1),  full_vol[start:end]))
        total_area += np.sum(np.minimum(area_matrix.sum(axis=1), full_area[start:end]))

    return total_vol, total_area


def main():
    base_dir = os.path.join(os.getcwd(), "test_files")
    files    = os.listdir(base_dir)

    datasets = {}
    for f in files:
        if not f.endswith(".xyzr"):
            continue
        key = f.replace("_AP.xyzr", "").replace("_void.xyzr", "")
        datasets.setdefault(key, []).append(f)

    rows = []

    print(f"\nVOID_MODE = {VOID_MODE!r}")
    print("  clipped   → void volume/area = portion inside AP grains only")
    print("  unclipped → void volume/area = raw sphere geometry\n")

    for key, file_list in sorted(datasets.items()):
        AP_xyzr   = np.zeros((0, 4))
        void_xyzr = np.zeros((0, 4))

        for file in file_list:
            full = os.path.join(base_dir, file)
            if "_AP" in file:
                AP_xyzr = np.vstack([AP_xyzr, read_xyzr(full)])
            elif "_void" in file:
                void_xyzr = np.vstack([void_xyzr, read_xyzr(full)])

        if len(AP_xyzr) == 0:
            continue

        has_voids = len(void_xyzr) > 0

        # ---- MWD ---------------------------------------------------------
        AP_r   = AP_xyzr[:, 3]
        AP_d   = 2 * AP_r
        AP_mwd = np.sum(AP_d**3) / np.sum(AP_d**2)

        # ---- AP volume fraction (domain-clipped, µm units) ---------------
        AP_vol_frac = (clipped_sphere_volumes(AP_xyzr, lo=0.0, hi=DOMAIN_SIZE_UM).sum()
                       / DOMAIN_VOLUME_UM3 * 100)

        # ---- Void metrics — computed ONCE in µm, scaled to SI ------------
        # This avoids calling void_metrics_clipped twice per dataset.
        if has_voids:
            if VOID_MODE == "clipped":
                void_vol_um3, void_area_um2 = void_metrics_clipped(void_xyzr, AP_xyzr)
                # Convert to SI for V/S calculation
                A_void = void_vol_um3  * 1e-18   # µm³ → m³
                P_void = void_area_um2 * 1e-12   # µm² → m²
            else:
                void_vol_um3 = clipped_sphere_volumes(
                    void_xyzr, lo=0.0, hi=DOMAIN_SIZE_UM).sum()
                void_xyzr_m  = void_xyzr * 1e-6
                hi_m_        = DOMAIN_SIZE_UM * 1e-6
                A_void = clipped_sphere_volumes(void_xyzr_m, lo=0.0, hi=hi_m_).sum()
                P_void = clipped_surface_areas(void_xyzr_m,  lo=0.0, hi=hi_m_).sum()
            void_fraction = void_vol_um3 / DOMAIN_VOLUME_UM3 * 100
        else:
            void_fraction = 0.0
            A_void = P_void = 0.0

        # ---- V/S calculation (SI units) ----------------------------------
        hi_m      = DOMAIN_SIZE_UM * 1e-6
        AP_xyzr_m = AP_xyzr * 1e-6

        A_AP = clipped_sphere_volumes(AP_xyzr_m, lo=0.0, hi=hi_m).sum()
        P_AP = clipped_surface_areas(AP_xyzr_m,  lo=0.0, hi=hi_m).sum()

        V        = A_AP - A_void
        S        = P_AP + P_void
        V_over_S = V / S

        mass_per_surface = DENSITY_KG_M3 * V_over_S
        specific_surface = 1.0 / mass_per_surface

        prefix       = key.split("_")[0]
        manual_value = MANUAL_VALUES.get(prefix, "")

        def fmt(val, sci=False):
            return f"{val:.4e}" if sci else f"{val:.4f}"

        rows.append([
            key,
            fmt(AP_mwd),
            fmt(AP_vol_frac),
            fmt(void_fraction),
            fmt(V_over_S * 1e6),
            fmt(V_over_S, sci=True),
            fmt(mass_per_surface, sci=True),
            fmt(specific_surface, sci=True),
            manual_value,
        ])

    headers = [
        "Dataset",
        "AP_mwd (µm)",
        "AP_vol (%)",
        f"Void_frac % ({VOID_MODE})",
        "V/S (µm)",
        "V/S (m)",
        "m/S (kg/m²)",
        "S/m (m²/kg)",
        "Kogha Sw (m²/kg)",
    ]

    table = tabulate(rows, headers=headers, tablefmt="simple")
    print("\n" + table + "\n")

    with open(OUTPUT_FILE, "w") as fh:
        fh.write(f"VOID_MODE = {VOID_MODE}\n\n")
        fh.write(table + "\n")

    if len(AP_xyzr) > 0:
        print(f"DEBUG raw AP vol sum: {clipped_sphere_volumes(AP_xyzr, lo=0.0, hi=DOMAIN_SIZE_UM).sum():.4f}")
        print(f"DEBUG DOMAIN_VOLUME_UM3: {DOMAIN_VOLUME_UM3}")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()