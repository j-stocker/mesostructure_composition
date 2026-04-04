#!/usr/bin/env python3

import os
import numpy as np
from tabulate import tabulate
import math

OUTPUT_FILE = "void_statistics_3D.txt"

DENSITY_KG_M3 = 1.95 * 1000  # 1950 kg/m³

MANUAL_VALUES = {
    "A_vf_11": 2100, 
    "B_vf_6": 1600,
    "C_vf_6": 1900,
    "D_vf_7": 2100,
    "E_vf_7": 1500,
    "F_vf_6": 900,
    "G_vf_14": 3100,
    "H_vf_13": 2600,
    "I": 2000,
    "J": 1800,
    "K": 1700,
    "R": 500,
    "S": 900,
    "T": 1900,
}
# ------------------------------------------------------------------------

DOMAIN_VOLUME_UM3 =  50 * 50 * 50#100 * 100 * 100  # µm³
DOMAIN_SIZE_UM    = 50.0            # domain side length in µm (cube assumed)


def read_xyzr(filename):
    """Read xyzr file, return array of shape (N, 4): x, y, z, r all in µm."""
    rows = []
    with open(filename, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z, r = [float(p) * 1e6 for p in parts[:4]]  # m → µm
                rows.append((x, y, z, r))
            except ValueError:
                continue
    return np.array(rows, dtype=float) if rows else np.zeros((0, 4))


def cap_vol(h, r):
    """Volume of a spherical cap of height h on a sphere of radius r."""
    h = np.asarray(h, dtype=float)
    result = np.where(h <= 0, 0.0,
             np.where(h >= 2*r, 4/3 * np.pi * r**3,
                      np.pi * h**2 * (3*r - h) / 3))
    return result


def clipped_sphere_volumes(xyzr, lo=0.0, hi=None):
    """
    Return clipped volume for each sphere, accounting for intersections
    with the axis-aligned box [lo, hi]^3. Uses independent cap subtraction
    (same approach as the generator — slightly underestimates near corners).
    """
    if hi is None:
        hi = DOMAIN_SIZE_UM
    x, y, z, r = xyzr[:, 0], xyzr[:, 1], xyzr[:, 2], xyzr[:, 3]
    V = 4/3 * np.pi * r**3
    V -= cap_vol(r - (x - lo), r)   # -x face
    V -= cap_vol(r - (hi - x), r)   # +x face
    V -= cap_vol(r - (y - lo), r)   # -y face
    V -= cap_vol(r - (hi - y), r)   # +y face
    V -= cap_vol(r - (z - lo), r)   # -z face
    V -= cap_vol(r - (hi - z), r)   # +z face
    return np.maximum(V, 0.0)       # guard against tiny negatives at corners


def full_sphere_volumes(xyzr):
    """Unclipped sphere volumes — used for voids to match generator accounting."""
    r = xyzr[:, 3]
    return (4/3) * np.pi * r**3


def process_dataset(dataset_path):

    AP_xyzr   = np.zeros((0, 4))
    void_xyzr = np.zeros((0, 4))

    for file in os.listdir(dataset_path):
        full = os.path.join(dataset_path, file)
        if file.endswith("_AP.xyzr"):
            AP_xyzr = np.vstack([AP_xyzr, read_xyzr(full)])
        elif file.endswith("_void.xyzr"):
            void_xyzr = np.vstack([void_xyzr, read_xyzr(full)])

    if len(AP_xyzr) == 0:
        return None

    dataset_name = os.path.basename(dataset_path)
    has_voids    = len(void_xyzr) > 0

    AP_r   = AP_xyzr[:, 3]
    AP_d   = 2 * AP_r
    AP_mwd = np.sum(AP_d**3) / np.sum(AP_d**2)

    # AP: clipped (grains cross domain boundary)
    # voids: unclipped (sit inside grains, consistent with generator tracking)
    AP_vol_frac   = clipped_sphere_volumes(AP_xyzr).sum() / DOMAIN_VOLUME_UM3 * 100
    void_fraction = full_sphere_volumes(void_xyzr).sum() / DOMAIN_VOLUME_UM3 * 100 if has_voids else 0.0

    # ---- SI --------------------------------------------------------------
    hi_m       = DOMAIN_SIZE_UM * 1e-6
    AP_xyzr_m  = AP_xyzr * 1e-6
    A_AP       = clipped_sphere_volumes(AP_xyzr_m, lo=0.0, hi=hi_m).sum()
    P_AP       = np.sum(4 * np.pi * AP_xyzr_m[:, 3]**2)

    if has_voids:
        void_xyzr_m = void_xyzr * 1e-6
        A_void      = full_sphere_volumes(void_xyzr_m).sum()
        P_void      = np.sum(4 * np.pi * void_xyzr_m[:, 3]**2)
    else:
        A_void = P_void = 0.0

    V        = A_AP - A_void
    S        = P_AP + P_void
    V_over_S = V / S

    mass_per_surface = DENSITY_KG_M3 * V_over_S
    specific_surface = 1.0 / mass_per_surface
    manual_value     = MANUAL_VALUES.get(dataset_name, "")

    def fmt(val, sci=False):
        return f"{val:.4e}" if sci else f"{val:.4f}"

    return [
        dataset_name,
        fmt(AP_mwd),
        fmt(AP_vol_frac),
        fmt(void_fraction),
        fmt(V_over_S * 1e6),
        fmt(V_over_S, sci=True),
        fmt(mass_per_surface, sci=True),
        fmt(specific_surface, sci=True),
        manual_value,
    ]


def main():
    base_dir = os.path.join(os.getcwd(), "3D_xyzrs")

    files = os.listdir(base_dir)

    # ---- group files by dataset prefix -----------------------------------
    datasets = {}
    for f in files:
        if not f.endswith(".xyzr"):
            continue
        key = f.replace("_AP.xyzr", "").replace("_void.xyzr", "")
        datasets.setdefault(key, []).append(f)

    rows = []

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

        AP_r   = AP_xyzr[:, 3]
        AP_d   = 2 * AP_r
        AP_mwd = np.sum(AP_d**3) / np.sum(AP_d**2)

        # AP: clipped (grains cross domain boundary)
        # voids: unclipped (sit inside grains, consistent with generator tracking)
        AP_vol_frac   = clipped_sphere_volumes(AP_xyzr).sum() / DOMAIN_VOLUME_UM3 * 100
        void_fraction = full_sphere_volumes(void_xyzr).sum() / DOMAIN_VOLUME_UM3 * 100 if has_voids else 0.0

        hi_m      = DOMAIN_SIZE_UM * 1e-6
        AP_xyzr_m = AP_xyzr * 1e-6
        A_AP      = clipped_sphere_volumes(AP_xyzr_m, lo=0.0, hi=hi_m).sum()
        P_AP      = np.sum(4 * np.pi * AP_xyzr_m[:, 3]**2)

        if has_voids:
            void_xyzr_m = void_xyzr * 1e-6
            A_void      = full_sphere_volumes(void_xyzr_m).sum()
            P_void      = np.sum(4 * np.pi * void_xyzr_m[:, 3]**2)
        else:
            A_void = P_void = 0.0

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
        "Void_frac (%)",
        "V/S (µm)",
        "V/S (m)",
        "m/S (kg/m²)",
        "S/m (m²/kg)",
        "Kogha Sw (m²/kg)",
    ]

    table = tabulate(rows, headers=headers, tablefmt="simple")
    print("\n" + table + "\n")

    with open(OUTPUT_FILE, "w") as fh:
        fh.write(table + "\n")

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()