#!/usr/bin/env python3

import os
import numpy as np
from tabulate import tabulate


# ------------------------------------------------------------------------
DATASET_PATH = "128x128"   # ← set this

DENSITY_KG_UM3   = 1.95e-15      # 1950 kg/m³ → kg/µm³
SAMPLE_AREA_UM2  = 200 * 200      # µm²
SAMPLE_DEPTH_UM  = 10             # µm
# ------------------------------------------------------------------------


def read_radii(filename):
    radii = []
    with open(filename, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                radii.append(float(parts[3]) * 1e6)  # m → µm
            except ValueError:
                continue
    return np.array(radii, dtype=float)


def main():
    ap_radii   = []
    void_radii = []

    for fname in sorted(os.listdir(DATASET_PATH)):
        fpath = os.path.join(DATASET_PATH, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.endswith("_AP.xyzr"):
            ap_radii.extend(read_radii(fpath))
        elif fname.endswith("_void.xyzr"):
            void_radii.extend(read_radii(fpath))

    ap_radii   = np.array(ap_radii,   dtype=float)
    void_radii = np.array(void_radii, dtype=float)

    if len(ap_radii) == 0:
        print("No *_AP.xyzr files found.")
        return

    has_voids = len(void_radii) > 0

    # ---- AP size distribution (µm) --------------------------------------
    ap_d   = 2 * ap_radii
    ap_mwd = np.sum(ap_d**3) / np.sum(ap_d**2)

    # ---- void fraction ---------------------------------------------------
    void_fraction = (
        np.sum(np.pi * void_radii**2) / (SAMPLE_DEPTH_UM * SAMPLE_AREA_UM2) * 100
        if has_voids else 0.0
    )

    # ---- geometry (all in µm) -------------------------------------------
    A_ap   = np.sum(np.pi * ap_radii**2)
    P_ap   = np.sum(2 * np.pi * ap_radii)
    A_void = np.sum(np.pi * void_radii**2) if has_voids else 0.0
    P_void = np.sum(2 * np.pi * void_radii) if has_voids else 0.0

    V = A_ap - A_void       # µm²
    S = P_ap + P_void       # µm
    V_over_S = V / S        # µm

    # ---- mass / surface (µm-based) --------------------------------------
    mass_per_surface = DENSITY_KG_UM3 * V_over_S   # kg/µm²
    specific_surface = 1.0 / mass_per_surface       # µm²/kg

    # ---- output ----------------------------------------------------------
    rows = [
        ("AP spheres",                  f"{len(ap_radii)}"),
        ("Void spheres",                f"{len(void_radii)}"),
        ("AP mean-weighted diam (µm)",  f"{ap_mwd:.4f}"),
        ("AP radius mean (µm)",         f"{ap_radii.mean():.4f}"),
        ("AP radius std  (µm)",         f"{ap_radii.std():.4f}"),
        ("AP radius min  (µm)",         f"{ap_radii.min():.4f}"),
        ("AP radius max  (µm)",         f"{ap_radii.max():.4f}"),
        ("Void fraction (%)",           f"{void_fraction:.4f}"),
        ("A_AP   (µm²)",                f"{A_ap:.4e}"),
        ("P_AP   (µm)",                 f"{P_ap:.4e}"),
        ("A_void (µm²)",                f"{A_void:.4e}"),
        ("P_void (µm)",                 f"{P_void:.4e}"),
        ("V = A_AP − A_void (µm²)",     f"{V:.4e}"),
        ("S = P_AP + P_void (µm)",      f"{S:.4e}"),
        ("V/S (µm)",                    f"{V_over_S:.4f}"),
        ("mass/S  (kg/µm²)",            f"{mass_per_surface:.4e}"),
        ("S/mass  (µm²/kg)",            f"{specific_surface:.4e}"),
    ]

    print(f"\n{DATASET_PATH}\n")
    print(tabulate(rows, headers=["Statistic", "Value"],
                   tablefmt="simple", colalign=("left", "right")))
    print()


if __name__ == "__main__":
    main()