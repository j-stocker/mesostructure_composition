#!/usr/bin/env python3

import os
import numpy as np
from tabulate import tabulate

OUTPUT_FILE = "void_statistics_2D.txt"

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


def compute_AP_only_ratio(radii):
    A = np.sum(np.pi * radii**2)
    P = np.sum(2 * np.pi * radii)
    return A / P


def process_dataset(dataset_path):

    AP_radii = []
    void_radii = []

    for file in os.listdir(dataset_path):
        full = os.path.join(dataset_path, file)
        if file.endswith("_AP.xyzr"):
            AP_radii.extend(read_radii(full))
        elif file.endswith("_void.xyzr"):
            void_radii.extend(read_radii(full))

    AP_radii   = np.array(AP_radii)
    void_radii = np.array(void_radii)

    if len(AP_radii) == 0:
        return None

    dataset_name = os.path.basename(dataset_path)

    has_voids = len(void_radii) > 0

    # ---- µm stats --------------------------------------------------------
    AP_d   = 2 * AP_radii
    AP_mwd = np.sum(AP_d**3) / np.sum(AP_d**2)

    if has_voids:
        void_fraction = np.sum(np.pi * void_radii**2) / (10 * 200 * 200) * 100
    else:
        void_fraction = 0.0

    # ---- SI --------------------------------------------------------------
    AP_radii_m = AP_radii * 1e-6

    A_AP   = np.sum(np.pi * AP_radii_m**2)
    P_AP   = np.sum(2 * np.pi * AP_radii_m)

    if has_voids:
        void_radii_m = void_radii * 1e-6
        A_void = np.sum(np.pi * void_radii_m**2)
        P_void = np.sum(2 * np.pi * void_radii_m)
    else:
        A_void = 0.0
        P_void = 0.0

    # ---- total geometry --------------------------------------------------
    V = A_AP - A_void          # 2D "volume"
    S = P_AP + P_void          # total surface

    V_over_S = V / S           # ALWAYS defined now

    # ---- mass ------------------------------------------------------------
    mass_kg = DENSITY_KG_M3 * V

    # ---- surface relations -----------------------------------------------
    mass_per_surface = DENSITY_KG_M3 * V_over_S   # kg/m²
    specific_surface = 1.0 / mass_per_surface     # m²/kg

    # ---- manual ----------------------------------------------------------
    manual_value = MANUAL_VALUES.get(dataset_name, "")

    # ---- formatting ------------------------------------------------------
    def fmt(val, sci=False):
        return f"{val:.4e}" if sci else f"{val:.4f}"

    return [
        str(dataset_name),
        fmt(AP_mwd),
        fmt(void_fraction),
        fmt(V_over_S * 1e6),
        fmt(V_over_S, sci=True),
        fmt(mass_per_surface, sci=True),
        fmt(specific_surface, sci=True),
        str(manual_value),   # <-- force to string
    ]


def main():
    base_dir = os.path.join(os.getcwd(), "new_datasets")

    datasets = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )
    print(f"Looking in: {base_dir}")
    print(f"Datasets found: {datasets}")

    for d in datasets:
        path = os.path.join(base_dir, d)
        files = os.listdir(path)
        print(f"\n{d}: {files}")

    rows = []
    
    for d in datasets:
        result = process_dataset(os.path.join(base_dir, d))
        if result:
            rows.append(result)

    headers = [
        "Dataset",
        "AP_mwd (µm)",
        "Void_frac (%)",
        "V/S (µm)",
        "V/S (m)",
        "m/S (kg/m²)",
        "S/m (m²/kg)",
        "Kogha Sw (m²/kg)",
    ]



    # Build colalign dynamically based on actual column count
    num_cols = len(headers)
    col_align = ("left",) + ("right",) * (num_cols - 1)  # or whatever alignment you want

    table = tabulate(
        rows,
        headers=headers,
        tablefmt="plain",   # try "plain" instead of "grid"
    )

    print("\n" + table + "\n")

    with open(OUTPUT_FILE, "w") as fh:
        fh.write(table + "\n")

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()