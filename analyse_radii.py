#!/usr/bin/env python3

import os
import numpy as np
from tabulate import tabulate

OUTPUT_FILE = "void_statistics.txt"


def read_radii(filename):

    radii = []

    with open(filename, "r") as f:
        for line in f:

            if not line.strip():
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            radii.append(float(parts[3]) * 1e6)

    return np.array(radii)


def process_dataset(dataset_path):

    AP_radii = []
    void_radii = []

    for file in os.listdir(dataset_path):

        full = os.path.join(dataset_path, file)

        if file.endswith("_AP.xyzr"):
            AP_radii.extend(read_radii(full))

        elif file.endswith("_void.xyzr"):
            void_radii.extend(read_radii(full))

    AP_radii = np.array(AP_radii)
    void_radii = np.array(void_radii)

    if len(AP_radii) == 0 or len(void_radii) == 0:
        return None

    AP_d = 2 * AP_radii
    AP_mwd = np.sum(AP_d**3) / np.sum(AP_d**2)
    AP_std = np.std(AP_radii)

    void_mean = np.mean(void_radii)
    void_std = np.std(void_radii)

    void_avg_size = void_mean**2
    void_area = void_radii**2 * np.pi
    void_fraction = sum(void_area) / (10 * 200*200) *100

    
    return [
        os.path.basename(dataset_path),
        len(AP_radii),
        len(void_radii),
        AP_mwd,
        AP_std,
        void_mean,
        void_std,
        void_fraction
    ]


    

def main():
    base_dir = os.path.join(os.getcwd(), "new_datasets")

    datasets = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(d) and "_vf_" in d
    )



    rows = []

    for d in datasets:

        result = process_dataset(d)

        if result:
            rows.append(result)

    headers = [
        "Dataset",
        "N_AP",
        "N_void",
        "AP_mean_w_diam (µm)",
        "AP_std",
        "Void_mean (µm)",
        "Void_std",
        "Void_fraction (%)"
    ]

    table = tabulate(
        rows,
        headers=headers,
        floatfmt=".4f",
        tablefmt="grid",
        colalign=("left","right","right","right","right","right","right","right")
    )

    print("\n" + table + "\n")

    with open(OUTPUT_FILE, "w") as f:
        f.write(table)

    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()