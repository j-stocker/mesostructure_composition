#Line of fit for data

import numpy as np
import matplotlib.pyplot as plt
import os

resultsA = "./generated_images/unimodal_varying_part_size/results.txt"
resultsB = "./generated_images/unimodal_varying_AP_ratio/results.txt"
save_dirA = "./generated_images/unimodal_varying_part_size/"
save_dirB = "./generated_images/unimodal_varying_AP_ratio/"

# Make sure the folder exists
os.makedirs(save_dirA, exist_ok=True)
os.makedirs(save_dirB, exist_ok=True)



def LoFA():
    radius_um = []
    interface_percentA = []

    with open(resultsA, "r") as f:
        next(f)  # skip header
        for line in f:
            cols = [float(x.strip()) for x in line.split(',')]
            radius_um.append(cols[0]/10)
            interface_percentA.append(cols[3])
    radius = np.array(radius)
    interface = np.array(interface)
    a, b, c = np.polyfit(radius, interface, 2)

    radius_fit = np.linspace(radius.min(), radius.max(), 500)
    interface_fit = a * radius_fit**2 + b * radius_fit + c

    plt.figure(figsize=(8,5))
    plt.plot(radius, interface, 'o', markersize=3, label="Data")
    plt.plot(radius_fit, interface_fit, linewidth=2, label="Linear fit")

    plt.xlabel("Radius (µm)")
    plt.ylabel("Interface (%)")
    plt.title("Interface vs Radius with Best Fit Line")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dirA, "interface_fitA.png")
    plt.savefig(save_path, dpi=300)
    plt.savefig("interface_fitA.png", dpi=300)
    print('Image saved.')

def LoFB():
    AP_achieved = []
    interface_percentB = []

    with open(resultsA, "r") as f:
        next(f)  # skip header
        for line in f:
            cols = [float(x.strip()) for x in line.split(',')]
            AP_achieved.append(cols[0])
            interface_percentB.append(cols[3])
    AP = np.array(AP_achieved)
    interface = np.array(interface)
    a, b, c = np.polyfit(AP, interface, 2)

    AP_fit = np.linspace(AP.min(), AP.max(), 500)
    interface_fit = a * AP_fit**2 + b * AP_fit + c

    plt.figure(figsize=(8,5))
    plt.plot(AP, interface, 'o', markersize=3, label="Data")
    plt.plot(AP_fit, interface_fit, linewidth=2, label="Linear fit")

    plt.xlabel("Radius (µm)")
    plt.ylabel("Interface (%)")
    plt.title("Interface vs Radius with Best Fit Line")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dirB, "interface_fitB.png")
    plt.savefig(save_path, dpi=300)
    plt.savefig("interface_fitB.png", dpi=300)
    print('Image saved.')

if __name__ == "__main__":
    LoFA()
    LoFB()
