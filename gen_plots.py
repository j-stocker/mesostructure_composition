#Generate plots for samples data

import matplotlib.pyplot as plt
import os
varying_size_dir = './generated_images/unimodal_varying_part_size/results.txt'
varying_AP_dir = './generated_images/unimodal_varying_AP_ratio/results.txt'
save_dir_A = "./generated_images/unimodal_varying_part_size/"
save_dir_B = "./generated_images/unimodal_varying_AP_ratio/"

#read from results files, plot % interface
def plotA():
    radius_um = []
    ap_percentA = []
    htpb_percentA = []
    interface_percentA = []

    with open(varying_size_dir, "r") as f:
        next(f)  # skip header
        for line in f:
            cols = [float(x.strip()) for x in line.split(',')]
            
            radius_um.append(cols[0])
            ap_percentA.append(cols[1])
            htpb_percentA.append(cols[2])
            interface_percentA.append(cols[3])

    plt.figure(figsize=(10, 6))
    plt.plot(radius_um, ap_percentA, label="AP %")
    plt.plot(radius_um, htpb_percentA, label="HTPB %")
    plt.plot(radius_um, interface_percentA, label="Interface %")

    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Results Plot")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir_A, "varying_size_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print('Saved plot as varying_size_plot.png')


def plotB():
    AP_achieved = []
    ap_percentB = []
    htpb_percentB = []
    interface_percentB = []


    with open(varying_AP_dir, "r") as f:
        next(f)  # skip header
        for line in f:
            cols = [float(x.strip()) for x in line.split(',')]
            
            AP_achieved.append(cols[0])
            ap_percentB.append(cols[1])
            htpb_percentB.append(cols[2])
            interface_percentB.append(cols[3])
    
    plt.figure(figsize=(10, 6))
    plt.plot(AP_achieved, ap_percentB, label="AP %")
    plt.plot(AP_achieved, htpb_percentB, label="HTPB %")
    plt.plot(AP_achieved, interface_percentB, label="Interface %")

    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Results Plot")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir_B, "varying_AP_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print('Saved plot as varying_AP_plot.png')



if __name__ == "__main__":
    plotB()
    
