#Generate plots for samples data

import matplotlib.pyplot as plt
import os
varying_size_dirA = './generated_images/unimodal_varying_part_size/results.txt'
varying_AP_dirB = './generated_images/unimodal_varying_AP_ratio/results.txt'
varying_size_dirC = './generated_images/bimodal_varying_part_size/results.txt'
varying_AP_dirD = './generated_images/bimodal_varying_AP_ratio/results.txt'
save_dir_A = "./generated_images/unimodal_varying_part_size/"
save_dir_B = "./generated_images/unimodal_varying_AP_ratio/"
save_dir_C = "./generated_images/bimodal_varying_part_size/"
save_dir_D = "./generated_images/bimodal_varying_AP_ratio/"

#read from results files, plot % interface
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_component_vs_x(
    results_file,         # path to CSV/TXT results
    save_dir,             # folder to save figure
    x_col=0,              # column index for x-axis
    y_col=3,              # column index for y-axis (component to plot)
    xlabel="X-axis",
    ylabel="% Component",
    title=None,
    plot_name="plot.png",
    scale_x=1.0           # optional factor to scale x values (e.g., radius / 10)
):
    """
    General function to plot a component (like Interface %) vs any x-value from results file.
    """
    # ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    # read data
    x_data, y_data = [], []
    with open(results_file, "r") as f:
        next(f)  # skip header
        for line in f:
            cols = [float(v.strip()) for v in line.split(',')]
            x_data.append(cols[x_col] * scale_x)
            y_data.append(cols[y_col])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'o-', label=ylabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title else f"{ylabel} vs {xlabel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save figure
    save_path = os.path.join(save_dir, plot_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot as {save_path}")




if __name__ == "__main__":
    #A
    plot_component_vs_x(
    results_file=varying_size_dirA,
    save_dir=save_dir_A,
    x_col=0,                   # average radius column
    y_col=3,                   # interface %
    xlabel="Average Particle Size (µm)",
    ylabel="Interface %",
    title="Particle Size vs Interface %, Unimodal",
    plot_name="varying_size_plotA.png",
    scale_x=0.1                # e.g., divide by 10 as in your original code
)

    #B
    plot_component_vs_x(
        results_file=varying_AP_dirB,
        save_dir=save_dir_B,
        x_col=0,                   # AP achieved column
        y_col=3,                   # interface %
        xlabel="AP Achieved (%)",
        ylabel="Interface %",
        title="AP Achieved vs Interface %, Unimodal",
        plot_name="varying_AP_plotB.png"
    )
    #C
    plot_component_vs_x(
    results_file=varying_size_dirC,
    save_dir=save_dir_C,
    x_col=0,                   # average radius column
    y_col=3,                   # interface %
    xlabel="Average Particle Size (µm)",
    ylabel="Interface %",
    title="Particle Size vs Interface %, Bimodal",
    plot_name="varying_size_plot.png",
    scale_x=0.1                # e.g., divide by 10 as in your original code
)
    #D
    # Plot D: Interface % vs AP achieved
    plot_component_vs_x(
        results_file=varying_AP_dirD,
        save_dir=save_dir_D,
        x_col=0,                   # AP achieved column
        y_col=3,                   # interface %
        xlabel="AP Achieved (%)",
        ylabel="Interface %",
        title="AP Achieved vs Interface %, Bimodal",
        plot_name="varying_AP_plot.png"
    )
