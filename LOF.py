import numpy as np
import matplotlib.pyplot as plt
import os

def plot_interface_fit(results_file, save_dir, x_col=0, y_col=3, xlabel="Radius (µm)", ylabel="Interface (%)", title=None, fit_order=2, plot_name="interface_fit.png"):
    """
    General function to read results, fit a polynomial, and plot data + fit line.

    Parameters
    ----------
    results_file : str
        Path to the CSV results file.
    save_dir : str
        Directory to save the figure.
    x_col : int
        Column index for x-axis data.
    y_col : int
        Column index for y-axis data.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    title : str or None
        Plot title.
    fit_order : int
        Order of polynomial fit (default quadratic).
    plot_name : str
        Name of the saved plot image.
    """
    # Make sure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Read data
    x_data, y_data = [], []
    with open(results_file, "r") as f:
        next(f)  # skip header
        for line in f:
            cols = [float(v.strip()) for v in line.split(',')]
            x_data.append(cols[x_col])
            y_data.append(cols[y_col])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Fit polynomial
    coeffs = np.polyfit(x_data, y_data, fit_order)
    x_fit = np.linspace(x_data.min(), x_data.max(), 500)
    y_fit = np.polyval(coeffs, x_fit)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(x_data, y_data, 'o', markersize=3, label="Data")
    plt.plot(x_fit, y_fit, linewidth=2, label=f"Poly fit (order {fit_order})")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title else f"{ylabel} vs {xlabel} with Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, plot_name)
    plt.savefig(save_path, dpi=300)
    plt.savefig(plot_name, dpi=300)
    plt.close()
    print(f"Image saved: {save_path}")


if __name__ == "__main__":
    resultsA = "./generated_images/unimodal_varying_part_size/results_uni_part_size.txt"
    resultsB = "./generated_images/unimodal_varying_AP_ratio/results_uni_AP_ratio.txt"
    resultsC = "./generated_images/bimodal_varying_part_size/results_bi_part_size.txt"
    resultsD = "./generated_images/bimodal_varying_AP_ratio/results_bi_AP_ratio.txt"

    save_dirA = "./generated_images/unimodal_varying_part_size/"
    save_dirB = "./generated_images/unimodal_varying_AP_ratio/"
    save_dirC = "./generated_images/bimodal_varying_part_size/"
    save_dirD = "./generated_images/bimodal_varying_AP_ratio/"

    plot_interface_fit(resultsA, save_dirA, x_col=0, y_col=3, xlabel="Radius (µm)", title="Interface vs Radius, Unimodal", plot_name="interface_fitA.png")
    plot_interface_fit(resultsB, save_dirB, x_col=0, y_col=3, xlabel="AP Achieved", title="Interface vs AP Achieved, Unimodal", plot_name="interface_fitB.png")
    plot_interface_fit(resultsC, save_dirC, x_col=0, y_col=3, xlabel="Radius (µm)", title="Interface vs Radius, Bimodal", plot_name="interface_fitC.png")
    plot_interface_fit(resultsD, save_dirD, x_col=0, y_col=3, xlabel="AP Achieved", title="Interface vs AP Achieved, Bimodal", plot_name="interface_fitD.png")

