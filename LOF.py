import numpy as np
import matplotlib.pyplot as plt
import os


def poly_eqn_string(coeffs, var="x"):
    """
    Build a readable polynomial equation string from np.polyfit coefficients.
    Works for any polynomial order.
    """
    order = len(coeffs) - 1
    terms = []

    for i, c in enumerate(coeffs):
        power = order - i

        if abs(c) < 1e-12:
            continue

        if power == 0:
            term = f"{c:.3e}"
        elif power == 1:
            term = f"{c:.3e} {var}"
        else:
            term = f"{c:.3e} {var}^{power}"

        terms.append(term)

    return "y = " + " + ".join(terms)


def plot_interface_fit(
    results_file,
    save_dir,
    x_col=0,
    y_col=3,
    xlabel="Radius (µm)",
    ylabel="Interface (%)",
    title=None,
    fit_order=2,
    plot_name="interface_fit.png"
):
    """
    Read results CSV, fit polynomial of arbitrary order, plot data and fit,
    and display the fitted equation on the plot.
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Read data
    x_data, y_data = [], []
    with open(results_file, "r") as f:
        next(f)  # skip header
        for line in f:
            cols = [float(v.strip()) for v in line.split(',')]
            x_data.append(cols[x_col]/100)
            y_data.append(cols[y_col])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Polynomial fit
    coeffs = np.polyfit(x_data, y_data, fit_order)
    x_fit = np.linspace(x_data.min(), x_data.max(), 500)
    y_fit = np.polyval(coeffs, x_fit)

    # Build equation string
    eqn = poly_eqn_string(coeffs)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_data, y_data, 'o', markersize=3, label="Data")
    plt.plot(x_fit, y_fit, linewidth=2, label=f"Poly fit (order {fit_order})")

    # Show equation on plot
    plt.text(
        0.02, 0.98, eqn,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round", alpha=0.25)
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title else f"{ylabel} vs {xlabel} with Fit")
    plt.legend(loc="center right")
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, plot_name)
    plt.savefig(save_path, dpi=300)
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

    plot_interface_fit(resultsA, save_dirA, x_col=0, y_col=3, xlabel="Radius (µm)", title="Interface vs Radius, Unimodal", fit_order=4, plot_name="interface_fitA.png")
    plot_interface_fit(resultsB, save_dirB, x_col=0, y_col=3, xlabel="AP Achieved", title="Interface vs AP Achieved, Unimodal", fit_order=2, plot_name="interface_fitB.png")
    plot_interface_fit(resultsC, save_dirC, x_col=4, y_col=3, xlabel="Radius (µm)", title="Interface vs Radius, Bimodal", fit_order=2, plot_name="interface_fitC.png")
    plot_interface_fit(resultsD, save_dirD, x_col=0, y_col=3, xlabel="AP Achieved", title="Interface vs AP Achieved, Bimodal", fit_order=1, plot_name="interface_fitD.png")

