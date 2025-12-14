#Generates an AP/HTPB image given avg radius and % dist

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from PIL import Image

#set variables

folder = "./generated_images"
os.makedirs(folder, exist_ok=True)  # create folder if it doesn't exist

# Full path for saving the figure
save_path = os.path.join(folder, "mesostructure.png")
save_path_untitled = os.path.join(folder, "mesostructure_untitled.png")
save_path_xyzr = os.path.join(folder, "mesostructure.xyzr")

img_size = 1 #square image, 1x1 (makes other values easier)
physical_size = 5e-4 #m
physical_mean_radius = 50e-6 #given as a portion of the image's height/width, 3%
ap_ratio = 0.7 #60% AP
#monomodal
rad_dev = 0.2 #standard deviation of radius sizes, normal distribution
rad_dev_bi = [0.2, 0.5]
mean_rad_bi = [125e-6, 20e-6]
max_attempts = 10000 #will try to generate an image 500000 times before it gives up
mix = 1/12 #coarse/fine particles


def save_xyzr(circles, filepath, img_size, physical_size):
    scale = physical_size/img_size #m/image unit
    with open(filepath, 'w') as f:
        for (x, y, r) in circles:
            xp = x * scale
            yp = y * scale
            rp = r * scale
            f.write(f"{xp:.8e} {yp:.8e} 0.0 {rp:.8e}\n")
def gen_struct(
    save_path, save_path_untitled, save_path_xyzr,
    img_size, physical_size, physical_mean_radius,
    ap_ratio, rad_dev, max_attempts,
    mode, mix=0.5, max_tries=80, interface_width=2e-6
):
    """
    Generate a 2D microstructure assuming perfect circles.
    """

    margin = 0.01
    rng = np.random.default_rng()

    # -----------------------------
    # Mean radius
    # -----------------------------
    if mode == 1:
        mean_rad = float(physical_mean_radius) / physical_size * img_size
        r_ref = physical_mean_radius
    else:
        mean_rad = [r / physical_size * img_size for r in physical_mean_radius]
        r_ref = min(physical_mean_radius)

    # -----------------------------
    # Area + bounds
    # -----------------------------
    target_AP_area = ap_ratio
    r_min = 1e-6 / physical_size * img_size
    r_max = 150e-6 / physical_size * img_size

    # -----------------------------
    # Mask (area estimation)
    # -----------------------------
    N_mask = 8192
    #N_mask = min(max(N_mask, 512), 2048)

    mask = np.zeros((N_mask, N_mask), dtype=bool)
    inv_N = 1.0 / N_mask

    # -----------------------------
    # Grid for overlap
    # -----------------------------
    cell_size = 40 * (mean_rad if mode == 1 else max(mean_rad))
    n_cells = max(1, int(math.ceil(img_size / cell_size)))
    grid = [[[] for _ in range(n_cells)] for __ in range(n_cells)]

    def cell_coords(x, y):
        cx = max(0, min(n_cells - 1, int(x / cell_size)))
        cy = max(0, min(n_cells - 1, int(y / cell_size)))
        return cx, cy

    def nearby_candidates(x, y, circles):
        cx, cy = cell_coords(x, y)
        for i in range(max(0, cx - 1), min(n_cells, cx + 2)):
            for j in range(max(0, cy - 1), min(n_cells, cy + 2)):
                for idx in grid[i][j]:
                    yield circles[idx]

    def sample_radius(mu, sigma):
        for _ in range(50):
            r = rng.lognormal(mu, sigma)
            if r_min <= r <= r_max:
                return r
        return np.clip(r, r_min, r_max)

    # -----------------------------
    # Pre-generate radii ONCE
    # -----------------------------
    radii_master = []

    if mode == 1:
        mu = np.log(mean_rad / np.sqrt(1 + rad_dev**2))
        sigma = np.sqrt(np.log(1 + rad_dev**2))
        radii_master = [sample_radius(mu, sigma) for _ in range(max_attempts)]

    else:
        dev0, dev1 = rad_dev
        fine_mean, coarse_mean = mean_rad

        mu_f = np.log(fine_mean / np.sqrt(1 + dev0**2))
        mu_c = np.log(coarse_mean / np.sqrt(1 + dev1**2))
        sigma_f = np.sqrt(np.log(1 + dev0**2))
        sigma_c = np.sqrt(np.log(1 + dev1**2))

        for _ in range(max_attempts):
            if rng.random() < mix:
                radii_master.append(sample_radius(mu_c, sigma_c))
            else:
                radii_master.append(sample_radius(mu_f, sigma_f))

    radii_master.sort(reverse=True)

    # -----------------------------
    # Retry loop
    # -----------------------------
    best_circles = []
    best_total_area = 0.0

    for _ in range(max_tries):
        circles = []
        total_area = 0.0
        grid = [[[] for _ in range(n_cells)] for __ in range(n_cells)]
        mask[:] = False

        for r in radii_master:
            if total_area >= target_AP_area:
                break

            r = min(r, 0.5 * img_size - margin)
            x = rng.uniform(-margin + r, img_size + margin - r)
            y = rng.uniform(-margin + r, img_size + margin - r)

            valid = True
            for (cx, cy, cr) in nearby_candidates(x, y, circles):
                tol = 1e-3 * min(r, cr)   # 0.1% of smaller radius
                #allow a little overlap
                if (cx - x)**2 + (cy - y)**2 < (cr + r - tol)**2:
                    valid = False
                    break

            if not valid:
                continue

            circles.append((x, y, r))
            gx, gy = cell_coords(x, y)
            grid[gx][gy].append(len(circles) - 1)

            # ---- pixel area ----
            x_px = int(x / img_size * N_mask)
            y_px = int(y / img_size * N_mask)
            r_px = int(math.ceil(r / img_size * N_mask))

            xmin = max(0, x_px - r_px)
            xmax = min(N_mask, x_px + r_px + 1)
            ymin = max(0, y_px - r_px)
            ymax = min(N_mask, y_px + r_px + 1)

            if xmax > xmin and ymax > ymin:
                xs = (np.arange(xmin, xmax) + 0.5) * inv_N
                ys = (np.arange(ymin, ymax) + 0.5) * inv_N
                XX, YY = np.meshgrid(xs, ys)
                patch = (XX - x)**2 + (YY - y)**2 <= r*r

                prev = mask[ymin:ymax, xmin:xmax]
                new = patch & (~prev)
                added = new.sum()
                if added:
                    mask[ymin:ymax, xmin:xmax] |= patch
                    total_area += added / (N_mask * N_mask)

        if abs(total_area - target_AP_area) < abs(best_total_area - target_AP_area):
            best_total_area = total_area
            best_circles = circles.copy()

        if abs(best_total_area - target_AP_area) < 0.001:
            break

    circles = best_circles
    total_area = best_total_area
    #untitled 1024 x 1024 
    #making the actual figure
    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=1024/6) #resolution 1024
    ax2.set_position([0, 0, 1, 1])
    ax2.set_axis_off()
    #boundaries
    ax2.set_xlim(0, img_size)
    ax2.set_ylim(0, img_size)
    ax2.set_aspect("equal", "box")
    ax2.axis("off")
    
    #HTPB background, blue
    ax2.add_patch(
        plt.Rectangle((0, 0), img_size, img_size, facecolor='#0000FF', zorder=0) 
    )
    
    #AP circles, red
    for (x, y, r) in circles:
        ax2.add_patch(
            Circle((x, y), r, facecolor='#FF0000', edgecolor='#800080', linewidth=interface_width/(physical_size*N_mask), zorder=10)
        )
    fig2.savefig(save_path_untitled, dpi=1024/6, bbox_inches=None, pad_inches=0.0)
    plt.close(fig2)
    
    #with title and padding
    fig = plt.figure(figsize=(6, 6), dpi=1024/6)
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire figure, no margins

    ax.set_axis_off()        # no ticks
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    
    #HTPB background, blue
    ax.add_patch(
        plt.Rectangle((0, 0), img_size, img_size, facecolor='#0000FF', zorder=0) 
    )
    
    #AP circles, red
    for (x, y, r) in circles:
        ax.add_patch(
            Circle((x, y), r, facecolor='#FF0000', edgecolor='#800080', linewidth=interface_width/(physical_size*1024), zorder=10)
        )
    if mode == 1:
        radius_str = f"{physical_mean_radius/1e-6:.1f}"
    else:
        # For bimodal, show both coarse and fine
        radius_str = f"{physical_mean_radius[0]/1e-6:.1f}/{physical_mean_radius[1]/1e-6:.1f}"

    ax.set_title(
        f"AP grains: mean radius = {radius_str} µm, target AP = {ap_ratio:.3f}\n"
        f"Placed {len(circles)} grains, achieved AP = {total_area:.3f}"
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    
    plt.close(fig)
    save_xyzr(circles, save_path_xyzr, img_size, physical_size)
    return save_path, save_path_untitled, total_area, save_path_xyzr



if __name__ == "__main__":
    save_path, save_path_untitled, _, _ = gen_struct(save_path, save_path_untitled, save_path_xyzr, img_size, physical_size, physical_mean_radius, ap_ratio, rad_dev_bi, max_attempts, 2, mix)
    print(f"\nSaved AP–HTPB microstructure images to: {save_path} and {save_path_untitled}")
    
