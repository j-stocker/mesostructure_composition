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
    mode=1, 
    mix=0.5, max_tries=80, interface_width=2e-6, N_mask=8192, dpi_highres=1000
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
                tol = 1e-4 * min(r, cr)   # 0.01% of smaller radius
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
                added_area = added / (N_mask * N_mask)

                if total_area + added_area > target_AP_area:
                    continue   # reject this circle

                mask[ymin:ymax, xmin:xmax] |= patch
                total_area += added_area

        if abs(total_area - target_AP_area) < abs(best_total_area - target_AP_area):
            best_total_area = total_area
            best_circles = circles.copy()

        if abs(best_total_area - target_AP_area) < 0.0001:
            break

    circles = best_circles
    total_area = best_total_area
    #untitled
    #making the actual figure
    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=dpi_highres) #high res
    ax2.set_position([0, 0, 1, 1])
    ax2.set_axis_off()
    #boundaries
    ax2.set_xlim(0, img_size)
    ax2.set_ylim(0, img_size)
    ax2.set_aspect("equal", "box")
    ax2.axis("off")
    antialiased=False
    #HTPB background, blue
    ax2.add_patch(
        plt.Rectangle((0, 0), img_size, img_size, facecolor='#0000FF', zorder=0) 
    )
    
    #AP circles, red
    for (x, y, r) in circles:
        ax2.add_patch(
            Circle((x, y), r, facecolor='#FF0000', edgecolor='#800080', linewidth=interface_width/(physical_size/img_size)*72, zorder=5)
        )
        ax2.add_patch(
            Circle((x, y), r, facecolor='none', edgecolor='#FFFFFF', linewidth=72/dpi_highres, zorder=10)
        )
    fig2.savefig(save_path_untitled, dpi=dpi_highres, bbox_inches=None, pad_inches=0.0)
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
            Circle((x, y), r, facecolor='#FF0000', edgecolor='#800080', linewidth=interface_width/(physical_size/img_size)*72, zorder=5)
        )
        ax.add_patch(
            Circle((x, y), r, facecolor='none', edgecolor='#FFFFFF', linewidth=1, zorder=10)
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

    fig.savefig(save_path, dpi=1024/6, bbox_inches="tight", pad_inches=0.02)
    
    plt.close(fig)
    save_xyzr(circles, save_path_xyzr, img_size, physical_size)
    return save_path, save_path_untitled, total_area, save_path_xyzr


'''
def gen_struct_combined(
    save_path, save_path_untitled, AP_xyzr, void_xyzr,
    img_size, physical_size,
    rad_dev, max_attempts,
    vol_percent_solid, vol_percent_hollow, vol_percent_porous, void_fraction,
    mean_rad_solid=120e-6, mean_rad_hollow=2e-6, mean_rad_porous=4.5e-6,
    max_tries=1, interface_width=1e-7, N_mask_base=1024, dpi_highres=100
):
    rng = np.random.default_rng()
    margin = 0.07

    # Convert mean radii to image units
    mean_rad_solid_img = mean_rad_solid / physical_size * img_size
    mean_rad_hollow_img = mean_rad_hollow / physical_size * img_size
    mean_rad_porous_img = mean_rad_porous / physical_size * img_size

    r_min = 0.2e-6 / physical_size * img_size
    r_max = 150e-6 / physical_size * img_size

    # Scale mask resolution for large domains
    N_mask = int(N_mask_base * (physical_size / 0.0005))
    mask = np.zeros((N_mask, N_mask), dtype=bool)

    # Adjust max_attempts for domain size
    max_attempts_scaled = int(max_attempts * (physical_size / 0.0005)**2)

    # Grid for overlap
    cell_size = 40 * mean_rad_solid_img
    n_cells = max(1, int(math.ceil(img_size / cell_size)))
    grid = [[[] for _ in range(n_cells)] for __ in range(n_cells)]

    def cell_coords(x, y):
        return max(0, min(n_cells - 1, int(x / cell_size))), max(0, min(n_cells - 1, int(y / cell_size)))

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

    circles = []
    voids = []
    
    # Track area for each particle type separately
    solid_area = 0.0
    hollow_area = 0.0
    porous_area = 0.0

    # -----------------------------
    # Place solid grains
    # -----------------------------
    mu_s = np.log(mean_rad_solid_img / np.sqrt(1 + rad_dev**2))
    sigma_s = np.sqrt(np.log(1 + rad_dev**2))
    radii_master_solid = [sample_radius(mu_s, sigma_s) for _ in range(max_attempts_scaled)]
    radii_master_solid.sort(reverse=True)

    consecutive_failures = 0
    max_consecutive_failures = 1500
    
    for r in radii_master_solid:
        if solid_area >= vol_percent_solid:
            break
        
        if consecutive_failures >= max_consecutive_failures:
            print(f"Stopping solid placement after {consecutive_failures} consecutive failures")
            break

        r = min(r, 0.5*img_size - margin)
        circle_area = math.pi*r**2 / (img_size*img_size)
        
        # Skip if this particle is too big for remaining space
        if solid_area + circle_area > vol_percent_solid:
            continue
        
        # Multiple attempts to place this particle
        placed = False
        for attempt in range(100):
            x = rng.uniform(-margin+r, img_size+margin-r)
            y = rng.uniform(-margin+r, img_size+margin-r)

            valid = True
            for (cx, cy, cr) in nearby_candidates(x, y, circles):
                tol = 1e-6 / physical_size * img_size
                if (cx-x)**2 + (cy-y)**2 < (cr + r - tol)**2:
                    valid = False
                    break
            
            if valid:
                circles.append((x, y, r))
                gx, gy = cell_coords(x, y)
                grid[gx][gy].append(len(circles)-1)

                # Update mask
                x_px = int(x / img_size * N_mask)
                y_px = int(y / img_size * N_mask)
                r_px = int(math.ceil(r / img_size * N_mask))
                xmin = max(0, x_px - r_px)
                xmax = min(N_mask, x_px + r_px + 1)
                ymin = max(0, y_px - r_px)
                ymax = min(N_mask, y_px + r_px + 1)
                if xmax > xmin and ymax > ymin:
                    ys = np.arange(ymin, ymax) - y_px
                    xs = np.arange(xmin, xmax) - x_px
                    XX, YY = np.meshgrid(xs, ys)
                    patch = (XX**2 + YY**2) <= r_px**2
                    mask[ymin:ymax, xmin:xmax] |= patch

                solid_area += circle_area
                placed = True
                consecutive_failures = 0
                break
        
        if not placed:
            consecutive_failures += 1

    print(f"After solid grains: solid area = {solid_area:.4f}")
    # -----------------------------
# Initialize global void tracking
# -----------------------------
    # -----------------------------
# Initialize global void tracking
# -----------------------------
    total_domain_area = img_size * img_size
    target_void_area = void_fraction * total_domain_area
    current_void_area = 0.0
    current_void_volume_3d = 0.0
# -----------------------------
# Place hollow grains
# -----------------------------
    if vol_percent_hollow > 0:
        mu_h = np.log(mean_rad_hollow_img / np.sqrt(1 + rad_dev**2))
        sigma_h = np.sqrt(np.log(1 + rad_dev**2))
        radii_master_hollow = [sample_radius(mu_h, sigma_h) for _ in range(max_attempts_scaled)]
        radii_master_hollow.sort(reverse=True)

        consecutive_failures = 0
        
        for r in radii_master_hollow:
            if hollow_area >= vol_percent_hollow:
                break
            if consecutive_failures >= max_consecutive_failures:
                print(f"Stopping hollow placement after {consecutive_failures} consecutive failures")
                break

            r = min(r, 0.5*img_size - margin)
            circle_area = math.pi*r**2 / (img_size*img_size)
            if hollow_area + circle_area > vol_percent_hollow:
                continue

            placed = False
            for attempt in range(100):
                x = rng.uniform(-margin+r, img_size+margin-r)
                y = rng.uniform(-margin+r, img_size+margin-r)

                valid = True
                for (cx, cy, cr) in nearby_candidates(x, y, circles):
                    tol = 1e-6 / physical_size * img_size
                    if (cx-x)**2 + (cy-y)**2 < (cr + r - tol)**2:
                        valid = False
                        break
                
                if valid:
                    # Compute void radius scaled to particle
                    void_radius = (void_fraction * r**3)**(1/3)

                    # Scale down if adding would exceed global void area
                    particle_void_area = math.pi * void_radius**2
                    if current_void_area + particle_void_area > target_void_area:
                        particle_void_area = max(0.0, target_void_area - current_void_area)
                        void_radius = np.sqrt(particle_void_area / np.pi)

                    circles.append((x, y, r))
                    voids.append((x, y, void_radius))
                    gx, gy = cell_coords(x, y)
                    grid[gx][gy].append(len(circles)-1)

                    # Update mask
                    x_px = int(x / img_size * N_mask)
                    y_px = int(y / img_size * N_mask)
                    r_px = int(math.ceil(r / img_size * N_mask))
                    vr_px = int(np.ceil(void_radius / img_size * N_mask))
                    xmin = max(0, x_px - r_px)
                    xmax = min(N_mask, x_px + r_px + 1)
                    ymin = max(0, y_px - r_px)
                    ymax = min(N_mask, y_px + r_px + 1)
                    if xmax > xmin and ymax > ymin:
                        ys = np.arange(ymin, ymax) - y_px
                        xs = np.arange(xmin, xmax) - x_px
                        XX, YY = np.meshgrid(xs, ys)
                        patch = (XX**2 + YY**2) <= r_px**2
                        void_patch = (XX**2 + YY**2) <= vr_px**2
                        solid_patch = patch & (~void_patch)
                        mask[ymin:ymax, xmin:xmax] |= solid_patch

                    hollow_area += circle_area
                    current_void_area += particle_void_area
                    # 3D volume tracking
                    # Convert back to physical size
                    # 3D volume tracking for hollow particle
                    void_radius_phys = void_radius * physical_size / img_size
                    current_void_volume_3d += (4/3) * np.pi * void_radius_phys**3



                    placed = True
                    consecutive_failures = 0
                    break

            if not placed:
                consecutive_failures += 1

    print(f"After hollow grains: hollow area = {hollow_area:.4f}, global void area = {current_void_area:.4f}")
    total_domain_volume_3d = physical_size**3  # m^3
    global_void_fraction_3d = current_void_volume_3d / total_domain_volume_3d
    print(f"Global void volume fraction =  {global_void_fraction_3d:.4f}")

# -----------------------------
# Place porous grains (non-overlapping pores, global void fraction)
# -----------------------------
    
    if vol_percent_porous > 0:
        mu_p = np.log(mean_rad_porous_img / np.sqrt(1 + rad_dev**2))
        sigma_p = np.sqrt(np.log(1 + rad_dev**2))
        radii_master_porous = [sample_radius(mu_p, sigma_p) for _ in range(max_attempts_scaled)]
        radii_master_porous.sort(reverse=True)

        consecutive_failures = 0

        for r in radii_master_porous:
            if porous_area >= vol_percent_porous:
                break
            if consecutive_failures >= max_consecutive_failures:
                print(f"Stopping porous placement after {consecutive_failures} consecutive failures")
                break

            r = min(r, 0.5 * img_size - margin)
            circle_area = math.pi * r**2 / (img_size * img_size)
            if porous_area + circle_area > vol_percent_porous:
                continue

            placed = False
            for attempt in range(100):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)

                valid = True
                for (cx, cy, cr) in nearby_candidates(x, y, circles):
                    tol = 1e-6 / physical_size * img_size
                    if (cx - x)**2 + (cy - y)**2 < (cr + r - tol)**2:
                        valid = False
                        break
                if not valid:
                    continue

                # -----------------------------
                # 3D-consistent pores
                # -----------------------------
                r_phys = r * physical_size / img_size
                particle_volume_3d = (4/3) * np.pi * r_phys**3
                particle_void_target_3d = void_fraction * particle_volume_3d

                max_pore_frac = 0.25
                max_pore_volume = (4/3) * np.pi * (r_phys * max_pore_frac)**3
                n_pores = max(1, int(np.ceil(particle_void_target_3d / max_pore_volume)))

                particle_void_current = 0.0
                placed_pores = []

                for _ in range(n_pores):
                    remaining_volume = particle_void_target_3d - particle_void_current
                    if remaining_volume <= 0:
                        break

                    pore_radius_phys = (3 * remaining_volume / (4 * np.pi) / (n_pores - len(placed_pores)))**(1/3)
                    pore_radius_img = pore_radius_phys / physical_size * img_size

                    for pore_attempt in range(50):
                        angle = rng.uniform(0, 2*np.pi)
                        dist = rng.uniform(0, r - pore_radius_img)
                        vx = x + dist * np.cos(angle)
                        vy = y + dist * np.sin(angle)

                        def is_valid_pore(vx, vy, vr, existing_pores):
                            for (px, py, pr) in existing_pores:
                                if (vx - px)**2 + (vy - py)**2 < (vr + pr)**2:
                                    return False
                            if (vx - x)**2 + (vy - y)**2 > (r - vr)**2:
                                return False
                            return True

                        if is_valid_pore(vx, vy, pore_radius_img, placed_pores):
                            placed_pores.append((vx, vy, pore_radius_img))
                            voids.append((vx, vy, pore_radius_img))
                            particle_void_current += (4/3) * np.pi * pore_radius_phys**3
                            current_void_volume_3d += (4/3) * np.pi * pore_radius_phys**3
                            current_void_area += np.pi * pore_radius_img**2
                            break

                # -----------------------------
                # Place particle and update mask
                # -----------------------------
                circles.append((x, y, r))
                gx, gy = cell_coords(x, y)
                grid[gx][gy].append(len(circles) - 1)

                x_px = int(x / img_size * N_mask)
                y_px = int(y / img_size * N_mask)
                r_px = int(np.ceil(r / img_size * N_mask))
                xmin = max(0, x_px - r_px)
                xmax = min(N_mask, x_px + r_px + 1)
                ymin = max(0, y_px - r_px)
                ymax = min(N_mask, y_px + r_px + 1)

                if xmax > xmin and ymax > ymin:
                    ys = np.arange(ymin, ymax) - y_px
                    xs = np.arange(xmin, xmax) - x_px
                    XX, YY = np.meshgrid(xs, ys)
                    mask_patch = (XX**2 + YY**2) <= r_px**2
                    mask[ymin:ymax, xmin:xmax] |= mask_patch

                    for (vx, vy, vr) in placed_pores:
                        vx_px = int(vx / img_size * N_mask)
                        vy_px = int(vy / img_size * N_mask)
                        vr_px = int(np.ceil(vr / img_size * N_mask))
                        XX_v, YY_v = np.meshgrid(xs, ys)  # local coordinates
                        void_patch = (XX_v + (vx_px - x_px))**2 + (YY_v + (vy_px - y_px))**2 <= vr_px**2
                        mask[ymin:ymax, xmin:xmax] &= ~void_patch

                porous_area += circle_area
                placed = True
                consecutive_failures = 0
                break

            if not placed:
                consecutive_failures += 1

        print(f"After porous grains: porous area = {porous_area:.4f}, global void area = {current_void_area:.4f}")
        total_domain_volume_3d = physical_size**3
        global_void_fraction_3d = current_void_volume_3d / total_domain_volume_3d
        print(f"Global void volume fraction = {global_void_fraction_3d:.4f}")




                # -----------------------------
                # Update mask (2D projection)
                # -----------------------------
    x_px = int(x / img_size * N_mask)
    y_px = int(y / img_size * N_mask)
    r_px = int(np.ceil(r / img_size * N_mask))
    xmin = max(0, x_px - r_px)
    xmax = min(N_mask, x_px + r_px + 1)
    ymin = max(0, y_px - r_px)
    ymax = min(N_mask, y_px + r_px + 1)

    if xmax > xmin and ymax > ymin:
        ys = np.arange(ymin, ymax) - y_px
        xs = np.arange(xmin, xmax) - x_px
        XX, YY = np.meshgrid(xs, ys)
        patch = (XX**2 + YY**2) <= r_px**2
        mask[ymin:ymax, xmin:xmax] |= patch

    for (vx, vy, vr) in placed_pores:
        vx_px = int(vx / img_size * N_mask)
        vy_px = int(vy / img_size * N_mask)
        vr_px = int(np.ceil(vr / img_size * N_mask))
        ys_v = np.arange(ymin, ymax) - vy_px
        xs_v = np.arange(xmin, xmax) - vx_px
        XX_v, YY_v = np.meshgrid(xs_v, ys_v)
        void_patch = (XX_v**2 + YY_v**2) <= vr_px**2
        mask[ymin:ymax, xmin:xmax] &= ~void_patch

        porous_area += circle_area
        placed = True
        consecutive_failures = 0
        break

    if not placed:
        consecutive_failures += 1

        print(f"After porous grains: porous area = {porous_area:.4f}, global void area = {current_void_area:.4f}")
        total_domain_volume_3d = physical_size**3  # m^3
        global_void_fraction_3d = current_void_volume_3d / total_domain_volume_3d
        print(f"Global void volume fraction =  {global_void_fraction_3d:.4f}")

        total_ap_area = solid_area + hollow_area + porous_area
        print(f"Total AP area = {total_ap_area:.4f}")

    # -----------------------------
    # Plotting
    # -----------------------------
    fig2, ax2 = plt.subplots(figsize=(6,6), dpi=dpi_highres)
    ax2.set_position([0,0,1,1])
    ax2.set_axis_off()
    ax2.set_xlim(0,img_size)
    ax2.set_ylim(0,img_size)
    ax2.set_aspect("equal")
    ax2.add_patch(plt.Rectangle((0,0), img_size, img_size, facecolor='#0000FF', zorder=0))
    for (x,y,r) in circles:
        ax2.add_patch(Circle((x,y), r, facecolor='#FF0000', edgecolor='#800080',
                              linewidth=interface_width/(physical_size/img_size)*72, zorder=5))
        ax2.add_patch(Circle((x,y), r, facecolor='none', edgecolor='#FFFFFF', linewidth=72/dpi_highres, zorder=10))
    for (x,y,r) in voids:
        ax2.add_patch(Circle((x,y), r, facecolor='#0000FF', edgecolor='#800080',
                              linewidth=interface_width/(physical_size/img_size)*72, zorder=5))
        ax2.add_patch(Circle((x,y), r, facecolor='none', edgecolor='#FFFFFF', linewidth=72/dpi_highres, zorder=10))
    fig2.savefig(save_path_untitled, dpi=dpi_highres, bbox_inches=None, pad_inches=0.0)
    plt.close(fig2)

    fig, ax = plt.subplots(figsize=(6,6), dpi=1024//6)
    ax.set_axis_off()
    ax.set_xlim(0,img_size)
    ax.set_ylim(0,img_size)
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0,0), img_size, img_size, facecolor='#0000FF', zorder=0))
    for (x,y,r) in circles:
        ax.add_patch(Circle((x,y), r, facecolor='#FF0000', edgecolor='#800080',
                            linewidth=interface_width/(physical_size/img_size)*72, zorder=5))
        ax.add_patch(Circle((x,y), r, facecolor='none', edgecolor='#FFFFFF', linewidth=1, zorder=10))
    for (x,y,r) in voids:
        ax.add_patch(Circle((x,y), r, facecolor='#0000FF', edgecolor='#800080',
                            linewidth=interface_width/(physical_size/img_size)*72, zorder=5))
        ax.add_patch(Circle((x,y), r, facecolor='none', edgecolor='#FFFFFF', linewidth=1, zorder=10))
    ax.set_title(f"Placed {len(circles)} grains, achieved AP = {total_ap_area:.3f}")
    fig.savefig(save_path, dpi=1024//6, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    save_xyzr(circles, AP_xyzr, img_size, physical_size)
    save_xyzr(voids, void_xyzr, img_size, physical_size)

    return save_path, save_path_untitled, total_ap_area, AP_xyzr, void_xyzr
'''

def save_xyzr(circles, filepath, img_size, physical_size):
    scale = physical_size / img_size
    with open(filepath, "w") as f:
        for (x, y, r) in circles:
            f.write(f"{x*scale:.6e} {y*scale:.6e} 0.0 {r*scale:.6e}\n")

def generate_pore_radii(
    target_area,
    parent_radius,
    rng,
    mean_frac=0.18,
    std_frac=0.35,
    r_min_frac=0.06,
    r_max_frac=0.45,
    max_pores=200
):
    radii = []
    area_sum = 0.0

    mu = np.log(mean_frac / np.sqrt(1 + std_frac**2))
    sigma = np.sqrt(np.log(1 + std_frac**2))

    for _ in range(max_pores):
        if area_sum >= 0.98 * target_area:
            break

        frac = rng.lognormal(mu, sigma)
        frac = np.clip(frac, r_min_frac, r_max_frac)
        r = frac * parent_radius
        a = np.pi * r**2

        if area_sum + a > target_area:
            a = target_area - area_sum
            if a <= 0:
                break
            r = np.sqrt(a / np.pi)

        radii.append(r)
        area_sum += np.pi * r**2

    return radii, area_sum


def place_pores_in_particle(
    cx, cy, parent_r,
    pore_radii,
    rng,
    max_attempts_per_pore=600,
    min_gap_frac=0.03,
    gap_relax=0.9,
    max_relax_steps=6
):
    placed = []

    for pr in pore_radii:
        placed_this = False
        gap_frac = min_gap_frac

        for _ in range(max_relax_steps):
            max_dist = parent_r - pr * 1.02
            if max_dist <= 0:
                break

            for _ in range(max_attempts_per_pore):
                theta = rng.uniform(0, 2*np.pi)
                dist = np.sqrt(rng.uniform(0, 1)) * max_dist
                x = cx + dist * np.cos(theta)
                y = cy + dist * np.sin(theta)

                valid = True
                for (px, py, pr0) in placed:
                    gap = gap_frac * min(pr, pr0)
                    if (x-px)**2 + (y-py)**2 < (pr + pr0 + gap)**2:
                        valid = False
                        break

                if valid:
                    placed.append((x, y, pr))
                    placed_this = True
                    break

            if placed_this:
                break

            gap_frac *= gap_relax

        # zero-gap fallback
        if not placed_this:
            for _ in range(max_attempts_per_pore):
                theta = rng.uniform(0, 2*np.pi)
                dist = np.sqrt(rng.uniform(0, 1)) * (parent_r - pr)
                x = cx + dist * np.cos(theta)
                y = cy + dist * np.sin(theta)

                valid = True
                for (px, py, pr0) in placed:
                    if (x-px)**2 + (y-py)**2 < (pr + pr0)**2:
                        valid = False
                        break

                if valid:
                    placed.append((x, y, pr))
                    placed_this = True
                    break

        if not placed_this:
            return placed, False

    return placed, True

'''

def gen_struct_combined(
    save_path, save_path_untitled, AP_xyzr, void_xyzr,
    img_size, physical_size,
    rad_dev, max_attempts,
    vol_percent_solid, vol_percent_hollow, vol_percent_porous, void_fraction,
    mean_rad_solid=120e-6, mean_rad_hollow=2e-6, mean_rad_porous=4.5e-6,
    max_tries=1, interface_width=1e-7, N_mask_base=1024, dpi_highres=100
):
    rng = np.random.default_rng()
    margin = 0.01

    # Convert mean radii to image units
    mean_rad_solid_img = mean_rad_solid / physical_size * img_size
    mean_rad_hollow_img = mean_rad_hollow / physical_size * img_size
    mean_rad_porous_img = mean_rad_porous / physical_size * img_size

    r_min = 0.2e-6 / physical_size * img_size
    r_max = 150e-6 / physical_size * img_size

    # Scale mask resolution for large domains
    N_mask = int(N_mask_base * (physical_size / 0.0005))
    mask = np.zeros((N_mask, N_mask), dtype=bool)

    # Adjust max_attempts for domain size
    max_attempts_scaled = int(max_attempts * (physical_size / 0.0005)**2)

    # Grid for overlap
    cell_size = 40 * mean_rad_solid_img
    n_cells = max(1, int(math.ceil(img_size / cell_size)))
    grid = [[[] for _ in range(n_cells)] for __ in range(n_cells)]

    def cell_coords(x, y):
        return max(0, min(n_cells - 1, int(x / cell_size))), max(0, min(n_cells - 1, int(y / cell_size)))

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

    circles = []
    voids = []
    
    # Track area for each particle type separately
    solid_area = 0.0
    hollow_area = 0.0
    porous_area = 0.0

    # -----------------------------
    # Place solid grains
    # -----------------------------
    mu_s = np.log(mean_rad_solid_img / np.sqrt(1 + rad_dev**2))
    sigma_s = np.sqrt(np.log(1 + rad_dev**2))
    radii_master_solid = [sample_radius(mu_s, sigma_s) for _ in range(max_attempts_scaled)]
    radii_master_solid.sort(reverse=True)

    consecutive_failures = 0
    max_consecutive_failures = 500
    
    for r in radii_master_solid:
        if solid_area >= vol_percent_solid:
            break
        
        if consecutive_failures >= max_consecutive_failures:
            print(f"Stopping solid placement after {consecutive_failures} consecutive failures")
            break

        r = min(r, 0.5*img_size - margin)
        circle_area = math.pi*r**2 / (img_size*img_size)
        
        # Skip if this particle is too big for remaining space
        if solid_area + circle_area > vol_percent_solid:
            continue
        
        # Multiple attempts to place this particle
        placed = False
        for attempt in range(100):
            x = rng.uniform(-margin+r, img_size+margin-r)
            y = rng.uniform(-margin+r, img_size+margin-r)

            valid = True
            for (cx, cy, cr) in nearby_candidates(x, y, circles):
                tol = 1e-6 / physical_size * img_size
                if (cx-x)**2 + (cy-y)**2 < (cr + r - tol)**2:
                    valid = False
                    break
            
            if valid:
                circles.append((x, y, r))
                gx, gy = cell_coords(x, y)
                grid[gx][gy].append(len(circles)-1)

                # Update mask
                x_px = int(x / img_size * N_mask)
                y_px = int(y / img_size * N_mask)
                r_px = int(math.ceil(r / img_size * N_mask))
                xmin = max(0, x_px - r_px)
                xmax = min(N_mask, x_px + r_px + 1)
                ymin = max(0, y_px - r_px)
                ymax = min(N_mask, y_px + r_px + 1)
                if xmax > xmin and ymax > ymin:
                    ys = np.arange(ymin, ymax) - y_px
                    xs = np.arange(xmin, xmax) - x_px
                    XX, YY = np.meshgrid(xs, ys)
                    patch = (XX**2 + YY**2) <= r_px**2
                    mask[ymin:ymax, xmin:xmax] |= patch

                solid_area += circle_area
                placed = True
                consecutive_failures = 0
                break
        
        if not placed:
            consecutive_failures += 1

    print(f"After solid grains: solid area = {solid_area:.4f}")

    # -----------------------------
    # Place hollow grains
    # -----------------------------
    if vol_percent_hollow > 0:
        mu_h = np.log(mean_rad_hollow_img / np.sqrt(1 + rad_dev**2))
        sigma_h = np.sqrt(np.log(1 + rad_dev**2))
        radii_master_hollow = [sample_radius(mu_h, sigma_h) for _ in range(max_attempts_scaled)]
        radii_master_hollow.sort(reverse=True)

        consecutive_failures = 0
        
        for r in radii_master_hollow:
            if hollow_area >= vol_percent_hollow:
                break
            
            if consecutive_failures >= max_consecutive_failures:
                print(f"Stopping hollow placement after {consecutive_failures} consecutive failures")
                break

            r = min(r, 0.5*img_size - margin)
            circle_area = math.pi*r**2 / (img_size*img_size)
            
            # Skip if this particle is too big for remaining space
            if hollow_area + circle_area > vol_percent_hollow:
                continue
            
            # Multiple attempts to place this particle
            placed = False
            for attempt in range(100):
                x = rng.uniform(-margin+r, img_size+margin-r)
                y = rng.uniform(-margin+r, img_size+margin-r)

                valid = True
                for (cx, cy, cr) in nearby_candidates(x, y, circles):
                    tol = 1e-6 / physical_size * img_size
                    if (cx-x)**2 + (cy-y)**2 < (cr + r - tol)**2:
                        valid = False
                        break
                
                if valid:
                    void_radius = (void_fraction * r**3)**(1/3)
                    circles.append((x, y, r))
                    voids.append((x, y, void_radius))
                    gx, gy = cell_coords(x, y)
                    grid[gx][gy].append(len(circles)-1)

                    # Update mask (only the solid shell, not the void)
                    x_px = int(x / img_size * N_mask)
                    y_px = int(y / img_size * N_mask)
                    r_px = int(math.ceil(r / img_size * N_mask))
                    vr_px = int(np.ceil(void_radius / img_size * N_mask))
                    xmin = max(0, x_px - r_px)
                    xmax = min(N_mask, x_px + r_px + 1)
                    ymin = max(0, y_px - r_px)
                    ymax = min(N_mask, y_px + r_px + 1)
                    if xmax > xmin and ymax > ymin:
                        ys = np.arange(ymin, ymax) - y_px
                        xs = np.arange(xmin, xmax) - x_px
                        XX, YY = np.meshgrid(xs, ys)
                        patch = (XX**2 + YY**2) <= r_px**2
                        void_patch = (XX**2 + YY**2) <= vr_px**2
                        solid_patch = patch & (~void_patch)
                        mask[ymin:ymax, xmin:xmax] |= solid_patch

                    hollow_area += circle_area
                    placed = True
                    consecutive_failures = 0
                    break
            
            if not placed:
                consecutive_failures += 1

    print(f"After hollow grains: hollow area = {hollow_area:.4f}")

    if vol_percent_porous > 0:
        mu = np.log(mean_rad_porous_img / np.sqrt(1 + rad_dev**2))
        sigma = np.sqrt(np.log(1 + rad_dev**2))
        radii = rng.lognormal(mu, sigma, max_attempts)
        radii.sort(reverse=True)

        for r in radii:
            if porous_area >= vol_percent_porous:
                break

            r = min(r, 0.5*img_size - margin)
            area_frac = np.pi*r**2 / total_domain_area
            if porous_area + area_frac > vol_percent_porous:
                continue

            x = rng.uniform(r, img_size-r)
            y = rng.uniform(r, img_size-r)

            particle_void_target = void_fraction * np.pi * r**2
            remaining_void = target_void_area - current_void_area
            particle_void_target = min(particle_void_target, remaining_void)

            pore_radii, _ = generate_pore_radii(
                particle_void_target, r, rng
            )

            placed_pores, success = place_pores_in_particle(
                x, y, r, pore_radii, rng
            )

            if not success:
                continue

            circles.append((x, y, r))
            porous_area += area_frac

            for (vx, vy, vr) in placed_pores:
                voids.append((vx, vy, vr))
                current_void_area += np.pi * vr**2

    save_xyzr(circles, AP_xyzr, img_size, physical_size)
    save_xyzr(voids, void_xyzr, img_size, physical_size)

    return save_path, save_path_untitled, porous_area, AP_xyzr, void_xyzr

'''



import numpy as np
import math
import os

# ------------------------------------------------------------
# Utility: save xyzr
# ------------------------------------------------------------
def save_xyzr(circles, filepath, img_size, physical_size):
    scale = physical_size / img_size
    with open(filepath, "w") as f:
        for (x, y, r) in circles:
            f.write(f"{x*scale:.6e} {y*scale:.6e} 0.0 {r*scale:.6e}\n")


# ------------------------------------------------------------
# Generate pore radii to hit target void area
# ------------------------------------------------------------
def generate_pore_radii(target_void_area, particle_radius, rng,
                        mean_ratio=0.15, max_ratio=0.4):
    """
    Generate pore radii whose total area matches target_void_area.
    """
    radii = []
    current_area = 0.0

    mean_r = mean_ratio * particle_radius
    max_r = max_ratio * particle_radius

    while current_area < target_void_area:
        r = rng.lognormal(np.log(mean_r), 0.4)
        r = min(r, max_r)

        area = np.pi * r**2
        if current_area + area > target_void_area:
            r = math.sqrt((target_void_area - current_area) / np.pi)
            if r <= 0:
                break
            area = np.pi * r**2

        radii.append(r)
        current_area += area

    return radii, current_area


# ------------------------------------------------------------
# Place pores inside a particle (retry instead of discard)
# ------------------------------------------------------------
def place_pores_in_particle(cx, cy, R, pore_radii, rng,
                            max_attempts_per_pore=200):
    placed = []

    for r in pore_radii:
        success = False

        for _ in range(max_attempts_per_pore):
            theta = rng.uniform(0, 2*np.pi)
            rho = rng.uniform(0, R - r)

            x = cx + rho * np.cos(theta)
            y = cy + rho * np.sin(theta)

            valid = True
            for (px, py, pr) in placed:
                if (px-x)**2 + (py-y)**2 < (pr + r)**2:
                    valid = False
                    break

            if valid:
                placed.append((x, y, r))
                success = True
                break

        if not success:
            return placed, False

    return placed, True


# ------------------------------------------------------------
# Main generator
# ------------------------------------------------------------


import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle



def gen_struct_combined_no_img(
    save_path, save_path_untitled, AP_xyzr, void_xyzr,
    img_size, physical_size,
    rad_dev, max_attempts,
    vol_percent_solid, vol_percent_hollow, vol_percent_porous, void_fraction,
    mean_rad_solid=120e-6, mean_rad_hollow=2e-6, mean_rad_porous=4.5e-6,
    max_tries=1, interface_width=1e-7,
    N_mask_base=1024, dpi_highres=100
):
    """
    2D microstructure generator with exact global void fraction control.
    Assumes 2D void fraction == 3D void fraction.

    Hollow grains: 1 concentric void
    Porous grains: 2-5 pores per particle, exact void area
    """

    rng = np.random.default_rng()
    margin = 0.07

    # -----------------------------
    # Global void bookkeeping
    # -----------------------------
    total_domain_area = img_size * img_size
    target_void_area = void_fraction * total_domain_area
    current_void_area = 0.0

    porosity_min = 0.05
    porosity_max = 0.25

    print(f"\n{'='*60}")
    print("TARGET PARAMETERS")
    print(f"{'='*60}")
    print(f"Target void fraction (global): {void_fraction:.4f}")
    print(f"Target void area (px²): {target_void_area:.2e}")
    print(f"{'='*60}\n")

    # -----------------------------
    # Radius helpers
    # -----------------------------
    def img_r(mu):
        return mu / physical_size * img_size

    def sample_radius(mu, sigma):
        return rng.lognormal(mu, sigma)

    def pore_radii_2_to_5(target_area, r_particle, rng):
        n_pores = rng.integers(2, 6)
        alpha = np.ones(n_pores)
        alpha[0] = 3.0  # bias first pore a bit larger
        areas = rng.dirichlet(alpha) * target_area
        return [min(np.sqrt(a / np.pi), 0.45 * r_particle) for a in areas]

    # -----------------------------
    # Spatial grid
    # -----------------------------
    cell_size = img_r(mean_rad_solid) * 4
    n_cells = max(1, int(math.ceil(img_size / cell_size)))
    grid = [[[] for _ in range(n_cells)] for __ in range(n_cells)]

    def cell_coords(x, y):
        return int(x // cell_size), int(y // cell_size)

    def nearby(x, y, circles):
        cx, cy = cell_coords(x, y)
        for i in range(max(0, cx-1), min(n_cells, cx+2)):
            for j in range(max(0, cy-1), min(n_cells, cy+2)):
                for idx in grid[i][j]:
                    yield circles[idx]

    def place_pores_for_particle(
        x0, y0, r_particle,
        pore_radii,
        existing_voids,
        max_attempts=500
    ):
        """
        Try to place ALL pores for a porous particle without shrinking.
        If any pore fails, the entire pore set is rejected.
        """
        placed_pores = []

        for rp in pore_radii:
            success = False

            for _ in range(max_attempts):
                theta = np.random.uniform(0, 2*np.pi)
                rad = np.random.uniform(0, r_particle - rp)

                xp = x0 + rad * np.cos(theta)
                yp = y0 + rad * np.sin(theta)

                # Must stay inside particle
                if np.hypot(xp - x0, yp - y0) + rp > r_particle:
                    continue

                # Must not overlap other pores in same particle
                if any(np.hypot(xp - px, yp - py) < rp + pr for (px, py, pr) in placed_pores):
                    continue

                # Must not overlap global voids
                if any(np.hypot(xp - vx, yp - vy) < rp + vr for (vx, vy, vr) in existing_voids):
                    continue

                placed_pores.append((xp, yp, rp))
                success = True
                break

            if not success:
                return None  # reject entire pore set

        return placed_pores

    circles = []
    voids = []

    solid_area = hollow_area = porous_area = 0.0

    # -----------------------------
    # SOLID GRAINS
    # -----------------------------
    print("Placing solid grains...")
    mu = math.log(img_r(mean_rad_solid))
    sigma = rad_dev

    for _ in range(max_attempts):
        if solid_area >= vol_percent_solid:
            break

        r = sample_radius(mu, sigma)
        A = math.pi * r**2 / total_domain_area
        if solid_area + A > vol_percent_solid:
            continue

        for _ in range(100):
            x = rng.uniform(-margin + r, img_size + margin - r)
            y = rng.uniform(-margin + r, img_size + margin - r)

            if all((x-cx)**2 + (y-cy)**2 >= (r+cr)**2 for cx,cy,cr in nearby(x,y,circles)):
                circles.append((x,y,r))
                grid[cell_coords(x,y)[0]][cell_coords(x,y)[1]].append(len(circles)-1)
                solid_area += A
                break

    print(f"  Solid area fraction: {solid_area:.4f}")

    # -----------------------------
    # HOLLOW GRAINS
    # -----------------------------
    print("\nPlacing hollow grains...")
    mu = math.log(img_r(mean_rad_hollow))

    for _ in range(max_attempts):
        if hollow_area >= vol_percent_hollow:
            break

        r = sample_radius(mu, sigma)
        A = math.pi * r**2 / total_domain_area
        if hollow_area + A > vol_percent_hollow:
            continue

        for _ in range(100):
            x = rng.uniform(r, img_size-r)
            y = rng.uniform(r, img_size-r)
            if not any((x-cx)**2 + (y-cy)**2 < (r+cr)**2 for cx,cy,cr in nearby(x,y,circles)):
                #convert void fraction ov erall to that of hollow particles
                f_void_hollow = void_fraction * total_domain_area / (vol_percent_hollow * total_domain_area)
              

                particle_void = f_void_hollow * math.pi * r**2
                particle_void = min(particle_void, target_void_area - current_void_area)

                rv = r * np.sqrt(f_void_hollow)


                circles.append((x,y,r))
                voids.append((x,y,rv))
                grid[cell_coords(x,y)[0]][cell_coords(x,y)[1]].append(len(circles)-1)

                hollow_area += A
                current_void_area += particle_void
                break

    print(f"  Hollow area fraction: {hollow_area:.4f}")
    print(f"  Void fraction so far: {current_void_area/total_domain_area:.4f}")

    # -----------------------------
    # POROUS GRAINS
    # -----------------------------
    print("\nPlacing porous grains...")
    mu = math.log(img_r(mean_rad_porous))

    for _ in range(max_attempts):
        if porous_area >= vol_percent_porous:
            break

        r = sample_radius(mu, sigma)
        A = math.pi * r**2 / total_domain_area
        if porous_area + A > vol_percent_porous:
            continue

        for _ in range(100):
            x = rng.uniform(r, img_size-r)
            y = rng.uniform(r, img_size-r)
            if any((x-cx)**2 + (y-cy)**2 < (r+cr)**2 for cx,cy,cr in nearby(x,y,circles)):
                continue

            phi_pore = rng.uniform(porosity_min, porosity_max)
            target_pore_area = phi_pore * np.pi * r**2
            target_pore_area = min(target_pore_area, target_void_area - current_void_area)

            for regen in range(10):
                pore_radii = pore_radii_2_to_5(target_pore_area, r, rng)
                placed = place_pores_for_particle(x, y, r, pore_radii, voids)
                if placed is not None:
                    voids.extend(placed)
                    current_void_area += target_pore_area
                    break
            else:
                raise RuntimeError("Failed to place pores without overlap")

            circles.append((x,y,r))
            grid[cell_coords(x,y)[0]][cell_coords(x,y)[1]].append(len(circles)-1)
            porous_area += A

            #print(f"  Porous grain r={r*physical_size/img_size*1e6:.1f} µm | pores={len(pore_radii)}")
            break

    # -----------------------------
    # FINAL SUMMARY
    # -----------------------------
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total grains: {len(circles)}")
    print(f"Total voids:  {len(voids)}")
    print(f"Void fraction: {current_void_area/total_domain_area:.4f}")
    print(f"Target:        {void_fraction:.4f}")
    print(f"Error:         {abs(current_void_area/total_domain_area - void_fraction):.2e}")
    print(f"{'='*60}\n")


    # -----------------------------
    # Save xyzr
    # -----------------------------
    save_xyzr(circles, AP_xyzr, img_size, physical_size)
    save_xyzr(voids, void_xyzr, img_size, physical_size)

    return current_void_area / total_domain_area

def gen_struct_combined(
    save_path, save_path_untitled, AP_xyzr, void_xyzr,
    img_size, physical_size,
    rad_dev, max_attempts,
    vol_percent_solid, vol_percent_hollow, vol_percent_porous, void_fraction,
    mean_rad_solid=120e-6, mean_rad_hollow=2e-6, mean_rad_porous=4.5e-6,
    max_tries=1, interface_width=1e-7,
    N_mask_base=1024, dpi_highres=100
):
    """
    2D microstructure generator with exact global void fraction control.
    Assumes 2D void fraction == 3D void fraction.

    Hollow grains: 1 concentric void
    Porous grains: 2-5 pores per particle, exact void area
    """

    rng = np.random.default_rng()
    margin = 0.07

    # -----------------------------
    # Global void bookkeeping
    # -----------------------------
    total_domain_area = img_size * img_size
    target_void_area = void_fraction * total_domain_area
    current_void_area = 0.0

    porosity_min = 0.05
    porosity_max = 0.25

    print(f"\n{'='*60}")
    print("TARGET PARAMETERS")
    print(f"{'='*60}")
    print(f"Target void fraction (global): {void_fraction:.4f}")
    print(f"Target void area (px²): {target_void_area:.2e}")
    print(f"{'='*60}\n")

    # -----------------------------
    # Radius helpers
    # -----------------------------
    def img_r(mu):
        return mu / physical_size * img_size

    def sample_radius(mu, sigma):
        return rng.lognormal(mu, sigma)

    def pore_radii_2_to_5(target_area, r_particle, rng):
        n_pores = rng.integers(2, 6)
        alpha = np.ones(n_pores)
        alpha[0] = 3.0  # bias first pore a bit larger
        areas = rng.dirichlet(alpha) * target_area
        return [min(np.sqrt(a / np.pi), 0.45 * r_particle) for a in areas]

    # -----------------------------
    # Spatial grid
    # -----------------------------
    cell_size = img_r(mean_rad_solid) * 4
    n_cells = max(1, int(math.ceil(img_size / cell_size)))
    grid = [[[] for _ in range(n_cells)] for __ in range(n_cells)]

    def cell_coords(x, y):
        return int(x // cell_size), int(y // cell_size)

    def nearby(x, y, circles):
        cx, cy = cell_coords(x, y)
        for i in range(max(0, cx-1), min(n_cells, cx+2)):
            for j in range(max(0, cy-1), min(n_cells, cy+2)):
                for idx in grid[i][j]:
                    yield circles[idx]

    def place_pores_for_particle(
        x0, y0, r_particle,
        pore_radii,
        existing_voids,
        max_attempts=500
    ):
        """
        Try to place ALL pores for a porous particle without shrinking.
        If any pore fails, the entire pore set is rejected.
        """
        placed_pores = []

        for rp in pore_radii:
            success = False

            for _ in range(max_attempts):
                theta = np.random.uniform(0, 2*np.pi)
                rad = np.random.uniform(0, r_particle - rp)

                xp = x0 + rad * np.cos(theta)
                yp = y0 + rad * np.sin(theta)

                # Must stay inside particle
                if np.hypot(xp - x0, yp - y0) + rp > r_particle:
                    continue

                # Must not overlap other pores in same particle
                if any(np.hypot(xp - px, yp - py) < rp + pr for (px, py, pr) in placed_pores):
                    continue

                # Must not overlap global voids
                if any(np.hypot(xp - vx, yp - vy) < rp + vr for (vx, vy, vr) in existing_voids):
                    continue

                placed_pores.append((xp, yp, rp))
                success = True
                break

            if not success:
                return None  # reject entire pore set

        return placed_pores

    circles = []
    voids = []

    solid_area = hollow_area = porous_area = 0.0

    # -----------------------------
    # SOLID GRAINS
    # -----------------------------
    print("Placing solid grains...")
    mu = math.log(img_r(mean_rad_solid))
    sigma = rad_dev

    for _ in range(max_attempts):
        if solid_area >= vol_percent_solid:
            break

        r = sample_radius(mu, sigma)
        A = math.pi * r**2 / total_domain_area
        if solid_area + A > vol_percent_solid:
            continue

        for _ in range(100):
            x = rng.uniform(-margin + r, img_size + margin - r)
            y = rng.uniform(-margin + r, img_size + margin - r)

            if all((x-cx)**2 + (y-cy)**2 >= (r+cr)**2 for cx,cy,cr in nearby(x,y,circles)):
                circles.append((x,y,r))
                grid[cell_coords(x,y)[0]][cell_coords(x,y)[1]].append(len(circles)-1)
                solid_area += A
                break

    print(f"  Solid area fraction: {solid_area:.4f}")

    # -----------------------------
    # HOLLOW GRAINS
    # -----------------------------
    print("\nPlacing hollow grains...")
    mu = math.log(img_r(mean_rad_hollow))

    for _ in range(max_attempts):
        if hollow_area >= vol_percent_hollow:
            break

        r = sample_radius(mu, sigma)
        A = math.pi * r**2 / total_domain_area
        if hollow_area + A > vol_percent_hollow:
            continue

        for _ in range(100):
            x = rng.uniform(r, img_size-r)
            y = rng.uniform(r, img_size-r)
            if not any((x-cx)**2 + (y-cy)**2 < (r+cr)**2 for cx,cy,cr in nearby(x,y,circles)):
                #convert void fraction ov erall to that of hollow particles
                f_void_hollow = void_fraction * total_domain_area / (vol_percent_hollow * total_domain_area)
              

                particle_void = f_void_hollow * math.pi * r**2
                particle_void = min(particle_void, target_void_area - current_void_area)

                rv = r * np.sqrt(f_void_hollow)


                circles.append((x,y,r))
                voids.append((x,y,rv))
                grid[cell_coords(x,y)[0]][cell_coords(x,y)[1]].append(len(circles)-1)

                hollow_area += A
                current_void_area += particle_void
                break

    print(f"  Hollow area fraction: {hollow_area:.4f}")
    print(f"  Void fraction so far: {current_void_area/total_domain_area:.4f}")

    # -----------------------------
    # POROUS GRAINS
    # -----------------------------
    print("\nPlacing porous grains...")
    mu = math.log(img_r(mean_rad_porous))

    for _ in range(max_attempts):
        if porous_area >= vol_percent_porous:
            break

        r = sample_radius(mu, sigma)
        A = math.pi * r**2 / total_domain_area
        if porous_area + A > vol_percent_porous:
            continue

        for _ in range(100):
            x = rng.uniform(r, img_size-r)
            y = rng.uniform(r, img_size-r)
            if any((x-cx)**2 + (y-cy)**2 < (r+cr)**2 for cx,cy,cr in nearby(x,y,circles)):
                continue

            phi_pore = rng.uniform(porosity_min, porosity_max)
            target_pore_area = phi_pore * np.pi * r**2
            target_pore_area = min(target_pore_area, target_void_area - current_void_area)

            for regen in range(10):
                pore_radii = pore_radii_2_to_5(target_pore_area, r, rng)
                placed = place_pores_for_particle(x, y, r, pore_radii, voids)
                if placed is not None:
                    voids.extend(placed)
                    current_void_area += target_pore_area
                    break
            else:
                raise RuntimeError("Failed to place pores without overlap")

            circles.append((x,y,r))
            grid[cell_coords(x,y)[0]][cell_coords(x,y)[1]].append(len(circles)-1)
            porous_area += A

            #print(f"  Porous grain r={r*physical_size/img_size*1e6:.1f} µm | pores={len(pore_radii)}")
            break

    # -----------------------------
    # FINAL SUMMARY
    # -----------------------------
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total grains: {len(circles)}")
    print(f"Total voids:  {len(voids)}")
    print(f"Void fraction: {current_void_area/total_domain_area:.4f}")
    print(f"Target:        {void_fraction:.4f}")
    print(f"Error:         {abs(current_void_area/total_domain_area - void_fraction):.2e}")
    print(f"{'='*60}\n")

    # -----------------------------
    # Plotting
    # -----------------------------
    
    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=dpi_highres)
    ax2.set_position([0, 0, 1, 1])
    ax2.set_axis_off()
    ax2.set_xlim(0, img_size)
    ax2.set_ylim(0, img_size)
    ax2.set_aspect("equal")

    # Background (binder / void)
    ax2.add_patch(plt.Rectangle((0, 0), img_size, img_size, facecolor='#0000FF', zorder=0))

    # AP grains
    for (x, y, r) in circles:
        ax2.add_patch(Circle((x, y), r, facecolor='#FF0000', edgecolor='#800080',
                              linewidth=interface_width / (physical_size / img_size) * 72, zorder=5))
        ax2.add_patch(Circle((x, y), r, facecolor='none', edgecolor='#FFFFFF',
                              linewidth=72 / dpi_highres, zorder=10))

    # Voids
    for (x, y, r) in voids:
        ax2.add_patch(Circle((x, y), r, facecolor='#0000FF', edgecolor='#800080',
                              linewidth=interface_width / (physical_size / img_size) * 72, zorder=6))
        ax2.add_patch(Circle((x, y), r, facecolor='none', edgecolor='#FFFFFF',
                              linewidth=72 / dpi_highres, zorder=10))

    fig2.savefig(save_path_untitled, dpi=dpi_highres, bbox_inches=None, pad_inches=0.0)
    plt.close(fig2)

    # -----------------------------
    # Annotated / lower-res figure
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 6), dpi=1024 // 6)
    ax.set_axis_off()
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    ax.set_aspect("equal")

    ax.add_patch(plt.Rectangle((0, 0), img_size, img_size, facecolor='#0000FF', zorder=0))

    for (x, y, r) in circles:
        ax.add_patch(Circle((x, y), r, facecolor='#FF0000', edgecolor='#800080',
                             linewidth=interface_width / (physical_size / img_size) * 72, zorder=5))
        ax.add_patch(Circle((x, y), r, facecolor='none', edgecolor='#FFFFFF',
                             linewidth=1, zorder=10))

    for (x, y, r) in voids:
        ax.add_patch(Circle((x, y), r, facecolor='#0000FF', edgecolor='#800080',
                             linewidth=interface_width / (physical_size / img_size) * 72, zorder=6))
        ax.add_patch(Circle((x, y), r, facecolor='none', edgecolor='#FFFFFF',
                             linewidth=1, zorder=10))

    void_frac_actual = current_void_area / total_domain_area
    ax.set_title(f"Placed {len(circles)} grains | AP = {(solid_area + hollow_area + porous_area):.3f} | Void = {void_frac_actual:.3f}")

    fig.savefig(save_path, dpi=1024 // 6, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    
    # -----------------------------
    # Save xyzr
    # -----------------------------
    save_xyzr(circles, AP_xyzr, img_size, physical_size)
    save_xyzr(voids, void_xyzr, img_size, physical_size)

    return current_void_area / total_domain_area



if __name__ == "__main__":
    gen_struct_combined(
        'test.png', 'test_untitled.png', 'test_AP.xyzr', 'test_voids.xyzr',
        1, 0.0012, 0.8, 10000, 0.3, 0.0, 0.3, 0.003
    )









if __name__ == "__main__":
    #save_path, save_path_untitled, _, _ = gen_struct(save_path, save_path_untitled, save_path_xyzr, img_size, physical_size, physical_mean_radius, ap_ratio, rad_dev_bi, max_attempts, 2, mix)
    #print(f"\nSaved AP–HTPB microstructure images to: {save_path} and {save_path_untitled}")
    gen_struct_combined(
        'test.png', 'test_untitled.png', 'test_AP.xyzr', 'test_voids.xyzr', 
        1, 0.0012, 0.8, 10000, 0.3, 0.0, 0.3, 0.003
    ) #solid, hollow, porous
    #mass_frac_to_vol_frac(0.8, 0.0, 0.0)
