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

    # -----------------------------
    # Place porous grains
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

            r = min(r, 0.5*img_size - margin)
            circle_area = math.pi*r**2 / (img_size*img_size)
            
            # Skip if this particle is too big for remaining space
            if porous_area + circle_area > vol_percent_porous:
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
                    # For porous particles, create multiple small voids distributed within the particle
                    # The number of pores scales with particle volume
                    n_pores = max(3, int(void_fraction * 20 * (r / mean_rad_porous_img)))
                    pore_radius = r * (void_fraction / n_pores)**(1/2) * 0.8  # Scale pore size
                    
                    circles.append((x, y, r))
                    gx, gy = cell_coords(x, y)
                    grid[gx][gy].append(len(circles)-1)

                    # Distribute pores within the particle
                    for _ in range(n_pores):
                        # Random position within particle (polar coordinates)
                        angle = rng.uniform(0, 2*np.pi)
                        # Pores can be anywhere within the particle
                        dist = rng.uniform(0, r * 0.95)
                        vx = x + dist * np.cos(angle)
                        vy = y + dist * np.sin(angle)
                        voids.append((vx, vy, pore_radius))

                    # Update mask (solid minus pores)
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
                        # Start with full particle
                        mask[ymin:ymax, xmin:xmax] |= patch
                        # Remove pores
                        for (vx, vy, vr) in voids[-n_pores:]:  # Only the pores we just added
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

    print(f"After porous grains: porous area = {porous_area:.4f}")
    
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


def mass_frac_to_vol_frac(m_Gr, m_Po_or_Ho, void_fraction, m_HTPB=0.2, rho_AP=1.95, rho_HTPB=0.93):
    '''get volume fractions for mixtures'''
    
    #convert to volumes for each component
    V_Gr = m_Gr/rho_AP
    V_Po_or_Ho = m_Po_or_Ho/rho_AP
    
    V_AP = V_Gr + V_Po_or_Ho
    V_AP = V_AP 
    
    V_HTPB = m_HTPB/rho_HTPB
    total_vol = (V_AP + V_HTPB) / (1 - void_fraction)
    total_rho = 1 / total_vol
    
    phi_Gr = V_Gr / total_vol
    phi_Po_or_Ho = V_Po_or_Ho / total_vol
    phi_HTPB = V_HTPB / total_vol
    phi_void = 1 - (phi_Gr + phi_Po_or_Ho + phi_HTPB)
    
    
    print(f"{phi_Gr*100:.4f}, {phi_Po_or_Ho*100:.4f}, {phi_HTPB*100:.4f}, {total_rho:.2f}")
    return total_rho



if __name__ == "__main__":
    #save_path, save_path_untitled, _, _ = gen_struct(save_path, save_path_untitled, save_path_xyzr, img_size, physical_size, physical_mean_radius, ap_ratio, rad_dev_bi, max_attempts, 2, mix)
    #print(f"\nSaved AP–HTPB microstructure images to: {save_path} and {save_path_untitled}")
    #gen_struct_combined(
    #    'test.png', 'test_untitled.png', 'test_AP.xyzr', 'test_voids.xyzr', 
    #    1, 0.0012, 0.8, 10000, 0.3, 0.0, 0.3, 0.003
    #) #solid, hollow, porous
    mass_frac_to_vol_frac(0.8, 0.0, 0.0)
