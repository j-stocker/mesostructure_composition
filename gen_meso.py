#Generates an AP/HTPB image given avg radius and % dist

import math
import numpy as np
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
max_attempts = 5000 #will try to generate an image 500000 times before it gives up
mix = 1/12 #coarse/fine particles


def save_xyzr(circles, filepath, img_size, physical_size):
    scale = physical_size/img_size #m/image unit
    with open(filepath, 'w') as f:
        for (x, y, r) in circles:
            xp = x * scale
            yp = y * scale
            rp = r * scale
            f.write(f"{xp:.8e} {yp:.8e} 0.0 {rp:.8e}\n")

def gen_struct(save_path, save_path_untitled, save_path_xyzr, img_size, physical_size, physical_mean_radius, ap_ratio, rad_dev, max_attempts, mode, mix=0.5):
    '''Generate a 2D microstructure assuming perfect circles, normal distribution
        returns untitled and titled images
        mode: 
            enter 1 for unimodal, 2 for bimodal'''
    margin = 0.05
    
    #random number generator to have various circle sizes
    rng = np.random.default_rng()
    #will place circles a bit outside domain
    # Determine mean_rad for unimodal or bimodal
    # Determine mean_rad for unimodal or bimodal
    if mode == 1:  # unimodal
        mean_rad = float(physical_mean_radius) / physical_size * img_size
    elif mode == 2:  # bimodal
        # ensure physical_mean_radius is a list of 2 floats
        if isinstance(physical_mean_radius[0], list) or isinstance(physical_mean_radius[1], list):
            raise ValueError("For bimodal, physical_mean_radius must be a list of two floats, not nested lists.")
        # convert both elements to scaled floats
        mean_rad = [r / physical_size * img_size for r in physical_mean_radius]


    #for image generation

    img_area = img_size **2 #make it a square
    
    target_AP_area = ap_ratio * img_area #absolute target area of AP
    
    #for calculating only area inside domain later
    N_mask = 1024  # generates 1024 pixel images
    mask = np.zeros((N_mask, N_mask), dtype=bool)
    inv_N = 1.0 / N_mask  
    
    #storage for loop
    circles = [] #will have x coord, y coord, radius for each circle 
    total_area = 0
    attempts = 0
    
    #use grid setup to check for overlap, rather than checking every grain every time
    if mode == 1:
        cell_size = 4 * mean_rad
    else:
        cell_size = 4 * max(mean_rad)  # use the larger radius
    n_cells = max(1, int(math.ceil(img_size / cell_size))) #numb er of cells in image
    
    grid = [[[] for _ in range(n_cells)] for __ in range(n_cells)] #empty grid
    
    
    def cell_coords(x, y):
        #ensure coordinates don't go outside the image's domain\
        #returns cell location in the grid (i.e., columns 2, row 3)
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        # clamp into valid grid range (handles negative/ >max coords from margin)
        cx = max(0, min(n_cells - 1, cx))
        cy = max(0, min(n_cells - 1, cy))
        return cx, cy 

    def nearby_candidates(x, y):
        #check for overlap in neighboring cells
        #returns the grains in the 3x3 around (x,y)
        cx, cy = cell_coords(x, y)
        for i in range(max(0, cx - 1), min(n_cells, cx + 2)): #check neighboring columns
            for j in range(max(0, cy - 1), min(n_cells, cy + 2)): #check neighboring rows
                for idx in grid[i][j]:
                    yield circles[idx] #return one grain at a time
    def sample_radius(r_min, r_max, mu, sigma, rng, max_resample=100):
        """Sample a log-normal radius with an upper/lower bound, fallback to clipping."""
        for _ in range(max_resample):
            r = float(rng.lognormal(mu, sigma))
            if r_min <= r <= r_max:
                return r
        # fallback if too many attempts fail
        r = np.clip(r, r_min, r_max)
        return r

    #main function: place AP circles until we get to target area of AP 
    
    max_tries = 5  # number of times to retry the whole image
    success = False
    best_circles = None
    best_total_area = 0
    for retry in range(max_tries):
        # reset grid and circles
        circles = []
        grid = [[[] for _ in range(n_cells)] for __ in range(n_cells)]
        total_area = 0
        attempts = 0
        num_samples = int(max_attempts * 1.2)

        radii_list = []

        for _ in range(num_samples):

            if mode == 1:  # unimodal
                mu0 = np.log(mean_rad / np.sqrt(1 + rad_dev**2))
                sigma0 = np.sqrt(np.log(1 + rad_dev**2))
                r = sample_radius(r_min, r_max, mu0, sigma0, rng)

            elif mode == 2:  # bimodal
                dev0, dev1 = float(rad_dev[0]), float(rad_dev[1])
                fine_mean = mean_rad_bi[0] / physical_size * img_size
                coarse_mean = mean_rad_bi[1] / physical_size * img_size

                mu0 = np.log(fine_mean / np.sqrt(1 + dev0**2))
                sigma0 = np.sqrt(np.log(1 + dev0**2))
                mu1 = np.log(coarse_mean / np.sqrt(1 + dev1**2))
                sigma1 = np.sqrt(np.log(1 + dev1**2))

                if rng.random() < mix:
                    r = sample_radius(r_min, r_max, mu0, sigma0, rng)
                else:
                    r = sample_radius(r_min, r_max, mu1, sigma1, rng)

            radii_list.append(r)

        # Sort largest -> smallest
        radii_list.sort(reverse=True)

        # Iterator over sorted radii
        try:
            r = next(r_iter)
        except StopIteration:
            break  # no more radii left

        while total_area < target_AP_area and attempts < max_attempts:
            attempts += 1
            ''' NORMAL DIST
            if mode == 1: #unimodal
                
                #normal distribution for radii
                #generate a number from normal dist
                r = float(rng.normal(mean_rad, rad_dev * mean_rad))
            elif mode == 2:
                dev0, dev1 = float(rad_dev[0]), float(rad_dev[1])
                #choose which mode to sample from 
                if rng.random() < mix:
                    r = float(rng.normal(mean_rad, dev0 * mean_rad))
                else:
                    r = float(rng.normal(mean_rad, dev1 * mean_rad))
            #ensure no negative values, minimum radius 
            r = max(r, 1e-6)
            #ensure no radii bigger than 1/4 the image's width
            r = min(r, 0.25 * img_size)
            '''
            
            #-----
            # LOG NORMAL DIST
            # Define min/max radius in image units
            r_min = 1e-6 / physical_size * img_size   # converts meters to image units
            r_max = 150e-6 / physical_size * img_size


            # Sample radius
            if mode == 1:  # unimodal log-normal
                mu0 = np.log(mean_rad / np.sqrt(1 + rad_dev**2))
                sigma0 = np.sqrt(np.log(1 + rad_dev**2))
                r = sample_radius(r_min, r_max, mu0, sigma0, rng)
            elif mode == 2:  # bimodal log-normal
                dev0, dev1 = float(rad_dev[0]), float(rad_dev[1])
                fine_mean = mean_rad_bi[0] / physical_size * img_size
                coarse_mean = mean_rad_bi[1] / physical_size * img_size

                mu0 = np.log(fine_mean / np.sqrt(1 + dev0**2))
                sigma0 = np.sqrt(np.log(1 + dev0**2))
                mu1 = np.log(coarse_mean / np.sqrt(1 + dev1**2))
                sigma1 = np.sqrt(np.log(1 + dev1**2))

                if rng.random() < mix:
                    r = sample_radius(r_min, r_max, mu0, sigma0, rng)
                else:
                    r = sample_radius(r_min, r_max, mu1, sigma1, rng)



            
            #random coordinate for center of circle
            '''want to see later if this works to plot some outside the domain,
            and have some circles overlap with the edge of the image'''
            
            x = rng.uniform(- margin + r, img_size + margin - r) #let it go out of bounds
            y = rng.uniform(- margin + r, img_size + margin - r)
            
            #assume placement is valid until overlaps
            valid = True
            
            #check for overlap
            for (cx, cy, cr) in nearby_candidates(x, y):
                if (cx - x)**2 + (cy - y)**2 < (cr + r)**2:
                    valid = False
                    break
            
            if not valid:
                continue #don't place
                    
            #accept placement, add to list
            circles.append((x, y, r))
            #add to grid
            gx, gy = cell_coords(x, y)
            grid[gx][gy].append(len(circles) - 1)
            
            x_px = int(x * N_mask)
            y_px = int(y * N_mask)
            r_px = int(math.ceil(r * N_mask))

            # bounding box in pixel indices (clamp to mask)
            xmin = max(0, x_px - r_px)
            xmax = min(N_mask, x_px + r_px + 1)
            ymin = max(0, y_px - r_px)
            ymax = min(N_mask, y_px + r_px + 1)

            # if bbox empty (circle entirely outside unit square), added_area = 0
            if xmin < xmax and ymin < ymax:
                xs = (np.arange(xmin, xmax) + 0.5) * inv_N  # x-coords of pixel centers
                ys = (np.arange(ymin, ymax) + 0.5) * inv_N  # y-coords of pixel centers
                XX, YY = np.meshgrid(xs, ys)
                patch = (XX - x)**2 + (YY - y)**2 <= r*r  # boolean mask of circle inside bbox

                prev = mask[ymin:ymax, xmin:xmax]
                new = patch & (~prev)           # pixels newly covered by this circle
                added_pixels = new.sum()
                if added_pixels:
                    mask[ymin:ymax, xmin:xmax] = prev | patch
                    added_area = added_pixels / (N_mask * N_mask)  # fraction of unit square
                    total_area += added_area
                        
                        #keep the best try
                    if best_circles is None or abs(total_area - target_AP_area) < abs(best_total_area - target_AP_area):
                        best_circles = circles.copy()
                        best_total_area = total_area
                        
                    if total_area >= target_AP_area - 0.001:
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
            Circle((x, y), r, facecolor='#FF0000', edgecolor='#800080', linewidth=1, zorder=10)
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
            Circle((x, y), r, facecolor='#FF0000', edgecolor='#800080', linewidth=1, zorder=10)
        )
    if mode == 1:
        radius_str = f"{physical_mean_radius/1e-6:.1f}"
    else:
        # For bimodal, show both coarse and fine
        radius_str = f"{physical_mean_radius[0]/1e-6:.1f}/{physical_mean_radius[1]/1e-6:.1f}"

    ax.set_title(
        f"AP grains: mean radius = {radius_str} µm, target AP = {ap_ratio:.3f}\n"
        f"Placed {len(circles)} grains, achieved AP = {total_area/img_area:.3f}"
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    
    plt.close(fig)
    save_xyzr(circles, save_path_xyzr, img_size, physical_size)
    return save_path, save_path_untitled, total_area, save_path_xyzr



if __name__ == "__main__":
    save_path, save_path_untitled, _, _ = gen_struct(save_path, save_path_untitled, save_path_xyzr, img_size, physical_size, physical_mean_radius, ap_ratio, rad_dev_bi, max_attempts, 2, mix)
    print(f"\nSaved AP–HTPB microstructure images to: {save_path} and {save_path_untitled}")
    