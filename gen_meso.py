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

img_size = 1 #square image, 1x1 (makes other values easier)
mean_radius = 0.03 #given as a portion of the image's height/width, 3%
ap_ratio = 0.6 #60% AP
#monomodal
rad_dev = 0.5 #standard deviation of radius sizes, normal distribution
max_attempts = 500000 #will try to generate an image 500000 times before it gives up


def gen_struct(save_path, save_path_untitled, img_size, mean_radius, ap_ratio, rad_dev, max_attempts):
    '''Generate a 2D microstructure assuming perfect circles, normal distribution
        returns untitled and titled images'''
    
    #random number generator to have various circle sizes
    rng = np.random.default_rng()
    
    #for image generation
    
    #mean radius from fraction of image size to absolute sizing
    mean_rad = mean_radius * img_size
    
    img_area = img_size **2 #make it a square
    
    target_AP_area = ap_ratio * img_area #absolute target area of AP
    
    #storage for loop
    circles = [] #will have x coord, y coord, radius for each circle 
    total_area = 0
    attempts = 0
    
    #use grid setup to check for overlap, rather than checking every grain every time
    cell_size = 4 * mean_rad
    n_cells = max(1, int(math.ceil(img_size / cell_size))) #numb er of cells in image
    
    grid = [[[] for _ in range(n_cells)] for __ in range(n_cells)] #empty grid
    
    
    def cell_coords(x, y):
        #ensure coordinates don't go outside the image's domain\
        #returns cell location in the grid (i.e., columns 2, row 3)
        cx = min(n_cells - 1, int(x / cell_size))
        cy = min(n_cells - 1, int(y / cell_size))
        return cx, cy 

    def nearby_candidates(x, y):
        #check for overlap in neighboring cells
        #returns the grains in the 3x3 around (x,y)
        cx, cy = cell_coords(x, y)
        for i in range(max(0, cx - 1), min(n_cells, cx + 2)): #check neighboring columns
            for j in range(max(0, cy - 1), min(n_cells, cy + 2)): #check neighboring rows
                for idx in grid[i][j]:
                    yield circles[idx] #return one grain at a time

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
        
        while total_area < target_AP_area and attempts < max_attempts:
            attempts += 1
            
            #normal distribution for radii
            #generate a number from normal dist
            r = float(rng.normal(mean_rad, rad_dev * mean_rad))
            #ensure no negative values, minimum radius 0.00001
            r = max(r, 0.00001)
            #ensure no radii bigger than 1/4 the image's width
            r = min(r, 0.25 * img_size)
            
            #random coordinate for center of circle
            '''want to see later if this works to plot some outside the domain,
            and have some circles overlap with the edge of the image'''
            
            x = rng.uniform(r, img_size - r) #make sure it stays inside bounds
            y = rng.uniform(r, img_size - r)
            
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
            
            total_area += math.pi * r**2
            
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
    ax.set_title(
            f"AP grains: mean radius = {mean_radius:.3f}, target AP = {ap_ratio:.3f}\n"
            f"Placed {len(circles)} grains, achieved AP = {total_area/img_area:.3f}"
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    
    plt.close(fig)
    
    return save_path, save_path_untitled



if __name__ == "__main__":
    save_path, save_path_untitled = gen_struct(save_path, save_path_untitled, img_size, mean_radius, ap_ratio, rad_dev, max_attempts)
    print(f"\nSaved AP–HTPB microstructure images to: {save_path} and {save_path_untitled}")
    