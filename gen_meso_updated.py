#sphere packing, updated to handle 2D or 3D

import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from typing import Literal
import void_analysis_3d

def save_xyzr(particles, filepath, img_size, physical_size):
    scale = physical_size / img_size
    with open(filepath, 'w') as f:
        for p in particles:
            if len(p) == 3:
                x, y, r = p
                f.write(f"{x*scale:.8e} {y*scale:.8e} 0.0 {r*scale:.8e}\n")
            else:
                x, y, z, r = p
                f.write(f"{x*scale:.8e} {y*scale:.8e} {z*scale:.8e} {r*scale:.8e}\n")

def gen_struct_combined_2or3D(
    AP_xyzr, void_xyzr,
    physical_size,
    rad_dev, max_attempts,
    vol_percent_solid, vol_percent_hollow, vol_percent_porous, void_fraction, mwd_target,
    dim: Literal[2,3],
    mean_rad_solid=60e-6, mean_rad_hollow=2e-6, mean_rad_porous=4.5e-6, #log-normal distribution
    mwd_tolerance=0.05e-6
):
    """
    3D microstructure generator with void fraction control.
    Can place combination of solid, hollow, and porous grains with assumed log-normal average radius
    Assumes log-normal distribution
    All pores are placed within AP particles

    Solid grains: no voids
    Hollow grains: 1 concentric void per particle
    Porous grains: several voids per particle, void radius ~15% of particle radius
    """
    if dim not in (2,3):
        raise ValueError("Dimension must be 2 or 3.")
    
    rng = np.random.default_rng()
    img_size = 1
    margin = 0.07 #place particles slightly outside domain to allow full particles at edges

    # -----------------------------
    # Global void bookkeeping
    # -----------------------------
    if dim == 2:
        total_domain_area = img_size * img_size
        target_void_area = void_fraction * total_domain_area
        current_void_area = 0.0

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


        # -----------------------------
        # Spatial grid
        # -----------------------------
        cell_size = img_r(mean_rad_solid) * 4
        n_cells = max(1, int(math.ceil(img_size / cell_size)))
        grid = [[[] for _ in range(n_cells)] for __ in range(n_cells)]

        def cell_coords(x, y):
            cx = int(x // cell_size)
            cy = int(y // cell_size)
            cx = max(0, min(n_cells - 1, cx))
            cy = max(0, min(n_cells - 1, cy))
            return cx, cy

        def nearby(x, y, circles):
            cx, cy = cell_coords(x, y)
            for i in range(max(0, cx-1), min(n_cells, cx+2)):
                for j in range(max(0, cy-1), min(n_cells, cy+2)):
                    for idx in grid[i][j]:
                        yield circles[idx]

        def clipped_circle_area(x, y, r, lo=0, hi=1):
            def segment_area(h):
                if h <= 0: return 0.0
                if h >= 2*r: return np.pi * r**2  # fully outside edge
                return r**2 * np.arccos((r - h) / r) - (r - h) * np.sqrt(2*r*h - h**2)

            A = np.pi * r**2
            A -= segment_area(r - (x - lo))  # lo x edge
            A -= segment_area(r - (hi - x))  # hi x edge
            A -= segment_area(r - (y - lo))  # lo y edge
            A -= segment_area(r - (hi - y))  # hi y edge
            return A

        circles = []
        voids = []

        solid_area = hollow_area = porous_area = 0.0

        # -----------------------------
        # SOLID GRAINS
        # -----------------------------
        print("Placing solid grains...")

        sigma_ln = np.sqrt(np.log(1 + rad_dev**2))
        for attempt in range(max_attempts):
            if attempt % 1000 == 0 and attempt != 0:
                print(f"Reached solid attempt {attempt}")
            if solid_area >= vol_percent_solid:
                break

            
            mu_ln = np.log(img_r(mean_rad_solid)) - 0.5 * sigma_ln**2  # adjust for mean weight radius
            r = rng.lognormal(mu_ln, sigma_ln)
            A = math.pi * r**2 / total_domain_area
            if solid_area + A > vol_percent_solid:
                continue

            for _ in range(100):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)

                if not any((x-cx)**2 + (y-cy)**2 < ((r+cr) * 0.999)**2 
            for cx,cy,cr in nearby(x,y,circles)):
                    circles.append((x,y,r))
                    grid[cell_coords(x,y)[0]][cell_coords(x,y)[1]].append(len(circles)-1)
                    solid_area += clipped_circle_area(x, y, r) 
                    break

        print(f"  Solid area fraction: {solid_area:.4f}")

    # -----------------------------
        # HOLLOW GRAINS
        # -----------------------------
        print("\nPlacing hollow grains...")

        for attempt in range(max_attempts):
            if attempt % 1000 == 0 and attempt != 0:
                print(f"Reached hollow attempt {attempt}, hollow area so far: {hollow_area:.4f}")
            if hollow_area >= vol_percent_hollow:
                break

            mu_ln = np.log(img_r(mean_rad_hollow)) - 0.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            A = math.pi * r**2 / total_domain_area
            if hollow_area + A > vol_percent_hollow:
                continue

            MAX_POSITION_TRIES = 4 if hollow_area < 0.6 * vol_percent_hollow else 30

            for _ in range(MAX_POSITION_TRIES):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)

                if not any((x-cx)**2 + (y-cy)**2 < ((r+cr) * 0.98)**2 
            for cx,cy,cr in nearby(x,y,circles)):
                    rv = r * np.sqrt(void_fraction/vol_percent_hollow)
                    circles.append((x, y, r))
                    voids.append((x, y, rv))
                    grid[cell_coords(x,y)[0]][cell_coords(x,y)[1]].append(len(circles)-1)
                    hollow_area += clipped_circle_area(x, y, r) 
                    current_void_area += clipped_circle_area(x, y, rv)
                    break

        print(f"  Hollow area fraction: {hollow_area:.4f}")
        print(f"  Void fraction so far: {current_void_area/total_domain_area:.4f}")
    # -----------------------------
        # POROUS GRAINS
        # -----------------------------
        print("\nPlacing porous grains...")
        mu = math.log(img_r(mean_rad_porous))

        for attempt in range(max_attempts):
            if attempt % 1000 == 0 and attempt != 0:
                print(f"Reached porous attempt {attempt}, porous area so far: {porous_area:.4f}")
            if porous_area >= vol_percent_porous:
                break

            mu_ln = np.log(img_r(mean_rad_porous)) - 1.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            A = math.pi * r**2 / total_domain_area
            if porous_area + A > vol_percent_porous:
                continue

            MAX_POSITION_TRIES = 4 if porous_area < 0.6 * vol_percent_porous else 40

            placed = False  # ← reset here every attempt
            for _ in range(MAX_POSITION_TRIES):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)
                if not any((x-cx)**2 + (y-cy)**2 < ((r+cr) * 0.999)**2
                        for cx,cy,cr in nearby(x,y,circles)):
                    circles.append((x, y, r))
                    grid[cell_coords(x,y)[0]][cell_coords(x,y)[1]].append(len(circles)-1)
                    porous_area += clipped_circle_area(x, y, r)
                    placed = True
                    if placed and porous_area >= vol_percent_porous:  # ← then exit outer loop
                        break

            if placed and porous_area >= vol_percent_porous:
                break

        print(f"  Porous area fraction: {porous_area:.4f}")

        # -----------------------------
        # MWD CHECK
        # -----------------------------
        save_xyzr(circles, AP_xyzr, img_size, physical_size)

        if mwd_tolerance is not None:
            radii = np.array([r for (x, y, r) in circles])
            mwd_actual = 2 * np.sum(radii**3) / np.sum(radii**2) * (physical_size / img_size)
            print(f"\nMWD check: {mwd_actual:.4e} m (target {mwd_target:.4e} m)")
            if abs(mwd_actual - mwd_target) > mwd_tolerance:
                print("MWD out of tolerance — skipping void placement.")
                return None

        # -----------------------------
        # PLACE VOIDS WITHIN POROUS GRAINS
        # -----------------------------
        print("\nPlacing voids within porous grains...")
        print(f"  Number of porous particles: {len(circles)}")
        print(f"  Target void area: {target_void_area:.4f}")
        if vol_percent_porous > 0:
            while current_void_area < target_void_area:

                progress = False

                for idx, (px, py, pr) in enumerate(circles):

                    if current_void_area >= target_void_area:
                        break
                    
                    
                    remaining_area = target_void_area - current_void_area

                    # choose pore radius based on remaining area and particle size
                    pore_r = rng.lognormal(math.log(pr * 0.1
                                                    ), 0.4) #7% pore radius, previously 15
                    pore_r = float(np.clip(pore_r, 0.01 * pr, 0.35 * pr))

                    pore_area = np.pi * pore_r**2

                    if pore_area > remaining_area:
                        pore_r = np.sqrt(remaining_area / np.pi)
                        pore_area = remaining_area

                    # try to place pore without overlap
                    success = False

                    for _ in range(500):

                        theta = rng.uniform(0, 2*np.pi)
                        rho   = rng.uniform(0, pr - pore_r)

                        vx = px + rho*np.cos(theta)
                        vy = py + rho*np.sin(theta)

                        # must remain inside particle
                        if np.hypot(vx-px, vy-py) + pore_r > pr:
                            continue

                        # must not overlap existing voids
                        if any(np.hypot(vx-xv, vy-yv) < pore_r + rv for xv,yv,rv in voids):
                            continue

                        success = True
                        break

                    if success:
                        voids.append((vx, vy, pore_r))
                        current_void_area += pore_area
                        progress = True

                print(f"  Void fraction so far: {current_void_area/total_domain_area:.4f}")

                if not progress:
                    print("WARNING: could not place more voids without overlap.")
                    break
        print(f"  Total voids placed: {len(voids)}")
        print(f"  Void fraction achieved: {current_void_area/total_domain_area:.4f}")
        print(f"  Target:                 {void_fraction:.4f}")

        # -----------------------------
        # FINAL SUMMARY
        # -----------------------------
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total grains: {len(circles)}")
        print(f"Total AP area fraction: {(solid_area + hollow_area + porous_area):.4f}")
        print(f"Solid area fraction: {solid_area:.4f}")
        print(f"Void fraction: {current_void_area/total_domain_area:.4f}")
        print(f"Target:        {void_fraction:.4f}")
        print(f"Error:         {abs(current_void_area/total_domain_area - void_fraction):.2e}")
        print(f"{'='*60}\n")

        save_xyzr(voids, void_xyzr, img_size, physical_size)

        return current_void_area / total_domain_area
    
    
    #------------------------------
    # 3D
    #------------------------------
    
    elif dim == 3:
        total_domain_vol = img_size * img_size * img_size
        target_void_vol = void_fraction * total_domain_vol
        current_void_vol = 0.0

        print(f"\n{'='*60}")
        print("TARGET PARAMETERS")
        print(f"{'='*60}")
        print(f"Target void fraction (global): {void_fraction:.4f}")
        print(f"Target void vol (px^3): {target_void_vol:.2e}")
        print(f"{'='*60}\n")

        # -----------------------------
        # Radius helpers
        # -----------------------------
        def img_r(mu):
            return mu / physical_size * img_size


        # -----------------------------
        # Spatial grid
        # -----------------------------
        cell_size = img_r(mean_rad_solid) * 4
        n_cells = max(1, int(math.ceil(img_size / cell_size)))
        grid = [[[[] for _ in range(n_cells)]
                for __ in range(n_cells)]
                for ___ in range(n_cells)]

        def cell_coords(x, y, z):
            i = int(x // cell_size)
            j = int(y // cell_size)
            k = int(z // cell_size)
            i = max(0, min(n_cells - 1, i))
            j = max(0, min(n_cells - 1, j))
            k = max(0, min(n_cells - 1, k))
            return i, j, k

        def nearby(x, y, z, spheres):
            cx, cy, cz = cell_coords(x, y, z)
            for i in range(max(0,cx-1), min(n_cells,cx+2)):
                for j in range(max(0,cy-1), min(n_cells,cy+2)):
                    for k in range(max(0,cz-1), min(n_cells,cz+2)):
                        for idx in grid[i][j][k]:
                            yield spheres[idx]

        def clipped_sphere_volume(x, y, z, r, lo=0, hi=1):
            def cap_vol(h):
                if h <= 0: return 0.0
                if h >= 2*r: return (4/3) * np.pi * r**3  # fully outside face
                return np.pi * h**2 * (3*r - h) / 3

            V = (4/3) * np.pi * r**3
            V -= cap_vol(r - (x - lo))  # lo x face
            V -= cap_vol(r - (hi - x))  # hi x face
            V -= cap_vol(r - (y - lo))  # lo y face
            V -= cap_vol(r - (hi - y))  # hi y face
            V -= cap_vol(r - (z - lo))  # lo z face
            V -= cap_vol(r - (hi - z))  # hi z face
            return V



        spheres = []
        voids = []

        solid_vol = hollow_vol = porous_vol = 0.0

        # -----------------------------
        # SOLID GRAINS
        # -----------------------------
        print("Placing solid grains...")
        #need to update to clip, and use spheres instead of circles
        sigma_ln = np.sqrt(np.log(1 + rad_dev**2))
        for attempt in range(max_attempts):
            if attempt % 1000 == 0 and attempt != 0:
                print(f"Reached solid attempt {attempt}, solid volume so far: {solid_vol:.4f}")
            if solid_vol >= vol_percent_solid:
                break

            
            mu_ln = np.log(img_r(mean_rad_solid)) - 1.5 * sigma_ln**2  # adjust for mean weight radius
            r = rng.lognormal(mu_ln, sigma_ln)
            V = 4/3 * math.pi * r**3 / total_domain_vol
            if solid_vol + V > vol_percent_solid:
                continue
            for _ in range(100): #attempts, place slightly outside of domain in all dimensions
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)
                z = rng.uniform(-margin + r, img_size + margin - r)

                if not any((x-cx)**2 + (y-cy)**2 + (z-cz)**2 < ((r+cr) * 0.98)**2 for cx,cy,cz,cr in nearby(x,y,z,spheres)):
                    spheres.append((x, y, z, r))
                    grid[cell_coords(x,y,z)[0]][cell_coords(x,y,z)[1]][cell_coords(x,y,z)[2]].append(len(spheres)-1)
                    porous_vol += clipped_sphere_volume(x, y, z, r)
                    placed = True
                    if porous_area >= vol_percent_porous:  # ← exit inner loop immediately
                        break

            if placed and porous_vol >= vol_percent_porous:
                break
        print(f"  Solid volume fraction: {solid_vol:.4f}")

    # -----------------------------
        # HOLLOW GRAINS
        # -----------------------------
        print("\nPlacing hollow grains...")

        for attempt in range(max_attempts):
            if attempt % 1000 == 0 and attempt != 0:
                print(f"Reached hollow attempt {attempt}, hollow volume so far: {hollow_vol:.4f}")
            if hollow_vol >= vol_percent_hollow:
                break

            mu_ln = np.log(img_r(mean_rad_hollow)) - 1.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            V = 4/3 * math.pi * r**3 / total_domain_vol
            if hollow_vol + V > vol_percent_hollow:
                continue

            MAX_POSITION_TRIES = 4 if hollow_vol < 0.6 * vol_percent_hollow else 30

            for _ in range(MAX_POSITION_TRIES):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)
                z = rng.uniform(-margin + r, img_size + margin - r)

                if not any((x-cx)**2 + (y-cy)**2 + (z-cz)**2 < ((r+cr) * 0.995)**2 
            for cx,cy,cz,cr in nearby(x,y,z,spheres)): #allow tiny overlap (touching)
                    rv = r * (void_fraction/vol_percent_hollow)**(1/3)
                    spheres.append((x, y, z, r))
                    voids.append((x, y, z, rv))
                    grid[cell_coords(x,y,z)[0]][cell_coords(x,y,z)[1]][cell_coords(x,y,z)[2]].append(len(spheres)-1)
                    hollow_vol += clipped_sphere_volume(x, y, z, r)
                    current_void_vol += clipped_sphere_volume(x, y, z, rv)
                    break

        print(f"  Hollow area fraction: {hollow_vol:.4f}")
        print(f"  Void fraction so far: {current_void_vol/total_domain_vol:.4f}")
    # -----------------------------
        # POROUS GRAINS
        # -----------------------------
        print("\nPlacing porous grains...")
        mu = math.log(img_r(mean_rad_porous))

        for attempt in range(max_attempts):
            if attempt % 1000 == 0 and attempt != 0:
                print(f"Reached porous attempt {attempt}, porous volume so far: {porous_vol:.4f}")
            if porous_vol >= vol_percent_porous:
                break

            mu_ln = np.log(img_r(mean_rad_porous)) - 1.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            V = 4/3 * math.pi * r**3 / total_domain_vol
            if porous_vol + V > vol_percent_porous:
                continue

            MAX_POSITION_TRIES = 50 if porous_vol < 0.5 * vol_percent_porous else 150

            placed = False  # ← reset here every attempt
            for _ in range(MAX_POSITION_TRIES):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)
                z = rng.uniform(-margin + r, img_size + margin - r)

                if not any((x-cx)**2 + (y-cy)**2 + (z-cz)**2 < ((r+cr) * 0.97)**2
                        for cx,cy,cz,cr in nearby(x,y,z,spheres)):
                    spheres.append((x, y, z, r))
                    grid[cell_coords(x,y,z)[0]][cell_coords(x,y,z)[1]][cell_coords(x,y,z)[2]].append(len(spheres)-1)
                    porous_vol += clipped_sphere_volume(x, y, z, r)
                    placed = True
                    if porous_vol >= vol_percent_porous:
                        break
                    

            if placed and porous_vol >= vol_percent_porous:
                break

        print(f"  Porous area fraction: {porous_vol:.4f}")

        # -----------------------------
        # MWD CHECK
        # -----------------------------
        save_xyzr(spheres, AP_xyzr, img_size, physical_size)

        if mwd_tolerance is not None:
            radii = np.array([r for (x, y, z, r) in spheres])
            mwd_actual = 2 * np.sum(radii**3) / np.sum(radii**2) * (physical_size / img_size)  #Sauter MWD
            print(f"\nMWD check: {mwd_actual:.4e} m (target {mwd_target:.4e} m)")
            if abs(mwd_actual - mwd_target) > mwd_tolerance:
                print("MWD out of tolerance — skipping void placement.")
                return None

        # -----------------------------
        # PLACE VOIDS WITHIN POROUS GRAINS
        # -----------------------------
        print("\nPlacing voids within porous grains...")
        print(f"  Number of porous particles: {len(spheres)}")
        print(f"  Target void area: {target_void_vol:.4f}")
        if vol_percent_porous > 0:
            while current_void_vol < target_void_vol:

                progress = False

                for idx, (px, py, pz, pr) in enumerate(spheres):

                    if current_void_vol >= target_void_vol:
                        break
                    
                    
                    remaining_vol = target_void_vol - current_void_vol

                    # choose pore radius based on remaining area and particle size
                    pore_r = rng.lognormal(math.log(pr * 0.15
                                                    ), 0.4) #15% rad
                    pore_r = float(np.clip(pore_r, 0.01 * pr, 0.35 * pr)) #can't be less than 1%or more than 35%

                    pore_vol = 4/3 * np.pi * pore_r**3

                    if pore_vol > remaining_vol:
                        pore_r = (remaining_vol / ((4/3) * np.pi))**(1/3)
                        pore_vol = remaining_vol

                    # try to place pore without overlap
                    success = False

                    for _ in range(500):

                        theta = rng.uniform(0, 2*np.pi)
                        phi   = rng.uniform(0, np.pi)
                        rho   = (pr - pore_r) * rng.uniform(0, 1)**(1/3) #avoid clustering at the center

                        vx = px + rho * np.sin(phi) * np.cos(theta)
                        vy = py + rho * np.sin(phi) * np.sin(theta)
                        vz = pz + rho * np.cos(phi)


                        # must remain inside particle
                        if np.sqrt((vx-px)**2 + (vy-py)**2 + (vz-pz)**2) + pore_r > pr:
                            continue

                        # must not overlap existing voids
                        if any(np.sqrt((vx-xv)**2 + (vy-yv)**2 + (vz-zv)**2) < pore_r + rv for xv,yv,zv,rv in voids):
                            continue

                        success = True
                        break

                    if success:
                        voids.append((vx, vy, vz, pore_r))
                        current_void_vol += clipped_sphere_volume(vx, vy, vz, pore_r)
                        progress = True

                print(f"  Void fraction so far: {current_void_vol/total_domain_vol:.4f}")

                if not progress:
                    print("WARNING: could not place more voids without overlap.")
                    break
        print(f"  Total voids placed: {len(voids)}")
        print(f"  Void fraction achieved: {current_void_vol/total_domain_vol:.4f}")
        print(f"  Target:                 {void_fraction:.4f}")

        # -----------------------------
        # FINAL SUMMARY
        # -----------------------------
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total grains: {len(spheres)}")
        print(f"Total AP area fraction: {(solid_vol + hollow_vol + porous_vol):.4f}")
        print(f"Solid volume fraction: {solid_vol:.4f}")
        print(f"Void fraction: {current_void_vol/total_domain_vol:.4f}")
        print(f"Target:        {void_fraction:.4f}")
        print(f"Error:         {abs(current_void_vol/total_domain_vol - void_fraction):.2e}")
        print(f"{'='*60}\n")

        save_xyzr(voids, void_xyzr, img_size, physical_size)

        return current_void_vol / total_domain_vol

def mean_weight_diameter(filename):
    """
    Reads an xyzr file and calculates the mean weight diameter.
    
    Parameters:
        filename (str): Path to the xyzr file (columns: x y z r)
    
    Returns:
        float: mean weight diameter
    """
    # Load r column (assuming last column is r)
    data = np.loadtxt(filename)
    radii = data[:, -1]  # last column
    
    # Calculate D_w
    numerator = np.sum(radii**3)
    denominator = np.sum(radii**2)
    D_w = 2 * numerator / denominator
    
    return D_w


def generate_structures_with_target_mwd(
    subfolder,
    target_mwd,
    mwd_tolerance=0.05e-6,
    n_target=1,
    max_total_attempts=200,
    base_name="A",
    dim: Literal[2, 3] = 2,  # new parameter
    # gen_struct_combined parameters
    physical_size=200e-6,
    rad_dev=0.4,
    max_attempts=800000,
    vol_percent_solid=0,
    vol_percent_hollow=0,
    vol_percent_porous=0.7141,
    void_fraction=0.1633,
    mean_rad_hollow=4.05e-6,
    mean_rad_porous=4.05e-6,
    mean_rad_solid=4.05e-6,
):
    os.makedirs(subfolder, exist_ok=True)
    subsubfolder = os.path.join(subfolder, "example_images")
    accepted = []
    attempt = 0
    accepted_idx = 0

    print(f"\n{'='*60}")
    print(f"Target MWD:     {target_mwd:.4e} m")
    print(f"Tolerance:      ±{mwd_tolerance:.4e} m")
    print(f"Target count:   {n_target}")
    print(f"Dimension:      {dim}D")          # new print
    print(f"{'='*60}\n")

    while len(accepted) < n_target and attempt < max_total_attempts:
        attempt += 1
        name = f"{base_name}_{accepted_idx:02d}"

        print(f"\n--- Attempt {attempt} (accepted so far: {len(accepted)}/{n_target}) ---")

        ap_xyzr            = os.path.join(subfolder, f"{name}_AP.xyzr")
        void_xyzr          = os.path.join(subfolder, f"{name}_void.xyzr")
        save_path          = os.path.join(subfolder, f"{name}.png")
        save_path_untitled = os.path.join(subfolder, f"{name}_untitled.png")

        try:
            void_frac = gen_struct_combined_2or3D(
                ap_xyzr,
                void_xyzr,
                physical_size=physical_size,
                rad_dev=rad_dev,
                max_attempts=max_attempts,
                vol_percent_solid=vol_percent_solid,
                vol_percent_hollow=vol_percent_hollow,
                vol_percent_porous=vol_percent_porous,
                void_fraction=void_fraction,
                mwd_target=target_mwd,
                dim=dim,                        # passed through
                mean_rad_solid=mean_rad_solid,
                mean_rad_hollow=mean_rad_hollow,
                mean_rad_porous=mean_rad_porous,
                mwd_tolerance=mwd_tolerance,
            )
        except Exception as e:

            print(f"  Generation failed: {e}")
            # Clean up any partial files
            for path in [ap_xyzr, void_xyzr, save_path, save_path_untitled]:
                if os.path.exists(path):
                    os.remove(path)
            continue
        if void_frac is None:
            if os.path.exists(ap_xyzr):
                os.remove(ap_xyzr)
            continue
        try:
            mwd = mean_weight_diameter(ap_xyzr)
        except Exception as e:
            print(f"  MWD calculation failed: {e}")
            for path in [ap_xyzr, void_xyzr, save_path, save_path_untitled]:
                if os.path.exists(path):
                    os.remove(path)
            continue

        mwd_error = abs(mwd - target_mwd)
        print(f"  MWD:        {mwd:.4e} m")
        print(f"  Target MWD: {target_mwd:.4e} m")
        print(f"  Error:      {mwd_error:.4e} m  ({'ACCEPTED ✓' if mwd_error <= mwd_tolerance else 'rejected ✗'})")

        if mwd_error <= mwd_tolerance:
            accepted.append({
                "index":     accepted_idx,
                "name":      name,
                "mwd":       mwd,
                "void_frac": void_frac,
                "attempt":   attempt,
            })
            print(f"  → Kept as {name}")
            if accepted_idx == 0:
                os.makedirs(subsubfolder, exist_ok=True)
                plot_from_xyzr(
                    ap_xyzr,
                    void_xyzr,
                    os.path.join(subsubfolder, f"{name}.png"),
                    dim=dim, physical_size=physical_size
                )
            accepted_idx += 1  # only increment on acceptance so naming stays contiguous
        else:
            for path in [ap_xyzr, void_xyzr, save_path, save_path_untitled]:
                if os.path.exists(path):
                    os.remove(path)

def plot_from_xyzr(ap_xyzr_path, void_xyzr_path, save_path,
                   physical_size=200e-6, img_size=1,
                   dim=2,
                   # 2D options
                   dpi=1024,
                   ap_alpha=1.0,
                   # 3D options
                   sphere_resolution=20, max_spheres=None,
                   elev=25, azim=45, alpha=0.6):

    def read_xyzr(path):
        pts = []
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"WARNING: xyzr file empty or missing: {path}")
            return pts
        with open(path, 'r') as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) == 3:
                    x, y, r = map(float, vals)
                    pts.append((x / physical_size * img_size,
                                y / physical_size * img_size,
                                0, r / physical_size * img_size))
                elif len(vals) >= 4:
                    x, y, z, r = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
                    pts.append((x / physical_size * img_size,
                                y / physical_size * img_size,
                                z / physical_size * img_size,
                                r / physical_size * img_size))
        return pts

    circles = read_xyzr(ap_xyzr_path)
    voids   = read_xyzr(void_xyzr_path)
    print(f"Plotting {len(circles)} particles and {len(voids)} voids")

    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.set_position([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.set_aspect("equal")

        ax.add_patch(plt.Rectangle((0, 0), img_size, img_size,
                                   facecolor='#0000FF', zorder=0))
        for (x, y, _, r) in circles:
            ax.add_patch(Circle((x, y), r,
                facecolor='#FF0000', edgecolor='none', alpha=ap_alpha, zorder=5))
        for (x, y, _, r) in voids:
            ax.add_patch(Circle((x, y), r,
                facecolor='#0000FF', edgecolor='none', zorder=6))

        fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0.0)
        plt.close(fig)

    elif dim == 3:
        if max_spheres is not None:
            circles = circles[:max_spheres]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        u = np.linspace(0, 2 * np.pi, sphere_resolution)
        v = np.linspace(0, np.pi,     sphere_resolution)
        u, v = np.meshgrid(u, v)

        for (x0, y0, z0, r) in circles:
            ax.plot_surface(x0 + r*np.cos(u)*np.sin(v),
                            y0 + r*np.sin(u)*np.sin(v),
                            z0 + r*np.cos(v),
                            color='#FF6B6B', linewidth=0, alpha=ap_alpha)
        for (x0, y0, z0, r) in voids:
            ax.plot_surface(x0 + r*np.cos(u)*np.sin(v),
                            y0 + r*np.sin(u)*np.sin(v),
                            z0 + r*np.cos(v),
                            color='#4ECDC4', linewidth=0, alpha=alpha)

        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.set_zlim(0, img_size)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)


def main():
    generate_structures_with_target_mwd('3D_xyzrs', target_mwd=9e-6, base_name= "ex", dim=3, physical_size=100e-6, mean_rad_porous=5e-6, mean_rad_solid=4.4e-6, void_fraction=0.06, vol_percent_porous=0.57779, mwd_tolerance=0.05e-6, rad_dev = 0.4)
    plot_from_xyzr("test/A_00_AP.xyzr", "test/A_00_void.xyzr", "test/A_00.png")
    #void_analysis_3d()
    
    

if __name__ == "__main__":
    main()