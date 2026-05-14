#sphere packing, updated to handle 2D or 3D — parallelized version

import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from typing import Literal
import concurrent.futures
from functools import partial
# import void_analysis_3d  # uncomment if available


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

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


def mean_weight_diameter(filename):
    """Reads an xyzr file and calculates the mean weight diameter."""
    data = np.loadtxt(filename)
    radii = data[:, -1]
    return 2 * np.sum(radii**3) / np.sum(radii**2)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def gen_struct_combined_2or3D(
    AP_xyzr, void_xyzr,
    physical_size,
    rad_dev, max_attempts,
    vol_percent_solid, vol_percent_hollow, vol_percent_porous, void_fraction, mwd_target,
    dim: Literal[2, 3],
    mean_rad_solid=60e-6, mean_rad_hollow=2e-6, mean_rad_porous=4.5e-6,
    mwd_tolerance=0.05e-6,
):
    """
    Microstructure generator (2-D or 3-D) with void fraction control.

    Parallelization notes
    ---------------------
    • Grain placement (solid / hollow / porous) is inherently sequential because
      each new grain must know the positions of all previously placed grains.
    • Void placement inside porous grains is embarrassingly parallel per grain
      and is handled with ThreadPoolExecutor (I/O-bound + NumPy releases the GIL).
    • The outer loop over multiple accepted structures is parallelized in
      generate_structures_with_target_mwd() via ProcessPoolExecutor.

    Performance fixes applied
    -------------------------
    • overlaps_fast / overlaps_fast_3d: replaced three np.fromiter generator
      calls with a single np.array(neighbors) + column-slice, eliminating
      redundant allocations on the hottest path.
    • cell_coords cached: the result computed for the density-check is reused
      for the grid insert; the second call has been removed in every placement
      loop (solid, hollow, porous) in both 2D and 3D.
    • 2D porous grain loop: removed the erroneous double for _ in range(max_pos)
      nesting (which ran max_pos² position attempts and fired the progress-print
      inside the inner loop). Replaced with a single position-attempt loop
      matching the 3D branch.
    • Void placement budget: per_grain_budget now divides by the number of
      porous grains only (solid and hollow grains cannot hold voids), so each
      porous grain receives a correctly-sized budget and fewer top-up iterations
      are needed.
    • Void placement iteration: the parallel and top-up loops now iterate over
      porous_circles / porous_spheres instead of all circles / spheres, avoiding
      wasted submissions for grain types that will never produce voids.
    """
    if dim not in (2, 3):
        raise ValueError("Dimension must be 2 or 3.")

    rng = np.random.default_rng()
    img_size = 1
    margin = 0.07
    def generate_radii(target_fraction, mean_rad, sigma_ln, dim):
        radii = []
        total = 0.0
        domain = 1.0 if dim == 2 else 1.0

        while total < target_fraction:
            r = np.random.lognormal(mean_rad, sigma_ln)

            if dim == 2:
                vol = math.pi * r**2
            else:
                vol = 4/3 * math.pi * r**3

            radii.append(r)
            total += vol

        return sorted(radii, reverse=True)  # 🔥 KEY
    # ------------------------------------------------------------------
    # Helper: place voids inside a single grain (used for parallel calls)
    # ------------------------------------------------------------------
    def _place_voids_in_grain(grain, existing_voids_snap, target_remaining,
                               dim, rng_seed):
        """
        Try to fill one grain with pores.
        Returns list of (x,y[,z],r) voids placed.
        Uses its own RNG seeded from rng_seed so workers don't share state.
        """
        
        local_rng = np.random.default_rng(rng_seed)
        placed = []
        cum_vol = 0.0

        if dim == 2:
            def overlaps_vec(x, y, r, nb):
                if len(nb) == 0:
                    return False
                dx = x - nb[:,0]
                dy = y - nb[:,1]
                return np.any(dx*dx + dy*dy < (r + nb[:,2])**2)
            px, py, pr = grain[:3]
            for _ in range(800):
                if cum_vol >= target_remaining:
                    break
                pore_r = local_rng.lognormal(math.log(pr * 0.1), 0.4)
                pore_r = float(np.clip(pore_r, 0.01 * pr, 0.35 * pr))
                pore_vol = math.pi * pore_r ** 2
                if cum_vol + pore_vol > target_remaining:
                    pore_r = math.sqrt((target_remaining - cum_vol) / math.pi)
                    pore_vol = target_remaining - cum_vol

                for _ in range(300):
                    theta = local_rng.uniform(0, 2 * math.pi)
                    rho   = local_rng.uniform(0, pr - pore_r)
                    vx = px + rho * math.cos(theta)
                    vy = py + rho * math.sin(theta)
                    if math.hypot(vx - px, vy - py) + pore_r > pr:
                        continue
                    if any(math.hypot(vx - xv, vy - yv) < pore_r + rv
                           for xv, yv, rv in existing_voids_snap):
                        continue
                    if any(math.hypot(vx - xv, vy - yv) < pore_r + rv
                           for xv, yv, rv in placed):
                        continue
                    placed.append((vx, vy, pore_r))
                    existing_voids_snap.append((vx, vy, pore_r))
                    cum_vol += pore_vol
                    break

        else:  # dim == 3
            px, py, pz, pr = grain[:4]
            for _ in range(800):
                if cum_vol >= target_remaining:
                    break
                pore_r = local_rng.lognormal(math.log(pr * 0.15), 0.4)
                pore_r = float(np.clip(pore_r, 0.01 * pr, 0.35 * pr))
                pore_vol = 4 / 3 * math.pi * pore_r ** 3
                if cum_vol + pore_vol > target_remaining:
                    pore_r = ((target_remaining - cum_vol) / (4 / 3 * math.pi)) ** (1 / 3)
                    pore_vol = target_remaining - cum_vol

                for _ in range(300):
                    theta = local_rng.uniform(0, 2 * math.pi)
                    phi   = local_rng.uniform(0, math.pi)
                    rho   = (pr - pore_r) * local_rng.uniform(0, 1) ** (1 / 3)
                    vx = px + rho * math.sin(phi) * math.cos(theta)
                    vy = py + rho * math.sin(phi) * math.sin(theta)
                    vz = pz + rho * math.cos(phi)
                    if math.sqrt((vx-px)**2 + (vy-py)**2 + (vz-pz)**2) + pore_r > pr:
                        continue
                    if any(math.sqrt((vx-xv)**2 + (vy-yv)**2 + (vz-zv)**2) < pore_r + rv
                           for xv, yv, zv, rv in existing_voids_snap):
                        continue
                    if any(math.sqrt((vx-xv)**2 + (vy-yv)**2 + (vz-zv)**2) < pore_r + rv
                           for xv, yv, zv, rv in placed):
                        continue
                    placed.append((vx, vy, vz, pore_r))
                    existing_voids_snap.append((vx, vy, vz, pore_r))
                    pore_vol_actual = 4 / 3 * math.pi * pore_r ** 3
                    cum_vol += pore_vol_actual
                    break

        return placed, cum_vol

    # ------------------------------------------------------------------
    # 2-D branch
    # ------------------------------------------------------------------
    if dim == 2:
        total_domain_area = img_size * img_size
        target_void_area  = void_fraction * total_domain_area
        current_void_area = 0.0

        print(f"\n{'='*60}")
        print("TARGET PARAMETERS (2D)")
        print(f"  Target void fraction: {void_fraction:.4f}")
        print(f"  Target void area:     {target_void_area:.2e}")
        print(f"{'='*60}\n")

        def img_r(mu):
            return mu / physical_size * img_size

        # Spatial grid
        cell_size = img_r(mean_rad_solid) * 4
        n_cells   = max(1, int(math.ceil(img_size / cell_size)))
        grid      = [[[] for _ in range(n_cells)] for __ in range(n_cells)]

        def cell_coords(x, y):
            cx = max(0, min(n_cells - 1, int(x // cell_size)))
            cy = max(0, min(n_cells - 1, int(y // cell_size)))
            return cx, cy

        def nearby(x, y, circles):
            cx, cy = cell_coords(x, y)
            for i in range(max(0, cx-1), min(n_cells, cx+2)):
                for j in range(max(0, cy-1), min(n_cells, cy+2)):
                    for idx in grid[i][j]:
                        yield circles[idx]

        # FIX 1: use np.array + column slicing instead of three np.fromiter calls.
        def overlaps_fast(x, y, r, neighbors, factor=0.999):
            if not neighbors:
                return False
            nb = np.array(neighbors, dtype=float)
            dx = x - nb[:, 0]
            dy = y - nb[:, 1]
            cr = nb[:, 2]
            return np.any(dx*dx + dy*dy < ((r + cr) * factor) ** 2)

        def clipped_circle_area(x, y, r, lo=0, hi=1):
            def segment_area(h):
                if h <= 0:    return 0.0
                if h >= 2*r:  return math.pi * r**2
                return r**2 * math.acos((r - h) / r) - (r - h) * math.sqrt(2*r*h - h**2)
            A  = math.pi * r**2
            A -= segment_area(r - (x - lo))
            A -= segment_area(r - (hi - x))
            A -= segment_area(r - (y - lo))
            A -= segment_area(r - (hi - y))
            return A

        circles       = []
        porous_circles = []   # FIX 4: track porous grains separately for void budget
        voids         = []
        solid_area = hollow_area = porous_area = 0.0
        sigma_ln   = math.sqrt(math.log(1 + rad_dev**2))

        # ---- Solid grains ----
        print("Placing solid grains...")
        for attempt in range(max_attempts):
            if solid_area >= vol_percent_solid:
                break
            mu_ln = math.log(img_r(mean_rad_solid)) - 0.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            if solid_area + math.pi * r**2 / total_domain_area > vol_percent_solid:
                continue
            for _ in range(100):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)
                # FIX 2: compute cell coords once; reuse for grid insert below.
                cx_cell, cy_cell = cell_coords(x, y)

                if len(grid[cx_cell][cy_cell]) > 15:
                    continue

                neighbors = list(nearby(x, y, circles))

                if not overlaps_fast(x, y, r, neighbors, factor=0.999):
                    circles.append((x, y, r))
                    grid[cx_cell][cy_cell].append(len(circles) - 1)  # FIX 2
                    if (x > r and x < img_size - r and
                        y > r and y < img_size - r):
                        solid_area += math.pi * r**2
                    else:
                        solid_area += clipped_circle_area(x, y, r)
                    break
        print(f"  Solid area fraction: {solid_area:.4f}")

        # ---- Hollow grains ----
        print("\nPlacing hollow grains...")
        for attempt in range(max_attempts):
            if hollow_area >= vol_percent_hollow:
                break
            mu_ln = math.log(img_r(mean_rad_hollow)) - 0.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            if hollow_area + math.pi * r**2 / total_domain_area > vol_percent_hollow:
                continue
            max_pos = 4 if hollow_area < 0.6 * vol_percent_hollow else 30
            for _ in range(max_pos):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)
                # FIX 2: cell coords already computed here; used for both density
                # check and grid insert (no second call needed).
                cx_cell, cy_cell = cell_coords(x, y)

                if len(grid[cx_cell][cy_cell]) > 15:
                    continue

                neighbors = list(nearby(x, y, circles))

                if not overlaps_fast(x, y, r, neighbors, factor=0.999):
                    rv = r * math.sqrt(void_fraction / vol_percent_hollow)

                    circles.append((x, y, r))
                    voids.append((x, y, rv))
                    grid[cx_cell][cy_cell].append(len(circles) - 1)

                    if (x > r and x < img_size - r and
                        y > r and y < img_size - r):
                        hollow_area += math.pi * r**2
                    else:
                        hollow_area += clipped_circle_area(x, y, r)

                    if (x > rv and x < img_size - rv and
                        y > rv and y < img_size - rv):
                        current_void_area += math.pi * rv**2
                    else:
                        current_void_area += clipped_circle_area(x, y, rv)
                    break
        print(f"  Hollow area fraction: {hollow_area:.4f}")
        print(f"  Void fraction so far: {current_void_area/total_domain_area:.4f}")

        # ---- Porous grains ----
        # FIX 3: replaced the erroneous double for _ in range(max_pos) nest
        # (which ran max_pos^2 position attempts and printed progress inside the
        # inner body) with a single position-attempt loop, matching the 3D branch.
        print("\nPlacing porous grains...")
        for attempt in range(max_attempts):
            if porous_area >= vol_percent_porous:
                break
            mu_ln = math.log(img_r(mean_rad_porous)) - 1.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            if porous_area + math.pi * r**2 / total_domain_area > vol_percent_porous:
                continue
            max_pos = 4 if porous_area < 0.6 * vol_percent_porous else 40
            placed = False
            for _ in range(max_pos):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)
                # FIX 2: single cell_coords call, reused for grid insert.
                cx_cell, cy_cell = cell_coords(x, y)

                if len(grid[cx_cell][cy_cell]) > 15:
                    continue

                neighbors = list(nearby(x, y, circles))

                if not overlaps_fast(x, y, r, neighbors, factor=0.97):
                    circles.append((x, y, r))
                    porous_circles.append((x, y, r))  # FIX 4
                    grid[cx_cell][cy_cell].append(len(circles) - 1)  # FIX 2

                    if (x > r and x < img_size - r and
                        y > r and y < img_size - r):
                        porous_area += math.pi * r**2
                    else:
                        porous_area += clipped_circle_area(x, y, r)

                    placed = True
                    break
            if placed and porous_area >= vol_percent_porous:
                break
        print(f"  Porous area fraction: {porous_area:.4f}")

        # ---- MWD check ----
        save_xyzr(circles, AP_xyzr, img_size, physical_size)
        if mwd_tolerance is not None:
            radii = np.array([r for (x, y, r) in circles])
            mwd_actual = 2 * np.sum(radii**3) / np.sum(radii**2) * (physical_size / img_size)
            print(f"\nMWD check: {mwd_actual:.4e} m (target {mwd_target:.4e} m)")
            if abs(mwd_actual - mwd_target) > mwd_tolerance:
                print("MWD out of tolerance — skipping void placement.")
                return None

        # ---- Void placement (parallelized per porous grain) ----
        # FIX 4 & 5: budget divided by number of porous grains only; loop
        # iterates over porous_circles rather than all circles.
        print("\nPlacing voids within porous grains (parallel)...")
        if vol_percent_porous > 0:
            per_grain_budget = (target_void_area - current_void_area) / max(len(porous_circles), 1)

            void_snap_base = [(x, y, rv) for (x, y, rv) in voids]

            futures_map = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for idx, grain in enumerate(porous_circles):
                    seed = int(rng.integers(0, 2**31))
                    fut = executor.submit(
                        _place_voids_in_grain,
                        grain,
                        list(void_snap_base),
                        per_grain_budget,
                        dim,
                        seed,
                    )
                    futures_map[fut] = idx

                for fut in concurrent.futures.as_completed(futures_map):
                    new_voids, vol_added = fut.result()
                    voids.extend(new_voids)
                    current_void_area += vol_added

            print(f"  Void fraction after parallel pass: {current_void_area/total_domain_area:.4f}")

            # Sequential top-up if still short
            max_topup = 20
            topup_iter = 0
            while current_void_area < target_void_area and topup_iter < max_topup:
                topup_iter += 1
                progress = False
                for grain in porous_circles:  # FIX 5: only porous grains
                    if current_void_area >= target_void_area:
                        break
                    remaining = target_void_area - current_void_area
                    new_voids, vol_added = _place_voids_in_grain(
                        grain, list((x, y, rv) for (x, y, rv) in voids),
                        remaining, dim, int(rng.integers(0, 2**31))
                    )
                    voids.extend(new_voids)
                    current_void_area += vol_added
                    if vol_added > 0:
                        progress = True
                if not progress:
                    print("WARNING: could not place more voids without overlap.")
                    break

        print(f"  Total voids placed: {len(voids)}")
        print(f"  Void fraction achieved: {current_void_area/total_domain_area:.4f}")
        print(f"  Target:                 {void_fraction:.4f}")

        print(f"\n{'='*60}")
        print("FINAL RESULTS (2D)")
        print(f"  Total grains:      {len(circles)}")
        print(f"  AP area fraction:  {solid_area+hollow_area+porous_area:.4f}")
        print(f"  Void fraction:     {current_void_area/total_domain_area:.4f}")
        print(f"  Error:             {abs(current_void_area/total_domain_area - void_fraction):.2e}")
        print(f"{'='*60}\n")

        save_xyzr(voids, void_xyzr, img_size, physical_size)
        return current_void_area / total_domain_area

    # ------------------------------------------------------------------
    # 3-D branch
    # ------------------------------------------------------------------
    else:
        total_domain_vol = img_size ** 3
        target_void_vol  = void_fraction * total_domain_vol
        current_void_vol = 0.0
        def overlaps_vec_3d(x, y, z, r, nb):
            if len(nb) == 0:
                return False
            d = nb[:,:3] - np.array([x,y,z])
            return np.any(np.sum(d*d, axis=1) < (r + nb[:,3])**2)

        print(f"\n{'='*60}")
        print("TARGET PARAMETERS (3D)")
        print(f"  Target void fraction: {void_fraction:.4f}")
        print(f"  Target void vol:      {target_void_vol:.2e}")
        print(f"{'='*60}\n")

        def img_r(mu):
            return mu / physical_size * img_size

        # Spatial grid
        cell_size = img_r(mean_rad_solid) * 4
        n_cells   = max(1, int(math.ceil(img_size / cell_size)))
        grid      = [[[[] for _ in range(n_cells)]
                      for __ in range(n_cells)]
                     for ___ in range(n_cells)]

        def cell_coords(x, y, z):
            i = max(0, min(n_cells-1, int(x // cell_size)))
            j = max(0, min(n_cells-1, int(y // cell_size)))
            k = max(0, min(n_cells-1, int(z // cell_size)))
            return i, j, k

        def nearby(x, y, z, spheres):
            cx, cy, cz = cell_coords(x, y, z)
            for i in range(max(0, cx-1), min(n_cells, cx+2)):
                for j in range(max(0, cy-1), min(n_cells, cy+2)):
                    for k in range(max(0, cz-1), min(n_cells, cz+2)):
                        for idx in grid[i][j][k]:
                            yield spheres[idx]

        # FIX 1: use np.array + column slicing instead of four np.fromiter calls.
        def overlaps_fast_3d(x, y, z, r, neighbors, factor=0.98):
            if not neighbors:
                return False
            nb = np.array(neighbors, dtype=float)
            dx = x - nb[:, 0]
            dy = y - nb[:, 1]
            dz = z - nb[:, 2]
            cr = nb[:, 3]
            return np.any(dx*dx + dy*dy + dz*dz < ((r + cr) * factor) ** 2)

        def clipped_sphere_volume(x, y, z, r, lo=0, hi=1):
            def cap_vol(h):
                if h <= 0:    return 0.0
                if h >= 2*r:  return 4/3 * math.pi * r**3
                return math.pi * h**2 * (3*r - h) / 3
            V  = 4/3 * math.pi * r**3
            V -= cap_vol(r - (x - lo))
            V -= cap_vol(r - (hi - x))
            V -= cap_vol(r - (y - lo))
            V -= cap_vol(r - (hi - y))
            V -= cap_vol(r - (z - lo))
            V -= cap_vol(r - (hi - z))
            return V

        spheres        = []
        porous_spheres = []   # FIX 4: track porous grains separately for void budget
        voids          = []
        solid_vol = hollow_vol = porous_vol = 0.0
        sigma_ln  = math.sqrt(math.log(1 + rad_dev**2))

        # ---- Solid grains ----
        print("Placing solid grains...")
        placed = False
        for attempt in range(max_attempts):
            if solid_vol >= vol_percent_solid:
                break
            mu_ln = math.log(img_r(mean_rad_solid)) - 1.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            if solid_vol + 4/3*math.pi*r**3/total_domain_vol > vol_percent_solid:
                continue
            for _ in range(100):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)
                z = rng.uniform(-margin + r, img_size + margin - r)
                # FIX 2: compute cell coords once; reuse for grid insert below.
                cx_cell, cy_cell, cz_cell = cell_coords(x, y, z)

                if len(grid[cx_cell][cy_cell][cz_cell]) > 20:
                    continue

                neighbors = list(nearby(x, y, z, spheres))

                if not overlaps_fast_3d(x, y, z, r, neighbors, factor=0.98):
                    spheres.append((x, y, z, r))
                    grid[cx_cell][cy_cell][cz_cell].append(len(spheres) - 1)  # FIX 2
                    if (x > r and x < img_size - r and
                        y > r and y < img_size - r and
                        z > r and z < img_size - r):
                        solid_vol += 4/3 * math.pi * r**3
                    else:
                        solid_vol += clipped_sphere_volume(x, y, z, r)
                    placed = True
                    break
            if placed and solid_vol >= vol_percent_solid:
                break
        print(f"  Solid volume fraction: {solid_vol:.4f}")

        # ---- Hollow grains ----
        print("\nPlacing hollow grains...")
        for attempt in range(max_attempts):
            if hollow_vol >= vol_percent_hollow:
                break
            mu_ln = math.log(img_r(mean_rad_hollow)) - 1.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            if hollow_vol + 4/3*math.pi*r**3/total_domain_vol > vol_percent_hollow:
                continue
            max_pos = 4 if hollow_vol < 0.6 * vol_percent_hollow else 30
            for _ in range(max_pos):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)
                z = rng.uniform(-margin + r, img_size + margin - r)
                # FIX 2: single cell_coords call, reused for grid insert.
                cx_cell, cy_cell, cz_cell = cell_coords(x, y, z)

                if len(grid[cx_cell][cy_cell][cz_cell]) > 20:
                    continue

                neighbors = list(nearby(x, y, z, spheres))

                if not overlaps_fast_3d(x, y, z, r, neighbors, factor=0.98):
                    rv = r * (void_fraction / vol_percent_hollow) ** (1/3)
                    spheres.append((x, y, z, r))
                    voids.append((x, y, z, rv))
                    grid[cx_cell][cy_cell][cz_cell].append(len(spheres) - 1)  # FIX 2
                    if (x > r and x < img_size - r and
                        y > r and y < img_size - r and
                        z > r and z < img_size - r):
                        hollow_vol += 4/3 * math.pi * r**3
                    else:
                        hollow_vol += clipped_sphere_volume(x, y, z, r)

                    if (x > rv and x < img_size - rv and
                        y > rv and y < img_size - rv and
                        z > rv and z < img_size - rv):
                        current_void_vol += 4/3 * math.pi * rv**3
                    else:
                        current_void_vol += clipped_sphere_volume(x, y, z, rv)
                    break
        print(f"  Hollow volume fraction: {hollow_vol:.4f}")
        print(f"  Void fraction so far:   {current_void_vol/total_domain_vol:.4f}")

        # ---- Porous grains ----
        print("\nPlacing porous grains...")
        for attempt in range(max_attempts):
            if attempt % 1000 == 0 and attempt != 0:
                print(f"Reached porous attempt {attempt}, porous volume so far: {porous_vol:.4f}")
            if porous_vol >= vol_percent_porous:
                break
            mu_ln = math.log(img_r(mean_rad_porous)) - 1.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            if porous_vol + 4/3*math.pi*r**3/total_domain_vol > vol_percent_porous:
                continue
            max_pos = 50 if porous_vol < 0.5 * vol_percent_porous else 150
            placed = False
            for _ in range(max_pos):
                x = rng.uniform(-margin + r, img_size + margin - r)
                y = rng.uniform(-margin + r, img_size + margin - r)
                z = rng.uniform(-margin + r, img_size + margin - r)
                # FIX 2: single cell_coords call, reused for grid insert.
                cx_cell, cy_cell, cz_cell = cell_coords(x, y, z)

                if len(grid[cx_cell][cy_cell][cz_cell]) > 20:
                    continue

                neighbors = list(nearby(x, y, z, spheres))

                if not overlaps_fast_3d(x, y, z, r, neighbors, factor=0.98):
                    spheres.append((x, y, z, r))
                    porous_spheres.append((x, y, z, r))  # FIX 4
                    grid[cx_cell][cy_cell][cz_cell].append(len(spheres) - 1)  # FIX 2
                    if (x > r and x < img_size - r and
                        y > r and y < img_size - r and
                        z > r and z < img_size - r):
                        porous_vol += 4/3 * math.pi * r**3
                    else:
                        porous_vol += clipped_sphere_volume(x, y, z, r)
                    placed = True
                    break
            if placed and porous_vol >= vol_percent_porous:
                break
        print(f"  Porous volume fraction: {porous_vol:.4f}")

        # ---- MWD check ----
        save_xyzr(spheres, AP_xyzr, img_size, physical_size)
        if mwd_tolerance is not None:
            radii = np.array([r for (x, y, z, r) in spheres])
            mwd_actual = 2 * np.sum(radii**3) / np.sum(radii**2) * (physical_size / img_size)
            print(f"\nMWD check: {mwd_actual:.4e} m (target {mwd_target:.4e} m)")
            if abs(mwd_actual - mwd_target) > mwd_tolerance:
                print("MWD out of tolerance — skipping void placement.")
                return None

        # ---- Void placement (parallelized per porous grain) ----
        # FIX 4 & 5: budget divided by number of porous grains only; loop
        # iterates over porous_spheres rather than all spheres.
        print("\nPlacing voids within porous grains (sequential with dynamic budget)...")
        if vol_percent_porous > 0:
            n_porous = len(porous_circles)
            for grain_idx, grain in enumerate(porous_circles):
                if current_void_area >= target_void_area:
                    break
                remaining_grains = n_porous - grain_idx
                per_grain_budget = (target_void_area - current_void_area) / remaining_grains

                new_voids, vol_added = _place_voids_in_grain(
                    grain,
                    [(x, y, rv) for (x, y, rv) in voids],  # fresh snapshot each grain
                    per_grain_budget,
                    dim,
                    int(rng.integers(0, 2**31)),
                )
                voids.extend(new_voids)
                current_void_area += vol_added

            print(f"  Void fraction after sequential pass: {current_void_area/total_domain_area:.4f}")

            # Sequential top-up if still short
            max_topup = 20
            topup_iter = 0
            while current_void_area < target_void_area and topup_iter < max_topup:
                topup_iter += 1
                progress = False
                for grain in porous_circles:
                    if current_void_area >= target_void_area:
                        break
                    remaining = target_void_area - current_void_area
                    new_voids, vol_added = _place_voids_in_grain(
                        grain, [(x, y, rv) for (x, y, rv) in voids],
                        remaining, dim, int(rng.integers(0, 2**31))
                    )
                    voids.extend(new_voids)
                    current_void_area += vol_added
                    if vol_added > 0:
                        progress = True
                if not progress:
                    print("WARNING: could not place more voids without overlap.")
                    break

            # Sequential top-up
            max_topup = 20
            topup_iter = 0
            while current_void_vol < target_void_vol and topup_iter < max_topup:
                topup_iter += 1
                progress = False
                void_list = [(x, y, z, rv) for (x, y, z, rv) in voids]
                for grain in porous_spheres:  # FIX 5: only porous grains
                    if current_void_vol >= target_void_vol:
                        break
                    remaining = target_void_vol - current_void_vol
                    new_voids, vol_added = _place_voids_in_grain(
                        grain, list(void_list), remaining, dim,
                        int(rng.integers(0, 2**31))
                    )
                    voids.extend(new_voids)
                    void_list.extend(new_voids)
                    current_void_vol += vol_added
                    if vol_added > 0:
                        progress = True
                if not progress:
                    print("WARNING: could not place more voids without overlap.")
                    break

        print(f"  Total voids placed: {len(voids)}")
        print(f"  Void fraction achieved: {current_void_vol/total_domain_vol:.4f}")
        print(f"  Target:                 {void_fraction:.4f}")

        print(f"\n{'='*60}")
        print("FINAL RESULTS (3D)")
        print(f"  Total grains:       {len(spheres)}")
        print(f"  AP volume fraction: {solid_vol+hollow_vol+porous_vol:.4f}")
        print(f"  Void fraction:      {current_void_vol/total_domain_vol:.4f}")
        print(f"  Error:              {abs(current_void_vol/total_domain_vol - void_fraction):.2e}")
        print(f"{'='*60}\n")

        save_xyzr(voids, void_xyzr, img_size, physical_size)
        return current_void_vol / total_domain_vol


# ---------------------------------------------------------------------------
# Worker function (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _run_one_attempt(args):
    """
    Generate a single structure and return metadata dict, or None on failure.
    Designed to run in a subprocess via ProcessPoolExecutor.
    """
    (attempt_idx, accepted_idx, name, subfolder,
     physical_size, rad_dev, max_attempts,
     vol_percent_solid, vol_percent_hollow, vol_percent_porous,
     void_fraction, mwd_target, mwd_tolerance, dim,
     mean_rad_solid, mean_rad_hollow, mean_rad_porous) = args

    ap_xyzr   = os.path.join(subfolder, f"{name}_AP.xyzr")
    void_xyzr = os.path.join(subfolder, f"{name}_void.xyzr")

    try:
        void_frac = gen_struct_combined_2or3D(
            ap_xyzr, void_xyzr,
            physical_size=physical_size,
            rad_dev=rad_dev,
            max_attempts=max_attempts,
            vol_percent_solid=vol_percent_solid,
            vol_percent_hollow=vol_percent_hollow,
            vol_percent_porous=vol_percent_porous,
            void_fraction=void_fraction,
            mwd_target=mwd_target,
            dim=dim,
            mean_rad_solid=mean_rad_solid,
            mean_rad_hollow=mean_rad_hollow,
            mean_rad_porous=mean_rad_porous,
            mwd_tolerance=mwd_tolerance,
        )
    except Exception as e:
        print(f"  [attempt {attempt_idx}] Generation failed: {e}")
        for path in [ap_xyzr, void_xyzr]:
            if os.path.exists(path):
                os.remove(path)
        return None

    if void_frac is None:
        if os.path.exists(ap_xyzr):
            os.remove(ap_xyzr)
        return None

    try:
        mwd = mean_weight_diameter(ap_xyzr)
    except Exception as e:
        print(f"  [attempt {attempt_idx}] MWD calculation failed: {e}")
        for path in [ap_xyzr, void_xyzr]:
            if os.path.exists(path):
                os.remove(path)
        return None

    mwd_error = abs(mwd - mwd_target)
    accepted  = mwd_error <= mwd_tolerance
    status    = "ACCEPTED ✓" if accepted else "rejected ✗"
    print(f"  [attempt {attempt_idx}] MWD {mwd:.4e} m  error {mwd_error:.4e}  {status}")

    if accepted:
        return {
            "index":     accepted_idx,
            "name":      name,
            "mwd":       mwd,
            "void_frac": void_frac,
            "attempt":   attempt_idx,
            "ap_xyzr":   ap_xyzr,
            "void_xyzr": void_xyzr,
        }
    else:
        for path in [ap_xyzr, void_xyzr]:
            if os.path.exists(path):
                os.remove(path)
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_structures_with_target_mwd(
    subfolder,
    target_mwd,
    mwd_tolerance=0.5e-6,
    n_target=1,
    max_total_attempts=200,
    base_name="A",
    dim: Literal[2, 3] = 2,
    # gen_struct_combined parameters
    physical_size=200e-6,
    rad_dev=0.4,
    max_attempts=800000,
    vol_percent_solid=0,
    vol_percent_hollow=0,
    vol_percent_porous=0.7141,
    void_fraction=0.1633,
    mean_rad_hollow=4.05e-6,
    mean_rad_porous=4.25e-6,
    mean_rad_solid=4.05e-6,
    n_workers=None,   # None → os.cpu_count()
):
    """
    Parallel version: dispatches up to n_workers structure-generation jobs
    simultaneously using ProcessPoolExecutor.  Jobs stream in as they finish;
    as soon as n_target accepted structures are collected, remaining futures
    are cancelled and the function returns.

    Parameters
    ----------
    n_workers : int or None
        Number of parallel worker processes.  Defaults to os.cpu_count().
        Set to 1 to disable parallelism (useful for debugging).
    """
    os.makedirs(subfolder, exist_ok=True)
    subsubfolder = os.path.join(subfolder, "example_images")

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    print(f"\n{'='*60}")
    print(f"Target MWD:     {target_mwd:.4e} m")
    print(f"Tolerance:      ±{mwd_tolerance:.4e} m")
    print(f"Target count:   {n_target}")
    print(f"Dimension:      {dim}D")
    print(f"Workers:        {n_workers}")
    print(f"{'='*60}\n")

    common = dict(
        physical_size=physical_size, rad_dev=rad_dev,
        max_attempts=max_attempts,
        vol_percent_solid=vol_percent_solid,
        vol_percent_hollow=vol_percent_hollow,
        vol_percent_porous=vol_percent_porous,
        void_fraction=void_fraction, mwd_target=target_mwd,
        mwd_tolerance=mwd_tolerance, dim=dim,
        mean_rad_solid=mean_rad_solid, mean_rad_hollow=mean_rad_hollow,
        mean_rad_porous=mean_rad_porous,
    )

    accepted      = []
    accepted_idx  = 0
    attempt_idx   = 0

    # We use a rolling window: keep up to n_workers futures in flight at once.
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        pending = {}   # future -> attempt_idx

        def _submit_next():
            nonlocal attempt_idx, accepted_idx
            name = f"{base_name}_{accepted_idx + len(pending):02d}_tmp{attempt_idx}"
            args = (
                attempt_idx, accepted_idx, name, subfolder,
                common["physical_size"], common["rad_dev"], common["max_attempts"],
                common["vol_percent_solid"], common["vol_percent_hollow"],
                common["vol_percent_porous"], common["void_fraction"],
                common["mwd_target"], common["mwd_tolerance"], common["dim"],
                common["mean_rad_solid"], common["mean_rad_hollow"],
                common["mean_rad_porous"],
            )
            fut = executor.submit(_run_one_attempt, args)
            pending[fut] = attempt_idx
            attempt_idx += 1

        # Seed the pool
        for _ in range(min(n_workers, max_total_attempts)):
            _submit_next()

        while pending and len(accepted) < n_target and attempt_idx <= max_total_attempts:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for fut in done:
                del pending[fut]
                result = fut.result()

                if result is not None:
                    # Rename files to canonical names
                    final_name    = f"{base_name}_{accepted_idx:02d}"
                    final_ap      = os.path.join(subfolder, f"{final_name}_AP.xyzr")
                    final_void    = os.path.join(subfolder, f"{final_name}_void.xyzr")
                    os.rename(result["ap_xyzr"],   final_ap)
                    os.rename(result["void_xyzr"],  final_void)

                    result["name"]      = final_name
                    result["ap_xyzr"]   = final_ap
                    result["void_xyzr"] = final_void
                    accepted.append(result)

                    print(f"  → Kept as {final_name}  "
                          f"(accepted {len(accepted)}/{n_target})")

                    # Plot example image for first accepted structure
                    if accepted_idx == 0:
                        os.makedirs(subsubfolder, exist_ok=True)
                        plot_from_xyzr(
                            final_ap, final_void,
                            os.path.join(subsubfolder, f"{final_name}.png"),
                            dim=dim, physical_size=physical_size,
                        )
                    accepted_idx += 1

                # Refill pool if more structures still needed
                if (len(accepted) < n_target
                        and attempt_idx < max_total_attempts
                        and len(pending) < n_workers):
                    _submit_next()

        # Cancel any still-running futures
        for fut in pending:
            fut.cancel()

    print(f"\nDone. {len(accepted)}/{n_target} structures accepted "
          f"in {attempt_idx} total attempts.")
    return accepted


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_from_xyzr(ap_xyzr_path, void_xyzr_path, save_path,
                   physical_size=200e-6, img_size=1,
                   dim=2,
                   dpi=1024,
                   ap_alpha=1.0,
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
                    pts.append((x/physical_size*img_size,
                                y/physical_size*img_size,
                                0, r/physical_size*img_size))
                elif len(vals) >= 4:
                    x, y, z, r = map(float, vals[:4])
                    pts.append((x/physical_size*img_size,
                                y/physical_size*img_size,
                                z/physical_size*img_size,
                                r/physical_size*img_size))
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
        ax  = fig.add_subplot(111, projection='3d')
        u   = np.linspace(0, 2*np.pi, sphere_resolution)
        v   = np.linspace(0, np.pi,   sphere_resolution)
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
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    generate_structures_with_target_mwd(
        '3D_xyzrs',
        target_mwd=8.1e-6,
        base_name="A",
        dim=3,
        physical_size=100e-6,
        mean_rad_porous=5e-6,
        mean_rad_solid=4.0e-6,
        void_fraction=0.11,
        vol_percent_porous=0.60025,
        mwd_tolerance=0.05e-6,
        n_workers=4,   # tune to your CPU count
    )


if __name__ == "__main__":
    main()