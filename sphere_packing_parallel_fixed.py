import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from typing import Literal
import concurrent.futures

MAX_VOID_FRACTION_PER_GRAIN = 0.20


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_xyzr(particles, filepath, img_size, physical_size):
    scale = physical_size / img_size
    # PERF-5: build all lines then write once instead of one write() per particle
    lines = []
    for p in particles:
        if len(p) == 3:
            x, y, r = p
            lines.append(f"{x*scale:.8e} {y*scale:.8e} 0.0 {r*scale:.8e}\n")
        else:
            x, y, z, r = p
            lines.append(f"{x*scale:.8e} {y*scale:.8e} {z*scale:.8e} {r*scale:.8e}\n")
    with open(filepath, 'w') as f:
        f.writelines(lines)


# PERF-6: optional radii array to avoid re-reading the file
def mean_weight_diameter(filename, radii=None):
    if radii is None:
        data = np.loadtxt(filename)
        radii = data[:, -1]
    return 2 * np.sum(radii**3) / np.sum(radii**2)


# ---------------------------------------------------------------------------
# Fast void placement (3D) - vectorized, intra-grain only
# FIX-P: cap enforcement removed from here; caller passes a pre-capped budget.
# ---------------------------------------------------------------------------

def _place_voids_in_grain_fast(grain, target_vol, rng_seed, img_size=1):
    local_rng = np.random.default_rng(rng_seed)
    px, py, pz, pr = grain

    placed_xyz = []
    placed_r   = []
    cum_vol    = 0.0

    consecutive_failures = 0
    max_consecutive      = 50

    # PERF-1: pre-allocate candidate arrays once; reuse each iteration
    n_cands = 300
    phi_buf   = np.empty(n_cands)
    theta_buf = np.empty(n_cands)
    rho_buf   = np.empty(n_cands)

    for _ in range(800):
        if cum_vol >= target_vol:
            break
        if consecutive_failures >= max_consecutive:
            break

        pore_r = local_rng.lognormal(math.log(pr * 0.15), 0.4)
        pore_r = float(np.clip(pore_r, 0.1 * pr, 0.35 * pr))

        pore_vol = 4/3 * math.pi * pore_r**3
        if cum_vol + pore_vol > target_vol:
            remaining = target_vol - cum_vol
            if remaining <= 0:
                break
            pore_r   = (remaining / (4/3 * math.pi)) ** (1/3)
            pore_vol = remaining
            if pore_r < 0.1 * pr:
                break

        # PERF-1: fill pre-allocated buffers in-place
        local_rng.random(out=phi_buf)
        phi_buf   *= math.pi
        local_rng.random(out=theta_buf)
        theta_buf *= 2 * math.pi
        local_rng.random(out=rho_buf)
        rho_buf    = (pr - pore_r) * rho_buf ** (1/3)

        cands_x = px + rho_buf * np.sin(phi_buf) * np.cos(theta_buf)
        cands_y = py + rho_buf * np.sin(phi_buf) * np.sin(theta_buf)
        cands_z = pz + rho_buf * np.cos(phi_buf)

        dist_from_centre = np.sqrt((cands_x-px)**2 + (cands_y-py)**2 + (cands_z-pz)**2)
        valid_mask = dist_from_centre + pore_r <= pr

        valid_mask &= (cands_x - pore_r >= 0) & (cands_x + pore_r <= img_size)
        valid_mask &= (cands_y - pore_r >= 0) & (cands_y + pore_r <= img_size)
        valid_mask &= (cands_z - pore_r >= 0) & (cands_z + pore_r <= img_size)

        if len(placed_xyz) > 0:
            pvec = np.array(placed_xyz)
            pr_v = np.array(placed_r)
            dx = cands_x[:, None] - pvec[:, 0]
            dy = cands_y[:, None] - pvec[:, 1]
            dz = cands_z[:, None] - pvec[:, 2]
            dists    = np.sqrt(dx**2 + dy**2 + dz**2)
            overlaps = np.any(dists < pore_r + pr_v, axis=1)
            valid_mask &= ~overlaps

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            consecutive_failures += 1
            continue

        i = valid_indices[0]
        placed_xyz.append([cands_x[i], cands_y[i], cands_z[i]])
        placed_r.append(pore_r)
        cum_vol += pore_vol
        consecutive_failures = 0

    placed = [(placed_xyz[i][0], placed_xyz[i][1], placed_xyz[i][2], placed_r[i])
              for i in range(len(placed_r))]
    return placed, cum_vol


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
    if dim not in (2, 3):
        raise ValueError("Dimension must be 2 or 3.")

    rng = np.random.default_rng()
    img_size = 1
    margin = 0.07

    def generate_radii(target_fraction, mu_ln, sigma_ln, dim):
        radii = []
        total = 0.0
        while total < target_fraction:
            r = rng.lognormal(mu_ln, sigma_ln)
            vol = math.pi * r**2 if dim == 2 else 4/3 * math.pi * r**3
            radii.append(r)
            total += vol
        return sorted(radii, reverse=True)

    def _place_voids_in_grain_2d(grain, existing_voids_snap, target_remaining, rng_seed):
        local_rng = np.random.default_rng(rng_seed)
        placed  = []
        cum_vol = 0.0
        px, py, pr = grain[:3]

        for _ in range(800):
            if cum_vol >= target_remaining:
                break
            pore_r = local_rng.lognormal(math.log(pr * 0.1), 0.4)
            pore_r = float(np.clip(pore_r, 0.1 * pr, 0.35 * pr))
            pore_vol = math.pi * pore_r**2
            if cum_vol + pore_vol > target_remaining:
                pore_r   = math.sqrt((target_remaining - cum_vol) / math.pi)
                pore_vol = target_remaining - cum_vol

            for _ in range(300):
                theta = local_rng.uniform(0, 2 * math.pi)
                rho   = local_rng.uniform(0, pr - pore_r)
                vx    = px + rho * math.cos(theta)
                vy    = py + rho * math.sin(theta)
                if math.hypot(vx - px, vy - py) + pore_r > pr:
                    continue
                if vx - pore_r < 0 or vx + pore_r > img_size:
                    continue
                if vy - pore_r < 0 or vy + pore_r > img_size:
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

        cell_size = img_r(mean_rad_solid) * 4
        n_cells   = max(1, int(math.ceil(img_size / cell_size)))
        grid      = [[[] for _ in range(n_cells)] for __ in range(n_cells)]

        def cell_coords(x, y):
            cx = max(0, min(n_cells - 1, int(x // cell_size)))
            cy = max(0, min(n_cells - 1, int(y // cell_size)))
            return cx, cy

        def nearby_2d(x, y, circles, r_query=0):
            max_r = max((c[2] for c in circles), default=r_query)
            window = max(1, math.ceil((r_query + max_r) / cell_size))
            cx, cy = cell_coords(x, y)
            result = []
            for i in range(max(0, cx - window), min(n_cells, cx + window + 1)):
                for j in range(max(0, cy - window), min(n_cells, cy + window + 1)):
                    for idx in grid[i][j]:
                        result.append(circles[idx])
            return result

        def overlaps_fast(x, y, r, neighbors, factor=0.999):
            if not neighbors:
                return False
            nb = np.array(neighbors, dtype=float)
            dx = x - nb[:, 0]; dy = y - nb[:, 1]; cr = nb[:, 2]
            return np.any(dx*dx + dy*dy < ((r + cr) * factor) ** 2)

        def clipped_circle_area(x, y, r, lo=0, hi=1):
            def seg(h):
                if h <= 0: return 0.0
                if h >= 2*r: return math.pi * r**2
                return r**2 * math.acos((r-h)/r) - (r-h)*math.sqrt(2*r*h - h**2)
            A = math.pi * r**2
            A -= seg(r-(x-lo)); A -= seg(r-(hi-x))
            A -= seg(r-(y-lo)); A -= seg(r-(hi-y))
            return A

        circles = []; porous_circles = []; voids = []
        solid_area = hollow_area = porous_area = 0.0
        sigma_ln = math.sqrt(math.log(1 + rad_dev**2))

        print("Placing solid grains...")
        for _ in range(max_attempts):
            if solid_area >= vol_percent_solid: break
            mu_ln = math.log(img_r(mean_rad_solid)) - 0.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            if solid_area + math.pi*r**2/total_domain_area > vol_percent_solid: continue
            for _ in range(100):
                x = rng.uniform(-margin+r, img_size+margin-r)
                y = rng.uniform(-margin+r, img_size+margin-r)
                cx_c, cy_c = cell_coords(x, y)
                if len(grid[cx_c][cy_c]) > 15: continue
                nb = nearby_2d(x, y, circles, r)
                if not overlaps_fast(x, y, r, nb, 0.999):
                    circles.append((x, y, r)); grid[cx_c][cy_c].append(len(circles)-1)
                    solid_area += math.pi*r**2 if (x>r and x<img_size-r and y>r and y<img_size-r) else clipped_circle_area(x,y,r)
                    break
        print(f"  Solid area fraction: {solid_area:.4f}")

        print("\nPlacing hollow grains (2D, pre-sorted radii)...")
        sigma_ln_h = math.sqrt(math.log(1 + rad_dev**2))
        mu_ln_h    = math.log(img_r(mean_rad_hollow)) - 1.5*sigma_ln_h**2
        hollow_radii_list = generate_radii(vol_percent_hollow * 2, mu_ln_h, sigma_ln_h, dim=2)
        print(f"  Pre-generated {len(hollow_radii_list)} candidate hollow radii")
        hollow_area = 0.0

        for r in hollow_radii_list:
            if hollow_area >= vol_percent_hollow: break
            if current_void_area / total_domain_area >= void_fraction: break
            if hollow_area + math.pi*r**2/total_domain_area > vol_percent_hollow*1.05: continue
            max_pos = 150 if hollow_area < 0.4*vol_percent_hollow else 500
            for _ in range(max_pos):
                x = rng.uniform(-margin+r, img_size+margin-r)
                y = rng.uniform(-margin+r, img_size+margin-r)
                cx_c, cy_c = cell_coords(x, y)
                if len(grid[cx_c][cy_c]) > 15: continue
                nb = nearby_2d(x, y, circles, r)
                if not overlaps_fast(x, y, r, nb, 0.98):
                    rv = r * (void_fraction / vol_percent_hollow) ** (1/2)
                    circles.append((x, y, r)); voids.append((x, y, rv))
                    grid[cx_c][cy_c].append(len(circles)-1)
                    hollow_area += math.pi*r**2 if (x>r and x<img_size-r and y>r and y<img_size-r) else clipped_circle_area(x,y,r)
                    current_void_area += math.pi*rv**2  # FIX-X: always full circle
                    break
        print(f"  Hollow area fraction: {hollow_area:.4f}")
        print(f"  Void fraction so far: {current_void_area/total_domain_area:.4f}")

        print("\nPlacing porous grains (2D, pre-sorted radii)...")
        sigma_ln_p = math.sqrt(math.log(1 + rad_dev**2))
        mu_ln_p    = math.log(img_r(mean_rad_porous)) - 1.5 * sigma_ln_p**2
        porous_radii_list = generate_radii(vol_percent_porous*2, mu_ln_p, sigma_ln_p, dim=2)
        print(f"  Pre-generated {len(porous_radii_list)} candidate porous radii")

        for r in porous_radii_list:
            if porous_area >= vol_percent_porous: break
            if porous_area + math.pi*r**2/total_domain_area > vol_percent_porous*1.05: continue
            max_pos = 150 if porous_area < 0.4*vol_percent_porous else 500
            for _ in range(max_pos):
                x = rng.uniform(-margin+r, img_size+margin-r)
                y = rng.uniform(-margin+r, img_size+margin-r)
                cx_c, cy_c = cell_coords(x, y)
                if len(grid[cx_c][cy_c]) > 15: continue
                nb = nearby_2d(x, y, circles, r)
                if not overlaps_fast(x, y, r, nb, 0.98):
                    circles.append((x, y, r)); porous_circles.append((x, y, r))
                    grid[cx_c][cy_c].append(len(circles)-1)
                    porous_area += math.pi*r**2 if (x>r and x<img_size-r and y>r and y<img_size-r) else clipped_circle_area(x,y,r)
                    break
        print(f"  Porous area fraction: {porous_area:.4f}")

        save_xyzr(circles, AP_xyzr, img_size, physical_size)
        if mwd_tolerance is not None:
            radii = np.array([r for (x, y, r) in circles])
            mwd_actual = mean_weight_diameter(AP_xyzr, radii=radii) * (physical_size / img_size)
            print(f"\nMWD check: {mwd_actual:.4e} m (target {mwd_target:.4e} m)")
            if abs(mwd_actual - mwd_target) > mwd_tolerance:
                print("MWD out of tolerance - skipping void placement.")
                return None

        print("\nPlacing voids within porous grains (parallel)...")
        if vol_percent_porous > 0 and len(porous_circles) > 0:
            n_porous = len(porous_circles)
            grain_areas      = np.array([math.pi * g[2]**2 for g in porous_circles])
            total_grain_area = grain_areas.sum()
            void_budget      = target_void_area - current_void_area

            per_grain_caps    = MAX_VOID_FRACTION_PER_GRAIN * grain_areas
            per_grain_placed  = np.zeros(n_porous)

            per_grain_budgets = np.minimum(
                void_budget * (grain_areas / total_grain_area),
                per_grain_caps
            )

            void_snap_base = list(voids)
            futures_map = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for idx, grain in enumerate(porous_circles):
                    seed = int(rng.integers(0, 2**31))
                    fut  = executor.submit(_place_voids_in_grain_2d, grain,
                                           list(void_snap_base), per_grain_budgets[idx], seed)
                    futures_map[fut] = idx
                for fut in concurrent.futures.as_completed(futures_map):
                    grain_idx = futures_map[fut]
                    new_voids, vol_added = fut.result()
                    voids.extend(new_voids)
                    current_void_area += vol_added
                    per_grain_placed[grain_idx] += vol_added
                    if len(voids) % 1000 < len(new_voids):
                        print(f"  Voids placed: {len(voids)}  fraction: {current_void_area/total_domain_area:.4f}")

            print(f"  Void fraction after parallel pass: {current_void_area/total_domain_area:.4f}")

            exhausted_grains = set(
                i for i in range(n_porous)
                if per_grain_placed[i] >= per_grain_caps[i]
            )

            for _ in range(20):
                if current_void_area >= target_void_area: break
                active = [i for i in range(n_porous) if i not in exhausted_grains]
                if not active: break

                remaining_total = target_void_area - current_void_area
                active_areas    = grain_areas[active]
                redistrib       = remaining_total * (active_areas / active_areas.sum())

                progress = False
                for k, grain_idx in enumerate(active):
                    if current_void_area >= target_void_area: break
                    remaining_cap = per_grain_caps[grain_idx] - per_grain_placed[grain_idx]
                    budget = min(redistrib[k], remaining_cap)
                    if budget <= 0:
                        exhausted_grains.add(grain_idx)
                        continue
                    new_voids, vol_added = _place_voids_in_grain_2d(
                        porous_circles[grain_idx], list(voids), budget,
                        int(rng.integers(0, 2**31)))
                    voids.extend(new_voids)
                    current_void_area += vol_added
                    per_grain_placed[grain_idx] += vol_added
                    if vol_added > 0:
                        progress = True
                    if per_grain_placed[grain_idx] >= per_grain_caps[grain_idx]:
                        exhausted_grains.add(grain_idx)
                    elif vol_added == 0:
                        exhausted_grains.add(grain_idx)
                if not progress:
                    print("WARNING: could not place more voids without overlap.")
                    break

        print(f"  Total voids placed: {len(voids)}")
        print(f"  Void fraction achieved: {current_void_area/total_domain_area:.4f}")
        print(f"  Target:                 {void_fraction:.4f}")

        if vol_percent_porous > 0 and len(porous_circles) > 0:
            fracs = per_grain_placed / grain_areas
            print(f"\n  Per-grain void fraction - min: {fracs.min():.3f}  "
                  f"max: {fracs.max():.3f}  mean: {fracs.mean():.3f}  "
                  f"std: {fracs.std():.3f}")
            over = np.sum(fracs > MAX_VOID_FRACTION_PER_GRAIN + 1e-9)
            if over:
                print(f"  WARNING: {over} grain(s) exceed the {MAX_VOID_FRACTION_PER_GRAIN*100:.0f}% cap!")
            else:
                print(f"  All grains within {MAX_VOID_FRACTION_PER_GRAIN*100:.0f}% cap. OK")

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

        print(f"\n{'='*60}")
        print("TARGET PARAMETERS (3D)")
        print(f"  Target void fraction: {void_fraction:.4f}")
        print(f"  Target void vol:      {target_void_vol:.2e}")
        print(f"{'='*60}\n")

        def img_r(mu):
            return mu / physical_size * img_size

        cell_size = img_r(mean_rad_solid) * 4
        n_cells   = max(1, int(math.ceil(img_size / cell_size)))
        grid      = [[[[] for _ in range(n_cells)] for __ in range(n_cells)] for ___ in range(n_cells)]

        def cell_coords(x, y, z):
            i = max(0, min(n_cells-1, int(x // cell_size)))
            j = max(0, min(n_cells-1, int(y // cell_size)))
            k = max(0, min(n_cells-1, int(z // cell_size)))
            return i, j, k

        def nearby_3d(x, y, z, spheres, r_query=0):
            max_r = max((s[3] for s in spheres), default=r_query)
            window = max(1, math.ceil((r_query + max_r) / cell_size))
            cx, cy, cz = cell_coords(x, y, z)
            result = []
            for i in range(max(0, cx - window), min(n_cells, cx + window + 1)):
                for j in range(max(0, cy - window), min(n_cells, cy + window + 1)):
                    for k in range(max(0, cz - window), min(n_cells, cz + window + 1)):
                        for idx in grid[i][j][k]:
                            result.append(spheres[idx])
            return result

        def overlaps_fast_3d(x, y, z, r, neighbors, factor=0.999):
            if not neighbors: return False
            nb = np.array(neighbors, dtype=float)
            dx = x-nb[:,0]; dy = y-nb[:,1]; dz = z-nb[:,2]; cr = nb[:,3]
            return np.any(dx*dx + dy*dy + dz*dz < ((r+cr)*factor)**2)

        def clipped_sphere_volume(x, y, z, r, lo=0, hi=1):
            def cap(h):
                if h <= 0: return 0.0
                if h >= 2*r: return 4/3*math.pi*r**3
                return math.pi*h**2*(3*r-h)/3
            V = 4/3*math.pi*r**3
            V -= cap(r-(x-lo)); V -= cap(r-(hi-x))
            V -= cap(r-(y-lo)); V -= cap(r-(hi-y))
            V -= cap(r-(z-lo)); V -= cap(r-(hi-z))
            return V

        def fully_inside(x, y, z, r):
            return (x>r and x<img_size-r and y>r and y<img_size-r and z>r and z<img_size-r)

        spheres = []; porous_spheres = []; voids = []
        solid_vol = hollow_vol = porous_vol = 0.0
        sigma_ln = math.sqrt(math.log(1 + rad_dev**2))

        # ---- Solid grains ----
        print("Placing solid grains...")
        for _ in range(max_attempts):
            if solid_vol >= vol_percent_solid: break
            mu_ln = math.log(img_r(mean_rad_solid)) - 1.5*sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            if solid_vol + 4/3*math.pi*r**3/total_domain_vol > vol_percent_solid: continue
            for _ in range(100):
                x = rng.uniform(-margin+r, img_size+margin-r)
                y = rng.uniform(-margin+r, img_size+margin-r)
                z = rng.uniform(-margin+r, img_size+margin-r)
                ci, cj, ck = cell_coords(x, y, z)
                if len(grid[ci][cj][ck]) > 20: continue
                nb = nearby_3d(x, y, z, spheres, r)
                if not overlaps_fast_3d(x, y, z, r, nb, 0.98):
                    spheres.append((x,y,z,r)); grid[ci][cj][ck].append(len(spheres)-1)
                    solid_vol += 4/3*math.pi*r**3 if fully_inside(x,y,z,r) else clipped_sphere_volume(x,y,z,r)
                    break
        print(f"  Solid volume fraction: {solid_vol:.4f}")

        # ---- Hollow grains ----
        # I integrated your standalone hollow void placement directly into the 3D branch
# of gen_struct_combined_2or3D WITHOUT changing anything else.
# Only the hollow void logic section is replaced to match your standalone script.

# --- ONLY SHOWING THE MODIFIED 3D HOLLOW SECTION ---
# Replace the "# ---- Hollow grains ----" block in your 3D branch with this:

        # ---- Hollow grains ----
        print("\nPlacing hollow grains (3D, pre-sorted radii)...")
        sigma_ln_h = math.sqrt(math.log(1 + rad_dev**2))
        mu_ln_h    = math.log(img_r(mean_rad_hollow)) - 1.5*sigma_ln_h**2
        hollow_radii_list = generate_radii(vol_percent_hollow * 2, mu_ln_h, sigma_ln_h, dim=3)
        print(f"  Pre-generated {len(hollow_radii_list)} candidate hollow radii")

        hollow_candidates = []

        for r in hollow_radii_list:
            if hollow_vol >= vol_percent_hollow: break
            if hollow_vol + 4/3*math.pi*r**3/total_domain_vol > vol_percent_hollow*1.05: continue
            max_pos = 150 if hollow_vol < 0.4*vol_percent_hollow else 500
            for _ in range(max_pos):
                x = rng.uniform(-margin+r, img_size+margin-r)
                y = rng.uniform(-margin+r, img_size+margin-r)
                z = rng.uniform(-margin+r, img_size+margin-r)
                ci, cj, ck = cell_coords(x, y, z)
                if len(grid[ci][cj][ck]) > 20: continue
                nb = nearby_3d(x, y, z, spheres, r)
                if not overlaps_fast_3d(x, y, z, r, nb, 0.98):
                    spheres.append((x,y,z,r))
                    hollow_candidates.append((x,y,z,r))
                    grid[ci][cj][ck].append(len(spheres)-1)
                    hollow_vol += 4/3*math.pi*r**3 if fully_inside(x,y,z,r) else clipped_sphere_volume(x,y,z,r)
                    break

        print(f"  Hollow volume fraction: {hollow_vol:.4f}")
        if len(hollow_candidates) > 0:
            total_candidate_vol = sum(
                (4/3*math.pi*r**3 if fully_inside(x,y,z,r)
                 else clipped_sphere_volume(x,y,z,r))
                for (x,y,z,r) in hollow_candidates
            )

            actual_hollow_frac = total_candidate_vol / total_domain_vol
            rv_r_ratio = (void_fraction / actual_hollow_frac) ** (1/3)

            print(f"  Actual hollow frac: {actual_hollow_frac:.4f}")
            print(f"  rv / r ratio:      {rv_r_ratio:.4f}")

            for (x,y,z,r) in hollow_candidates:
                rv = r * rv_r_ratio
                voids.append((x,y,z,rv))
                current_void_vol += 4/3 * math.pi * rv**3

        print(f"  Void fraction so far:   {current_void_vol/total_domain_vol:.4f}")

        # ---- Porous grains ----
        print("\nPlacing porous grains (3D, pre-sorted radii)...")
        sigma_ln_p = math.sqrt(math.log(1 + rad_dev**2))
        mu_ln_p    = math.log(img_r(mean_rad_porous)) - 1.5*sigma_ln_p**2
        porous_radii_list = generate_radii(vol_percent_porous*2, mu_ln_p, sigma_ln_p, dim=3)
        print(f"  Pre-generated {len(porous_radii_list)} candidate porous radii")

        for r in porous_radii_list:
            if porous_vol >= vol_percent_porous: break
            if porous_vol + 4/3*math.pi*r**3/total_domain_vol > vol_percent_porous*1.05: continue
            max_pos = 150 if porous_vol < 0.4*vol_percent_porous else 500
            for _ in range(max_pos):
                x = rng.uniform(-margin+r, img_size+margin-r)
                y = rng.uniform(-margin+r, img_size+margin-r)
                z = rng.uniform(-margin+r, img_size+margin-r)
                ci, cj, ck = cell_coords(x, y, z)
                if len(grid[ci][cj][ck]) > 20: continue
                nb = nearby_3d(x, y, z, spheres, r)
                if not overlaps_fast_3d(x, y, z, r, nb, 0.98):
                    spheres.append((x,y,z,r)); porous_spheres.append((x,y,z,r))
                    grid[ci][cj][ck].append(len(spheres)-1)
                    porous_vol += 4/3*math.pi*r**3 if fully_inside(x,y,z,r) else clipped_sphere_volume(x,y,z,r)
                    if len(porous_spheres) % 1000 == 0:
                            print(f"  Porous grains placed: {len(porous_spheres)}  volume fraction: {porous_vol:.4f}")
                    break
        print(f"  Porous volume fraction: {porous_vol:.4f}")

        # ---- MWD check ----
        save_xyzr(spheres, AP_xyzr, img_size, physical_size)
        if mwd_tolerance is not None:
            radii = np.array([r for (x,y,z,r) in spheres])
            mwd_actual = mean_weight_diameter(AP_xyzr, radii=radii) * (physical_size / img_size)
            print(f"\nMWD check: {mwd_actual:.4e} m (target {mwd_target:.4e} m)")
            if abs(mwd_actual - mwd_target) > mwd_tolerance:
                print("MWD out of tolerance - skipping void placement.")
                return None

        # ---- Void placement in porous grains ----
        print("\nPlacing voids within porous grains (parallel, ProcessPoolExecutor)...")
        if vol_percent_porous > 0 and len(porous_spheres) > 0:
            n_porous = len(porous_spheres)
            print(f"  Porous grains: {n_porous}")

            grain_vols       = np.array([4/3*math.pi*g[3]**3 for g in porous_spheres])
            total_grain_vol  = grain_vols.sum()
            void_budget      = target_void_vol - current_void_vol

            per_grain_caps    = MAX_VOID_FRACTION_PER_GRAIN * grain_vols
            per_grain_placed  = np.zeros(n_porous)

            per_grain_budgets = np.minimum(
                void_budget * (grain_vols / total_grain_vol),
                per_grain_caps
            )

            exhausted_grains = set()

            futures_map = {}
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for grain_idx, grain in enumerate(porous_spheres):
                    if current_void_vol >= target_void_vol:
                        break
                    seed = int(rng.integers(0, 2**31))
                    fut = executor.submit(
                        _place_voids_in_grain_fast,
                        grain, per_grain_budgets[grain_idx],
                        seed, img_size
                    )
                    futures_map[fut] = grain_idx

                for fut in concurrent.futures.as_completed(futures_map):
                    grain_idx = futures_map[fut]
                    new_voids, vol_added = fut.result()
                    voids.extend(new_voids)
                    current_void_vol += vol_added
                    per_grain_placed[grain_idx] += vol_added
                    if per_grain_placed[grain_idx] >= per_grain_caps[grain_idx]:
                        exhausted_grains.add(grain_idx)
                    elif vol_added == 0:
                        exhausted_grains.add(grain_idx)
                    if len(voids) % 1000 < len(new_voids):
                        print(f"  Voids placed: {len(voids)}  fraction: {current_void_vol/total_domain_vol:.4f} / {void_fraction:.4f}")

            print(f"  Void fraction after parallel pass: {current_void_vol/total_domain_vol:.4f}")

            for _ in range(20):
                if current_void_vol >= target_void_vol: break
                active = [i for i in range(n_porous) if i not in exhausted_grains]
                if not active: break
                remaining_total = target_void_vol - current_void_vol
                active_vols     = grain_vols[active]
                redistrib       = remaining_total * (active_vols / active_vols.sum())

                progress = False
                for k, grain_idx in enumerate(active):
                    if current_void_vol >= target_void_vol: break
                    remaining_cap = per_grain_caps[grain_idx] - per_grain_placed[grain_idx]
                    budget = min(redistrib[k], remaining_cap)
                    if budget <= 0:
                        exhausted_grains.add(grain_idx)
                        continue
                    new_voids, vol_added = _place_voids_in_grain_fast(
                        porous_spheres[grain_idx], budget,
                        int(rng.integers(0, 2**31)), img_size=img_size)
                    voids.extend(new_voids)
                    current_void_vol += vol_added
                    per_grain_placed[grain_idx] += vol_added
                    if vol_added > 0:
                        progress = True
                    if per_grain_placed[grain_idx] >= per_grain_caps[grain_idx]:
                        exhausted_grains.add(grain_idx)
                    elif vol_added == 0:
                        exhausted_grains.add(grain_idx)
                if not progress:
                    print("WARNING: could not place more voids without overlap.")
                    break

        print(f"  Total voids placed: {len(voids)}")
        print(f"  Void fraction achieved: {current_void_vol/total_domain_vol:.4f}")
        print(f"  Target:                 {void_fraction:.4f}")

        if vol_percent_porous > 0 and len(porous_spheres) > 0:
            fracs = per_grain_placed / grain_vols
            print(f"\n  Per-grain void fraction - min: {fracs.min():.3f}  "
                  f"max: {fracs.max():.3f}  mean: {fracs.mean():.3f}  "
                  f"std: {fracs.std():.3f}")
            over = np.sum(fracs > MAX_VOID_FRACTION_PER_GRAIN + 1e-9)
            if over:
                print(f"  WARNING: {over} grain(s) exceed the {MAX_VOID_FRACTION_PER_GRAIN*100:.0f}% cap!")
            else:
                print(f"  All grains within {MAX_VOID_FRACTION_PER_GRAIN*100:.0f}% cap. OK")

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
            physical_size=physical_size, rad_dev=rad_dev, max_attempts=max_attempts,
            vol_percent_solid=vol_percent_solid, vol_percent_hollow=vol_percent_hollow,
            vol_percent_porous=vol_percent_porous, void_fraction=void_fraction,
            mwd_target=mwd_target, dim=dim,
            mean_rad_solid=mean_rad_solid, mean_rad_hollow=mean_rad_hollow,
            mean_rad_porous=mean_rad_porous, mwd_tolerance=mwd_tolerance,
        )
    except Exception as e:
        print(f"  [attempt {attempt_idx}] Generation failed: {e}")
        for path in [ap_xyzr, void_xyzr]:
            if os.path.exists(path): os.remove(path)
        return None

    if void_frac is None:
        if os.path.exists(ap_xyzr): os.remove(ap_xyzr)
        return None

    try:
        mwd = mean_weight_diameter(ap_xyzr)
    except Exception as e:
        print(f"  [attempt {attempt_idx}] MWD calculation failed: {e}")
        for path in [ap_xyzr, void_xyzr]:
            if os.path.exists(path): os.remove(path)
        return None

    mwd_error = abs(mwd - mwd_target)
    accepted  = mwd_error <= mwd_tolerance
    print(f"  [attempt {attempt_idx}] MWD {mwd:.4e} m  error {mwd_error:.4e}  "
          f"{'ACCEPTED' if accepted else 'rejected'}")

    if accepted:
        return {"index": accepted_idx, "name": name, "mwd": mwd,
                "void_frac": void_frac, "attempt": attempt_idx,
                "ap_xyzr": ap_xyzr, "void_xyzr": void_xyzr}
    else:
        for path in [ap_xyzr, void_xyzr]:
            if os.path.exists(path): os.remove(path)
        return None



def clipped_sphere_volume(x, y, z, r, lo=0.0, hi=1.0):
    def cap(h):
        if h <= 0:   return 0.0
        if h >= 2*r: return 4/3 * math.pi * r**3
        return math.pi * h**2 * (3*r - h) / 3
    V  = 4/3 * math.pi * r**3
    V -= cap(r - (x - lo));  V -= cap(r - (hi - x))
    V -= cap(r - (y - lo));  V -= cap(r - (hi - y))
    V -= cap(r - (z - lo));  V -= cap(r - (hi - z))
    return max(V, 0.0)

def compute_ap_volume_fraction_clipped(xyzr_path, physical_size, img_size=1.0):
    data = np.loadtxt(xyzr_path)

    total_domain_vol = img_size ** 3
    scale = img_size / physical_size

    total_vol = 0.0

    for row in data:
        if len(row) < 4:
            continue
        x, y, z, r = row[:4]

        # convert to normalized space
        x *= scale
        y *= scale
        z *= scale
        r *= scale

        if (x > r and x < img_size - r and
            y > r and y < img_size - r and
            z > r and z < img_size - r):
            total_vol += 4/3 * np.pi * r**3
        else:
            total_vol += clipped_sphere_volume(x, y, z, r, 0, img_size)

    return total_vol / total_domain_vol


def _run_one_attempt(args):
    (attempt_idx, accepted_idx, name, subfolder,
     physical_size, rad_dev, max_attempts,
     vol_percent_solid, vol_percent_hollow, vol_percent_porous,
     void_fraction, mwd_target, mwd_tolerance, dim,
     mean_rad_solid, mean_rad_hollow, mean_rad_porous,
     ap_vol_tolerance) = args

    ap_xyzr   = os.path.join(subfolder, f"{name}_AP.xyzr")
    void_xyzr = os.path.join(subfolder, f"{name}_void.xyzr")

    try:
        void_frac = gen_struct_combined_2or3D(
            ap_xyzr, void_xyzr,
            physical_size=physical_size, rad_dev=rad_dev, max_attempts=max_attempts,
            vol_percent_solid=vol_percent_solid, vol_percent_hollow=vol_percent_hollow,
            vol_percent_porous=vol_percent_porous, void_fraction=void_fraction,
            mwd_target=mwd_target, dim=dim,
            mean_rad_solid=mean_rad_solid, mean_rad_hollow=mean_rad_hollow,
            mean_rad_porous=mean_rad_porous, mwd_tolerance=mwd_tolerance,
        )
    except Exception as e:
        print(f"  [attempt {attempt_idx}] Generation failed: {e}")
        for path in [ap_xyzr, void_xyzr]:
            if os.path.exists(path): os.remove(path)
        return None

    if void_frac is None:
        if os.path.exists(ap_xyzr): os.remove(ap_xyzr)
        return None

    try:
        mwd = mean_weight_diameter(ap_xyzr)
    except Exception as e:
        print(f"  [attempt {attempt_idx}] MWD calculation failed: {e}")
        for path in [ap_xyzr, void_xyzr]:
            if os.path.exists(path): os.remove(path)
        return None

    # --- NEW CONSISTENT AP VOLUME CHECK ---
    ap_vol = compute_ap_volume_fraction_clipped(ap_xyzr, physical_size)
    target_ap_vol = (vol_percent_solid + vol_percent_hollow + vol_percent_porous)

    mwd_error = abs(mwd - mwd_target)
    ap_error  = abs(ap_vol - target_ap_vol)

    accepted = (mwd_error <= mwd_tolerance) and (ap_error <= ap_vol_tolerance)

    print(f"  [attempt {attempt_idx}] "
          f"MWD {mwd:.4e} (err {mwd_error:.2e}) | "
          f"AP {ap_vol:.4f} (err {ap_error:.2e}) "
          f"{'ACCEPTED' if accepted else 'rejected'}")

    if accepted:
        return {"index": accepted_idx, "name": name, "mwd": mwd,
                "void_frac": void_frac, "ap_vol": ap_vol,
                "attempt": attempt_idx,
                "ap_xyzr": ap_xyzr, "void_xyzr": void_xyzr}
    else:
        for path in [ap_xyzr, void_xyzr]:
            if os.path.exists(path): os.remove(path)
        return None

def generate_structures_with_target_mwd(
    subfolder,
    target_mwd,
    mwd_tolerance=0.5e-6,
    ap_vol_tolerance=0.1,
    n_target=1,
    max_total_attempts=200,
    base_name="A",
    dim: Literal[2, 3] = 2,
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
    n_workers=None,
):

    os.makedirs(subfolder, exist_ok=True)

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    common = dict(
        physical_size=physical_size, rad_dev=rad_dev, max_attempts=max_attempts,
        vol_percent_solid=vol_percent_solid, vol_percent_hollow=vol_percent_hollow,
        vol_percent_porous=vol_percent_porous, void_fraction=void_fraction,
        mwd_target=target_mwd, mwd_tolerance=mwd_tolerance,
        ap_vol_tolerance=ap_vol_tolerance,
        dim=dim,
        mean_rad_solid=mean_rad_solid, mean_rad_hollow=mean_rad_hollow,
        mean_rad_porous=mean_rad_porous,
    )

    accepted = []; accepted_idx = 0; attempt_idx = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        pending = {}

        def _submit_next():
            nonlocal attempt_idx, accepted_idx
            name = f"{base_name}_{accepted_idx + len(pending):02d}_tmp{attempt_idx}"
            args = (
                attempt_idx, accepted_idx, name, subfolder,
                common["physical_size"], common["rad_dev"], common["max_attempts"],
                common["vol_percent_solid"], common["vol_percent_hollow"],
                common["vol_percent_porous"], common["void_fraction"],
                common["mwd_target"], common["mwd_tolerance"], common["dim"],
                common["mean_rad_solid"], common["mean_rad_hollow"], common["mean_rad_porous"],
                common["ap_vol_tolerance"],
            )
            fut = executor.submit(_run_one_attempt, args)
            pending[fut] = attempt_idx
            attempt_idx += 1

        for _ in range(min(n_workers, max_total_attempts)):
            _submit_next()

        while pending and len(accepted) < n_target and attempt_idx <= max_total_attempts:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                del pending[fut]
                result = fut.result()
                if result is not None:
                    final_name = f"{base_name}_{accepted_idx:02d}"
                    final_ap   = os.path.join(subfolder, f"{final_name}_AP.xyzr")
                    final_void = os.path.join(subfolder, f"{final_name}_void.xyzr")
                    os.rename(result["ap_xyzr"], final_ap)
                    os.rename(result["void_xyzr"], final_void)
                    accepted.append(result)
                    accepted_idx += 1
                if len(accepted) < n_target and attempt_idx < max_total_attempts and len(pending) < n_workers:
                    _submit_next()

        for fut in pending:
            fut.cancel()

    return accepted



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_from_xyzr(ap_xyzr_path, void_xyzr_path, save_path,
                   physical_size=200e-6, img_size=1, dim=2, dpi=1024,
                   ap_alpha=1.0, sphere_resolution=20, max_spheres=None,
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
                    pts.append((x/physical_size*img_size, y/physical_size*img_size, 0, r/physical_size*img_size))
                elif len(vals) >= 4:
                    x, y, z, r = map(float, vals[:4])
                    pts.append((x/physical_size*img_size, y/physical_size*img_size,
                                z/physical_size*img_size, r/physical_size*img_size))
        return pts

    circles = read_xyzr(ap_xyzr_path)
    voids   = read_xyzr(void_xyzr_path)
    print(f"Plotting {len(circles)} particles and {len(voids)} voids")

    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.set_position([0, 0, 1, 1]); ax.set_axis_off()
        ax.set_xlim(0, img_size); ax.set_ylim(0, img_size); ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((0,0), img_size, img_size, facecolor='#0000FF', zorder=0))
        for (x,y,_,r) in circles:
            ax.add_patch(Circle((x,y), r, facecolor='#FF0000', edgecolor='none', alpha=ap_alpha, zorder=5))
        for (x,y,_,r) in voids:
            ax.add_patch(Circle((x,y), r, facecolor='#0000FF', edgecolor='none', zorder=6))
        fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0.0)
        plt.close(fig)

    elif dim == 3:
        if max_spheres is not None: circles = circles[:max_spheres]
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2*np.pi, sphere_resolution)
        v = np.linspace(0, np.pi,   sphere_resolution)
        u, v = np.meshgrid(u, v)
        for (x0,y0,z0,r) in circles:
            ax.plot_surface(x0+r*np.cos(u)*np.sin(v), y0+r*np.sin(u)*np.sin(v),
                            z0+r*np.cos(v), color='#FF6B6B', linewidth=0, alpha=ap_alpha)
        for (x0,y0,z0,r) in voids:
            ax.plot_surface(x0+r*np.cos(u)*np.sin(v), y0+r*np.sin(u)*np.sin(v),
                            z0+r*np.cos(v), color='#4ECDC4', linewidth=0, alpha=alpha)
        ax.set_xlim(0, img_size); ax.set_ylim(0, img_size); ax.set_zlim(0, img_size)
        ax.set_box_aspect([1,1,1]); ax.view_init(elev=elev, azim=azim)
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
        target_mwd=4.0e-6,
        base_name="H",
        dim=3,
        physical_size=50e-6,
        mean_rad_porous = 1.3e-6 / (1.2 * math.exp(math.sqrt(math.log(1 + 0.4 ** 2)) ** 2)), #target mwd
        mean_rad_hollow = 2.2e-6 / (1.2 * math.exp(math.sqrt(math.log(1 + 0.4 ** 2)) ** 2)),
        mean_rad_solid=2e-6,
        void_fraction=0.13,
        vol_percent_porous=0.0,
        vol_percent_hollow=0.60923,
        mwd_tolerance=0.20e-6,
        n_workers=4,
    )


if __name__ == "__main__":
    main()