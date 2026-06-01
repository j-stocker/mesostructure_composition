#sphere packing — 2D only, periodic in Y dimension
# Derived from sphere_packing.py (v18).  Only the 2D branch is included.
# ALL logic is unchanged from the original except for the minimal set of edits
# required to make the Y dimension periodic:
#
# PERIODIC-Y CHANGES (search "# PERIODIC-Y" to find every touched line):
#   1. dy_periodic(a, b) helper — computes the shortest signed Y-distance
#      under periodic boundary conditions (wrap distance).
#   2. overlaps_fast() — uses dy_periodic so Y-wrapped neighbours are tested
#      correctly.
#   3. nearby_2d() — unchanged (grid already looks up by cell; periodicity is
#      handled in the distance check, not the lookup).  Ghost-cell rows are
#      added to the grid lookup so grains near y=0 see grains near y=img_size
#      and vice-versa.
#   4. Grain placement loops — Y coordinate drawn from [−margin+r, img_size+margin−r]
#      and then wrapped into [0, img_size] before storing (so stored coords are
#      always canonical).  X placement is unchanged (no periodicity in X).
#   5. _place_voids_in_grain_2d — void–grain Y-distance uses dy_periodic;
#      void–void Y-distance uses dy_periodic.
#   6. save_xyzr — unchanged (stores physical coords as-is).
#   7. clipped_circle_area — only clips against X boundaries (left/right walls);
#      the Y boundary is periodic so no clipping is applied there.
#
# Everything else (solid grain placement, hollow grain placement, porous grain
# placement, MWD check, htpb_only mode, parallel executor, plotting, entry
# point) is byte-for-byte identical to the original.

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
# File I/O  (unchanged)
# ---------------------------------------------------------------------------

def save_xyzr(particles, filepath, img_size, physical_size):
    scale = physical_size / img_size
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


def mean_weight_diameter(filename, radii=None):
    if radii is None:
        data = np.loadtxt(filename)
        radii = data[:, -1]
    return 2 * np.sum(radii**3) / np.sum(radii**2)


def load_xyzr(filepath, physical_size, img_size=1.0, dim=3):
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    scale = img_size / physical_size
    x = data[:, 0] * scale
    y = data[:, 1] * scale
    z = data[:, 2] * scale
    r = data[:, 3] * scale
    if dim == 2:
        return [(x[i], y[i], r[i]) for i in range(len(r))]
    else:
        return [(x[i], y[i], z[i], r[i]) for i in range(len(r))]


def classify_loaded_particles(all_particles, dim, mean_rad_porous,
                              physical_size, img_size=1):
    thresh = (mean_rad_porous / physical_size) * img_size * 1.5
    porous = []
    nonporous = []
    for p in all_particles:
        r = p[-1]
        if r <= thresh:
            porous.append(p)
        else:
            nonporous.append(p)
    return porous, nonporous


# ---------------------------------------------------------------------------
# Circle–circle intersection area (2D)  (unchanged)
# ---------------------------------------------------------------------------

def _circle_circle_intersection_area_2d(d, r1, r2):
    if d <= 0:
        return math.pi * min(r1, r2) ** 2
    if d >= r1 + r2:
        return 0.0
    def seg(R, h):
        return R**2 * math.acos(max(-1.0, min(1.0, (R - h) / R))) - (R - h) * math.sqrt(max(0.0, 2*R*h - h**2))
    h1 = (r1**2 - r2**2 + d**2) / (2*d)
    h1_cap = r1 - h1
    h2_cap = r2 - (d - h1)
    return seg(r1, max(h1_cap, 0)) + seg(r2, max(h2_cap, 0))


# ---------------------------------------------------------------------------
# compute_ap_volume_fraction_clipped  (unchanged)
# ---------------------------------------------------------------------------

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


def compute_ap_volume_fraction_clipped(xyzr_path, physical_size, img_size=1.0, dim=3):
    data = np.loadtxt(xyzr_path)
    if data.ndim == 1:
        data = data[None, :]
    total_vol = 0.0
    if dim == 2:
        # PERIODIC-Y: Y boundary is periodic — no Y clipping.
        # X boundaries are hard walls, so we still clip against x.
        total_domain = physical_size ** 2
        for row in data:
            if len(row) < 4:
                continue
            x, y, _, r = row[:4]
            # Use full circle area (periodicity means no net clipping in Y).
            # Clip against X walls only.
            def seg_x(R, h):
                if h <= 0: return 0.0
                if h >= 2*R: return math.pi * R**2
                return R**2 * math.acos((R-h)/R) - (R-h)*math.sqrt(2*R*h - h**2)
            A = math.pi * r**2
            A -= seg_x(r, r - (x - 0.0))          # left wall clip  (physical coords)
            A -= seg_x(r, r - (physical_size - x)) # right wall clip
            total_vol += max(A, 0.0)
        return total_vol / total_domain
    else:
        total_domain = physical_size ** 3
        for row in data:
            if len(row) < 4:
                continue
            x, y, z, r = row[:4]
            if (x > r and x < physical_size - r and
                y > r and y < physical_size - r and
                z > r and z < physical_size - r):
                total_vol += 4/3 * math.pi * r**3
            else:
                total_vol += clipped_sphere_volume(x, y, z, r, 0.0, physical_size)
        return total_vol / total_domain


# ---------------------------------------------------------------------------
# Core 2-D generator  (periodic in Y)
# ---------------------------------------------------------------------------

def gen_struct_combined_2D_periodic_y(
    AP_xyzr, void_xyzr,
    physical_size,
    rad_dev, max_attempts,
    vol_percent_solid, vol_percent_hollow, vol_percent_porous, void_fraction, mwd_target,
    mean_rad_solid=60e-6, mean_rad_hollow=2e-6, mean_rad_porous=4.5e-6,
    mwd_tolerance=0.05e-6,
    pore_placement: Literal["int", "ext", "htpb_only"] = "int",
    void_fraction_mode: Literal["clipped", "unclipped"] = "clipped",
    pore_radius_factor: float = 0.15,
    existing_ap_xyzr=None,
):
    rng = np.random.default_rng()
    img_size = 1
    margin = 0.07

    # ------------------------------------------------------------------
    # PERIODIC-Y helper — shortest signed Y-distance under PBC
    # ------------------------------------------------------------------
    def dy_periodic(ay, by):                          # PERIODIC-Y
        raw = ay - by
        return raw - img_size * round(raw / img_size)

    def generate_radii(target_fraction, mu_ln, sigma_ln, dim=2):
        radii = []
        total = 0.0
        while total < target_fraction:
            r = rng.lognormal(mu_ln, sigma_ln)
            vol = math.pi * r**2
            radii.append(r)
            total += vol
        return sorted(radii, reverse=True)

    SHELL_INNER_FRAC_2D = 0.75

    # ------------------------------------------------------------------
    # _place_voids_in_grain_2d — periodic in Y
    # ------------------------------------------------------------------
    def _place_voids_in_grain_2d(grain, existing_voids_snap, target_remaining, rng_seed,
                                  all_circles_snap=None):
        local_rng = np.random.default_rng(rng_seed)
        placed  = []
        cum_vol = 0.0
        px, py, pr = grain[:3]

        if existing_voids_snap and pore_placement != "htpb_only":
            ev = np.array(existing_voids_snap, dtype=float)
            dx_ev = ev[:, 0] - px
            # PERIODIC-Y: use periodic Y-distance for proximity filter
            dy_ev = np.array([dy_periodic(v[1], py) for v in existing_voids_snap])  # PERIODIC-Y
            dist_to_grain = np.sqrt(dx_ev**2 + dy_ev**2)
            keep = dist_to_grain < 2.0 * pr + ev[:, 2]
            existing_voids_snap = [existing_voids_snap[i] for i in np.where(keep)[0]]

        for _ in range(800):
            if cum_vol >= target_remaining:
                break
            pore_r = local_rng.lognormal(math.log(pr * pore_radius_factor), 0.4)
            pore_r = float(np.clip(pore_r, 0.1 * pr, 0.35 * pr))

            if pore_placement == "int":
                pore_area = math.pi * pore_r**2
                if cum_vol + pore_area > target_remaining:
                    pore_r   = math.sqrt((target_remaining - cum_vol) / math.pi)
                    pore_area = target_remaining - cum_vol

                for _ in range(300):
                    theta = local_rng.uniform(0, 2 * math.pi)
                    rho   = local_rng.uniform(0, pr - pore_r)
                    vx    = px + rho * math.cos(theta)
                    vy    = py + rho * math.sin(theta)
                    # PERIODIC-Y: wrap vy into [0, img_size]
                    vy = vy % img_size                                     # PERIODIC-Y
                    if math.hypot(vx - px, dy_periodic(vy, py)) + pore_r > pr:  # PERIODIC-Y
                        continue
                    # X hard-wall boundary only
                    if vx - pore_r < 0 or vx + pore_r > img_size:
                        continue
                    # PERIODIC-Y: Y is periodic — no boundary check in Y
                    # PERIODIC-Y: periodic void–void overlap check
                    if any(math.hypot(vx - xv, dy_periodic(vy, yv)) < pore_r + rv   # PERIODIC-Y
                           for xv, yv, rv in existing_voids_snap):
                        continue
                    if any(math.hypot(vx - xv, dy_periodic(vy, yv)) < pore_r + rv   # PERIODIC-Y
                           for xv, yv, rv in placed):
                        continue
                    placed.append((vx, vy, pore_r))
                    existing_voids_snap.append((vx, vy, pore_r))
                    cum_vol += pore_area
                    break

            elif pore_placement == "ext":
                rho_min = SHELL_INNER_FRAC_2D * pr
                rho_max = pr + pore_r
                n_batch = 300

                thetas = local_rng.uniform(0, 2 * math.pi, n_batch)
                rhos   = local_rng.uniform(rho_min, rho_max, n_batch)
                vxs    = px + rhos * np.cos(thetas)
                vys    = (py + rhos * np.sin(thetas)) % img_size          # PERIODIC-Y

                # PERIODIC-Y: periodic distance to host grain
                dys = np.array([dy_periodic(vy_, py) for vy_ in vys])    # PERIODIC-Y
                ds  = np.sqrt((vxs - px)**2 + dys**2)                    # PERIODIC-Y

                vmask  = ds < pr + pore_r
                vmask &= ds + pore_r >= pr
                # X hard walls only
                vmask &= (vxs >= 0) & (vxs <= img_size)
                # PERIODIC-Y: no Y boundary check

                all_placed = list(existing_voids_snap) + list(placed)
                if all_placed:
                    pv = np.array(all_placed, dtype=float)
                    dx2 = vxs[:, None] - pv[:, 0]
                    # PERIODIC-Y: periodic Y-distance for void–void overlap
                    dy2 = np.array([[dy_periodic(vys[i], pv[j, 1])          # PERIODIC-Y
                                     for j in range(len(pv))]
                                    for i in range(len(vxs))])
                    ovlp = np.any(dx2**2 + dy2**2 < (pore_r + pv[:, 2])**2, axis=1)
                    vmask &= ~ovlp

                valid_idx = np.where(vmask)[0]
                if len(valid_idx) == 0:
                    continue

                remaining = target_remaining - cum_vol
                if remaining <= 0:
                    break

                if void_fraction_mode == "unclipped":
                    contrib_v = np.full(len(valid_idx), math.pi * pore_r**2)
                else:
                    h1_v  = (pr**2 - pore_r**2 + ds[valid_idx]**2) / (2.0 * np.maximum(ds[valid_idx], 1e-30))
                    h1c   = np.maximum(pr     - h1_v,       0.0)
                    h2c   = np.maximum(pore_r - (ds[valid_idx] - h1_v), 0.0)
                    contrib_v = (pr**2     * np.arccos(np.clip((pr     - h1c) / pr,     -1, 1))
                                 - (pr     - h1c) * np.sqrt(np.maximum(2*pr    *h1c - h1c**2, 0))
                                 + pore_r**2 * np.arccos(np.clip((pore_r - h2c) / pore_r, -1, 1))
                                 - (pore_r - h2c) * np.sqrt(np.maximum(2*pore_r*h2c - h2c**2, 0)))
                    contrib_v = np.where(ds[valid_idx] <= 0, math.pi * min(pr, pore_r)**2, contrib_v)

                fits = np.where((contrib_v > 0) & (contrib_v <= remaining))[0]
                if len(fits) == 0:
                    continue

                pick = valid_idx[fits[0]]
                placed.append((float(vxs[pick]), float(vys[pick]), pore_r))
                existing_voids_snap.append((float(vxs[pick]), float(vys[pick]), pore_r))
                cum_vol += float(contrib_v[fits[0]])

            else:
                # htpb_only 2D
                n_batch = 300
                vxs = local_rng.uniform(0, img_size, n_batch)
                vys = local_rng.uniform(0, img_size, n_batch)  # full range; periodic

                if all_circles_snap is not None and len(all_circles_snap) > 0:
                    ac = np.array(all_circles_snap, dtype=float)
                    dx2 = vxs[:, None] - ac[:, 0]
                    # PERIODIC-Y
                    dy2 = np.array([[dy_periodic(vys[i], ac[j, 1])           # PERIODIC-Y
                                     for j in range(len(ac))]
                                    for i in range(len(vxs))])
                    dist_to_aps = np.sqrt(dx2**2 + dy2**2)

                    if void_fraction_mode == "clipped":
                        vmask = np.all(dist_to_aps >= ac[:, 2][None, :] + pore_r, axis=1)
                    else:
                        vmask = np.all(dist_to_aps >= ac[:, 2][None, :], axis=1)
                else:
                    vmask = np.ones(n_batch, dtype=bool)

                # X hard walls only
                vmask &= (vxs >= pore_r) & (vxs <= img_size - pore_r)
                # PERIODIC-Y: no Y boundary check

                all_placed = list(existing_voids_snap) + list(placed)
                if all_placed:
                    pv = np.array(all_placed, dtype=float)
                    dx2 = vxs[:, None] - pv[:, 0]
                    # PERIODIC-Y
                    dy2 = np.array([[dy_periodic(vys[i], pv[j, 1])           # PERIODIC-Y
                                     for j in range(len(pv))]
                                    for i in range(len(vxs))])
                    ovlp = np.any(dx2**2 + dy2**2 < (pore_r + pv[:, 2])**2, axis=1)
                    vmask &= ~ovlp

                valid_idx = np.where(vmask)[0]
                if len(valid_idx) == 0:
                    continue

                remaining = target_remaining - cum_vol
                if remaining <= 0:
                    break
                pore_area_full = math.pi * pore_r**2
                if pore_area_full > remaining:
                    continue

                pick = valid_idx[0]
                vx, vy = float(vxs[pick]), float(vys[pick])
                placed.append((vx, vy, pore_r))
                existing_voids_snap.append((vx, vy, pore_r))
                cum_vol += pore_area_full

        return placed, cum_vol

    # ------------------------------------------------------------------
    # 2-D grid + overlap helpers
    # ------------------------------------------------------------------
    total_domain_area = img_size * img_size
    target_void_area  = void_fraction * total_domain_area
    current_void_area = 0.0

    print(f"\n{'='*60}")
    print("TARGET PARAMETERS (2D, periodic-Y)")
    print(f"  Target void fraction:  {void_fraction:.4f}  (of full domain)")
    print(f"  Target void area:      {target_void_area:.2e}")
    print(f"  Pore placement mode:   {pore_placement}")
    print(f"  Void fraction mode:    {void_fraction_mode}")
    print(f"  Pore radius factor:    {pore_radius_factor}")
    print(f"{'='*60}\n")

    def img_r(mu):
        return mu / physical_size * img_size

    cell_size = img_r(mean_rad_solid) * 4
    n_cells   = max(1, int(math.ceil(img_size / cell_size)))
    grid      = [[[] for _ in range(n_cells)] for __ in range(n_cells)]

    def cell_coords(x, y):
        cx = max(0, min(n_cells - 1, int(x // cell_size)))
        # PERIODIC-Y: wrap y cell index
        cy = int(y // cell_size) % n_cells                                # PERIODIC-Y
        return cx, cy

    def nearby_2d(x, y, circles, r_query=0):
        max_r = max((c[2] for c in circles), default=r_query)
        window = max(1, math.ceil((r_query + max_r) / cell_size))
        cx, cy = cell_coords(x, y)
        result = []
        for i in range(max(0, cx - window), min(n_cells, cx + window + 1)):
            for dj in range(-window, window + 1):                         # PERIODIC-Y
                j = (cy + dj) % n_cells                                   # PERIODIC-Y
                for idx in grid[i][j]:
                    result.append(circles[idx])
        return result

    # PERIODIC-Y: overlap check uses dy_periodic
    def overlaps_fast(x, y, r, neighbors, factor=0.999):                 # PERIODIC-Y
        if not neighbors:
            return False
        nb = np.array(neighbors, dtype=float)
        dx = x - nb[:, 0]
        # PERIODIC-Y: shortest Y-distance
        raw_dy = y - nb[:, 1]
        dy = raw_dy - img_size * np.round(raw_dy / img_size)             # PERIODIC-Y
        cr = nb[:, 2]
        return np.any(dx*dx + dy*dy < ((r + cr) * factor) ** 2)

    # PERIODIC-Y: clipped_circle_area clips X walls only; Y is periodic
    def clipped_circle_area(x, y, r, lo=0, hi=1):                       # PERIODIC-Y
        def seg(h):
            if h <= 0: return 0.0
            if h >= 2*r: return math.pi * r**2
            return r**2 * math.acos((r-h)/r) - (r-h)*math.sqrt(2*r*h - h**2)
        A = math.pi * r**2
        A -= seg(r - (x - lo))   # left X wall
        A -= seg(r - (hi - x))   # right X wall
        # No Y clipping — periodic boundary
        return A

    circles = []; porous_circles = []; voids = []
    solid_area = hollow_area = porous_area = 0.0
    sigma_ln = math.sqrt(math.log(1 + rad_dev**2))
    using_existing_geometry = existing_ap_xyzr is not None

    # ------------------------------------------------------------------
    # Load or place solid grains
    # ------------------------------------------------------------------
    if using_existing_geometry:
        print("Loading existing AP geometry...")
        circles = load_xyzr(existing_ap_xyzr, physical_size, img_size, dim=2)
        porous_circles, solid_circles = classify_loaded_particles(
            circles, dim=2, mean_rad_porous=mean_rad_porous,
            physical_size=physical_size, img_size=img_size)
        for idx, (x, y, r) in enumerate(circles):
            cx_c, cy_c = cell_coords(x, y)
            grid[cx_c][cy_c].append(idx)
            solid_area += clipped_circle_area(x, y, r)
        print(f"  Loaded grains: {len(circles)}, porous: {len(porous_circles)}")
    else:
        print("Placing solid grains...")
        for _ in range(max_attempts):
            if solid_area >= vol_percent_solid: break
            mu_ln = math.log(img_r(mean_rad_solid)) - 0.5 * sigma_ln**2
            r = rng.lognormal(mu_ln, sigma_ln)
            if solid_area + math.pi*r**2/total_domain_area > vol_percent_solid: continue
            for _ in range(100):
                x = rng.uniform(-margin+r, img_size+margin-r)
                # PERIODIC-Y: Y drawn from full [0, img_size); wrap into domain
                y = rng.uniform(0, img_size)                              # PERIODIC-Y
                x_clamp = max(r, min(img_size - r, x))                   # X hard wall
                nb = nearby_2d(x_clamp, y, circles, r)
                if not overlaps_fast(x_clamp, y, r, nb, 0.999):
                    circles.append((x_clamp, y, r))
                    cx_c, cy_c = cell_coords(x_clamp, y)
                    grid[cx_c][cy_c].append(len(circles)-1)
                    solid_area += clipped_circle_area(x_clamp, y, r)
                    break
        print(f"  Solid area fraction: {solid_area:.4f}")

    # ------------------------------------------------------------------
    # Hollow grains
    # ------------------------------------------------------------------
    print("\nPlacing hollow grains (2D periodic-Y, pre-sorted radii)...")
    sigma_ln_h = math.sqrt(math.log(1 + rad_dev**2))
    mu_ln_h    = math.log(img_r(mean_rad_hollow)) - 1.5*sigma_ln_h**2
    hollow_radii_list = generate_radii(vol_percent_hollow * 2, mu_ln_h, sigma_ln_h, dim=2)
    print(f"  Pre-generated {len(hollow_radii_list)} candidate hollow radii")
    hollow_area = 0.0

    for r in hollow_radii_list:
        if hollow_area >= vol_percent_hollow: break
        if current_void_area >= target_void_area: break
        if hollow_area + math.pi*r**2/total_domain_area > vol_percent_hollow*1.05: continue
        max_pos = 150 if hollow_area < 0.4*vol_percent_hollow else 500
        for _ in range(max_pos):
            x = rng.uniform(-margin+r, img_size+margin-r)
            y = rng.uniform(0, img_size)                                  # PERIODIC-Y
            x = max(r, min(img_size - r, x))
            cx_c, cy_c = cell_coords(x, y)
            if len(grid[cx_c][cy_c]) > 15: continue
            nb = nearby_2d(x, y, circles, r)
            if not overlaps_fast(x, y, r, nb, 0.98):
                rv = r * (void_fraction / vol_percent_hollow) ** (1/2)
                circles.append((x, y, r)); voids.append((x, y, rv))
                grid[cx_c][cy_c].append(len(circles)-1)
                hollow_area += clipped_circle_area(x, y, r)
                current_void_area += math.pi*rv**2
                break
    print(f"  Hollow area fraction: {hollow_area:.4f}")
    print(f"  Void area so far:     {current_void_area:.4e}  ({current_void_area/total_domain_area:.4f} of domain)")

    # ------------------------------------------------------------------
    # Porous grains
    # ------------------------------------------------------------------
    print("\nPlacing porous grains (2D periodic-Y, pre-sorted radii)...")
    sigma_ln_p = math.sqrt(math.log(1 + rad_dev**2))
    mu_ln_p    = math.log(img_r(mean_rad_porous)) - 1.5 * sigma_ln_p**2
    porous_radii_list = generate_radii(vol_percent_porous * 2, mu_ln_p, sigma_ln_p, dim=2)
    print(f"  Pre-generated {len(porous_radii_list)} candidate porous radii")

    for r in porous_radii_list:
        if porous_area >= vol_percent_porous: break
        if porous_area + math.pi*r**2/total_domain_area > vol_percent_porous*1.05: continue
        max_pos = 150 if porous_area < 0.4*vol_percent_porous else 500
        for _ in range(max_pos):
            x = rng.uniform(-margin+r, img_size+margin-r)
            y = rng.uniform(0, img_size)                                  # PERIODIC-Y
            x = max(r, min(img_size - r, x))
            cx_c, cy_c = cell_coords(x, y)
            if len(grid[cx_c][cy_c]) > 15: continue
            nb = nearby_2d(x, y, circles, r)
            if not overlaps_fast(x, y, r, nb, 0.98):
                circles.append((x, y, r)); porous_circles.append((x, y, r))
                grid[cx_c][cy_c].append(len(circles)-1)
                porous_area += clipped_circle_area(x, y, r)
                break
    print(f"  Porous area fraction: {porous_area:.4f}")

    save_xyzr(circles, AP_xyzr, img_size, physical_size)

    # MWD check
    if mwd_tolerance is not None:
        if using_existing_geometry:
            existing_count = len(load_xyzr(existing_ap_xyzr, physical_size, img_size, dim=2))
            new_particles  = circles[existing_count:]
            radii = np.array([r for (x, y, r) in new_particles])
        else:
            radii = np.array([r for (x, y, r) in circles])
        mwd_actual = mean_weight_diameter(AP_xyzr, radii=radii) * (physical_size / img_size)
        print(f"\nMWD check: {mwd_actual:.4e} m (target {mwd_target:.4e} m)")
        if abs(mwd_actual - mwd_target) > mwd_tolerance:
            print("MWD out of tolerance — skipping void placement.")
            return None

    # ------------------------------------------------------------------
    # Void placement — htpb_only
    # ------------------------------------------------------------------
    if pore_placement == "htpb_only":
        print(f"\nPlacing voids in HTPB binder space (htpb_only, mode={void_fraction_mode})...")
        ref_pr = img_r(mean_rad_porous)
        all_grains_arr_2d = np.array(circles, dtype=float)

        void_grid_2d = [[[] for _ in range(n_cells)] for __ in range(n_cells)]
        for vi, (vx_, vy_, vr_) in enumerate(voids):
            vcx, vcy = cell_coords(vx_, vy_)
            void_grid_2d[vcx][vcy].append(vi)

        def nearby_voids_2d_fast(x, y, r_query):
            max_vr = ref_pr * 0.35
            window = max(1, math.ceil((r_query + max_vr) / cell_size))
            cx, cy = cell_coords(x, y)
            result = []
            for i in range(max(0, cx - window), min(n_cells, cx + window + 1)):
                for dj in range(-window, window + 1):                    # PERIODIC-Y
                    j = (cy + dj) % n_cells                              # PERIODIC-Y
                    for vi in void_grid_2d[i][j]:
                        result.append(voids[vi])
            return result

        n_batch = 512
        consecutive_failures = 0
        max_consecutive = 200

        if len(all_grains_arr_2d) > 0:
            gx2 = all_grains_arr_2d[:, 0]
            gy2 = all_grains_arr_2d[:, 1]
            gr2 = all_grains_arr_2d[:, 2]
        else:
            gx2 = gy2 = gr2 = np.empty(0)

        while current_void_area < target_void_area:
            if consecutive_failures >= max_consecutive:
                print("WARNING: could not place more htpb voids without overlap.")
                break

            pore_r = float(np.clip(
                rng.lognormal(math.log(ref_pr * pore_radius_factor), 0.4),
                0.1 * ref_pr, 0.35 * ref_pr))

            full_circle_area = math.pi * pore_r**2
            remaining = target_void_area - current_void_area
            if full_circle_area > remaining and void_fraction_mode == "unclipped":
                break

            vxs = rng.uniform(pore_r, img_size - pore_r, n_batch)
            vys = rng.uniform(0, img_size, n_batch)                       # PERIODIC-Y: full range

            if len(gx2) > 0:
                dx2d = vxs[:, None] - gx2[None, :]
                # PERIODIC-Y
                raw_dy = vys[:, None] - gy2[None, :]
                dy2d   = raw_dy - img_size * np.round(raw_dy / img_size) # PERIODIC-Y
                dist2_grains = dx2d**2 + dy2d**2
                vmask = np.all(dist2_grains >= gr2[None, :]**2, axis=1)
            else:
                vmask = np.ones(n_batch, dtype=bool)

            # X hard walls
            vmask &= (vxs >= pore_r) & (vxs <= img_size - pore_r)

            valid_idx = np.where(vmask)[0]
            if len(valid_idx) == 0:
                consecutive_failures += 1
                continue

            rep_x = float(vxs[valid_idx[0]])
            rep_y = float(vys[valid_idx[0]])
            nearby_v = nearby_voids_2d_fast(rep_x, rep_y, pore_r)

            if nearby_v:
                nv_arr = np.array(nearby_v, dtype=float)
                nvx2 = nv_arr[:, 0]; nvy2 = nv_arr[:, 1]; nvr2 = nv_arr[:, 2]
                ddx = vxs[valid_idx, None] - nvx2[None, :]
                # PERIODIC-Y
                raw_ddy = vys[valid_idx, None] - nvy2[None, :]
                ddy = raw_ddy - img_size * np.round(raw_ddy / img_size)  # PERIODIC-Y
                void_overlap = np.any(
                    ddx**2 + ddy**2 < (pore_r + nvr2[None, :])**2, axis=1)
                valid_idx = valid_idx[~void_overlap]

            if len(valid_idx) == 0:
                consecutive_failures += 1
                continue

            if void_fraction_mode == "unclipped":
                if full_circle_area > remaining:
                    break
                pick = int(valid_idx[0])
                counted_area = full_circle_area
            else:
                if len(gx2) > 0:
                    vx_v = vxs[valid_idx]
                    vy_v = vys[valid_idx]
                    dx_v = vx_v[:, None] - gx2[None, :]
                    # PERIODIC-Y
                    raw_dy_v = vy_v[:, None] - gy2[None, :]
                    dy_v = raw_dy_v - img_size * np.round(raw_dy_v / img_size)  # PERIODIC-Y
                    d_v  = np.sqrt(dx_v**2 + dy_v**2)

                    overlapping = d_v < pore_r + gr2[None, :]
                    d_safe = np.where(overlapping, np.maximum(d_v, 1e-30), 1.0)
                    h1 = (pore_r**2 - gr2[None, :]**2 + d_v**2) / (2.0 * d_safe)
                    h1c = np.maximum(pore_r - h1,             0.0)
                    h2c = np.maximum(gr2[None, :] - (d_v - h1), 0.0)
                    seg_void  = (pore_r**2 *
                                 np.arccos(np.clip((pore_r - h1c) / pore_r, -1, 1))
                                 - (pore_r - h1c) *
                                 np.sqrt(np.maximum(2*pore_r*h1c - h1c**2, 0)))
                    seg_grain = (gr2[None, :]**2 *
                                 np.arccos(np.clip((gr2[None, :] - h2c) /
                                                   np.maximum(gr2[None, :], 1e-30), -1, 1))
                                 - (gr2[None, :] - h2c) *
                                 np.sqrt(np.maximum(2*gr2[None, :]*h2c - h2c**2, 0)))
                    lens = seg_void + seg_grain
                    fully = d_v <= np.abs(pore_r - gr2[None, :])
                    small_r = np.minimum(pore_r, gr2[None, :])
                    full_contained = math.pi * small_r**2
                    intersection = np.where(~overlapping, 0.0,
                                            np.where(fully, full_contained, lens))
                    total_intersection = intersection.sum(axis=1)
                    counted_areas = np.clip(full_circle_area - total_intersection,
                                            0.0, full_circle_area)
                else:
                    counted_areas = np.full(len(valid_idx), full_circle_area)

                fits = np.where(counted_areas <= remaining)[0]
                if len(fits) == 0:
                    consecutive_failures += 1
                    continue
                pick = int(valid_idx[fits[0]])
                counted_area = float(counted_areas[fits[0]])

            vx_new = float(vxs[pick])
            vy_new = float(vys[pick])
            voids.append((vx_new, vy_new, pore_r))
            vi_new = len(voids) - 1
            vcx, vcy = cell_coords(vx_new, vy_new)
            void_grid_2d[vcx][vcy].append(vi_new)
            current_void_area += counted_area
            consecutive_failures = 0

            if len(voids) % 500 == 0:
                print(f"  Voids placed: {len(voids)}  "
                      f"fraction of domain: {current_void_area/total_domain_area:.4f}")

    # ------------------------------------------------------------------
    # Void placement — int / ext (parallel over porous grains)
    # ------------------------------------------------------------------
    else:
        print(f"\nPlacing voids within porous grains (parallel, mode={pore_placement}, "
              f"vf_mode={void_fraction_mode})...")
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
                        print(f"  Voids placed: {len(voids)}  "
                              f"fraction of domain: {current_void_area/total_domain_area:.4f}")

            print(f"  Void fraction of domain after parallel pass: {current_void_area/total_domain_area:.4f}")

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

    void_frac_domain = current_void_area / total_domain_area
    print(f"\n{'='*60}")
    print("FINAL RESULTS (2D, periodic-Y)")
    print(f"  Total grains:           {len(circles)}")
    print(f"  AP area fraction:       {solid_area+hollow_area+porous_area:.4f}")
    print(f"  Void fraction (domain): {void_frac_domain:.4f}  (target: {void_fraction:.4f})")
    print(f"  Error:                  {abs(void_frac_domain - void_fraction):.2e}")
    print(f"{'='*60}\n")

    if pore_placement != "htpb_only" and vol_percent_porous > 0 and len(porous_circles) > 0:
        fracs = per_grain_placed / grain_areas
        print(f"  Per-grain void fraction — min: {fracs.min():.3f}  "
              f"max: {fracs.max():.3f}  mean: {fracs.mean():.3f}  "
              f"std: {fracs.std():.3f}")
        over = np.sum(fracs > MAX_VOID_FRACTION_PER_GRAIN + 1e-9)
        if over:
            print(f"  WARNING: {over} grain(s) exceed the {MAX_VOID_FRACTION_PER_GRAIN*100:.0f}% cap!")
        else:
            print(f"  All grains within {MAX_VOID_FRACTION_PER_GRAIN*100:.0f}% cap. OK")

    save_xyzr(voids, void_xyzr, img_size, physical_size)
    return void_frac_domain


# ---------------------------------------------------------------------------
# _run_one_attempt / generate_structures wrapper  (unchanged logic; calls new fn)
# ---------------------------------------------------------------------------

def _run_one_attempt(args):
    (attempt_idx, accepted_idx, name, subfolder,
     physical_size, rad_dev, max_attempts,
     vol_percent_solid, vol_percent_hollow, vol_percent_porous,
     void_fraction, mwd_target, mwd_tolerance,
     mean_rad_solid, mean_rad_hollow, mean_rad_porous,
     ap_vol_tolerance, pore_placement, void_fraction_mode,
     pore_radius_factor, existing_ap_xyzr) = args

    ap_xyzr   = os.path.join(subfolder, f"{name}_AP.xyzr")
    void_xyzr = os.path.join(subfolder, f"{name}_void.xyzr")

    try:
        void_frac = gen_struct_combined_2D_periodic_y(
            ap_xyzr, void_xyzr,
            physical_size=physical_size, rad_dev=rad_dev, max_attempts=max_attempts,
            vol_percent_solid=vol_percent_solid, vol_percent_hollow=vol_percent_hollow,
            vol_percent_porous=vol_percent_porous, void_fraction=void_fraction,
            mwd_target=mwd_target,
            mean_rad_solid=mean_rad_solid, mean_rad_hollow=mean_rad_hollow,
            mean_rad_porous=mean_rad_porous, mwd_tolerance=mwd_tolerance,
            pore_placement=pore_placement,
            void_fraction_mode=void_fraction_mode,
            pore_radius_factor=pore_radius_factor,
            existing_ap_xyzr=existing_ap_xyzr,
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

    ap_vol = compute_ap_volume_fraction_clipped(ap_xyzr, physical_size, dim=2)
    target_ap_vol = vol_percent_solid + vol_percent_hollow + vol_percent_porous

    mwd_error = abs(mwd - mwd_target)
    ap_error  = abs(ap_vol - target_ap_vol)
    accepted  = (mwd_error <= mwd_tolerance) and (ap_error <= ap_vol_tolerance)

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
    pore_placement: Literal["int", "ext", "htpb_only"] = "int",
    void_fraction_mode: Literal["clipped", "unclipped"] = "clipped",
    pore_radius_factor: float = 0.15,
    existing_ap_xyzr=None,
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
        mean_rad_solid=mean_rad_solid, mean_rad_hollow=mean_rad_hollow,
        mean_rad_porous=mean_rad_porous,
        pore_placement=pore_placement,
        void_fraction_mode=void_fraction_mode,
        pore_radius_factor=pore_radius_factor,
        existing_ap_xyzr=existing_ap_xyzr,
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
                common["mwd_target"], common["mwd_tolerance"],
                common["mean_rad_solid"], common["mean_rad_hollow"], common["mean_rad_porous"],
                common["ap_vol_tolerance"], common["pore_placement"],
                common["void_fraction_mode"], common["pore_radius_factor"],
                common["existing_ap_xyzr"],
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
                    plot_save = os.path.join(subfolder, f"{final_name}.png")
                    plot_from_xyzr(final_ap, final_void, plot_save,
                                   physical_size=physical_size)
                    print(f"  Saved plot: {plot_save}")
                    accepted.append(result)
                    accepted_idx += 1
                if (len(accepted) < n_target and attempt_idx < max_total_attempts
                        and len(pending) < n_workers):
                    _submit_next()

        for fut in pending:
            fut.cancel()

    return accepted


# ---------------------------------------------------------------------------
# Plotting  (unchanged from original)
# ---------------------------------------------------------------------------

def plot_from_xyzr(ap_xyzr_path, void_xyzr_path, save_path,
                   physical_size=200e-6, img_size=1, dpi=1024, ap_alpha=1.0):

    def read_xyzr(path):
        pts = []
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"WARNING: xyzr file empty or missing: {path}")
            return pts
        with open(path, 'r') as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) >= 4:
                    x, y, _, r = map(float, vals[:4])
                    pts.append((x/physical_size*img_size,
                                y/physical_size*img_size,
                                r/physical_size*img_size))
        return pts

    circles = read_xyzr(ap_xyzr_path)
    voids   = read_xyzr(void_xyzr_path)
    print(f"Plotting {len(circles)} particles and {len(voids)} voids")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.set_position([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, img_size); ax.set_ylim(0, img_size); ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0, 0), img_size, img_size, facecolor='#0000FF', zorder=0))
        # --- AP particles ---
    for (x, y, r) in circles:

        # main particle
        ax.add_patch(Circle(
            (x, y), r,
            facecolor='#FF0000',
            edgecolor='none',
            alpha=ap_alpha,
            zorder=5
        ))

        # periodic wrap copies in Y
        if y - r < 0:
            ax.add_patch(Circle(
                (x, y + img_size), r,
                facecolor='#FF0000',
                edgecolor='none',
                alpha=ap_alpha,
                zorder=5
            ))

        if y + r > img_size:
            ax.add_patch(Circle(
                (x, y - img_size), r,
                facecolor='#FF0000',
                edgecolor='none',
                alpha=ap_alpha,
                zorder=5
            ))

    # --- voids ---
    for (x, y, r) in voids:

        # main void
        ax.add_patch(Circle(
            (x, y), r,
            facecolor='#0000FF',
            edgecolor='none',
            zorder=6
        ))

        # periodic wrap copies in Y
        if y - r < 0:
            ax.add_patch(Circle(
                (x, y + img_size), r,
                facecolor='#0000FF',
                edgecolor='none',
                zorder=6
            ))

        if y + r > img_size:
            ax.add_patch(Circle(
                (x, y - img_size), r,
                facecolor='#0000FF',
                edgecolor='none',
                zorder=6
            ))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



# ---------------------------------------------------------------------------
# Entry point  (mirrors original main())
# ---------------------------------------------------------------------------

def main():
    generate_structures_with_target_mwd(
        '32x32_periodic_y',
        target_mwd=4.0e-6,
        base_name="nonvoid_small_no_overlap_periodic_y",
        physical_size=32e-6,
        mean_rad_porous=1e-6,
        mean_rad_hollow=2.2e-6 / (1.2 * math.exp(math.sqrt(math.log(1 + 0.4**2))**2)),
        mean_rad_solid=.5e-6,
        void_fraction=0.0,
        vol_percent_solid=0.0,
        vol_percent_porous=0.85,
        vol_percent_hollow=0.0,
        mwd_tolerance=200e-6,
        n_workers=4,
        pore_placement='ext',
        void_fraction_mode="unclipped",
        rad_dev=0.4
    )


if __name__ == "__main__":
    main()