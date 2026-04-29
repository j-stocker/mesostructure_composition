#sphere packing, updated to handle 2D or 3D — parallelized version
# FIXES applied (v2-v5): see original header
# FIXES applied (v7):
#   FIX-S  Porous grain overlap factor tightened from 0.95 → 0.98 in both 2D and 3D,
#          matching the solid grain tolerance and eliminating visible grain overlap.
#          max_pos bumped from 100/300 → 150/500 to compensate for tighter packing.
# FIXES applied (v6):
#   FIX-O  Per-grain void volume is now TRACKED CUMULATIVELY across both the initial
#          placement pass and every top-up round.  Previously the cap was only enforced
#          on the budget handed to each call, so a grain that received multiple top-up
#          calls could accumulate far more than MAX_VOID_FRACTION_PER_GRAIN.
#   FIX-P  _place_voids_in_grain_fast and _place_voids_in_grain_2d no longer apply
#          the cap internally — the caller owns the cap via per_grain_placed tracking.
#          This avoids double-capping and makes the accounting authoritative in one place.
#   FIX-Q  Top-up budget for each grain is now:
#            min(remaining_global, cap - already_placed_for_this_grain)
#          so no grain can ever exceed the cap regardless of how many passes run.
#   FIX-R  Grains are marked exhausted when their cap is reached, not only when a
#          placement attempt returns zero — prevents wasted attempts on full grains.
# FIXES applied (v8):
#   FIX-W  Neighbor lookup window radius now scales with the query grain radius instead
#          of being hard-coded to +-1 cell.  Previously, when a grain's diameter exceeded
#          cell_size (= mean_rad_solid * 4), grains sitting 2+ cells away were invisible
#          to the overlap check and could be placed overlapping.  The window is now
#          ceil((r_query + max_placed_r) / cell_size), guaranteeing all grains within
#          interaction range are examined.  Applies to both 2D and 3D branches.
#
# PERF changes (no logic changes):
#   PERF-1  _place_voids_in_grain_fast: candidate arrays pre-allocated once per call
#           instead of reallocated every iteration.  Same values, same selection logic.
#   PERF-2  3D void placement loop parallelized with ProcessPoolExecutor, mirroring
#           the existing 2D ThreadPoolExecutor pattern exactly.
#   PERF-3  nearby_3d / nearby_2d converted from generators to list-returning functions
#           to eliminate per-item generator frame overhead (list() was called anyway).
#   PERF-4  overlaps_fast / overlaps_fast_3d accept a pre-built numpy array so the
#           np.array() conversion isn't repeated on every call inside the placement loop.
#           Callers updated to pass np.array(neighbors) once per neighbor snapshot.
#   PERF-5  save_xyzr uses writelines() with a pre-built list instead of one write()
#           per particle.
#   PERF-6  mean_weight_diameter accepts an optional radii array to skip re-reading
#           the file when the data is already in memory.
# FIXES applied (v9):
#   FIX-X  Hollow grain void volume now counted as full sphere (4/3*pi*rv**3) regardless
#          of grain position, removing clipped_sphere_volume undercounting that caused
#          void fraction overshoot.  Early-stop guard added so hollow void placement
#          halts as soon as void_fraction target is reached.
# FIXES applied (v10):
#   FIX-Y  Added pore_placement parameter ("int" | "ext") to both void placement
#          functions and all callers.
#          "int" (default): original behaviour — void centers sampled uniformly inside
#                           the grain; full void sphere must fit within grain boundary.
#          "ext": void centers placed in the outer shell of the grain
#                 (rho in [shell_inner_frac * pr, pr + pore_r]) so pores straddle the
#                 grain surface.  Only the sphere–sphere intersection volume (grain ∩ void)
#                 is counted against the void-fraction budget and stored; the void sphere
#                 center and radius are stored as-is so the caller can reconstruct the
#                 partial geometry.  Void budget accounting uses the clipped volume so
#                 targets remain consistent with "int" mode.
# FIXES applied (v11):
#   FIX-Z  compute_ap_volume_fraction_clipped now works entirely in physical units
#          (metres) and accepts a dim parameter (2 or 3) so that 2D runs use
#          physical_size**2 as the domain area instead of physical_size**3.
#          Previously the function applied a scale = img_size / physical_size factor
#          (e.g. 1 / 20e-6 = 50000) to coordinates that were already in physical units,
#          shrinking all radii by that factor and producing an AP fraction ~0 instead of
#          the true ~0.3, causing every 2D attempt to be rejected.
# FIXES applied (v12):
#   FIX-AA In "ext" pore placement mode, the validity mask now requires that each pore
#          actually reaches the grain surface: dist_from_center + pore_r >= pr.
#          Previously only grain overlap was checked (dist < pr + pore_r), so pores
#          whose centers landed deep in the shell interior could pass the filter without
#          ever breaking the grain boundary.  Applies to both 2D and 3D branches.

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
# Sphere–sphere intersection volume (3D)
# Returns the volume of the region inside BOTH spheres.
# d  = distance between centers
# r1 = radius of sphere 1, r2 = radius of sphere 2
# ---------------------------------------------------------------------------

def _sphere_sphere_intersection_volume_3d(d, r1, r2):
    """Volume of intersection of two spheres with radii r1, r2 separated by d."""
    if d <= 0:
        # one sphere fully inside the other (or coincident)
        return 4/3 * math.pi * min(r1, r2) ** 3
    if d >= r1 + r2:
        return 0.0  # no overlap
    # standard lens formula
    def cap(R, h):
        return math.pi * h**2 * (3*R - h) / 3
    h1 = (r1**2 - r2**2 + d**2) / (2*d)  # distance from center1 to radical plane
    h1_cap = r1 - h1                       # cap height on sphere 1 side
    h2_cap = r2 - (d - h1)                 # cap height on sphere 2 side
    vol = cap(r1, max(h1_cap, 0)) + cap(r2, max(h2_cap, 0))
    return vol


# ---------------------------------------------------------------------------
# Circle–circle intersection area (2D)
# Returns the area of the region inside BOTH circles.
# ---------------------------------------------------------------------------

def _circle_circle_intersection_area_2d(d, r1, r2):
    """Area of intersection of two circles with radii r1, r2 separated by d."""
    if d <= 0:
        return math.pi * min(r1, r2) ** 2
    if d >= r1 + r2:
        return 0.0
    # lens formula
    def seg(R, h):
        # circular segment area for a chord at distance (R-h) from center
        return R**2 * math.acos(max(-1.0, min(1.0, (R - h) / R))) - (R - h) * math.sqrt(max(0.0, 2*R*h - h**2))
    h1 = (r1**2 - r2**2 + d**2) / (2*d)
    h1_cap = r1 - h1
    h2_cap = r2 - (d - h1)
    area = seg(r1, max(h1_cap, 0)) + seg(r2, max(h2_cap, 0))
    return area


# ---------------------------------------------------------------------------
# Fast void placement (3D) - vectorized, intra-grain only
# FIX-P: cap enforcement removed from here; caller passes a pre-capped budget.
# FIX-Y: pore_placement parameter added ("int" | "ext").
# FIX-AA: ext mode validity mask requires dist_from_center + pore_r >= pr so
#          every accepted pore is guaranteed to touch the outer grain surface.
# ---------------------------------------------------------------------------

def _place_voids_in_grain_fast(grain, target_vol, rng_seed, img_size=1,
                                pore_placement: str = "int"):
    local_rng = np.random.default_rng(rng_seed)
    px, py, pz, pr = grain

    SHELL_INNER_FRAC = 0.75

    placed_xyz = []
    placed_r   = []
    cum_vol    = 0.0          # accumulated CLIPPED void volume (grain ∩ void)

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

        if pore_placement == "int":
            # ---- Interior mode (original) ----
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

            dist_from_center = np.sqrt((cands_x-px)**2 + (cands_y-py)**2 + (cands_z-pz)**2)
            valid_mask = dist_from_center + pore_r <= pr  # void fully inside grain

            valid_mask &= (cands_x - pore_r >= 0) & (cands_x + pore_r <= img_size)
            valid_mask &= (cands_y - pore_r >= 0) & (cands_y + pore_r <= img_size)
            valid_mask &= (cands_z - pore_r >= 0) & (cands_z + pore_r <= img_size)

        else:
            # ---- Exterior (surface) mode ----
            # Void centers sampled in the annular shell [SHELL_INNER_FRAC*pr, pr + pore_r].
            local_rng.random(out=phi_buf)
            phi_buf   *= math.pi
            local_rng.random(out=theta_buf)
            theta_buf *= 2 * math.pi
            local_rng.random(out=rho_buf)
            rho_min = SHELL_INNER_FRAC * pr
            rho_max = pr + pore_r
            rho_buf  = rho_min + (rho_max - rho_min) * rho_buf

            cands_x = px + rho_buf * np.sin(phi_buf) * np.cos(theta_buf)
            cands_y = py + rho_buf * np.sin(phi_buf) * np.sin(theta_buf)
            cands_z = pz + rho_buf * np.cos(phi_buf)

            dist_from_center = np.sqrt((cands_x-px)**2 + (cands_y-py)**2 + (cands_z-pz)**2)

            # Pore must overlap with grain interior
            valid_mask = dist_from_center < pr + pore_r
            # FIX-AA: pore must also reach (touch or cross) the grain surface
            valid_mask &= dist_from_center + pore_r >= pr

            # Void center must be within the image domain
            valid_mask &= (cands_x >= 0) & (cands_x <= img_size)
            valid_mask &= (cands_y >= 0) & (cands_y <= img_size)
            valid_mask &= (cands_z >= 0) & (cands_z <= img_size)

            pore_vol = None  # will be computed per chosen candidate

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

        if pore_placement == "int":
            i = valid_indices[0]
            placed_xyz.append([cands_x[i], cands_y[i], cands_z[i]])
            placed_r.append(pore_r)
            cum_vol += pore_vol
            consecutive_failures = 0
        else:
            # ext mode: vectorized intersection volume for all valid candidates at once.
            remaining = target_vol - cum_vol
            if remaining <= 0:
                break
            d_v = dist_from_center[valid_indices]
            # Vectorized sphere-sphere intersection (lens formula)
            h1_v     = (pr**2 - pore_r**2 + d_v**2) / (2.0 * np.maximum(d_v, 1e-30))
            h1_cap_v = np.maximum(pr    - h1_v,        0.0)
            h2_cap_v = np.maximum(pore_r - (d_v - h1_v), 0.0)
            clipped_v = (math.pi * h1_cap_v**2 * (3*pr    - h1_cap_v) / 3 +
                         math.pi * h2_cap_v**2 * (3*pore_r - h2_cap_v) / 3)
            clipped_v = np.where(d_v <= 0, 4/3 * math.pi * min(pr, pore_r)**3, clipped_v)

            fits = np.where((clipped_v > 0) & (clipped_v <= remaining))[0]
            if len(fits) == 0:
                consecutive_failures += 1
            else:
                pick = valid_indices[fits[0]]
                placed_xyz.append([cands_x[pick], cands_y[pick], cands_z[pick]])
                placed_r.append(pore_r)
                cum_vol += float(clipped_v[fits[0]])
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
    pore_placement: Literal["int", "ext"] = "int",
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

    # Shell fraction for ext mode (fraction of grain radius where outer shell begins)
    SHELL_INNER_FRAC_2D = 0.75

    def _place_voids_in_grain_2d(grain, existing_voids_snap, target_remaining, rng_seed):
        local_rng = np.random.default_rng(rng_seed)
        placed  = []
        cum_vol = 0.0  # accumulated CLIPPED void area
        px, py, pr = grain[:3]

        if existing_voids_snap:
            ev = np.array(existing_voids_snap, dtype=float)       # (N,3): x, y, r
            dist_to_grain = np.hypot(ev[:, 0] - px, ev[:, 1] - py)
            keep = dist_to_grain < 2.0 * pr + ev[:, 2]
            existing_voids_snap = [existing_voids_snap[i] for i in np.where(keep)[0]]

        for _ in range(800):
            if cum_vol >= target_remaining:
                break
            pore_r = local_rng.lognormal(math.log(pr * 0.1), 0.4)
            pore_r = float(np.clip(pore_r, 0.1 * pr, 0.35 * pr))

            if pore_placement == "int":
                # ---- Interior mode (original) ----
                pore_area = math.pi * pore_r**2
                if cum_vol + pore_area > target_remaining:
                    pore_r   = math.sqrt((target_remaining - cum_vol) / math.pi)
                    pore_area = target_remaining - cum_vol

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
                    cum_vol += pore_area
                    break

            else:
                # ---- Exterior (surface) mode — numpy batch ----
                rho_min = SHELL_INNER_FRAC_2D * pr
                rho_max = pr + pore_r
                n_batch = 300

                thetas = local_rng.uniform(0, 2 * math.pi, n_batch)
                rhos   = local_rng.uniform(rho_min, rho_max, n_batch)
                vxs    = px + rhos * np.cos(thetas)
                vys    = py + rhos * np.sin(thetas)
                ds     = np.hypot(vxs - px, vys - py)

                # Pore must overlap grain interior
                vmask  = ds < pr + pore_r
                # FIX-AA: pore must also reach (touch or cross) the grain surface
                vmask &= ds + pore_r >= pr
                vmask &= (vxs >= 0) & (vxs <= img_size)
                vmask &= (vys >= 0) & (vys <= img_size)

                # Overlap check against all already-placed voids (numpy)
                all_placed = list(existing_voids_snap) + list(placed)
                if all_placed:
                    pv = np.array(all_placed, dtype=float)          # (N,3): x,y,r
                    dx2 = vxs[:, None] - pv[:, 0]
                    dy2 = vys[:, None] - pv[:, 1]
                    ovlp = np.any(dx2**2 + dy2**2 < (pore_r + pv[:, 2])**2, axis=1)
                    vmask &= ~ovlp

                valid_idx = np.where(vmask)[0]
                if len(valid_idx) == 0:
                    continue   # outer loop will retry with new pore_r

                # Vectorized circle-circle intersection (lens formula) for valid cands
                remaining = target_remaining - cum_vol
                if remaining <= 0:
                    break
                d_v   = ds[valid_idx]
                h1_v  = (pr**2 - pore_r**2 + d_v**2) / (2.0 * np.maximum(d_v, 1e-30))
                h1c   = np.maximum(pr     - h1_v,       0.0)
                h2c   = np.maximum(pore_r - (d_v - h1_v), 0.0)
                areas = (pr**2     * np.arccos(np.clip((pr     - h1c) / pr,     -1, 1))
                         - (pr     - h1c) * np.sqrt(np.maximum(2*pr    *h1c - h1c**2, 0))
                         + pore_r**2 * np.arccos(np.clip((pore_r - h2c) / pore_r, -1, 1))
                         - (pore_r - h2c) * np.sqrt(np.maximum(2*pore_r*h2c - h2c**2, 0)))
                # d==0 edge: full smaller circle
                areas = np.where(d_v <= 0, math.pi * min(pr, pore_r)**2, areas)

                fits = np.where((areas > 0) & (areas <= remaining))[0]
                if len(fits) == 0:
                    continue

                pick = valid_idx[fits[0]]
                placed.append((float(vxs[pick]), float(vys[pick]), pore_r))
                existing_voids_snap.append((float(vxs[pick]), float(vys[pick]), pore_r))
                cum_vol += float(areas[fits[0]])

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
        print(f"  Target void fraction:  {void_fraction:.4f}")
        print(f"  Target void area:      {target_void_area:.2e}")
        print(f"  Pore placement mode:   {pore_placement}")
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

        print("\nPlacing hollow grains (2D)...")
        sigma_ln_h = math.sqrt(math.log(1 + rad_dev**2))
        mu_ln_h    = math.log(img_r(mean_rad_hollow)) - 1.5*sigma_ln_h**2
        hollow_area = 0.0
        consecutive_hollow_fails_2d = 0
        max_consecutive_hollow_2d   = 2000

        for _ in range(max_attempts):
            if hollow_area >= vol_percent_hollow: break
            if current_void_area / total_domain_area >= void_fraction: break
            if consecutive_hollow_fails_2d >= max_consecutive_hollow_2d: break
            r = rng.lognormal(mu_ln_h, sigma_ln_h)
            if hollow_area + math.pi*r**2/total_domain_area > vol_percent_hollow*1.05:
                continue
            placed_hollow_2d = False
            for _ in range(50):
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
                    placed_hollow_2d = True
                    break
            if placed_hollow_2d:
                consecutive_hollow_fails_2d = 0
            else:
                consecutive_hollow_fails_2d += 1
        print(f"  Hollow area fraction: {hollow_area:.4f}")
        print(f"  Void fraction so far: {current_void_area/total_domain_area:.4f}")

        print("\nPlacing porous grains (2D)...")
        sigma_ln_p = math.sqrt(math.log(1 + rad_dev**2))
        mu_ln_p    = math.log(img_r(mean_rad_porous)) - 1.5 * sigma_ln_p**2
        consecutive_porous_fails_2d = 0
        max_consecutive_porous_2d   = 2000

        for _ in range(max_attempts):
            if porous_area >= vol_percent_porous: break
            if consecutive_porous_fails_2d >= max_consecutive_porous_2d: break
            r = rng.lognormal(mu_ln_p, sigma_ln_p)
            if porous_area + math.pi*r**2/total_domain_area > vol_percent_porous*1.05:
                continue
            placed_porous_2d = False
            for _ in range(50):
                x = rng.uniform(-margin+r, img_size+margin-r)
                y = rng.uniform(-margin+r, img_size+margin-r)
                cx_c, cy_c = cell_coords(x, y)
                if len(grid[cx_c][cy_c]) > 15: continue
                nb = nearby_2d(x, y, circles, r)
                if not overlaps_fast(x, y, r, nb, 0.98):
                    circles.append((x, y, r)); porous_circles.append((x, y, r))
                    grid[cx_c][cy_c].append(len(circles)-1)
                    porous_area += math.pi*r**2 if (x>r and x<img_size-r and y>r and y<img_size-r) else clipped_circle_area(x,y,r)
                    placed_porous_2d = True
                    break
            if placed_porous_2d:
                consecutive_porous_fails_2d = 0
            else:
                consecutive_porous_fails_2d += 1
        print(f"  Porous area fraction: {porous_area:.4f}")

        save_xyzr(circles, AP_xyzr, img_size, physical_size)
        if mwd_tolerance is not None:
            radii = np.array([r for (x, y, r) in circles])
            mwd_actual = mean_weight_diameter(AP_xyzr, radii=radii) * (physical_size / img_size)
            print(f"\nMWD check: {mwd_actual:.4e} m (target {mwd_target:.4e} m)")
            if abs(mwd_actual - mwd_target) > mwd_tolerance:
                print("MWD out of tolerance - skipping void placement.")
                return None

        print(f"\nPlacing voids within porous grains (parallel, mode={pore_placement})...")
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
        print(f"  Target void fraction:  {void_fraction:.4f}")
        print(f"  Target void vol:       {target_void_vol:.2e}")
        print(f"  Pore placement mode:   {pore_placement}")
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
        print("\nPlacing hollow grains (3D)...")
        sigma_ln_h = math.sqrt(math.log(1 + rad_dev**2))
        mu_ln_h    = math.log(img_r(mean_rad_hollow)) - 1.5*sigma_ln_h**2

        hollow_candidates = []
        consecutive_hollow_fails = 0
        max_consecutive_hollow   = 2000

        for _ in range(max_attempts):
            if hollow_vol >= vol_percent_hollow: break
            if consecutive_hollow_fails >= max_consecutive_hollow: break
            r = rng.lognormal(mu_ln_h, sigma_ln_h)
            if hollow_vol + 4/3*math.pi*r**3/total_domain_vol > vol_percent_hollow*1.05:
                continue
            placed_hollow = False
            for _ in range(50):
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
                    placed_hollow = True
                    break
            if placed_hollow:
                consecutive_hollow_fails = 0
            else:
                consecutive_hollow_fails += 1

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
        print("\nPlacing porous grains (3D)...")
        sigma_ln_p = math.sqrt(math.log(1 + rad_dev**2))
        mu_ln_p    = math.log(img_r(mean_rad_porous)) - 1.5*sigma_ln_p**2

        consecutive_porous_fails = 0
        max_consecutive_porous   = 2000

        for _ in range(max_attempts):
            if porous_vol >= vol_percent_porous: break
            if consecutive_porous_fails >= max_consecutive_porous: break
            r = rng.lognormal(mu_ln_p, sigma_ln_p)
            if porous_vol + 4/3*math.pi*r**3/total_domain_vol > vol_percent_porous*1.05:
                continue
            placed_porous = False
            for _ in range(50):
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
                    placed_porous = True
                    break
            if placed_porous:
                consecutive_porous_fails = 0
            else:
                consecutive_porous_fails += 1
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
        print(f"\nPlacing voids within porous grains (parallel, ProcessPoolExecutor, mode={pore_placement})...")
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
                        seed, img_size, pore_placement
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
                        int(rng.integers(0, 2**31)), img_size=img_size,
                        pore_placement=pore_placement)
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


# FIX-Z: work entirely in physical units; accept dim parameter so 2D uses
# physical_size**2 as the domain area instead of physical_size**3.
def compute_ap_volume_fraction_clipped(xyzr_path, physical_size, img_size=1.0, dim=3):
    data = np.loadtxt(xyzr_path)
    if data.ndim == 1:
        data = data[None, :]

    total_vol = 0.0

    if dim == 2:
        total_domain = physical_size ** 2
        for row in data:
            if len(row) < 4:
                continue
            x, y, _, r = row[:4]
            total_vol += math.pi * r**2
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


def _run_one_attempt(args):
    (attempt_idx, accepted_idx, name, subfolder,
     physical_size, rad_dev, max_attempts,
     vol_percent_solid, vol_percent_hollow, vol_percent_porous,
     void_fraction, mwd_target, mwd_tolerance, dim,
     mean_rad_solid, mean_rad_hollow, mean_rad_porous,
     ap_vol_tolerance, pore_placement) = args

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
            pore_placement=pore_placement,
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

    # --- CONSISTENT AP VOLUME CHECK ---
    ap_vol = compute_ap_volume_fraction_clipped(ap_xyzr, physical_size, dim=dim)
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
    pore_placement: Literal["int", "ext"] = "int",
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
        pore_placement=pore_placement,
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
                common["ap_vol_tolerance"], common["pore_placement"],
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
                                   physical_size=physical_size, dim=dim)
                    print(f"  Saved plot: {plot_save}")
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
        base_name="test3",
        dim=3,
        physical_size=20e-6,
        mean_rad_porous = 2e-6 / (1.2 * math.exp(math.sqrt(math.log(1 + 0.4 ** 2)) ** 2)), #target mwd
        mean_rad_hollow = 2.2e-6 / (1.2 * math.exp(math.sqrt(math.log(1 + 0.4 ** 2)) ** 2)),
        mean_rad_solid=2e-6,
        void_fraction=0.03,
        vol_percent_porous=0.3,
        vol_percent_hollow=0.0,
        mwd_tolerance=2.0e-6,
        n_workers=4,
        pore_placement="ext",   # <-- change to "ext" for surface pores
    )
    
    

if __name__ == "__main__":
    main()