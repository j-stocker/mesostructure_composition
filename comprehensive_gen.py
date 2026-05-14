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
# FIXES applied (v14):
#   FIX-AC Hollow and porous grain placement in both 2D and 3D restored to the
#          pre-sorted radii pattern from the working older version.  Radii are
#          pre-generated (2× target volume budget), sorted largest-first, then
#          placed in that order.  Large grains fill space first; small grains
#          fill gaps — enabling >60% AP packing that the random-draw loop could
#          not reach.  max_pos scales from 150 → 500 as the domain fills.
#          Nothing else changed.
# FIXES applied (v13):
#   FIX-AB Added void_fraction_mode parameter ("clipped" | "unclipped") to both void
#          placement functions and all callers.
#          "clipped" (default): existing behaviour — only the grain∩void intersection
#                               volume/area is counted against the budget (relevant for
#                               "ext" mode; "int" mode is unaffected as voids are always
#                               fully inside the grain).
#          "unclipped": the full void sphere/circle volume/area (4/3*pi*r^3 or pi*r^2)
#                       is always counted against the budget, regardless of how much of
#                       the void actually overlaps the grain.  Only meaningful for "ext"
#                       mode; "int" mode behaviour is identical to "clipped" since the
#                       void is fully contained.
# FIXES applied (v15):
#   FIX-AD Added "htpb_only" as a third pore_placement mode.  Void centers are placed
#          in the binder (HTPB) space — i.e. outside all AP grain surfaces.
#          The pore size distribution is the same lognormal used for "ext"/"int" pores
#          (relative to a reference radius derived from mean_rad_porous).
#          Two behaviours depending on void_fraction_mode:
#            "unclipped": void center must be outside every grain (dist >= grain_r for
#                         all grains); the void sphere MAY clip into grain surfaces.
#                         The full sphere volume is counted against the budget.
#            "clipped":   void center must be outside every grain AND the void sphere
#                         must not clip any grain surface (dist >= grain_r + pore_r for
#                         all grains).  The full sphere volume is counted.
#          In both cases void budget accounting uses the full sphere volume/area.
#          The "htpb_only" path does NOT use the per-grain cap mechanism (there is no
#          host grain); voids are placed globally until the domain budget is exhausted.
#          A domain-level spatial grid is used so that candidate–grain distance checks
#          scale as O(1) rather than O(N_grains).
# FIXES applied (v17):
#   FIX-AF void_fraction always means void_volume / total_domain_volume for BOTH modes.
#          In "clipped" mode, the ACCUMULATOR (cum_vol / current_void_vol/area) only
#          counts the portion of each void that is geometrically inside an AP particle
#          (grain∩void intersection volume/area).  The TARGET is still
#          void_fraction * total_domain_vol/area so the user input is always a fraction
#          of the full domain.  In "unclipped" mode, the full void sphere/circle volume
#          is counted (unchanged from before).
#          "int" pore placement is unaffected because voids are fully inside grains,
#          so the intersection volume equals the full void volume in both modes.
#          Removed FIX-AE logic that changed the denominator to AP grain volume.
# FIXES applied (v18):
#   FIX-AG In "ext" + "clipped" mode, void budget accounting now sums the intersection
#          volume with ALL overlapping AP grains (capped at full sphere volume) rather
#          than only the host grain.  This matches the void_statistics_3D.py accounting
#          so that input void_fraction == output void_fraction.  all_grains_arr is now
#          passed from the parallel executor and top-up loop to _place_voids_in_grain_fast.

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


# ---------------------------------------------------------------------------
# Sphere–sphere intersection volume (3D)
# ---------------------------------------------------------------------------

def _sphere_sphere_intersection_volume_3d(d, r1, r2):
    if d <= 0:
        return 4/3 * math.pi * min(r1, r2) ** 3
    if d >= r1 + r2:
        return 0.0
    def cap(R, h):
        return math.pi * h**2 * (3*R - h) / 3
    h1 = (r1**2 - r2**2 + d**2) / (2*d)
    h1_cap = r1 - h1
    h2_cap = r2 - (d - h1)
    return cap(r1, max(h1_cap, 0)) + cap(r2, max(h2_cap, 0))


# ---------------------------------------------------------------------------
# Circle–circle intersection area (2D)
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
# Fast void placement (3D) - vectorized
# FIX-AF: In "clipped" mode, cum_vol accumulates grain∩void intersection volume
#         so that the budget (target_vol = void_fraction * domain_vol) is met
#         in terms of void volume actually inside AP particles.
#         In "unclipped" mode, cum_vol accumulates the full sphere volume.
#         "int" mode is identical in both cases (void fully inside grain).
# FIX-AG: In "ext" + "clipped" mode, intersection is summed across ALL overlapping
#         grains (capped at full sphere) rather than only the host grain.
# ---------------------------------------------------------------------------

def _place_voids_in_grain_fast(grain, target_vol, rng_seed, img_size=1,
                                pore_placement: str = "int",
                                void_fraction_mode: str = "clipped",
                                all_grains_arr=None,
                                pore_radius_factor: float = 0.15):
    local_rng = np.random.default_rng(rng_seed)
    px, py, pz, pr = grain

    SHELL_INNER_FRAC = 0.75

    placed_xyz = []
    placed_r   = []
    cum_vol    = 0.0

    consecutive_failures = 0
    max_consecutive      = 50

    n_cands = 300
    phi_buf   = np.empty(n_cands)
    theta_buf = np.empty(n_cands)
    rho_buf   = np.empty(n_cands)

    for _ in range(800):
        if cum_vol >= target_vol:
            break
        if consecutive_failures >= max_consecutive:
            break

        pore_r = local_rng.lognormal(math.log(pr * pore_radius_factor), 0.4)
        pore_r = float(np.clip(pore_r, 0.1 * pr, 0.35 * pr))

        if pore_placement == "int":
            # Void fully inside grain — intersection == full sphere in both modes
            pore_vol = 4/3 * math.pi * pore_r**3
            if cum_vol + pore_vol > target_vol:
                remaining = target_vol - cum_vol
                if remaining <= 0:
                    break
                pore_r   = (remaining / (4/3 * math.pi)) ** (1/3)
                pore_vol = remaining
                if pore_r < 0.1 * pr:
                    break

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
            valid_mask = dist_from_center + pore_r <= pr

            valid_mask &= (cands_x - pore_r >= 0) & (cands_x + pore_r <= img_size)
            valid_mask &= (cands_y - pore_r >= 0) & (cands_y + pore_r <= img_size)
            valid_mask &= (cands_z - pore_r >= 0) & (cands_z + pore_r <= img_size)

        elif pore_placement == "ext":
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

            valid_mask = dist_from_center < pr + pore_r
            valid_mask &= dist_from_center + pore_r >= pr

            valid_mask &= (cands_x >= 0) & (cands_x <= img_size)
            valid_mask &= (cands_y >= 0) & (cands_y <= img_size)
            valid_mask &= (cands_z >= 0) & (cands_z <= img_size)

            pore_vol = None

        else:
            # htpb_only
            local_rng.random(out=phi_buf)
            local_rng.random(out=theta_buf)
            local_rng.random(out=rho_buf)

            cands_x = local_rng.uniform(0, img_size, n_cands)
            cands_y = local_rng.uniform(0, img_size, n_cands)
            cands_z = local_rng.uniform(0, img_size, n_cands)

            if all_grains_arr is not None and len(all_grains_arr) > 0:
                gx = all_grains_arr[:, 0]
                gy = all_grains_arr[:, 1]
                gz = all_grains_arr[:, 2]
                gr = all_grains_arr[:, 3]
                dx = cands_x[:, None] - gx[None, :]
                dy = cands_y[:, None] - gy[None, :]
                dz = cands_z[:, None] - gz[None, :]
                dist_to_grains = np.sqrt(dx**2 + dy**2 + dz**2)

                if void_fraction_mode == "clipped":
                    valid_mask = np.all(dist_to_grains >= gr[None, :] + pore_r, axis=1)
                else:
                    valid_mask = np.all(dist_to_grains >= gr[None, :], axis=1)
            else:
                valid_mask = np.ones(n_cands, dtype=bool)

            valid_mask &= (cands_x >= pore_r) & (cands_x <= img_size - pore_r)
            valid_mask &= (cands_y >= pore_r) & (cands_y <= img_size - pore_r)
            valid_mask &= (cands_z >= pore_r) & (cands_z <= img_size - pore_r)

            pore_vol = None

        # Overlap check against already-placed voids
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

        elif pore_placement == "ext":
            remaining = target_vol - cum_vol
            if remaining <= 0:
                break

            if void_fraction_mode == "unclipped":
                contrib_v = np.full(len(valid_indices), 4/3 * math.pi * pore_r**3)
            else:
                # FIX-AG: sum intersection with ALL overlapping grains, cap at full sphere
                full_sphere_vol = 4/3 * math.pi * pore_r**3
                contrib_v = np.zeros(len(valid_indices))

                if all_grains_arr is not None and len(all_grains_arr) > 0:
                    gx = all_grains_arr[:, 0]
                    gy = all_grains_arr[:, 1]
                    gz = all_grains_arr[:, 2]
                    gr = all_grains_arr[:, 3]

                    for vi, idx in enumerate(valid_indices):
                        cx = cands_x[idx]; cy = cands_y[idx]; cz = cands_z[idx]
                        d_all = np.sqrt((cx - gx)**2 + (cy - gy)**2 + (cz - gz)**2)
                        overlapping = d_all < pore_r + gr
                        if not np.any(overlapping):
                            continue
                        d_ov  = d_all[overlapping]
                        gr_ov = gr[overlapping]
                        d_safe = np.where(d_ov > 0, d_ov, 1e-30)
                        h1v  = (pore_r**2 - gr_ov**2 + d_ov**2) / (2.0 * d_safe)
                        h1c  = np.maximum(pore_r - h1v,        0.0)
                        h2c  = np.maximum(gr_ov  - (d_ov - h1v), 0.0)
                        # fully contained cases
                        fully = d_ov <= np.abs(pore_r - gr_ov)
                        lens  = (math.pi * h1c**2 * (3*pore_r - h1c) / 3 +
                                 math.pi * h2c**2 * (3*gr_ov  - h2c) / 3)
                        small_r  = np.minimum(pore_r, gr_ov)
                        full_con = 4/3 * math.pi * small_r**3
                        per_grain = np.where(fully, full_con, lens)
                        contrib_v[vi] = min(per_grain.sum(), full_sphere_vol)
                else:
                    # fallback: host grain only
                    d_v = np.sqrt((cands_x[valid_indices] - px)**2 +
                                  (cands_y[valid_indices] - py)**2 +
                                  (cands_z[valid_indices] - pz)**2)
                    d_safe = np.where(d_v > 0, d_v, 1e-30)
                    h1v = (pore_r**2 - pr**2 + d_v**2) / (2.0 * d_safe)
                    h1c = np.maximum(pore_r - h1v,      0.0)
                    h2c = np.maximum(pr     - (d_v - h1v), 0.0)
                    contrib_v = (math.pi * h1c**2 * (3*pore_r - h1c) / 3 +
                                 math.pi * h2c**2 * (3*pr     - h2c) / 3)

            fits = np.where((contrib_v > 0) & (contrib_v <= remaining))[0]
            if len(fits) == 0:
                consecutive_failures += 1
            else:
                pick = valid_indices[fits[0]]
                placed_xyz.append([cands_x[pick], cands_y[pick], cands_z[pick]])
                placed_r.append(pore_r)
                cum_vol += float(contrib_v[fits[0]])
                consecutive_failures = 0

        else:
            # htpb_only — always full sphere volume
            remaining = target_vol - cum_vol
            if remaining <= 0:
                break
            pore_vol_full = 4/3 * math.pi * pore_r**3
            if pore_vol_full > remaining:
                consecutive_failures += 1
                continue
            i = valid_indices[0]
            placed_xyz.append([cands_x[i], cands_y[i], cands_z[i]])
            placed_r.append(pore_r)
            cum_vol += pore_vol_full
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
    pore_placement: Literal["int", "ext", "htpb_only"] = "int",
    void_fraction_mode: Literal["clipped", "unclipped"] = "clipped",
    pore_radius_factor: float = 0.15,
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

    SHELL_INNER_FRAC_2D = 0.75

    def _place_voids_in_grain_2d(grain, existing_voids_snap, target_remaining, rng_seed,
                                  all_circles_snap=None):
        local_rng = np.random.default_rng(rng_seed)
        placed  = []
        cum_vol = 0.0
        px, py, pr = grain[:3]

        if existing_voids_snap and pore_placement != "htpb_only":
            ev = np.array(existing_voids_snap, dtype=float)
            dist_to_grain = np.hypot(ev[:, 0] - px, ev[:, 1] - py)
            keep = dist_to_grain < 2.0 * pr + ev[:, 2]
            existing_voids_snap = [existing_voids_snap[i] for i in np.where(keep)[0]]

        for _ in range(800):
            if cum_vol >= target_remaining:
                break
            pore_r = local_rng.lognormal(math.log(pr * pore_radius_factor), 0.4)
            pore_r = float(np.clip(pore_r, 0.1 * pr, 0.35 * pr))

            if pore_placement == "int":
                # Fully inside grain — intersection == full circle in both modes
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

            elif pore_placement == "ext":
                rho_min = SHELL_INNER_FRAC_2D * pr
                rho_max = pr + pore_r
                n_batch = 300

                thetas = local_rng.uniform(0, 2 * math.pi, n_batch)
                rhos   = local_rng.uniform(rho_min, rho_max, n_batch)
                vxs    = px + rhos * np.cos(thetas)
                vys    = py + rhos * np.sin(thetas)
                ds     = np.hypot(vxs - px, vys - py)

                vmask  = ds < pr + pore_r
                vmask &= ds + pore_r >= pr
                vmask &= (vxs >= 0) & (vxs <= img_size)
                vmask &= (vys >= 0) & (vys <= img_size)

                all_placed = list(existing_voids_snap) + list(placed)
                if all_placed:
                    pv = np.array(all_placed, dtype=float)
                    dx2 = vxs[:, None] - pv[:, 0]
                    dy2 = vys[:, None] - pv[:, 1]
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
                vys = local_rng.uniform(0, img_size, n_batch)

                if all_circles_snap is not None and len(all_circles_snap) > 0:
                    ac = np.array(all_circles_snap, dtype=float)
                    dx2 = vxs[:, None] - ac[:, 0]
                    dy2 = vys[:, None] - ac[:, 1]
                    dist_to_aps = np.sqrt(dx2**2 + dy2**2)

                    if void_fraction_mode == "clipped":
                        vmask = np.all(dist_to_aps >= ac[:, 2][None, :] + pore_r, axis=1)
                    else:
                        vmask = np.all(dist_to_aps >= ac[:, 2][None, :], axis=1)
                else:
                    vmask = np.ones(n_batch, dtype=bool)

                vmask &= (vxs >= pore_r) & (vxs <= img_size - pore_r)
                vmask &= (vys >= pore_r) & (vys <= img_size - pore_r)

                all_placed = list(existing_voids_snap) + list(placed)
                if all_placed:
                    pv = np.array(all_placed, dtype=float)
                    dx2 = vxs[:, None] - pv[:, 0]
                    dy2 = vys[:, None] - pv[:, 1]
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
    # 2-D branch
    # ------------------------------------------------------------------
    if dim == 2:
        total_domain_area = img_size * img_size
        target_void_area  = void_fraction * total_domain_area
        current_void_area = 0.0

        print(f"\n{'='*60}")
        print("TARGET PARAMETERS (2D)")
        print(f"  Target void fraction:  {void_fraction:.4f}  (of full domain)")
        print(f"  Target void area:      {target_void_area:.2e}")
        print(f"  Pore placement mode:   {pore_placement}")
        print(f"  Void fraction mode:    {void_fraction_mode}")
        print(f"    clipped → accumulate grain∩void intersection area")
        print(f"    unclipped → accumulate full circle area")
        print(f"  Pore radius factor:    {pore_radius_factor}")
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
            if current_void_area >= target_void_area: break
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
                    current_void_area += math.pi*rv**2
                    break
        print(f"  Hollow area fraction: {hollow_area:.4f}")
        print(f"  Void area so far:     {current_void_area:.4e}  ({current_void_area/total_domain_area:.4f} of domain)")

        print("\nPlacing porous grains (2D, pre-sorted radii)...")
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

        # ------------------------------------------------------------------
        # Void placement — three modes
        # ------------------------------------------------------------------
        if pore_placement == "htpb_only":
            print(f"\\nPlacing voids in HTPB binder space (htpb_only, "
                  f"mode={void_fraction_mode})...")
            ref_pr = img_r(mean_rad_porous)
 
            # Build a grain array once — updated when a void is added so the
            # void–void grid stays current, but the grain array is fixed.
            all_grains_arr_2d = np.array(circles, dtype=float)  # (N, 3): x, y, r
 
            # Separate spatial grid for already-placed voids so the
            # void–void overlap check is O(1) instead of O(N_voids).
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
                    for j in range(max(0, cy - window), min(n_cells, cy + window + 1)):
                        for vi in void_grid_2d[i][j]:
                            result.append(voids[vi])
                return result
 
            n_batch = 512
            consecutive_failures = 0
            max_consecutive = 200
 
            # Pre-extract grain columns for fast numpy broadcasting
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
 
                # --- Sample candidates ---
                vxs = rng.uniform(pore_r, img_size - pore_r, n_batch)
                vys = rng.uniform(pore_r, img_size - pore_r, n_batch)
 
                # --- Grain rejection (vectorized) ---
                # Centers must be strictly outside every AP grain surface.
                if len(gx2) > 0:
                    dx2d = vxs[:, None] - gx2[None, :]   # (n_batch, N_grains)
                    dy2d = vys[:, None] - gy2[None, :]
                    dist2_grains = dx2d**2 + dy2d**2       # squared distances
                    # center outside grain: dist >= grain_r  (both modes)
                    vmask = np.all(dist2_grains >= gr2[None, :]**2, axis=1)
                else:
                    vmask = np.ones(n_batch, dtype=bool)
 
                valid_idx = np.where(vmask)[0]
                if len(valid_idx) == 0:
                    consecutive_failures += 1
                    continue
 
                # --- Void–void overlap check (vectorized per valid candidate) ---
                # Collect nearby placed voids once per batch attempt.
                # Use a representative center to gather neighbors; candidates
                # are close enough that this is conservative.
                rep_x = float(vxs[valid_idx[0]])
                rep_y = float(vys[valid_idx[0]])
                nearby_v = nearby_voids_2d_fast(rep_x, rep_y, pore_r)
 
                if nearby_v:
                    nv_arr = np.array(nearby_v, dtype=float)
                    nvx2 = nv_arr[:, 0]; nvy2 = nv_arr[:, 1]; nvr2 = nv_arr[:, 2]
                    ddx = vxs[valid_idx, None] - nvx2[None, :]
                    ddy = vys[valid_idx, None] - nvy2[None, :]
                    void_overlap = np.any(
                        ddx**2 + ddy**2 < (pore_r + nvr2[None, :])**2, axis=1)
                    valid_idx = valid_idx[~void_overlap]
 
                if len(valid_idx) == 0:
                    consecutive_failures += 1
                    continue
 
                # --- Budget accounting ---
                if void_fraction_mode == "unclipped":
                    # Full circle area for every valid candidate; pick first that fits.
                    pick_mask = full_circle_area <= remaining
                    if not pick_mask:
                        break
                    pick = int(valid_idx[0])
                    counted_area = full_circle_area
 
                else:
                    # "clipped": count only the binder-side portion.
                    # counted = full_area - sum(circle∩grain intersections)
                    if len(gx2) > 0:
                        # Compute per-candidate binder area for all valid candidates.
                        vx_v = vxs[valid_idx]
                        vy_v = vys[valid_idx]
                        dx_v = vx_v[:, None] - gx2[None, :]   # (n_valid, N_grains)
                        dy_v = vy_v[:, None] - gy2[None, :]
                        d_v  = np.sqrt(dx_v**2 + dy_v**2)     # dist void-center → grain-center
 
                        # Circle–circle intersection area for each (void, grain) pair.
                        # Only pairs where d < pore_r + gr contribute.
                        overlapping = d_v < pore_r + gr2[None, :]  # (n_valid, N_grains)
 
                        # Vectorized lens formula
                        d_safe = np.where(overlapping, np.maximum(d_v, 1e-30), 1.0)
                        h1 = (pore_r**2 - gr2[None, :]**2 + d_v**2) / (2.0 * d_safe)
                        h1c = np.maximum(pore_r - h1,             0.0)
                        h2c = np.maximum(gr2[None, :] - (d_v - h1), 0.0)
 
                        # Segment areas
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
 
                        # Fully-contained case: smaller circle entirely inside larger
                        fully = d_v <= np.abs(pore_r - gr2[None, :])
                        small_r = np.minimum(pore_r, gr2[None, :])
                        full_contained = math.pi * small_r**2
 
                        intersection = np.where(
                            ~overlapping, 0.0,
                            np.where(fully, full_contained, lens))
 
                        total_intersection = intersection.sum(axis=1)   # (n_valid,)
                        # Clamp: counted area ∈ [0, full_circle_area]
                        counted_areas = np.clip(
                            full_circle_area - total_intersection,
                            0.0, full_circle_area)
                    else:
                        counted_areas = np.full(len(valid_idx), full_circle_area)
 
                    fits = np.where(counted_areas <= remaining)[0]
                    if len(fits) == 0:
                        consecutive_failures += 1
                        continue
                    pick = int(valid_idx[fits[0]])
                    counted_area = float(counted_areas[fits[0]])
 
                # --- Accept the void ---
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

        else:
            print(f"\nPlacing voids within porous grains (parallel, mode={pore_placement}, vf_mode={void_fraction_mode})...")
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
                            print(f"  Voids placed: {len(voids)}  fraction of domain: {current_void_area/total_domain_area:.4f}")

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
        print("FINAL RESULTS (2D)")
        print(f"  Total grains:           {len(circles)}")
        print(f"  AP area fraction:       {solid_area+hollow_area+porous_area:.4f}")
        print(f"  Void fraction (domain): {void_frac_domain:.4f}  (target: {void_fraction:.4f})")
        print(f"  Error:                  {abs(void_frac_domain - void_fraction):.2e}")
        print(f"  Note: in clipped mode, numerator = grain∩void intersection area only")
        print(f"{'='*60}\n")

        if pore_placement != "htpb_only" and vol_percent_porous > 0 and len(porous_circles) > 0:
            fracs = per_grain_placed / grain_areas
            print(f"  Per-grain void fraction - min: {fracs.min():.3f}  "
                  f"max: {fracs.max():.3f}  mean: {fracs.mean():.3f}  "
                  f"std: {fracs.std():.3f}")
            over = np.sum(fracs > MAX_VOID_FRACTION_PER_GRAIN + 1e-9)
            if over:
                print(f"  WARNING: {over} grain(s) exceed the {MAX_VOID_FRACTION_PER_GRAIN*100:.0f}% cap!")
            else:
                print(f"  All grains within {MAX_VOID_FRACTION_PER_GRAIN*100:.0f}% cap. OK")

        save_xyzr(voids, void_xyzr, img_size, physical_size)
        return void_frac_domain

    # ------------------------------------------------------------------
    # 3-D branch
    # ------------------------------------------------------------------
    else:
        total_domain_vol = img_size ** 3
        target_void_vol  = void_fraction * total_domain_vol
        current_void_vol = 0.0

        print(f"\n{'='*60}")
        print("TARGET PARAMETERS (3D)")
        print(f"  Target void fraction:  {void_fraction:.4f}  (of full domain)")
        print(f"  Target void vol:       {target_void_vol:.2e}")
        print(f"  Pore placement mode:   {pore_placement}")
        print(f"  Void fraction mode:    {void_fraction_mode}")
        print(f"    clipped → accumulate grain∩void intersection volume")
        print(f"    unclipped → accumulate full sphere volume")
        print(f"  Pore radius factor:    {pore_radius_factor}")
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
            print(f"  rv / r ratio:       {rv_r_ratio:.4f}")

            for (x,y,z,r) in hollow_candidates:
                rv = r * rv_r_ratio
                voids.append((x,y,z,rv))
                current_void_vol += 4/3 * math.pi * rv**3

        print(f"  Void fraction of domain so far: {current_void_vol/total_domain_vol:.4f}")

        # ---- Porous grains ----
        print("\nPlacing porous grains (3D, pre-sorted radii)...")
        sigma_ln_p = math.sqrt(math.log(1 + rad_dev**2))
        mu_ln_p    = math.log(img_r(mean_rad_porous)) - 1.5*sigma_ln_p**2
        porous_radii_list = generate_radii(vol_percent_porous * 2, mu_ln_p, sigma_ln_p, dim=3)
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

        # ---- Void placement — three modes ----
        if pore_placement == "htpb_only":
            print(f"\\nPlacing voids in HTPB binder space (htpb_only, "
                  f"mode={void_fraction_mode})...")
            ref_pr = img_r(mean_rad_porous)
 
            # Fixed grain array — centres are outside all grains in both modes.
            all_grains_arr_3d = np.array(spheres, dtype=float)  # (N, 4): x, y, z, r
 
            # Spatial grid for placed voids only (void–void overlap).
            void_grid_3d = [[[[] for _ in range(n_cells)]
                              for __ in range(n_cells)]
                             for ___ in range(n_cells)]
            for vi, (vx_, vy_, vz_, vr_) in enumerate(voids):
                vci, vcj, vck = cell_coords(vx_, vy_, vz_)
                void_grid_3d[vci][vcj][vck].append(vi)
 
            def nearby_voids_3d_fast(x, y, z, r_query):
                max_vr = ref_pr * 0.35
                window = max(1, math.ceil((r_query + max_vr) / cell_size))
                cx, cy, cz = cell_coords(x, y, z)
                result = []
                for i in range(max(0, cx - window), min(n_cells, cx + window + 1)):
                    for j in range(max(0, cy - window), min(n_cells, cy + window + 1)):
                        for k in range(max(0, cz - window), min(n_cells, cz + window + 1)):
                            for vi in void_grid_3d[i][j][k]:
                                result.append(voids[vi])
                return result
 
            n_batch = 512
            consecutive_failures = 0
            max_consecutive = 200
 
            # Pre-extract grain columns for broadcasting
            if len(all_grains_arr_3d) > 0:
                gx3 = all_grains_arr_3d[:, 0]
                gy3 = all_grains_arr_3d[:, 1]
                gz3 = all_grains_arr_3d[:, 2]
                gr3 = all_grains_arr_3d[:, 3]
            else:
                gx3 = gy3 = gz3 = gr3 = np.empty(0)
 
            while current_void_vol < target_void_vol:
                if consecutive_failures >= max_consecutive:
                    print("WARNING: could not place more htpb voids without overlap.")
                    break
 
                pore_r = float(np.clip(
                    rng.lognormal(math.log(ref_pr * pore_radius_factor), 0.4),
                    0.1 * ref_pr, 0.35 * ref_pr))
 
                full_sphere_vol = 4/3 * math.pi * pore_r**3
                remaining = target_void_vol - current_void_vol
                if full_sphere_vol > remaining and void_fraction_mode == "unclipped":
                    break
 
                # --- Sample candidates ---
                vxs = rng.uniform(pore_r, img_size - pore_r, n_batch)
                vys = rng.uniform(pore_r, img_size - pore_r, n_batch)
                vzs = rng.uniform(pore_r, img_size - pore_r, n_batch)
 
                # --- Grain rejection (vectorized) ---
                # Centers must be strictly outside every AP grain surface in both modes.
                if len(gx3) > 0:
                    dx3d = vxs[:, None] - gx3[None, :]   # (n_batch, N_grains)
                    dy3d = vys[:, None] - gy3[None, :]
                    dz3d = vzs[:, None] - gz3[None, :]
                    dist2_grains = dx3d**2 + dy3d**2 + dz3d**2
                    vmask = np.all(dist2_grains >= gr3[None, :]**2, axis=1)
                else:
                    vmask = np.ones(n_batch, dtype=bool)
 
                valid_idx = np.where(vmask)[0]
                if len(valid_idx) == 0:
                    consecutive_failures += 1
                    continue
 
                # --- Void–void overlap check (vectorized per valid subset) ---
                rep_x = float(vxs[valid_idx[0]])
                rep_y = float(vys[valid_idx[0]])
                rep_z = float(vzs[valid_idx[0]])
                nearby_v = nearby_voids_3d_fast(rep_x, rep_y, rep_z, pore_r)
 
                if nearby_v:
                    nv_arr = np.array(nearby_v, dtype=float)
                    nvx3 = nv_arr[:, 0]; nvy3 = nv_arr[:, 1]
                    nvz3 = nv_arr[:, 2]; nvr3 = nv_arr[:, 3]
                    ddx = vxs[valid_idx, None] - nvx3[None, :]
                    ddy = vys[valid_idx, None] - nvy3[None, :]
                    ddz = vzs[valid_idx, None] - nvz3[None, :]
                    void_overlap = np.any(
                        ddx**2 + ddy**2 + ddz**2 < (pore_r + nvr3[None, :])**2,
                        axis=1)
                    valid_idx = valid_idx[~void_overlap]
 
                if len(valid_idx) == 0:
                    consecutive_failures += 1
                    continue
 
                # --- Budget accounting ---
                if void_fraction_mode == "unclipped":
                    if full_sphere_vol > remaining:
                        break
                    pick = int(valid_idx[0])
                    counted_vol = full_sphere_vol
 
                else:
                    # "clipped": count only the binder-side portion.
                    # counted = full_vol - sum(sphere∩grain intersections)
                    if len(gx3) > 0:
                        vx_v = vxs[valid_idx]
                        vy_v = vys[valid_idx]
                        vz_v = vzs[valid_idx]
                        dx_v = vx_v[:, None] - gx3[None, :]   # (n_valid, N_grains)
                        dy_v = vy_v[:, None] - gy3[None, :]
                        dz_v = vz_v[:, None] - gz3[None, :]
                        d_v  = np.sqrt(dx_v**2 + dy_v**2 + dz_v**2)
 
                        # Sphere–sphere intersection volume for each (void, grain) pair
                        overlapping = d_v < pore_r + gr3[None, :]
                        d_safe = np.where(overlapping, np.maximum(d_v, 1e-30), 1.0)
 
                        h1 = (pore_r**2 - gr3[None, :]**2 + d_v**2) / (2.0 * d_safe)
                        h1c = np.maximum(pore_r - h1,              0.0)
                        h2c = np.maximum(gr3[None, :] - (d_v - h1), 0.0)
 
                        cap_void  = math.pi * h1c**2 * (3*pore_r         - h1c) / 3
                        cap_grain = math.pi * h2c**2 * (3*gr3[None, :] - h2c) / 3
                        lens = cap_void + cap_grain
 
                        fully = d_v <= np.abs(pore_r - gr3[None, :])
                        small_r = np.minimum(pore_r, gr3[None, :])
                        full_contained = 4/3 * math.pi * small_r**3
 
                        intersection = np.where(
                            ~overlapping, 0.0,
                            np.where(fully, full_contained, lens))
 
                        total_intersection = intersection.sum(axis=1)   # (n_valid,)
                        counted_vols = np.clip(
                            full_sphere_vol - total_intersection,
                            0.0, full_sphere_vol)
                    else:
                        counted_vols = np.full(len(valid_idx), full_sphere_vol)
 
                    fits = np.where(counted_vols <= remaining)[0]
                    if len(fits) == 0:
                        consecutive_failures += 1
                        continue
                    pick = int(valid_idx[fits[0]])
                    counted_vol = float(counted_vols[fits[0]])
 
                # --- Accept the void ---
                vx_new = float(vxs[pick])
                vy_new = float(vys[pick])
                vz_new = float(vzs[pick])
                voids.append((vx_new, vy_new, vz_new, pore_r))
                vi_new = len(voids) - 1
                vci, vcj, vck = cell_coords(vx_new, vy_new, vz_new)
                void_grid_3d[vci][vcj][vck].append(vi_new)
                current_void_vol += counted_vol
                consecutive_failures = 0
 
                if len(voids) % 500 == 0:
                    print(f"  Voids placed: {len(voids)}  "
                          f"fraction of domain: {current_void_vol/total_domain_vol:.4f}")

        else:
            print(f"\nPlacing voids within porous grains (parallel, ProcessPoolExecutor, mode={pore_placement}, vf_mode={void_fraction_mode})...")
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

                # FIX-AG: pass all_grains_arr so each worker can sum intersections
                # across all overlapping grains in ext+clipped mode.
                all_grains_arr = np.array(spheres)

                futures_map = {}
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for grain_idx, grain in enumerate(porous_spheres):
                        if current_void_vol >= target_void_vol:
                            break
                        seed = int(rng.integers(0, 2**31))
                        fut = executor.submit(
                            _place_voids_in_grain_fast,
                            grain, per_grain_budgets[grain_idx],
                            seed, img_size, pore_placement, void_fraction_mode,
                            all_grains_arr,   # FIX-AG: was None
                            pore_radius_factor,
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
                            print(f"  Voids placed: {len(voids)}  fraction of domain: {current_void_vol/total_domain_vol:.4f} / {void_fraction:.4f}")

                print(f"  Void fraction of domain after parallel pass: {current_void_vol/total_domain_vol:.4f}")

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
                            pore_placement=pore_placement,
                            void_fraction_mode=void_fraction_mode,
                            all_grains_arr=all_grains_arr,   # FIX-AG: was missing
                            pore_radius_factor=pore_radius_factor)
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

        void_frac_domain = current_void_vol / total_domain_vol
        print(f"\n{'='*60}")
        print("FINAL RESULTS (3D)")
        print(f"  Total grains:            {len(spheres)}")
        print(f"  AP volume fraction:      {solid_vol+hollow_vol+porous_vol:.4f}")
        print(f"  Void fraction (domain):  {void_frac_domain:.4f}  (target: {void_fraction:.4f})")
        print(f"  Error:                   {abs(void_frac_domain - void_fraction):.2e}")
        print(f"  Note: in clipped mode, numerator = grain∩void intersection volume only")
        print(f"{'='*60}\n")

        if pore_placement != "htpb_only" and vol_percent_porous > 0 and len(porous_spheres) > 0:
            fracs = per_grain_placed / grain_vols
            print(f"  Per-grain void fraction - min: {fracs.min():.3f}  "
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
     ap_vol_tolerance, pore_placement, void_fraction_mode,
     pore_radius_factor) = args

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
            void_fraction_mode=void_fraction_mode,
            pore_radius_factor=pore_radius_factor,
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
    pore_placement: Literal["int", "ext", "htpb_only"] = "int",
    void_fraction_mode: Literal["clipped", "unclipped"] = "clipped",
    pore_radius_factor: float = 0.15,
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
        void_fraction_mode=void_fraction_mode,
        pore_radius_factor=pore_radius_factor,
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
                common["void_fraction_mode"], common["pore_radius_factor"],
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
        'test_files',
        target_mwd=4.0e-6,
        base_name="htpb_only_clipped_4um_vf03_ap55",
        dim=3,
        physical_size=50e-6,
        mean_rad_porous = 2e-6 / (1.2 * math.exp(math.sqrt(math.log(1 + 0.4 ** 2)) ** 2)),
        mean_rad_hollow = 2.2e-6 / (1.2 * math.exp(math.sqrt(math.log(1 + 0.4 ** 2)) ** 2)),
        mean_rad_solid=2e-6,
        void_fraction=0.03,
        vol_percent_porous=0.55,
        vol_percent_hollow=0.0,
        mwd_tolerance=0.2e-6,
        n_workers=4,
        pore_placement='htpb_only',
        void_fraction_mode="clipped",
    )


if __name__ == "__main__":
    main()