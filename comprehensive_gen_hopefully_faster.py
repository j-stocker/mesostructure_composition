import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from typing import Literal
import concurrent.futures
from collections import deque
from scipy.spatial import cKDTree

MAX_VOID_FRACTION_PER_GRAIN = 0.20

# ---------------------------------------------------------------------------
# Placement performance constants
# ---------------------------------------------------------------------------
GUIDED_ONSET       = 0.18
GUIDED_BATCH       = 256
BASE_ATTEMPTS      = 200
ATTEMPT_SCALE      = 5.0
MAX_ATTEMPTS_CAP   = 1500
STALL_WINDOW       = 30
STALL_THRESHOLD    = 1000
STALL_BOOST_MULT   = 3   # multiplier on MAX_ATTEMPTS_CAP once stall is detected

# Pre-allocation chunk size for dynamic arrays
_ALLOC_CHUNK = 65536


# ---------------------------------------------------------------------------
# Dynamic pre-allocated array helper
# ---------------------------------------------------------------------------

class DynamicArray:
    """
    Pre-allocated NumPy array that doubles capacity as needed.
    Avoids O(n^2) copies from repeated np.vstack / np.append.
    """
    def __init__(self, ncols: int, dtype=np.float64, initial=_ALLOC_CHUNK):
        self._data  = np.empty((initial, ncols), dtype=dtype)
        self._count = 0
        self._ncols = ncols

    def append(self, row):
        if self._count == len(self._data):
            self._data = np.resize(self._data, (len(self._data) * 2, self._ncols))
        self._data[self._count] = row
        self._count += 1

    @property
    def arr(self):
        return self._data[:self._count]

    def __len__(self):
        return self._count


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_xyzr(particles, filepath, img_size, physical_size):
    scale = physical_size / img_size
    if not particles:
        open(filepath, 'w').close()
        return
    arr = np.array(particles, dtype=float)
    if arr.shape[1] == 3:
        out = np.zeros((len(arr), 4))
        out[:, 0] = arr[:, 0] * scale
        out[:, 1] = arr[:, 1] * scale
        out[:, 3] = arr[:, 2] * scale
    else:
        out = arr * scale
    np.savetxt(filepath, out, fmt='%.8e')


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
    porous   = [p for p in all_particles if p[-1] <= thresh]
    nonporous = [p for p in all_particles if p[-1] > thresh]
    return porous, nonporous


# ---------------------------------------------------------------------------
# Sphere–sphere / Circle–circle intersection helpers
# ---------------------------------------------------------------------------

def _sphere_sphere_intersection_volume_3d(d, r1, r2):
    if d <= 0:
        return 4/3 * math.pi * min(r1, r2) ** 3
    if d >= r1 + r2:
        return 0.0
    def cap(R, h):
        return math.pi * h**2 * (3*R - h) / 3
    h1 = (r1**2 - r2**2 + d**2) / (2*d)
    return cap(r1, max(r1 - h1, 0)) + cap(r2, max(r2 - (d - h1), 0))


def _circle_circle_intersection_area_2d(d, r1, r2):
    if d <= 0:
        return math.pi * min(r1, r2) ** 2
    if d >= r1 + r2:
        return 0.0
    def seg(R, h):
        return R**2 * math.acos(max(-1.0, min(1.0, (R-h)/R))) - (R-h)*math.sqrt(max(0.0, 2*R*h-h**2))
    h1 = (r1**2 - r2**2 + d**2) / (2*d)
    return seg(r1, max(r1-h1, 0)) + seg(r2, max(r2-(d-h1), 0))


# ---------------------------------------------------------------------------
# Neighbor-cell overlap check helpers (cell-restricted, avoids O(n) scan)
# ---------------------------------------------------------------------------

def _neighbor_indices_3d(ci, cj, ck, window, n_cells, cell_counts, grid):
    """Collect placed-sphere indices from cells within `window` of (ci,cj,ck)."""
    indices = []
    for di in range(-window, window + 1):
        ii = ci + di
        if ii < 0 or ii >= n_cells: continue
        for dj in range(-window, window + 1):
            jj = cj + dj
            if jj < 0 or jj >= n_cells: continue
            for dk in range(-window, window + 1):
                kk = ck + dk
                if kk < 0 or kk >= n_cells: continue
                indices.extend(grid[ii][jj][kk])
    return indices


def _neighbor_indices_2d(ci, cj, window, n_cells, grid):
    indices = []
    for di in range(-window, window + 1):
        ii = ci + di
        if ii < 0 or ii >= n_cells: continue
        for dj in range(-window, window + 1):
            jj = cj + dj
            if jj < 0 or jj >= n_cells: continue
            indices.extend(grid[ii][jj])
    return indices


def find_first_valid_3d(cands_x, cands_y, cands_z, r, spheres_arr, factor=0.98,
                        grid=None, cell_size=None, n_cells=None, cell_counts=None):
    """
    Check each candidate against neighbours only (when grid provided),
    falling back to full scan if not.  Returns index into cands arrays or None.
    """
    if len(spheres_arr) == 0:
        return 0 if len(cands_x) > 0 else None

    if grid is not None:
        window = max(1, math.ceil(2 * r / cell_size) + 1)
        for i in range(len(cands_x)):
            cx = cands_x[i]; cy = cands_y[i]; cz = cands_z[i]
            ci = max(0, min(n_cells - 1, int(cx // cell_size)))
            cj = max(0, min(n_cells - 1, int(cy // cell_size)))
            ck = max(0, min(n_cells - 1, int(cz // cell_size)))
            nb_idx = _neighbor_indices_3d(ci, cj, ck, window, n_cells, cell_counts, grid)
            if not nb_idx:
                return i
            nb = spheres_arr[nb_idx]
            dx = cx - nb[:, 0]; dy = cy - nb[:, 1]; dz = cz - nb[:, 2]
            if not np.any(dx**2 + dy**2 + dz**2 < ((r + nb[:, 3]) * factor) ** 2):
                return i
        return None

    # Full-scan fallback
    gx = spheres_arr[:, 0]; gy = spheres_arr[:, 1]
    gz = spheres_arr[:, 2]; gr = spheres_arr[:, 3]
    dx = cands_x[:, None] - gx[None, :]
    dy = cands_y[:, None] - gy[None, :]
    dz = cands_z[:, None] - gz[None, :]
    dist2 = dx**2 + dy**2 + dz**2
    threshold2 = ((r + gr) * factor) ** 2
    has_overlap = np.any(dist2 < threshold2[None, :], axis=1)
    valid = np.where(~has_overlap)[0]
    return int(valid[0]) if len(valid) > 0 else None


def find_first_valid_2d(cands_x, cands_y, r, circles_arr, factor=0.98,
                        grid=None, cell_size=None, n_cells=None):
    if len(circles_arr) == 0:
        return 0 if len(cands_x) > 0 else None

    if grid is not None:
        window = max(1, math.ceil(2 * r / cell_size) + 1)
        for i in range(len(cands_x)):
            cx = cands_x[i]; cy = cands_y[i]
            ci = max(0, min(n_cells - 1, int(cx // cell_size)))
            cj = max(0, min(n_cells - 1, int(cy // cell_size)))
            nb_idx = _neighbor_indices_2d(ci, cj, window, n_cells, grid)
            if not nb_idx:
                return i
            nb = circles_arr[nb_idx]
            dx = cx - nb[:, 0]; dy = cy - nb[:, 1]
            if not np.any(dx**2 + dy**2 < ((r + nb[:, 2]) * factor) ** 2):
                return i
        return None

    # Full-scan fallback
    gx = circles_arr[:, 0]; gy = circles_arr[:, 1]; gr = circles_arr[:, 2]
    dx = cands_x[:, None] - gx[None, :]
    dy = cands_y[:, None] - gy[None, :]
    dist2 = dx**2 + dy**2
    threshold2 = ((r + gr) * factor) ** 2
    has_overlap = np.any(dist2 < threshold2[None, :], axis=1)
    valid = np.where(~has_overlap)[0]
    return int(valid[0]) if len(valid) > 0 else None


# ---------------------------------------------------------------------------
# Neighbor-guided candidate generators
# ---------------------------------------------------------------------------

def guided_candidates_3d(spheres_arr, r_new, rng, n_cands=GUIDED_BATCH,
                          margin=0.07, img_size=1.0):
    if len(spheres_arr) == 0:
        lo = -margin + r_new; hi = img_size + margin - r_new
        return (rng.uniform(lo, hi, n_cands),
                rng.uniform(lo, hi, n_cands),
                rng.uniform(lo, hi, n_cands))
    radii = spheres_arr[:, 3]
    weights = radii ** 2; weights /= weights.sum()
    idx = rng.choice(len(spheres_arr), size=n_cands, replace=True, p=weights)
    ax = spheres_arr[idx, 0]; ay = spheres_arr[idx, 1]
    az = spheres_arr[idx, 2]; ar = spheres_arr[idx, 3]
    phi   = rng.uniform(0, math.pi,   n_cands)
    theta = rng.uniform(0, 2*math.pi, n_cands)
    gap   = rng.uniform(0.0, 0.15 * r_new, n_cands)
    d     = ar + r_new + gap
    return (ax + d * np.sin(phi) * np.cos(theta),
            ay + d * np.sin(phi) * np.sin(theta),
            az + d * np.cos(phi))


def guided_candidates_2d(circles_arr, r_new, rng, n_cands=GUIDED_BATCH,
                          margin=0.07, img_size=1.0):
    if len(circles_arr) == 0:
        lo = -margin + r_new; hi = img_size + margin - r_new
        return rng.uniform(lo, hi, n_cands), rng.uniform(lo, hi, n_cands)
    radii = circles_arr[:, 2]
    weights = radii ** 2; weights /= weights.sum()
    idx = rng.choice(len(circles_arr), size=n_cands, replace=True, p=weights)
    ax = circles_arr[idx, 0]; ay = circles_arr[idx, 1]; ar = circles_arr[idx, 2]
    theta = rng.uniform(0, 2*math.pi, n_cands)
    gap   = rng.uniform(0.0, 0.15 * r_new, n_cands)
    d     = ar + r_new + gap
    return ax + d * np.cos(theta), ay + d * np.sin(theta)


# ---------------------------------------------------------------------------
# Adaptive attempt budget
# ---------------------------------------------------------------------------

def adaptive_max_pos(current_frac, target_frac,
                     base=BASE_ATTEMPTS, scale=ATTEMPT_SCALE, cap=MAX_ATTEMPTS_CAP):
    if target_frac <= 0:
        return base
    ratio = min(current_frac / target_frac, 1.0)
    return min(cap, int(base * math.exp(scale * ratio)))


# ---------------------------------------------------------------------------
# Vectorised generate_radii  (OPT: binary-search cutoff, doubling batch)
# ---------------------------------------------------------------------------

def _generate_radii(target_fraction, mu_ln, sigma_ln, dim, rng):
    radii = []
    total = 0.0
    batch = 1024
    while total < target_fraction:
        rs   = rng.lognormal(mu_ln, sigma_ln, batch)
        vols = (math.pi * rs**2) if dim == 2 else (4/3 * math.pi * rs**3)
        cumvol = np.cumsum(vols)
        cutoff = int(np.searchsorted(cumvol, target_fraction - total))
        take   = min(cutoff + 1, len(rs))
        radii.extend(rs[:take].tolist())
        total += float(cumvol[take - 1])
        batch  = min(batch * 2, 65536)
    return sorted(radii, reverse=True)


# ---------------------------------------------------------------------------
# Fast void placement (3D) — vectorized, KD-tree for self-overlap
# ---------------------------------------------------------------------------

def _place_voids_in_grain_fast(grain, target_vol, rng_seed, img_size=1,
                                pore_placement: str = "int",
                                void_fraction_mode: str = "clipped",
                                all_grains_arr=None,
                                pore_radius_factor: float = 0.15,
                                max_iter: int = 800,
                                max_consecutive: int = None):
    local_rng = np.random.default_rng(rng_seed)
    px, py, pz, pr = grain
    SHELL_INNER_FRAC = 0.75

    # Use DynamicArray instead of vstack
    placed_dyn = DynamicArray(4)   # (x, y, z, r)
    cum_vol    = 0.0
    consecutive_failures = 0
    if max_consecutive is None:
        max_consecutive = 200 if pore_placement == "ext" else 50
    n_cands = 300

    # KD-tree for self-overlap (rebuilt lazily every 50 additions)
    _kdtree      = None
    _kdtree_pts  = None
    _kdtree_at   = 0

    def _get_kdtree():
        nonlocal _kdtree, _kdtree_pts, _kdtree_at
        placed_arr = placed_dyn.arr
        if len(placed_arr) == 0:
            return None, None
        if _kdtree is None or len(placed_arr) - _kdtree_at > 50:
            _kdtree_pts = placed_arr[:, :3].copy()
            _kdtree     = cKDTree(_kdtree_pts)
            _kdtree_at  = len(placed_arr)
        return _kdtree, placed_arr

    for _ in range(max_iter):
        if cum_vol >= target_vol:
            break
        if consecutive_failures >= max_consecutive:
            break

        pore_r = local_rng.lognormal(math.log(pr * pore_radius_factor), 0.4)
        pore_r = float(np.clip(pore_r, 0.1 * pr, 0.35 * pr))
        remaining_now = target_vol - cum_vol
        max_r_from_budget = (remaining_now / (4/3 * math.pi)) ** (1/3)
        if pore_r > max_r_from_budget > 0.05 * pr:
            pore_r = max_r_from_budget

        if pore_placement == "int":
            pore_vol = 4/3 * math.pi * pore_r**3
            if cum_vol + pore_vol > target_vol:
                remaining = target_vol - cum_vol
                if remaining <= 0:
                    break
                pore_r   = (remaining / (4/3 * math.pi)) ** (1/3)
                pore_vol = remaining
                if pore_r < 0.1 * pr:
                    break
            phi   = local_rng.uniform(0, math.pi,   n_cands)
            theta = local_rng.uniform(0, 2*math.pi, n_cands)
            rho   = (pr - pore_r) * local_rng.random(n_cands) ** (1/3)
            cands_x = px + rho * np.sin(phi) * np.cos(theta)
            cands_y = py + rho * np.sin(phi) * np.sin(theta)
            cands_z = pz + rho * np.cos(phi)
            dist_fc  = np.sqrt((cands_x-px)**2 + (cands_y-py)**2 + (cands_z-pz)**2)
            valid_mask = (dist_fc + pore_r <= pr)
            valid_mask &= (cands_x - pore_r >= 0) & (cands_x + pore_r <= img_size)
            valid_mask &= (cands_y - pore_r >= 0) & (cands_y + pore_r <= img_size)
            valid_mask &= (cands_z - pore_r >= 0) & (cands_z + pore_r <= img_size)

        elif pore_placement == "ext":
            phi   = local_rng.uniform(0, math.pi,   n_cands)
            theta = local_rng.uniform(0, 2*math.pi, n_cands)
            rho_min = SHELL_INNER_FRAC * pr
            rho_max = pr + pore_r
            rho   = rho_min + (rho_max - rho_min) * local_rng.random(n_cands)
            cands_x = px + rho * np.sin(phi) * np.cos(theta)
            cands_y = py + rho * np.sin(phi) * np.sin(theta)
            cands_z = pz + rho * np.cos(phi)
            dist_fc  = np.sqrt((cands_x-px)**2 + (cands_y-py)**2 + (cands_z-pz)**2)
            valid_mask  = (dist_fc < pr + pore_r)
            valid_mask &= (dist_fc + pore_r >= pr)
            valid_mask &= (cands_x >= 0) & (cands_x <= img_size)
            valid_mask &= (cands_y >= 0) & (cands_y <= img_size)
            valid_mask &= (cands_z >= 0) & (cands_z <= img_size)
            pore_vol = None

        else:  # htpb_only
            cands_x = local_rng.uniform(0, img_size, n_cands)
            cands_y = local_rng.uniform(0, img_size, n_cands)
            cands_z = local_rng.uniform(0, img_size, n_cands)
            if all_grains_arr is not None and len(all_grains_arr) > 0:
                gx = all_grains_arr[:, 0]; gy = all_grains_arr[:, 1]
                gz = all_grains_arr[:, 2]; gr = all_grains_arr[:, 3]
                dx = cands_x[:, None] - gx; dy = cands_y[:, None] - gy
                dz = cands_z[:, None] - gz
                dist_to_grains = np.sqrt(dx**2 + dy**2 + dz**2)
                thresh = gr + pore_r if void_fraction_mode == "clipped" else gr
                valid_mask = np.all(dist_to_grains >= thresh, axis=1)
            else:
                valid_mask = np.ones(n_cands, dtype=bool)
            valid_mask &= (cands_x >= pore_r) & (cands_x <= img_size - pore_r)
            valid_mask &= (cands_y >= pore_r) & (cands_y <= img_size - pore_r)
            valid_mask &= (cands_z >= pore_r) & (cands_z <= img_size - pore_r)
            pore_vol = None

        # --- KD-tree self-overlap check ---
        kdt, placed_arr = _get_kdtree()
        if kdt is not None:
            cands_xyz = np.column_stack([cands_x, cands_y, cands_z])
            max_r_placed = placed_arr[:, 3].max()
            hits = kdt.query_ball_point(cands_xyz, pore_r + max_r_placed)
            for i, hit in enumerate(hits):
                if not valid_mask[i]: continue
                if not hit: continue
                nb = placed_arr[hit]
                dx = cands_x[i] - nb[:, 0]
                dy = cands_y[i] - nb[:, 1]
                dz = cands_z[i] - nb[:, 2]
                if np.any(dx**2 + dy**2 + dz**2 < (pore_r + nb[:, 3])**2):
                    valid_mask[i] = False

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            consecutive_failures += 1
            continue

        if pore_placement == "int":
            i = valid_indices[0]
            placed_dyn.append([cands_x[i], cands_y[i], cands_z[i], pore_r])
            cum_vol   += pore_vol
            consecutive_failures = 0
            _kdtree = None  # invalidate

        elif pore_placement == "ext":
            remaining = target_vol - cum_vol
            if remaining <= 0:
                break
            if void_fraction_mode == "unclipped":
                contrib_v = np.full(len(valid_indices), 4/3 * math.pi * pore_r**3)
            else:
                full_sphere_vol = 4/3 * math.pi * pore_r**3
                contrib_v = np.zeros(len(valid_indices))
                if all_grains_arr is not None and len(all_grains_arr) > 0:
                    gx = all_grains_arr[:, 0]; gy = all_grains_arr[:, 1]
                    gz = all_grains_arr[:, 2]; gr = all_grains_arr[:, 3]
                    for vi, idx in enumerate(valid_indices):
                        cx = cands_x[idx]; cy = cands_y[idx]; cz = cands_z[idx]
                        d_all = np.sqrt((cx-gx)**2 + (cy-gy)**2 + (cz-gz)**2)
                        overlapping = d_all < pore_r + gr
                        if not np.any(overlapping):
                            continue
                        d_ov = d_all[overlapping]; gr_ov = gr[overlapping]
                        d_safe = np.where(d_ov > 0, d_ov, 1e-30)
                        h1v  = (pore_r**2 - gr_ov**2 + d_ov**2) / (2.0 * d_safe)
                        h1c  = np.maximum(pore_r - h1v,          0.0)
                        h2c  = np.maximum(gr_ov  - (d_ov - h1v), 0.0)
                        fully = d_ov <= np.abs(pore_r - gr_ov)
                        lens  = (math.pi * h1c**2 * (3*pore_r - h1c) / 3 +
                                 math.pi * h2c**2 * (3*gr_ov  - h2c) / 3)
                        small_r  = np.minimum(pore_r, gr_ov)
                        full_con = 4/3 * math.pi * small_r**3
                        contrib_v[vi] = min(np.where(fully, full_con, lens).sum(), full_sphere_vol)
                else:
                    d_v  = np.sqrt((cands_x[valid_indices]-px)**2 +
                                   (cands_y[valid_indices]-py)**2 +
                                   (cands_z[valid_indices]-pz)**2)
                    d_safe = np.where(d_v > 0, d_v, 1e-30)
                    h1v = (pore_r**2 - pr**2 + d_v**2) / (2.0 * d_safe)
                    h1c = np.maximum(pore_r - h1v,       0.0)
                    h2c = np.maximum(pr     - (d_v - h1v), 0.0)
                    contrib_v = (math.pi * h1c**2 * (3*pore_r - h1c) / 3 +
                                 math.pi * h2c**2 * (3*pr     - h2c) / 3)
            fits = np.where((contrib_v > 0) & (contrib_v <= remaining))[0]
            if len(fits) == 0:
                consecutive_failures += 1
            else:
                pick = valid_indices[fits[0]]
                placed_dyn.append([cands_x[pick], cands_y[pick], cands_z[pick], pore_r])
                cum_vol   += float(contrib_v[fits[0]])
                consecutive_failures = 0
                _kdtree = None  # invalidate

        else:  # htpb_only
            remaining = target_vol - cum_vol
            if remaining <= 0:
                break
            pore_vol_full = 4/3 * math.pi * pore_r**3
            if pore_vol_full > remaining:
                consecutive_failures += 1
                continue
            i = valid_indices[0]
            placed_dyn.append([cands_x[i], cands_y[i], cands_z[i], pore_r])
            cum_vol   += pore_vol_full
            consecutive_failures = 0
            _kdtree = None  # invalidate

    pa = placed_dyn.arr
    placed = [(pa[i, 0], pa[i, 1], pa[i, 2], pa[i, 3]) for i in range(len(pa))]
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
    existing_ap_xyzr=None,
):
    if dim not in (2, 3):
        raise ValueError("Dimension must be 2 or 3.")

    rng      = np.random.default_rng()
    img_size = 1
    margin   = 0.07

    # ==================================================================
    # 2-D branch
    # ==================================================================
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
        print(f"  Pore radius factor:    {pore_radius_factor}")
        print(f"{'='*60}\n")

        def img_r(mu):
            return mu / physical_size * img_size

        cell_size = img_r(mean_rad_solid) * 4
        n_cells   = max(1, int(math.ceil(img_size / cell_size)))
        grid      = [[[] for _ in range(n_cells)] for __ in range(n_cells)]
        cell_counts = np.zeros((n_cells, n_cells), dtype=np.int32)

        def cell_coords_2d(x, y):
            return (max(0, min(n_cells-1, int(x // cell_size))),
                    max(0, min(n_cells-1, int(y // cell_size))))

        def clipped_circle_area(x, y, r, lo=0, hi=1):
            def seg(h):
                if h <= 0: return 0.0
                if h >= 2*r: return math.pi * r**2
                return r**2 * math.acos((r-h)/r) - (r-h)*math.sqrt(2*r*h - h**2)
            A = math.pi * r**2
            A -= seg(r-(x-lo)); A -= seg(r-(hi-x))
            A -= seg(r-(y-lo)); A -= seg(r-(hi-y))
            return A

        def _dynamic_cell_cap_2d(r):
            grain_area = math.pi * r**2
            if grain_area <= 0: return 30
            return max(30, int((cell_size**2 / grain_area) * 0.64 * 1.5))

        circles        = []
        porous_circles = []
        voids          = []
        solid_area = hollow_area = porous_area = 0.0
        sigma_ln   = math.sqrt(math.log(1 + rad_dev**2))

        # OPT: use DynamicArray instead of repeated vstack
        circles_dyn = DynamicArray(3)
        using_existing_geometry = existing_ap_xyzr is not None

        if using_existing_geometry:
            print("Loading existing AP geometry...")
            circles = load_xyzr(existing_ap_xyzr, physical_size, img_size, dim=2)
            porous_circles, _ = classify_loaded_particles(
                circles, dim=2, mean_rad_porous=mean_rad_porous,
                physical_size=physical_size, img_size=img_size)
            for idx, (x, y, r) in enumerate(circles):
                cx_c, cy_c = cell_coords_2d(x, y)
                grid[cx_c][cy_c].append(idx)
                cell_counts[cx_c, cy_c] += 1
                circles_dyn.append([x, y, r])
                area = (math.pi*r**2 if (x>r and x<img_size-r and y>r and y<img_size-r)
                        else clipped_circle_area(x, y, r))
                solid_area += area
            print(f"  Loaded grains: {len(circles)}")
            print(f"  Porous grains: {len(porous_circles)}")
        else:
            print("Placing solid grains...")
            for _ in range(max_attempts):
                if solid_area >= vol_percent_solid: break
                mu_ln = math.log(img_r(mean_rad_solid)) - 0.5*sigma_ln**2
                r = rng.lognormal(mu_ln, sigma_ln)
                if solid_area + math.pi*r**2/total_domain_area > vol_percent_solid: continue
                for _ in range(100):
                    x = rng.uniform(-margin+r, img_size+margin-r)
                    y = rng.uniform(-margin+r, img_size+margin-r)
                    cx_c, cy_c = cell_coords_2d(x, y)
                    if cell_counts[cx_c, cy_c] > 15: continue
                    if find_first_valid_2d(np.array([x]), np.array([y]),
                                           r, circles_dyn.arr, 0.999,
                                           grid, cell_size, n_cells) is not None:
                        circles.append((x, y, r))
                        circles_dyn.append([x, y, r])
                        grid[cx_c][cy_c].append(len(circles)-1)
                        cell_counts[cx_c, cy_c] += 1
                        solid_area += (math.pi*r**2
                                       if (x>r and x<img_size-r and y>r and y<img_size-r)
                                       else clipped_circle_area(x, y, r))
                        break
            print(f"  Solid area fraction: {solid_area:.4f}")

        # ------------------------------------------------------------------
        # Grain-placement loop (2D)
        # ------------------------------------------------------------------
        def place_grains_2d(radii_list, vol_target, vol_frac_name):
            nonlocal circles, cell_counts
            acc_frac     = 0.0
            recent_costs = deque()
            stalled      = False
            placed_count = 0

            def _guided_pass_2d(r, cap, n_rounds):
                for _ in range(n_rounds):
                    cx_arr, cy_arr = guided_candidates_2d(
                        circles_dyn.arr, r, rng,
                        n_cands=GUIDED_BATCH, margin=margin, img_size=img_size)
                    bm = ((cx_arr >= -margin+r) & (cx_arr <= img_size+margin-r) &
                          (cy_arr >= -margin+r) & (cy_arr <= img_size+margin-r))
                    cx_arr = cx_arr[bm]; cy_arr = cy_arr[bm]
                    if len(cx_arr) == 0: continue
                    ci_arr = np.clip((cx_arr // cell_size).astype(int), 0, n_cells-1)
                    cj_arr = np.clip((cy_arr // cell_size).astype(int), 0, n_cells-1)
                    cf = cell_counts[ci_arr, cj_arr] <= cap
                    cx_arr = cx_arr[cf]; cy_arr = cy_arr[cf]
                    if len(cx_arr) == 0: continue
                    idx = find_first_valid_2d(cx_arr, cy_arr, r, circles_dyn.arr,
                                             grid=grid, cell_size=cell_size, n_cells=n_cells)
                    if idx is not None:
                        return float(cx_arr[idx]), float(cy_arr[idx])
                return None

            for r in radii_list:
                if acc_frac >= vol_target: break
                grain_area = math.pi * r**2 / total_domain_area
                if acc_frac + grain_area > vol_target * 1.05: continue

                max_pos    = adaptive_max_pos(acc_frac, vol_target)
                use_guided = (acc_frac >= GUIDED_ONSET * vol_target) or stalled
                cap        = _dynamic_cell_cap_2d(r)
                attempt_count = 0
                result        = None

                if use_guided:
                    n_rounds = max(1, max_pos // GUIDED_BATCH)
                    result   = _guided_pass_2d(r, cap, n_rounds)
                    attempt_count += n_rounds * GUIDED_BATCH

                    if result is None and not stalled:
                        for _ in range(min(200, max_pos // 4)):
                            attempt_count += 1
                            x = rng.uniform(-margin+r, img_size+margin-r)
                            y = rng.uniform(-margin+r, img_size+margin-r)
                            cx_c, cy_c = cell_coords_2d(x, y)
                            if cell_counts[cx_c, cy_c] > cap: continue
                            if find_first_valid_2d(np.array([x]), np.array([y]),
                                                   r, circles_dyn.arr, 0.999,
                                                   grid, cell_size, n_cells) is not None:
                                result = (x, y); break

                    # FIX: was incorrectly calling _guided_pass_3d (bug in original)
                    if result is None and stalled:
                        boosted = STALL_BOOST_MULT * (MAX_ATTEMPTS_CAP // GUIDED_BATCH)
                        print(f"  [{vol_frac_name}] Boosted recovery "
                              f"({boosted} rounds) for r={r:.5f}...")
                        result = _guided_pass_2d(r, cap, boosted)
                        attempt_count += boosted * GUIDED_BATCH
                        if result is None:
                            print(f"  [{vol_frac_name}] Recovery failed for "
                                  f"r={r:.5f} — skipping grain.")
                            placed_count += 1
                            if placed_count % 1000 == 0:
                                print(f"  [{vol_frac_name}] Placed {placed_count} grains  "
                                      f"vol_frac={acc_frac:.4f}")
                else:
                    for _ in range(max_pos):
                        attempt_count += 1
                        x = rng.uniform(-margin+r, img_size+margin-r)
                        y = rng.uniform(-margin+r, img_size+margin-r)
                        cx_c, cy_c = cell_coords_2d(x, y)
                        if cell_counts[cx_c, cy_c] > cap: continue
                        if find_first_valid_2d(np.array([x]), np.array([y]),
                                               r, circles_dyn.arr, 0.999,
                                               grid, cell_size, n_cells) is not None:
                            result = (x, y); break

                if result is not None:
                    x, y = result
                    circles.append((x, y, r))
                    circles_dyn.append([x, y, r])
                    cx_c, cy_c = cell_coords_2d(x, y)
                    grid[cx_c][cy_c].append(len(circles)-1)
                    cell_counts[cx_c, cy_c] += 1
                    acc_area = (math.pi*r**2
                                if (x>r and x<img_size-r and y>r and y<img_size-r)
                                else clipped_circle_area(x, y, r))
                    acc_frac += acc_area / total_domain_area

                    placed_count += 1
                    if placed_count % 1000 == 0:
                        print(f"  [{vol_frac_name}] Placed {placed_count} grains  "
                              f"vol_frac={acc_frac:.4f}")

                    recent_costs.append(attempt_count)
                    if len(recent_costs) > STALL_WINDOW:
                        recent_costs.popleft()
                    if (len(recent_costs) == STALL_WINDOW and
                            np.mean(recent_costs) > STALL_THRESHOLD):
                        if not stalled:
                            print(f"  [{vol_frac_name}] Stall detected at "
                                  f"{acc_frac:.4f} — switching to boosted guided-only.")
                            stalled = True
                    elif stalled and attempt_count < STALL_THRESHOLD // 2:
                        stalled = False
                        print(f"  [{vol_frac_name}] Stall cleared at {acc_frac:.4f}.")

            return acc_frac

        # ---- Void placement helper (2D) — KD-tree self-overlap ----
        SHELL_INNER_FRAC_2D = 0.75

        def _place_voids_in_grain_2d(grain, existing_voids_snap, target_remaining, rng_seed,
                                      all_circles_snap=None):
            local_rng = np.random.default_rng(rng_seed)
            placed  = []
            cum_vol = 0.0
            px, py, pr = grain[:3]

            if existing_voids_snap and pore_placement != "htpb_only":
                ev = np.array(existing_voids_snap, dtype=float)
                dist_to_grain = np.hypot(ev[:, 0]-px, ev[:, 1]-py)
                keep = dist_to_grain < 2.0*pr + ev[:, 2]
                existing_voids_snap = [existing_voids_snap[i] for i in np.where(keep)[0]]

            # KD-tree for void self-overlap
            _void_kdt    = None
            _void_kdt_at = 0
            all_voids    = list(existing_voids_snap)

            def _rebuild_kdt():
                nonlocal _void_kdt, _void_kdt_at
                if all_voids:
                    pts = np.array([[v[0], v[1]] for v in all_voids])
                    _void_kdt = cKDTree(pts)
                else:
                    _void_kdt = None
                _void_kdt_at = len(all_voids)

            _rebuild_kdt()

            def _overlaps_any(vx, vy, r):
                nonlocal _void_kdt, _void_kdt_at
                if len(all_voids) == 0: return False
                if len(all_voids) - _void_kdt_at > 30:
                    _rebuild_kdt()
                if _void_kdt is None: return False
                max_r = max(v[2] for v in all_voids)
                hits = _void_kdt.query_ball_point([vx, vy], r + max_r)
                for h in hits:
                    xv, yv, rv = all_voids[h]
                    if math.hypot(vx-xv, vy-yv) < r + rv:
                        return True
                return False

            for _ in range(800):
                if cum_vol >= target_remaining: break
                pore_r = local_rng.lognormal(math.log(pr * pore_radius_factor), 0.4)
                pore_r = float(np.clip(pore_r, 0.1*pr, 0.35*pr))

                if pore_placement == "int":
                    pore_area = math.pi * pore_r**2
                    if cum_vol + pore_area > target_remaining:
                        pore_r    = math.sqrt((target_remaining - cum_vol) / math.pi)
                        pore_area = target_remaining - cum_vol
                    for _ in range(300):
                        theta = local_rng.uniform(0, 2*math.pi)
                        rho   = local_rng.uniform(0, pr - pore_r)
                        vx = px + rho*math.cos(theta)
                        vy = py + rho*math.sin(theta)
                        if math.hypot(vx-px, vy-py) + pore_r > pr: continue
                        if vx-pore_r < 0 or vx+pore_r > img_size: continue
                        if vy-pore_r < 0 or vy+pore_r > img_size: continue
                        if _overlaps_any(vx, vy, pore_r): continue
                        placed.append((vx, vy, pore_r))
                        all_voids.append((vx, vy, pore_r))
                        cum_vol += pore_area
                        break

                elif pore_placement == "ext":
                    rho_min = SHELL_INNER_FRAC_2D * pr
                    rho_max = pr + pore_r
                    n_batch = 300
                    thetas = local_rng.uniform(0, 2*math.pi, n_batch)
                    rhos   = local_rng.uniform(rho_min, rho_max, n_batch)
                    vxs = px + rhos*np.cos(thetas); vys = py + rhos*np.sin(thetas)
                    ds  = np.hypot(vxs-px, vys-py)
                    vmask  = ds < pr + pore_r
                    vmask &= ds + pore_r >= pr
                    vmask &= (vxs >= 0) & (vxs <= img_size)
                    vmask &= (vys >= 0) & (vys <= img_size)
                    if all_voids:
                        av = np.array(all_voids, dtype=float)
                        dx2 = vxs[:, None]-av[:, 0]; dy2 = vys[:, None]-av[:, 1]
                        vmask &= ~np.any(dx2**2+dy2**2 < (pore_r+av[:, 2])**2, axis=1)
                    valid_idx = np.where(vmask)[0]
                    if len(valid_idx) == 0: continue
                    remaining = target_remaining - cum_vol
                    if remaining <= 0: break
                    if void_fraction_mode == "unclipped":
                        contrib_v = np.full(len(valid_idx), math.pi*pore_r**2)
                    else:
                        h1_v  = (pr**2 - pore_r**2 + ds[valid_idx]**2) / (2.0 * np.maximum(ds[valid_idx], 1e-30))
                        h1c   = np.maximum(pr     - h1_v, 0.0)
                        h2c   = np.maximum(pore_r - (ds[valid_idx] - h1_v), 0.0)
                        contrib_v = (pr**2 * np.arccos(np.clip((pr-h1c)/pr, -1, 1))
                                     - (pr-h1c)*np.sqrt(np.maximum(2*pr*h1c - h1c**2, 0))
                                     + pore_r**2 * np.arccos(np.clip((pore_r-h2c)/pore_r, -1, 1))
                                     - (pore_r-h2c)*np.sqrt(np.maximum(2*pore_r*h2c - h2c**2, 0)))
                        contrib_v = np.where(ds[valid_idx] <= 0, math.pi*min(pr, pore_r)**2, contrib_v)
                    fits = np.where((contrib_v > 0) & (contrib_v <= remaining))[0]
                    if len(fits) == 0: continue
                    pick = valid_idx[fits[0]]
                    placed.append((float(vxs[pick]), float(vys[pick]), pore_r))
                    all_voids.append((float(vxs[pick]), float(vys[pick]), pore_r))
                    cum_vol += float(contrib_v[fits[0]])

                else:  # htpb_only 2D
                    n_batch = 300
                    vxs = local_rng.uniform(0, img_size, n_batch)
                    vys = local_rng.uniform(0, img_size, n_batch)
                    if all_circles_snap is not None and len(all_circles_snap) > 0:
                        ac = np.array(all_circles_snap, dtype=float)
                        dx2 = vxs[:, None]-ac[:, 0]; dy2 = vys[:, None]-ac[:, 1]
                        dist_to_aps = np.sqrt(dx2**2 + dy2**2)
                        thresh = ac[:, 2] + pore_r if void_fraction_mode == "clipped" else ac[:, 2]
                        vmask = np.all(dist_to_aps >= thresh, axis=1)
                    else:
                        vmask = np.ones(n_batch, dtype=bool)
                    vmask &= (vxs >= pore_r) & (vxs <= img_size-pore_r)
                    vmask &= (vys >= pore_r) & (vys <= img_size-pore_r)
                    if all_voids:
                        av = np.array(all_voids, dtype=float)
                        dx2 = vxs[:, None]-av[:, 0]; dy2 = vys[:, None]-av[:, 1]
                        vmask &= ~np.any(dx2**2+dy2**2 < (pore_r+av[:, 2])**2, axis=1)
                    valid_idx = np.where(vmask)[0]
                    if len(valid_idx) == 0: continue
                    remaining = target_remaining - cum_vol
                    if remaining <= 0: break
                    if math.pi*pore_r**2 > remaining: continue
                    pick = valid_idx[0]
                    placed.append((float(vxs[pick]), float(vys[pick]), pore_r))
                    all_voids.append((float(vxs[pick]), float(vys[pick]), pore_r))
                    cum_vol += math.pi*pore_r**2

            return placed, cum_vol

        # ---- Hollow grains (2D) ----
        print("\nPlacing hollow grains (2D)...")
        sigma_ln_h   = math.sqrt(math.log(1+rad_dev**2))
        mu_ln_h      = math.log(img_r(mean_rad_hollow)) - 1.5*sigma_ln_h**2
        hollow_radii = _generate_radii(vol_percent_hollow*2, mu_ln_h, sigma_ln_h, 2, rng)
        print(f"  Pre-generated {len(hollow_radii)} candidate hollow radii")
        circles_before_hollow = len(circles)
        hollow_area = place_grains_2d(hollow_radii, vol_percent_hollow, "hollow")
        hollow_candidates_2d = circles[circles_before_hollow:]

        if len(hollow_candidates_2d) > 0:
            total_hollow_area = sum(
                math.pi*r**2 if (x>r and x<img_size-r and y>r and y<img_size-r)
                else clipped_circle_area(x, y, r)
                for (x, y, r) in hollow_candidates_2d)
            rv_scale = (math.sqrt(void_fraction*total_domain_area/total_hollow_area)
                        if total_hollow_area > 0 else 0)
            for (x, y, r) in hollow_candidates_2d:
                rv = r * rv_scale
                voids.append((x, y, rv))
                current_void_area += math.pi * rv**2

        print(f"  Hollow area fraction: {hollow_area:.4f}")
        print(f"  Void area so far:     {current_void_area:.4e}  "
              f"({current_void_area/total_domain_area:.4f} of domain)")

        # ---- Porous grains (2D) ----
        print("\nPlacing porous grains (2D)...")
        sigma_ln_p   = math.sqrt(math.log(1+rad_dev**2))
        mu_ln_p      = math.log(img_r(mean_rad_porous)) - 1.5*sigma_ln_p**2
        porous_radii = _generate_radii(vol_percent_porous*2, mu_ln_p, sigma_ln_p, 2, rng)
        print(f"  Pre-generated {len(porous_radii)} candidate porous radii")
        circles_before_porous = len(circles)
        porous_area   = place_grains_2d(porous_radii, vol_percent_porous, "porous")
        porous_circles = list(circles[circles_before_porous:])
        print(f"  Porous area fraction: {porous_area:.4f}")

        save_xyzr(circles, AP_xyzr, img_size, physical_size)
        if mwd_tolerance is not None:
            if using_existing_geometry:
                existing_count = len(load_xyzr(existing_ap_xyzr, physical_size, img_size, dim=2))
                radii = np.array([r for (x, y, r) in circles[existing_count:]])
            else:
                radii = np.array([r for (x, y, r) in circles])
            mwd_actual = mean_weight_diameter(AP_xyzr, radii=radii) * (physical_size/img_size)
            print(f"\nMWD check: {mwd_actual:.4e} m (target {mwd_target:.4e} m)")
            if abs(mwd_actual - mwd_target) > mwd_tolerance:
                print("MWD out of tolerance - skipping void placement.")
                return None

        # ---- Void placement (2D) ----
        if pore_placement == "htpb_only":
            print(f"\nPlacing voids in HTPB binder space (htpb_only, mode={void_fraction_mode})...")
            ref_pr = img_r(mean_rad_porous)
            all_grains_arr_2d = circles_dyn.arr

            void_grid_2d = [[[] for _ in range(n_cells)] for __ in range(n_cells)]
            for vi, (vx_, vy_, vr_) in enumerate(voids):
                vcx, vcy = cell_coords_2d(vx_, vy_)
                void_grid_2d[vcx][vcy].append(vi)

            # Build KD-tree for grain exclusion
            if len(all_grains_arr_2d) > 0:
                grain_kdt_2d = cKDTree(all_grains_arr_2d[:, :2])
                gr2 = all_grains_arr_2d[:, 2]
            else:
                grain_kdt_2d = None; gr2 = np.empty(0)

            void_pts_2d  = [(v[0], v[1]) for v in voids]
            void_kdt_2d  = cKDTree(void_pts_2d) if void_pts_2d else None
            void_rs_2d   = [v[2] for v in voids]

            n_batch = 512
            consecutive_failures = 0
            max_consecutive = 200

            while current_void_area < target_void_area:
                if consecutive_failures >= max_consecutive:
                    print("WARNING: could not place more htpb voids without overlap.")
                    break
                pore_r = float(np.clip(
                    rng.lognormal(math.log(ref_pr*pore_radius_factor), 0.4),
                    0.1*ref_pr, 0.35*ref_pr))
                full_circle_area = math.pi * pore_r**2
                remaining = target_void_area - current_void_area
                if full_circle_area > remaining and void_fraction_mode == "unclipped":
                    break
                vxs = rng.uniform(pore_r, img_size-pore_r, n_batch)
                vys = rng.uniform(pore_r, img_size-pore_r, n_batch)

                # Grain exclusion via KD-tree
                if grain_kdt_2d is not None:
                    cands_xy = np.column_stack([vxs, vys])
                    max_gr   = gr2.max()
                    hits_g   = grain_kdt_2d.query_ball_point(cands_xy, pore_r + max_gr)
                    vmask = np.ones(n_batch, dtype=bool)
                    for i, hg in enumerate(hits_g):
                        if not hg: continue
                        nb = all_grains_arr_2d[hg]
                        dx = vxs[i]-nb[:, 0]; dy = vys[i]-nb[:, 1]
                        if np.any(dx**2+dy**2 < nb[:, 2]**2):
                            vmask[i] = False
                else:
                    vmask = np.ones(n_batch, dtype=bool)

                valid_idx = np.where(vmask)[0]
                if len(valid_idx) == 0:
                    consecutive_failures += 1; continue

                # Void self-overlap via KD-tree
                if void_kdt_2d is not None and void_rs_2d:
                    max_vr = max(void_rs_2d)
                    cands_v = np.column_stack([vxs[valid_idx], vys[valid_idx]])
                    hits_v  = void_kdt_2d.query_ball_point(cands_v, pore_r + max_vr)
                    keep = []
                    for i, hv in enumerate(hits_v):
                        if not hv:
                            keep.append(valid_idx[i]); continue
                        nv = np.array([(voids[h][0], voids[h][1], voids[h][2]) for h in hv])
                        dx = vxs[valid_idx[i]]-nv[:, 0]; dy = vys[valid_idx[i]]-nv[:, 1]
                        if not np.any(dx**2+dy**2 < (pore_r+nv[:, 2])**2):
                            keep.append(valid_idx[i])
                    valid_idx = np.array(keep, dtype=int)

                if len(valid_idx) == 0:
                    consecutive_failures += 1; continue

                if void_fraction_mode == "unclipped":
                    if full_circle_area > remaining: break
                    pick = int(valid_idx[0]); counted_area = full_circle_area
                else:
                    if grain_kdt_2d is not None:
                        vx_v = vxs[valid_idx]; vy_v = vys[valid_idx]
                        cands_v2 = np.column_stack([vx_v, vy_v])
                        hits2 = grain_kdt_2d.query_ball_point(cands_v2, pore_r + gr2.max())
                        counted_areas = np.full(len(valid_idx), full_circle_area)
                        for i, hg in enumerate(hits2):
                            if not hg: continue
                            nb = all_grains_arr_2d[hg]
                            dx = vx_v[i]-nb[:, 0]; dy = vy_v[i]-nb[:, 1]
                            d_v  = np.sqrt(dx**2 + dy**2)
                            grs  = nb[:, 2]
                            over = d_v < pore_r + grs
                            if not np.any(over): continue
                            d_safe = np.where(over, np.maximum(d_v, 1e-30), 1.0)
                            h1  = (pore_r**2 - grs**2 + d_v**2) / (2.0*d_safe)
                            h1c = np.maximum(pore_r - h1,    0.0)
                            h2c = np.maximum(grs - (d_v-h1), 0.0)
                            seg_void  = (pore_r**2 * np.arccos(np.clip((pore_r-h1c)/pore_r, -1, 1))
                                         - (pore_r-h1c)*np.sqrt(np.maximum(2*pore_r*h1c-h1c**2, 0)))
                            seg_grain = (grs**2 * np.arccos(np.clip((grs-h2c)/np.maximum(grs, 1e-30), -1, 1))
                                         - (grs-h2c)*np.sqrt(np.maximum(2*grs*h2c-h2c**2, 0)))
                            lens = seg_void + seg_grain
                            fully = d_v <= np.abs(pore_r - grs)
                            small_r = np.minimum(pore_r, grs)
                            intersection = np.where(~over, 0.0,
                                                    np.where(fully, math.pi*small_r**2, lens))
                            counted_areas[i] = max(0.0, full_circle_area - intersection.sum())
                    else:
                        counted_areas = np.full(len(valid_idx), full_circle_area)
                    fits = np.where(counted_areas <= remaining)[0]
                    if len(fits) == 0:
                        consecutive_failures += 1; continue
                    pick = int(valid_idx[fits[0]]); counted_area = float(counted_areas[fits[0]])

                vx_new = float(vxs[pick]); vy_new = float(vys[pick])
                voids.append((vx_new, vy_new, pore_r))
                void_pts_2d.append((vx_new, vy_new))
                void_rs_2d.append(pore_r)
                # Rebuild KD-tree periodically
                if len(voids) % 200 == 0:
                    void_kdt_2d = cKDTree(void_pts_2d)
                vi_new = len(voids)-1
                vcx, vcy = cell_coords_2d(vx_new, vy_new)
                void_grid_2d[vcx][vcy].append(vi_new)
                current_void_area += counted_area
                consecutive_failures = 0
                if len(voids) % 1000 == 0:
                    print(f"  Voids placed: {len(voids)}  "
                          f"fraction of domain: {current_void_area/total_domain_area:.4f}")

        else:
            print(f"\nPlacing voids within porous grains (parallel, "
                  f"mode={pore_placement}, vf_mode={void_fraction_mode})...")
            if vol_percent_porous > 0 and len(porous_circles) > 0:
                n_porous         = len(porous_circles)
                grain_areas      = np.array([math.pi*g[2]**2 for g in porous_circles])
                total_grain_area = grain_areas.sum()
                void_budget      = target_void_area - current_void_area
                per_grain_caps   = MAX_VOID_FRACTION_PER_GRAIN * grain_areas
                per_grain_placed = np.zeros(n_porous)
                per_grain_budgets = np.minimum(
                    void_budget*(grain_areas/total_grain_area), per_grain_caps)

                void_snap_base = list(voids)
                futures_map    = {}
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for idx, grain in enumerate(porous_circles):
                        seed = int(rng.integers(0, 2**31))
                        fut  = executor.submit(_place_voids_in_grain_2d, grain,
                                               list(void_snap_base),
                                               per_grain_budgets[idx], seed)
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

                print(f"  Void fraction after parallel pass: "
                      f"{current_void_area/total_domain_area:.4f}")

                exhausted_grains = set(
                    i for i in range(n_porous)
                    if per_grain_placed[i] >= per_grain_caps[i])

                for _ in range(20):
                    if current_void_area >= target_void_area: break
                    active = [i for i in range(n_porous) if i not in exhausted_grains]
                    if not active: break
                    remaining_total = target_void_area - current_void_area
                    active_areas    = grain_areas[active]
                    redistrib       = remaining_total*(active_areas/active_areas.sum())
                    progress = False
                    for k, grain_idx in enumerate(active):
                        if current_void_area >= target_void_area: break
                        remaining_cap = per_grain_caps[grain_idx]-per_grain_placed[grain_idx]
                        budget = min(redistrib[k], remaining_cap)
                        if budget <= 0:
                            exhausted_grains.add(grain_idx); continue
                        new_voids, vol_added = _place_voids_in_grain_2d(
                            porous_circles[grain_idx], list(voids), budget,
                            int(rng.integers(0, 2**31)))
                        voids.extend(new_voids)
                        current_void_area += vol_added
                        per_grain_placed[grain_idx] += vol_added
                        if vol_added > 0: progress = True
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
        print(f"{'='*60}\n")

        if pore_placement != "htpb_only" and vol_percent_porous > 0 and len(porous_circles) > 0:
            fracs = per_grain_placed / grain_areas
            print(f"  Per-grain void fraction - min: {fracs.min():.3f}  "
                  f"max: {fracs.max():.3f}  mean: {fracs.mean():.3f}  "
                  f"std: {fracs.std():.3f}")
            over = np.sum(fracs > MAX_VOID_FRACTION_PER_GRAIN + 1e-9)
            if over:
                print(f"  WARNING: {over} grain(s) exceed the "
                      f"{MAX_VOID_FRACTION_PER_GRAIN*100:.0f}% cap!")
            else:
                print(f"  All grains within {MAX_VOID_FRACTION_PER_GRAIN*100:.0f}% cap. OK")

        save_xyzr(voids, void_xyzr, img_size, physical_size)
        return void_frac_domain

    # ==================================================================
    # 3-D branch
    # ==================================================================
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
        print(f"  Pore radius factor:    {pore_radius_factor}")
        print(f"{'='*60}\n")

        def img_r(mu):
            return mu / physical_size * img_size

        cell_size = img_r(mean_rad_solid) * 4
        n_cells   = max(1, int(math.ceil(img_size / cell_size)))
        grid      = [[[[] for _ in range(n_cells)]
                      for __ in range(n_cells)]
                     for ___ in range(n_cells)]
        cell_counts = np.zeros((n_cells, n_cells, n_cells), dtype=np.int32)

        def cell_coords_3d(x, y, z):
            return (max(0, min(n_cells-1, int(x // cell_size))),
                    max(0, min(n_cells-1, int(y // cell_size))),
                    max(0, min(n_cells-1, int(z // cell_size))))

        def clipped_sphere_volume_local(x, y, z, r, lo=0, hi=1):
            def cap(h):
                if h <= 0: return 0.0
                if h >= 2*r: return 4/3*math.pi*r**3
                return math.pi*h**2*(3*r-h)/3
            V  = 4/3*math.pi*r**3
            V -= cap(r-(x-lo)); V -= cap(r-(hi-x))
            V -= cap(r-(y-lo)); V -= cap(r-(hi-y))
            V -= cap(r-(z-lo)); V -= cap(r-(hi-z))
            return V

        def fully_inside(x, y, z, r):
            return (x>r and x<img_size-r and y>r and y<img_size-r
                    and z>r and z<img_size-r)

        def _dynamic_cell_cap_3d(r):
            grain_vol = 4/3*math.pi*r**3
            if grain_vol <= 0: return 40
            return max(40, int((cell_size**3 / grain_vol) * 0.64 * 1.5))

        spheres        = []
        porous_spheres = []
        voids          = []
        solid_vol = hollow_vol = porous_vol = 0.0
        sigma_ln  = math.sqrt(math.log(1+rad_dev**2))

        # OPT: DynamicArray instead of repeated vstack
        spheres_dyn = DynamicArray(4)
        using_existing_geometry = existing_ap_xyzr is not None

        if using_existing_geometry:
            print("Loading existing AP geometry...")
            spheres = load_xyzr(existing_ap_xyzr, physical_size, img_size, dim=3)
            porous_spheres, _ = classify_loaded_particles(
                spheres, dim=3, mean_rad_porous=mean_rad_porous,
                physical_size=physical_size, img_size=img_size)
            for idx, (x, y, z, r) in enumerate(spheres):
                ci, cj, ck = cell_coords_3d(x, y, z)
                grid[ci][cj][ck].append(idx)
                cell_counts[ci, cj, ck] += 1
                spheres_dyn.append([x, y, z, r])
                vol = (4/3*math.pi*r**3 if fully_inside(x, y, z, r)
                       else clipped_sphere_volume_local(x, y, z, r))
                solid_vol += vol
            print(f"  Loaded grains: {len(spheres)}")
            print(f"  Porous grains: {len(porous_spheres)}")
        else:
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
                    ci, cj, ck = cell_coords_3d(x, y, z)
                    if cell_counts[ci, cj, ck] > 20: continue
                    if find_first_valid_3d(np.array([x]), np.array([y]), np.array([z]),
                                           r, spheres_dyn.arr, 0.98,
                                           grid, cell_size, n_cells, cell_counts) is not None:
                        spheres.append((x, y, z, r))
                        spheres_dyn.append([x, y, z, r])
                        grid[ci][cj][ck].append(len(spheres)-1)
                        cell_counts[ci, cj, ck] += 1
                        solid_vol += (4/3*math.pi*r**3 if fully_inside(x, y, z, r)
                                      else clipped_sphere_volume_local(x, y, z, r))
                        break
            print(f"  Solid volume fraction: {solid_vol:.4f}")

        # ------------------------------------------------------------------
        # Grain-placement loop (3D)
        # ------------------------------------------------------------------
        def place_grains_3d(radii_list, vol_target, vol_frac_name):
            nonlocal spheres, cell_counts
            acc_frac     = 0.0
            recent_costs = deque()
            stalled      = False
            placed_count = 0

            def _guided_pass_3d(r, cap, n_rounds):
                for _ in range(n_rounds):
                    cx_arr, cy_arr, cz_arr = guided_candidates_3d(
                        spheres_dyn.arr, r, rng,
                        n_cands=GUIDED_BATCH, margin=margin, img_size=img_size)
                    bm = ((cx_arr >= -margin+r) & (cx_arr <= img_size+margin-r) &
                          (cy_arr >= -margin+r) & (cy_arr <= img_size+margin-r) &
                          (cz_arr >= -margin+r) & (cz_arr <= img_size+margin-r))
                    cx_arr = cx_arr[bm]; cy_arr = cy_arr[bm]; cz_arr = cz_arr[bm]
                    if len(cx_arr) == 0: continue
                    ci_arr = np.clip((cx_arr//cell_size).astype(int), 0, n_cells-1)
                    cj_arr = np.clip((cy_arr//cell_size).astype(int), 0, n_cells-1)
                    ck_arr = np.clip((cz_arr//cell_size).astype(int), 0, n_cells-1)
                    cf = cell_counts[ci_arr, cj_arr, ck_arr] <= cap
                    cx_arr = cx_arr[cf]; cy_arr = cy_arr[cf]; cz_arr = cz_arr[cf]
                    if len(cx_arr) == 0: continue
                    idx = find_first_valid_3d(cx_arr, cy_arr, cz_arr, r, spheres_dyn.arr,
                                              grid=grid, cell_size=cell_size,
                                              n_cells=n_cells, cell_counts=cell_counts)
                    if idx is not None:
                        return float(cx_arr[idx]), float(cy_arr[idx]), float(cz_arr[idx])
                return None

            for r in radii_list:
                if acc_frac >= vol_target: break
                grain_vol_frac = (4/3*math.pi*r**3) / total_domain_vol
                if acc_frac + grain_vol_frac > vol_target*1.05: continue

                max_pos    = adaptive_max_pos(acc_frac, vol_target)
                use_guided = (acc_frac >= GUIDED_ONSET * vol_target) or stalled
                cap        = _dynamic_cell_cap_3d(r)
                attempt_count = 0
                result        = None

                if use_guided:
                    n_rounds = max(1, max_pos // GUIDED_BATCH)
                    result   = _guided_pass_3d(r, cap, n_rounds)
                    attempt_count += n_rounds * GUIDED_BATCH

                    if result is None and not stalled:
                        for _ in range(min(200, max_pos // 4)):
                            attempt_count += 1
                            x = rng.uniform(-margin+r, img_size+margin-r)
                            y = rng.uniform(-margin+r, img_size+margin-r)
                            z = rng.uniform(-margin+r, img_size+margin-r)
                            ci, cj, ck = cell_coords_3d(x, y, z)
                            if cell_counts[ci, cj, ck] > cap: continue
                            if find_first_valid_3d(np.array([x]), np.array([y]),
                                                   np.array([z]), r, spheres_dyn.arr, 0.98,
                                                   grid, cell_size, n_cells, cell_counts) is not None:
                                result = (x, y, z); break

                    if result is None and stalled:
                        boosted = STALL_BOOST_MULT * (MAX_ATTEMPTS_CAP // GUIDED_BATCH)
                        print(f"  [{vol_frac_name}] Boosted recovery "
                              f"({boosted} rounds) for r={r:.5f}...")
                        result = _guided_pass_3d(r, cap, boosted)
                        attempt_count += boosted * GUIDED_BATCH
                        if result is None:
                            print(f"  [{vol_frac_name}] Recovery failed for "
                                  f"r={r:.5f} — skipping grain.")
                else:
                    for _ in range(max_pos):
                        attempt_count += 1
                        x = rng.uniform(-margin+r, img_size+margin-r)
                        y = rng.uniform(-margin+r, img_size+margin-r)
                        z = rng.uniform(-margin+r, img_size+margin-r)
                        ci, cj, ck = cell_coords_3d(x, y, z)
                        if cell_counts[ci, cj, ck] > cap: continue
                        if find_first_valid_3d(np.array([x]), np.array([y]),
                                               np.array([z]), r, spheres_dyn.arr, 0.98,
                                               grid, cell_size, n_cells, cell_counts) is not None:
                            result = (x, y, z); break

                if result is not None:
                    x, y, z = result
                    spheres.append((x, y, z, r))
                    spheres_dyn.append([x, y, z, r])
                    ci, cj, ck = cell_coords_3d(x, y, z)
                    grid[ci][cj][ck].append(len(spheres)-1)
                    cell_counts[ci, cj, ck] += 1
                    v = (4/3*math.pi*r**3 if fully_inside(x, y, z, r)
                         else clipped_sphere_volume_local(x, y, z, r))
                    acc_frac += v / total_domain_vol

                    placed_count += 1
                    if placed_count % 1000 == 0:
                        print(f"  [{vol_frac_name}] Placed {placed_count} grains  "
                              f"vol_frac={acc_frac:.4f}")

                    recent_costs.append(attempt_count)
                    if len(recent_costs) > STALL_WINDOW:
                        recent_costs.popleft()
                    if (len(recent_costs) == STALL_WINDOW and
                            np.mean(recent_costs) > STALL_THRESHOLD):
                        if not stalled:
                            print(f"  [{vol_frac_name}] Stall detected at "
                                  f"{acc_frac:.4f} — switching to boosted guided-only.")
                            stalled = True
                    elif stalled and attempt_count < STALL_THRESHOLD // 2:
                        stalled = False
                        print(f"  [{vol_frac_name}] Stall cleared at {acc_frac:.4f}.")

            return acc_frac

        # ---- Hollow grains (3D) ----
        print("\nPlacing hollow grains (3D)...")
        sigma_ln_h   = math.sqrt(math.log(1+rad_dev**2))
        mu_ln_h      = math.log(img_r(mean_rad_hollow)) - 1.5*sigma_ln_h**2
        hollow_radii = _generate_radii(vol_percent_hollow*2, mu_ln_h, sigma_ln_h, 3, rng)
        print(f"  Pre-generated {len(hollow_radii)} candidate hollow radii")
        spheres_before_hollow = len(spheres)
        hollow_vol = place_grains_3d(hollow_radii, vol_percent_hollow, "hollow")
        hollow_candidates = spheres[spheres_before_hollow:]

        if len(hollow_candidates) > 0:
            total_candidate_vol = sum(
                (4/3*math.pi*r**3 if fully_inside(x, y, z, r)
                 else clipped_sphere_volume_local(x, y, z, r))
                for (x, y, z, r) in hollow_candidates)
            actual_hollow_frac = total_candidate_vol / total_domain_vol
            rv_r_ratio = ((void_fraction / actual_hollow_frac)**(1/3)
                          if actual_hollow_frac > 0 else 0)
            print(f"  Actual hollow frac: {actual_hollow_frac:.4f}")
            print(f"  rv / r ratio:       {rv_r_ratio:.4f}")
            for (x, y, z, r) in hollow_candidates:
                rv = r * rv_r_ratio
                voids.append((x, y, z, rv))
                current_void_vol += 4/3*math.pi*rv**3

        print(f"  Hollow volume fraction: {hollow_vol:.4f}")
        print(f"  Void fraction so far:   {current_void_vol/total_domain_vol:.4f}")

        # ---- Porous grains (3D) ----
        print("\nPlacing porous grains (3D)...")
        sigma_ln_p   = math.sqrt(math.log(1+rad_dev**2))
        mu_ln_p      = math.log(img_r(mean_rad_porous)) - 1.5*sigma_ln_p**2
        porous_radii = _generate_radii(vol_percent_porous*2, mu_ln_p, sigma_ln_p, 3, rng)
        print(f"  Pre-generated {len(porous_radii)} candidate porous radii")
        spheres_before_porous = len(spheres)
        porous_vol = place_grains_3d(porous_radii, vol_percent_porous, "porous")
        porous_spheres = list(spheres[spheres_before_porous:])
        print(f"  Porous volume fraction: {porous_vol:.4f}")

        # ---- MWD check ----
        save_xyzr(spheres, AP_xyzr, img_size, physical_size)
        if mwd_tolerance is not None:
            if using_existing_geometry:
                existing_count = len(load_xyzr(existing_ap_xyzr, physical_size, img_size, dim=3))
                radii = np.array([r for (x, y, z, r) in spheres[existing_count:]])
            else:
                radii = np.array([r for (x, y, z, r) in spheres])
            mwd_actual = mean_weight_diameter(AP_xyzr, radii=radii) * (physical_size/img_size)
            print(f"\nMWD check: {mwd_actual:.4e} m (target {mwd_target:.4e} m)")
            if abs(mwd_actual - mwd_target) > mwd_tolerance:
                print("MWD out of tolerance - skipping void placement.")
                return None

        # ---- Void placement (3D) ----
        try:
            if pore_placement == "htpb_only":
                print(f"\nPlacing voids in HTPB binder space (htpb_only, mode={void_fraction_mode})...")
                ref_pr            = img_r(mean_rad_porous)
                all_grains_arr_3d = spheres_dyn.arr

                void_grid_3d = [[[[] for _ in range(n_cells)]
                                 for __ in range(n_cells)]
                                for ___ in range(n_cells)]
                for vi, (vx_, vy_, vz_, vr_) in enumerate(voids):
                    vci, vcj, vck = cell_coords_3d(vx_, vy_, vz_)
                    void_grid_3d[vci][vcj][vck].append(vi)

                # KD-trees for grains and voids
                if len(all_grains_arr_3d) > 0:
                    grain_kdt_3d = cKDTree(all_grains_arr_3d[:, :3])
                    gr3 = all_grains_arr_3d[:, 3]
                else:
                    grain_kdt_3d = None; gr3 = np.empty(0)

                void_pts_3d = [(v[0], v[1], v[2]) for v in voids]
                void_kdt_3d = cKDTree(void_pts_3d) if void_pts_3d else None
                void_rs_3d  = [v[3] for v in voids]

                n_batch = 512
                consecutive_failures = 0
                max_consecutive = 200

                while current_void_vol < target_void_vol:
                    if consecutive_failures >= max_consecutive:
                        print("WARNING: could not place more htpb voids without overlap.")
                        break
                    pore_r = float(np.clip(
                        rng.lognormal(math.log(ref_pr*pore_radius_factor), 0.4),
                        0.1*ref_pr, 0.35*ref_pr))
                    full_sphere_vol = 4/3*math.pi*pore_r**3
                    remaining = target_void_vol - current_void_vol
                    if full_sphere_vol > remaining and void_fraction_mode == "unclipped":
                        break
                    vxs = rng.uniform(pore_r, img_size-pore_r, n_batch)
                    vys = rng.uniform(pore_r, img_size-pore_r, n_batch)
                    vzs = rng.uniform(pore_r, img_size-pore_r, n_batch)

                    # Grain exclusion via KD-tree
                    if grain_kdt_3d is not None:
                        cands_xyz = np.column_stack([vxs, vys, vzs])
                        max_gr    = gr3.max()
                        hits_g    = grain_kdt_3d.query_ball_point(cands_xyz, pore_r + max_gr)
                        vmask = np.ones(n_batch, dtype=bool)
                        for i, hg in enumerate(hits_g):
                            if not hg: continue
                            nb = all_grains_arr_3d[hg]
                            dx = vxs[i]-nb[:, 0]; dy = vys[i]-nb[:, 1]; dz = vzs[i]-nb[:, 2]
                            if np.any(dx**2+dy**2+dz**2 < nb[:, 3]**2):
                                vmask[i] = False
                    else:
                        vmask = np.ones(n_batch, dtype=bool)

                    valid_idx = np.where(vmask)[0]
                    if len(valid_idx) == 0:
                        consecutive_failures += 1; continue

                    # Void self-overlap via KD-tree
                    if void_kdt_3d is not None and void_rs_3d:
                        max_vr   = max(void_rs_3d)
                        cands_v  = np.column_stack([vxs[valid_idx], vys[valid_idx], vzs[valid_idx]])
                        hits_v   = void_kdt_3d.query_ball_point(cands_v, pore_r + max_vr)
                        keep = []
                        for i, hv in enumerate(hits_v):
                            if not hv:
                                keep.append(valid_idx[i]); continue
                            nv = np.array([(voids[h][0], voids[h][1], voids[h][2], voids[h][3]) for h in hv])
                            dx = vxs[valid_idx[i]]-nv[:, 0]
                            dy = vys[valid_idx[i]]-nv[:, 1]
                            dz = vzs[valid_idx[i]]-nv[:, 2]
                            if not np.any(dx**2+dy**2+dz**2 < (pore_r+nv[:, 3])**2):
                                keep.append(valid_idx[i])
                        valid_idx = np.array(keep, dtype=int)

                    if len(valid_idx) == 0:
                        consecutive_failures += 1; continue

                    if void_fraction_mode == "unclipped":
                        if full_sphere_vol > remaining: break
                        pick = int(valid_idx[0]); counted_vol = full_sphere_vol
                    else:
                        if grain_kdt_3d is not None:
                            vx_v = vxs[valid_idx]; vy_v = vys[valid_idx]; vz_v = vzs[valid_idx]
                            cands_v3 = np.column_stack([vx_v, vy_v, vz_v])
                            hits2 = grain_kdt_3d.query_ball_point(cands_v3, pore_r + gr3.max())
                            counted_vols = np.full(len(valid_idx), full_sphere_vol)
                            for i, hg in enumerate(hits2):
                                if not hg: continue
                                nb  = all_grains_arr_3d[hg]
                                dx  = vx_v[i]-nb[:, 0]; dy = vy_v[i]-nb[:, 1]; dz = vz_v[i]-nb[:, 2]
                                d_v = np.sqrt(dx**2+dy**2+dz**2)
                                grs = nb[:, 3]
                                over = d_v < pore_r + grs
                                if not np.any(over): continue
                                d_safe = np.where(over, np.maximum(d_v, 1e-30), 1.0)
                                h1  = (pore_r**2-grs**2+d_v**2)/(2.0*d_safe)
                                h1c = np.maximum(pore_r-h1,    0.0)
                                h2c = np.maximum(grs-(d_v-h1), 0.0)
                                lens = math.pi*h1c**2*(3*pore_r-h1c)/3 + math.pi*h2c**2*(3*grs-h2c)/3
                                fully = d_v <= np.abs(pore_r-grs)
                                small_r = np.minimum(pore_r, grs)
                                intersection = np.where(~over, 0.0,
                                                        np.where(fully, 4/3*math.pi*small_r**3, lens))
                                counted_vols[i] = max(0.0, full_sphere_vol - intersection.sum())
                        else:
                            counted_vols = np.full(len(valid_idx), full_sphere_vol)
                        fits = np.where(counted_vols <= remaining)[0]
                        if len(fits) == 0:
                            consecutive_failures += 1; continue
                        pick = int(valid_idx[fits[0]]); counted_vol = float(counted_vols[fits[0]])

                    vx_new = float(vxs[pick]); vy_new = float(vys[pick]); vz_new = float(vzs[pick])
                    voids.append((vx_new, vy_new, vz_new, pore_r))
                    void_pts_3d.append((vx_new, vy_new, vz_new))
                    void_rs_3d.append(pore_r)
                    if len(voids) % 200 == 0:
                        void_kdt_3d = cKDTree(void_pts_3d)
                    vci, vcj, vck = cell_coords_3d(vx_new, vy_new, vz_new)
                    void_grid_3d[vci][vcj][vck].append(len(voids)-1)
                    current_void_vol += counted_vol
                    consecutive_failures = 0
                    if len(voids) % 1000 == 0:
                        print(f"  Voids placed: {len(voids)}  "
                              f"fraction of domain: {current_void_vol/total_domain_vol:.4f}")

            else:
                print(f"\nPlacing voids within porous grains (parallel, "
                      f"ProcessPoolExecutor, mode={pore_placement}, vf_mode={void_fraction_mode})...")
                if vol_percent_porous > 0 and len(porous_spheres) > 0:
                    n_porous         = len(porous_spheres)
                    print(f"  Porous grains: {n_porous}")
                    grain_vols       = np.array([4/3*math.pi*g[3]**3 for g in porous_spheres])
                    total_grain_vol  = grain_vols.sum()
                    void_budget      = target_void_vol - current_void_vol
                    per_grain_caps   = MAX_VOID_FRACTION_PER_GRAIN * grain_vols
                    per_grain_placed = np.zeros(n_porous)
                    per_grain_budgets = np.minimum(
                        void_budget*(grain_vols/total_grain_vol), per_grain_caps)
                    exhausted_grains = set()
                    all_grains_arr   = spheres_dyn.arr

                    futures_map = {}
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        for grain_idx, grain in enumerate(porous_spheres):
                            if current_void_vol >= target_void_vol: break
                            seed = int(rng.integers(0, 2**31))
                            fut  = executor.submit(
                                _place_voids_in_grain_fast,
                                grain, per_grain_budgets[grain_idx],
                                seed, img_size, pore_placement, void_fraction_mode,
                                all_grains_arr, pore_radius_factor, 3000, 500)
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
                                print(f"  Voids placed: {len(voids)}  "
                                      f"fraction of domain: "
                                      f"{current_void_vol/total_domain_vol:.4f} / {void_fraction:.4f}")

                    print(f"  Void fraction after parallel pass: "
                          f"{current_void_vol/total_domain_vol:.4f}")

                    no_progress_rounds = 0
                    for _ in range(50):
                        if current_void_vol >= target_void_vol: break
                        active = [i for i in range(n_porous) if i not in exhausted_grains]
                        if not active: break
                        remaining_total = target_void_vol - current_void_vol
                        active_vols = grain_vols[active]
                        redistrib = remaining_total*(active_vols/active_vols.sum())
                        order = sorted(range(len(active)),
                                       key=lambda k: grain_vols[active[k]], reverse=True)
                        progress = False
                        for k in order:
                            grain_idx = active[k]
                            if current_void_vol >= target_void_vol: break
                            remaining_cap = per_grain_caps[grain_idx] - per_grain_placed[grain_idx]
                            budget = min(redistrib[k], remaining_cap,
                                         target_void_vol - current_void_vol)
                            if budget <= 0:
                                exhausted_grains.add(grain_idx); continue
                            new_voids, vol_added = _place_voids_in_grain_fast(
                                porous_spheres[grain_idx], budget,
                                int(rng.integers(0, 2**31)), img_size=img_size,
                                pore_placement=pore_placement,
                                void_fraction_mode=void_fraction_mode,
                                all_grains_arr=all_grains_arr,
                                pore_radius_factor=pore_radius_factor,
                                max_iter=3000, max_consecutive=500)
                            voids.extend(new_voids)
                            current_void_vol += vol_added
                            per_grain_placed[grain_idx] += vol_added
                            if vol_added > 0: progress = True
                            if per_grain_placed[grain_idx] >= per_grain_caps[grain_idx]:
                                exhausted_grains.add(grain_idx)
                        if not progress:
                            no_progress_rounds += 1
                            if no_progress_rounds >= 3:
                                print("WARNING: could not place more voids without overlap.")
                                break
                        else:
                            no_progress_rounds = 0

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught — saving partial results...")
        finally:
            save_xyzr(spheres, AP_xyzr, img_size, physical_size)
            save_xyzr(voids, void_xyzr, img_size, physical_size)
            print(f"  Partial AP saved:   {AP_xyzr}  ({len(spheres)} grains)")
            print(f"  Partial void saved: {void_xyzr}  ({len(voids)} voids)")

        void_frac_domain = current_void_vol / total_domain_vol
        print(f"\n{'='*60}")
        print("FINAL RESULTS (3D)")
        print(f"  Total grains:            {len(spheres)}")
        print(f"  AP volume fraction:      {solid_vol+hollow_vol+porous_vol:.4f}")
        print(f"  Void fraction (domain):  {void_frac_domain:.4f}  (target: {void_fraction:.4f})")
        print(f"  Error:                   {abs(void_frac_domain-void_fraction):.2e}")
        print(f"{'='*60}\n")

        if pore_placement != "htpb_only" and vol_percent_porous > 0 and len(porous_spheres) > 0:
            fracs = per_grain_placed / grain_vols
            print(f"  Per-grain void fraction - min: {fracs.min():.3f}  "
                  f"max: {fracs.max():.3f}  mean: {fracs.mean():.3f}  "
                  f"std: {fracs.std():.3f}")
            over = np.sum(fracs > MAX_VOID_FRACTION_PER_GRAIN + 1e-9)
            if over:
                print(f"  WARNING: {over} grain(s) exceed the "
                      f"{MAX_VOID_FRACTION_PER_GRAIN*100:.0f}% cap!")
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
    V -= cap(r-(x-lo));  V -= cap(r-(hi-x))
    V -= cap(r-(y-lo));  V -= cap(r-(hi-y))
    V -= cap(r-(z-lo));  V -= cap(r-(hi-z))
    return max(V, 0.0)


def compute_ap_volume_fraction_clipped(xyzr_path, physical_size, img_size=1.0, dim=3):
    data = np.loadtxt(xyzr_path)
    if data.ndim == 1:
        data = data[None, :]
    total_vol = 0.0
    if dim == 2:
        total_domain = physical_size ** 2
        for row in data:
            if len(row) < 4: continue
            x, y, _, r = row[:4]
            total_vol += math.pi * r**2
        return total_vol / total_domain
    else:
        total_domain = physical_size ** 3
        for row in data:
            if len(row) < 4: continue
            x, y, z, r = row[:4]
            if (x>r and x<physical_size-r and y>r and y<physical_size-r
                    and z>r and z<physical_size-r):
                total_vol += 4/3*math.pi*r**3
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
     pore_radius_factor, existing_ap_xyzr) = args

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

    ap_vol = compute_ap_volume_fraction_clipped(ap_xyzr, physical_size, dim=dim)
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
    existing_ap_xyzr=None,
):
    os.makedirs(subfolder, exist_ok=True)
    if n_workers is None:
        n_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)

    common = dict(
        physical_size=physical_size, rad_dev=rad_dev, max_attempts=max_attempts,
        vol_percent_solid=vol_percent_solid, vol_percent_hollow=vol_percent_hollow,
        vol_percent_porous=vol_percent_porous, void_fraction=void_fraction,
        mwd_target=target_mwd, mwd_tolerance=mwd_tolerance,
        ap_vol_tolerance=ap_vol_tolerance, dim=dim,
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
            name = f"{base_name}_{accepted_idx+len(pending):02d}_tmp{attempt_idx}"
            args = (
                attempt_idx, accepted_idx, name, subfolder,
                common["physical_size"], common["rad_dev"], common["max_attempts"],
                common["vol_percent_solid"], common["vol_percent_hollow"],
                common["vol_percent_porous"], common["void_fraction"],
                common["mwd_target"], common["mwd_tolerance"], common["dim"],
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
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED)
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
                if (len(accepted) < n_target and attempt_idx < max_total_attempts
                        and len(pending) < n_workers):
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
                    pts.append((x/physical_size*img_size, y/physical_size*img_size,
                                0, r/physical_size*img_size))
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
        for (x, y, _, r) in circles:
            ax.add_patch(Circle((x, y), r, facecolor='#FF0000', edgecolor='none',
                                alpha=ap_alpha, zorder=5))
        for (x, y, _, r) in voids:
            ax.add_patch(Circle((x, y), r, facecolor='#0000FF', edgecolor='none', zorder=6))
        fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0.0)
        plt.close(fig)

    elif dim == 3:
        if max_spheres is not None: circles = circles[:max_spheres]
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2*np.pi, sphere_resolution)
        v = np.linspace(0,   np.pi, sphere_resolution)
        u, v = np.meshgrid(u, v)
        for (x0, y0, z0, r) in circles:
            ax.plot_surface(x0+r*np.cos(u)*np.sin(v), y0+r*np.sin(u)*np.sin(v),
                            z0+r*np.cos(v), color='#FF6B6B', linewidth=0, alpha=ap_alpha)
        for (x0, y0, z0, r) in voids:
            ax.plot_surface(x0+r*np.cos(u)*np.sin(v), y0+r*np.sin(u)*np.sin(v),
                            z0+r*np.cos(v), color='#4ECDC4', linewidth=0, alpha=alpha)
        ax.set_xlim(0, img_size); ax.set_ylim(0, img_size); ax.set_zlim(0, img_size)
        ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    generate_structures_with_target_mwd(
        '50x200_domains',
        target_mwd=4.0e-6,
        base_name="nonvoid_200_cubic_4um",
        dim=3,
        physical_size=200e-6,
        mean_rad_porous=2e-6/(1.2*math.exp(math.sqrt(math.log(1+0.4**2))**2)),
        mean_rad_hollow=2.2e-6/(1.2*math.exp(math.sqrt(math.log(1+0.4**2))**2)),
        mean_rad_solid=2.2e-6/(1.2*math.exp(math.sqrt(math.log(1+0.4**2))**2)),
        void_fraction=0.0,
        vol_percent_solid=0.0,
        vol_percent_porous=0.55,
        vol_percent_hollow=0.0,
        mwd_tolerance=0.2e-6,
        n_workers=4,
        pore_placement='ext',
        void_fraction_mode="unclipped",
    )


if __name__ == "__main__":
    main()