import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter
from scipy.interpolate import splprep, splev
from numba import njit, prange
from rasterio.windows import Window
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS (TUNING WORKSPACE)
# ==========================================
CANVAS_W, CANVAS_H = 1200, 800
CROP_RADIUS = 1500        

# Camera Projection
CAMERA_PITCH_DEG = 15.0   
CAMERA_YAW_DEG = 180.0      
EXAGGERATE_Z = 1.0        
DEM_16BIT_MAX = 65535.0   

# Analytical Lighting 
LIGHT_ALTITUDE = 15.0     
LIGHT_AZIMUTH = 135.0     

pitch_rad, yaw_rad = np.radians(CAMERA_PITCH_DEG), np.radians(CAMERA_YAW_DEG)
cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
Y_COMPRESSION = np.tan(pitch_rad) if CAMERA_PITCH_DEG > 0.5 else 0.01

# Aesthetic Styling
UNIFIED_THICKNESS = 0.5  
UNIFIED_COLOR = "#1a1a1a" 
BG_COLOR = "#fbfbf9"      

# Flowline Tracing Parameters
SHADOW_MIN_SPACING = 1.5   
HIGHLIGHT_MAX_SPACING = 3.5

PASS_SPACINGS = [12, 8, 6, 4, 3, 2, 1] 
STEP_SIZE = 0.5          
MAX_STEPS = 3000          
MIN_LINE_LENGTH = 20      
SLOPE_THRESHOLD = 0.001   

# Ridgeline Settings
RIDGE_SMOOTHING_FACTOR = 50.0  
MAX_GAP_PIXELS = 15.0  
DEPTH_TOLERANCE = 8.0  

# DEM-Space Vignette (in raster pixels from peak center)
VIGNETTE_DEM_OUTER_A = 1500   
VIGNETTE_DEM_OUTER_B = 1200    
VIGNETTE_DEM_INNER_RATIO = 0.70  

# Wave Fragmentation
FRAGMENT_FREQUENCY = 0.18   
MIN_FRAGMENT_VERTS = 4      


# ==========================================
# 2. NUMBA-ACCELERATED BATCH FRAGMENTATION ENGINE
# ==========================================
@njit(cache=True, parallel=True)
def batch_fragment_paths(
    flat_screen, flat_dem, line_offsets, line_lengths,
    outer_a, outer_b, inner_ratio, frequency, min_verts, peak_r, peak_c
):
    """
    Processes all geometry segments simultaneously using parallel CPU execution.
    Returns flattened fragmentation results and indexing meta-arrays.
    """
    num_lines = len(line_offsets)
    valid_counts = np.zeros(num_lines, dtype=np.int32)
    
    max_total_pts = len(flat_screen)
    out_frag_screen = np.zeros((max_total_pts, 2), dtype=np.float64)
    out_offsets = np.zeros(max_total_pts, dtype=np.int32)
    out_lengths = np.zeros(max_total_pts, dtype=np.int32)
    
    write_pt_offset = 0
    frag_counter = 0

    for idx in prange(num_lines):
        start = line_offsets[idx]
        length = line_lengths[idx]
        if length < 2:
            continue
            
        arc = np.zeros(length, dtype=np.float64)
        cum_sum = 0.0
        for i in range(1, length):
            dx = flat_screen[start + i, 0] - flat_screen[start + i - 1, 0]
            dy = flat_screen[start + i, 1] - flat_screen[start + i - 1, 1]
            cum_sum += np.sqrt(dx*dx + dy*dy)
            arc[i] = cum_sum

        pen_down = np.zeros(length, dtype=np.bool_)
        phase = np.sin(start) * 3.14159265
        
        for i in range(length):
            dem_x = flat_dem[start + i, 0]
            dem_y = flat_dem[start + i, 1]
            
            dx_p = (dem_x - peak_c) / outer_a
            dy_p = (dem_y - peak_r) / outer_b
            r = np.sqrt(dx_p*dx_p + dy_p*dy_p)
            
            t = (r - inner_ratio) / (1.0 - inner_ratio + 1e-8)
            if t < 0.0: t = 0.0
            if t > 1.0: t = 1.0
            vignette = 1.0 - t * t * (3.0 - 2.0 * t)
            
            threshold = 1.0 - 2.0 * vignette
            wave = np.sin(arc[i] * frequency + phase)
            pen_down[i] = wave >= threshold

        p_start = -1
        for i in range(length):
            if pen_down[i]:
                if p_start == -1:
                    p_start = i
            else:
                if p_start != -1:
                    frag_len = i - p_start
                    if frag_len >= min_verts:
                        pass
                    p_start = -1
                    
    for idx in range(num_lines):
        start = line_offsets[idx]
        length = line_lengths[idx]
        if length < 2: continue
        
        arc_val = 0.0
        phase = np.sin(start) * 3.14159265
        p_start = -1
        
        for i in range(length):
            if i > 0:
                dx = flat_screen[start + i, 0] - flat_screen[start + i - 1, 0]
                dy = flat_screen[start + i, 1] - flat_screen[start + i - 1, 1]
                arc_val += np.sqrt(dx*dx + dy*dy)
                
            dx_p = (flat_dem[start + i, 0] - peak_c) / outer_a
            dy_p = (flat_dem[start + i, 1] - peak_r) / outer_b
            r = np.sqrt(dx_p*dx_p + dy_p*dy_p)
            t = (r - inner_ratio) / (1.0 - inner_ratio + 1e-8)
            if t < 0.0: t = 0.0
            elif t > 1.0: t = 1.0
            vignette = 1.0 - t * t * (3.0 - 2.0 * t)
            
            is_down = np.sin(arc_val * frequency + phase) >= (1.0 - 2.0 * vignette)
            
            if is_down:
                if p_start == -1: p_start = i
            else:
                if p_start != -1:
                    f_len = i - p_start
                    if f_len >= min_verts:
                        out_offsets[frag_counter] = write_pt_offset
                        out_lengths[frag_counter] = f_len
                        out_frag_screen[write_pt_offset : write_pt_offset + f_len] = flat_screen[start + p_start : start + i]
                        write_pt_offset += f_len
                        frag_counter += 1
                    p_start = -1
                    
        if p_start != -1:
            f_len = length - p_start
            if f_len >= min_verts:
                out_offsets[frag_counter] = write_pt_offset
                out_lengths[frag_counter] = f_len
                out_frag_screen[write_pt_offset : write_pt_offset + f_len] = flat_screen[start + p_start : start + length]
                write_pt_offset += f_len
                frag_counter += 1

    return out_frag_screen[:write_pt_offset], out_offsets[:frag_counter], out_lengths[:frag_counter]


# ==========================================
# 3. RASTER CROP & TONE MAP GENERATION
# ==========================================
print("Analyzing raster structure...")
tif_path = '../data/mt_hood/aoi.tif'

with rasterio.open(tif_path) as src:
    overview = src.read(1, out_shape=(1, src.height // 4, src.width // 4))
    low_res_y, low_res_x = np.unravel_index(np.argmax(overview), overview.shape)
    
    crop_window = Window(
        max(0, min(low_res_x * 4 - CROP_RADIUS, src.width - 1)),
        max(0, min(low_res_y * 4 - CROP_RADIUS, src.height - 1)),
        min(CROP_RADIUS * 2, src.width),
        min(CROP_RADIUS * 2, src.height)
    )
    dem = src.read(1, window=crop_window).astype(float)

rows, cols = dem.shape
scale_factor = (CANVAS_W * 0.72) / max(rows, cols)

dem_guide = gaussian_filter(dem, sigma=10.0)      
dem_feature = gaussian_filter(dem, sigma=1.5)    

dy, dx = np.gradient(dem_guide)
slope_map = np.sqrt(dx**2 + dy**2)
mag = np.where(slope_map == 0, 1.0, slope_map)
flow_y, flow_x = -dy / mag, -dx / mag

dy_feat, dx_feat = np.gradient(dem_feature)
slope_magnitude = np.sqrt(dx_feat**2 + dy_feat**2)
aspect_rad = np.arctan2(-dy_feat, dx_feat)
light_az_rad = np.radians((360.0 - LIGHT_AZIMUTH + 90.0) % 360.0)

directional = np.cos(light_az_rad - aspect_rad)
raw_directional_tone = (1.0 - directional) / 2.0

max_slope = np.percentile(slope_magnitude, 98) if np.max(slope_magnitude) > 0 else 1.0
slope_weight = gaussian_filter(np.clip(slope_magnitude / max_slope, 0.0, 1.0), sigma=1.0)

VALLEY_BLENDING_TONE = 0.25 
blended_tone = (slope_weight * raw_directional_tone) + ((1.0 - slope_weight) * VALLEY_BLENDING_TONE)
tone_map = np.clip(blended_tone ** 0.85, 0.0, 1.0)


# ==========================================
# 4. COMPUTE PROJECTION DEPTH BUFFER & PEAK LOCATIONS
# ==========================================
print("Building depth buffer maps...")
rc, cc = np.mgrid[0:rows, 0:cols]
px_flat, py_flat = cc.ravel(), rc.ravel()

xc = (px_flat - cols / 2.0) * scale_factor
yc = (py_flat - rows / 2.0) * scale_factor
horizontal_span = cols * scale_factor
baseline_height_cap = horizontal_span * 0.22

z_normalized = dem_feature.ravel() / DEM_16BIT_MAX
max_true_signal = np.max(z_normalized) if np.max(z_normalized) > 0 else 1.0
zc = (z_normalized / max_true_signal) * baseline_height_cap * EXAGGERATE_Z

rot_xc = xc * cos_y - yc * sin_y
rot_yc = xc * sin_y + yc * cos_y

screen_x = rot_xc + (CANVAS_W / 2.0)
screen_y = (CANVAS_H * 0.50) + (rot_yc * Y_COMPRESSION * sin_p - zc * cos_p)
depth_vals = -(rot_yc * cos_p + zc * sin_p)

sx_idx = np.clip(np.round(screen_x).astype(np.int32), 0, CANVAS_W - 1)
sy_idx = np.clip(np.round(screen_y).astype(np.int32), 0, CANVAS_H - 1)

depth_buffer = np.full((CANVAS_H, CANVAS_W), np.inf, dtype=np.float32)
np.minimum.at(depth_buffer, (sy_idx, sx_idx), depth_vals.astype(np.float32))

filled_depth = minimum_filter(depth_buffer, size=3)
depth_buffer = np.where((depth_buffer == np.inf) & (filled_depth < np.inf), filled_depth, depth_buffer)
depth_buffer = np.where(depth_buffer == np.inf, depth_buffer[depth_buffer < np.inf].max() + 9999.0, depth_buffer)

peak_r_dem, peak_c_dem = np.unravel_index(np.argmax(dem_feature), dem_feature.shape)


# ==========================================
# 5. GENERATE STRUCTURAL RIDGELINES
# ==========================================
print("Extracting spine networks...")
inverted_dem = np.max(dem_guide) - dem_guide
flat_sorted_indices = np.argsort(inverted_dem.ravel())[::-1] 

@njit(cache=True)
def compute_topographic_spines(inv_dem, flat_idx):
    h, w = inv_dem.shape
    acc = np.ones((h, w), dtype=np.float32)
    dr, dc = [-1, -1, -1, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 1, -1, 0, 1]
    down_r, down_c = np.full((h, w), -1, dtype=np.int32), np.full((h, w), -1, dtype=np.int32)
    
    for idx in flat_idx:
        r, c = idx // w, idx % w
        best_slope, best_n = 0.0, -1
        for i in range(8):
            nr, nc = r + dr[i], c + dc[i]
            if 0 <= nr < h and 0 <= nc < w:
                slope = inv_dem[r, c] - inv_dem[nr, nc]
                if slope > best_slope:
                    best_slope, best_n = slope, i
        if best_n != -1:
            nr, nc = r + dr[best_n], c + dc[best_n]
            down_r[r, c], down_c[r, c] = nr, nc
            acc[nr, nc] += acc[r, c]
    return acc, down_r, down_c

acc, down_r, down_c = compute_topographic_spines(inverted_dem, flat_sorted_indices)

@njit(cache=True)
def extract_clean_spine_paths(acc, down_r, down_c, thresh, min_len):
    h, w = acc.shape
    visited = np.zeros((h, w), dtype=np.bool_)
    out_lines = np.zeros((1500, 500, 2), dtype=np.float32)
    out_counts = np.zeros(1500, dtype=np.int32)
    l_idx = 0
    
    for r in range(0, h):
        for c in range(0, w):
            if acc[r, c] >= thresh and not visited[r, c]:
                curr_r, curr_c, pt_cnt = r, c, 0
                while curr_r != -1 and pt_cnt < 500:
                    if visited[curr_r, curr_c] and pt_cnt > 0:
                        out_lines[l_idx, pt_cnt] = [curr_c, curr_r]
                        pt_cnt += 1
                        break
                    visited[curr_r, curr_c] = True
                    out_lines[l_idx, pt_cnt] = [curr_c, curr_r]
                    pt_cnt += 1
                    curr_r, curr_c = down_r[curr_r, curr_c], down_c[curr_r, curr_c]
                    
                if pt_cnt >= min_len and l_idx < 1500:
                    out_counts[l_idx] = pt_cnt
                    l_idx += 1
    return out_lines[:l_idx], out_counts[:l_idx]

out_lines, out_counts = extract_clean_spine_paths(acc, down_r, down_c, float(cols * 1.5), 45)

# Flat buffers tracking geometries for batch Numba acceleration execution loops
ridge_screen_list, ridge_dem_list, ridge_depths = [], [], []

# Matrix used to block flowlines from routing over topographic ridges
shared_occupied_grid = np.zeros((rows, cols), dtype=np.bool_)

for idx in range(len(out_counts)):
    path = out_lines[idx, :out_counts[idx]]
    valid = np.ones(len(path), dtype=bool)
    valid[1:] = np.any(np.diff(path, axis=0) != 0, axis=1)
    path = path[valid]
    if len(path) < 5: continue
        
    try:
        tck, u = splprep([path[:, 0], path[:, 1]], s=RIDGE_SMOOTHING_FACTOR)
        smoothed = np.column_stack(splev(np.linspace(0, 1, max(len(path) * 3, 100)), tck))
    except:
        smoothed = path

    px_r, py_r = smoothed[:, 0], smoothed[:, 1]
    
    # Inject ridgeline traces natively into the exclusion layer before flow computation loops
    for pr, pc in zip(py_r, px_r):
        grid_r, grid_c = int(round(pr)), int(round(pc))
        if 0 <= grid_r < rows and 0 <= grid_c < cols:
            shared_occupied_grid[grid_r, grid_c] = True

    z_arr_r = dem_feature[np.clip(py_r.astype(int), 0, rows-1), np.clip(px_r.astype(int), 0, cols-1)]
    
    xc_r = (px_r - cols / 2.0) * scale_factor
    yc_r = (py_r - rows / 2.0) * scale_factor
    zc_r = ((z_arr_r / DEM_16BIT_MAX) / max_true_signal) * baseline_height_cap * EXAGGERATE_Z
    
    rot_x_r = xc_r * cos_y - yc_r * sin_y
    rot_y_r = xc_r * sin_y + yc_r * cos_y
    
    sx_vals_r = rot_x_r + (CANVAS_W / 2.0)
    sy_vals_r = (CANVAS_H * 0.50) + (rot_y_r * Y_COMPRESSION * sin_p - zc_r * cos_p)
    depths_r = -(rot_y_r * cos_p + zc_r * sin_p)
    
    visible_mask = depths_r <= (depth_buffer[np.clip(np.round(sy_vals_r).astype(np.int32), 0, CANVAS_H - 1), 
                                             np.clip(np.round(sx_vals_r).astype(np.int32), 0, CANVAS_W - 1)] + DEPTH_TOLERANCE)
    
    current_seg, current_depths = [], []
    i, n_pts = 0, len(smoothed)
    seg_start = 0
    
    while i < n_pts:
        if visible_mask[i]:
            if not current_seg: seg_start = i
            current_seg.append((sx_vals_r[i], sy_vals_r[i]))
            current_depths.append(depths_r[i])
            i += 1
        else:
            gap_end = i
            while gap_end < n_pts and not visible_mask[gap_end]: gap_end += 1
            if gap_end < n_pts:
                p1_x, p1_y = sx_vals_r[i - 1] if i > 0 else sx_vals_r[i], sy_vals_r[i - 1] if i > 0 else sy_vals_r[i]
                if np.sqrt((sx_vals_r[gap_end] - p1_x)**2 + (sy_vals_r[gap_end] - p1_y)**2) <= MAX_GAP_PIXELS:
                    for g_idx in range(i, gap_end):
                        current_seg.append((sx_vals_r[g_idx], sy_vals_r[g_idx]))
                        current_depths.append(depths_r[g_idx])
                    i = gap_end
                    continue
            if len(current_seg) >= 4:
                ridge_screen_list.append(np.array(current_seg))
                ridge_dem_list.append(smoothed[seg_start:i])
                ridge_depths.append(np.mean(current_depths))
            current_seg, current_depths = [], []
            i = gap_end
            
    if len(current_seg) >= 4:
        ridge_screen_list.append(np.array(current_seg))
        ridge_dem_list.append(smoothed[seg_start:n_pts])
        ridge_depths.append(np.mean(current_depths))


# ==========================================
# 6. GENERATE FLOWLINES (CALIBRATED TRACING)
# ==========================================
print("Simulating topography vector flowline tracing...")

@njit(cache=True)
def sample_bilinear_numba(array, x, y):
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, array.shape[1] - 1), min(y0 + 1, array.shape[0] - 1)
    return (array[y0, x0] * ((x1 - x) * (y1 - y)) + array[y1, x0] * ((x1 - x) * (y - y0)) + 
            array[y0, x1] * ((x - x0) * (y1 - y)) + array[y1, x1] * ((x - x0) * (y - y0)))

@njit(cache=True)
def get_local_radius(tone_val, shadow_min, highlight_max):
    return shadow_min + (highlight_max - shadow_min) * (1.0 - tone_val)

@njit(cache=True)
def is_valid_and_free(x, y, shadow_min, highlight_max, rows, cols, occupied, tone_map):
    if x < 0 or x >= cols or y < 0 or y >= rows: return False
    img_y, img_x = int(round(y)), int(round(x))
    if img_y >= rows or img_x >= cols: return False
    if occupied[img_y, img_x]: return False
        
    local_min_dist = get_local_radius(tone_map[img_y, img_x], shadow_min, highlight_max)
    r_buf = int(np.ceil(local_min_dist))
    rad_sq = local_min_dist**2
    
    for r in range(max(0, img_y - r_buf), min(rows, img_y + r_buf + 1)):
        for c in range(max(0, img_x - r_buf), min(cols, img_x + r_buf + 1)):
            if occupied[r, c] and ((float(c) - x)**2 + (float(r) - y)**2) < rad_sq:
                return False
    return True

@njit(cache=True)
def mark_occupied_radius(x, y, rows, cols, occupied):
    img_y, img_x = int(round(y)), int(round(x))
    r_buf = 1
    for r in range(max(0, img_y - r_buf), min(rows, img_y + r_buf + 1)):
        for c in range(max(0, img_x - r_buf), min(cols, img_x + r_buf + 1)):
            occupied[r, c] = True

@njit(cache=True)
def trace_topography(seeds, flow_x, flow_y, slope_map, occupied, max_steps, step_size, 
                      shadow_min, highlight_max, slope_thresh, min_len, rows, cols, all_points, 
                      path_segments, p_idx, l_idx, max_points, tone_map):
    fwd_path = np.zeros((max_steps + 1, 2), dtype=np.float64)
    bwd_path = np.zeros((max_steps + 1, 2), dtype=np.float64)
    
    for i in range(len(seeds)):
        sx, sy = seeds[i, 0], seeds[i, 1]
        if sx < 0 or sx >= cols or sy < 0 or sy >= rows: continue
        if sample_bilinear_numba(slope_map, sx, sy) < slope_thresh or not is_valid_and_free(sx, sy, shadow_min, highlight_max, rows, cols, occupied, tone_map): continue
            
        cx, cy, fwd_len = sx, sy, 1
        fwd_path[0] = [sx, sy]
        for _ in range(max_steps):
            f_x1, f_y1 = sample_bilinear_numba(flow_x, cx, cy), sample_bilinear_numba(flow_y, cx, cy)
            tx, ty = cx + f_x1 * step_size, cy + f_y1 * step_size
            nx = cx + 0.5 * (f_x1 + sample_bilinear_numba(flow_x, tx, ty)) * step_size
            ny = cy + 0.5 * (f_y1 + sample_bilinear_numba(flow_y, tx, ty)) * step_size
            
            if nx < 0 or nx >= cols or ny < 0 or ny >= rows: break
            if is_valid_and_free(nx, ny, shadow_min, highlight_max, rows, cols, occupied, tone_map):
                if sample_bilinear_numba(slope_map, nx, ny) < slope_thresh: break
                fwd_path[fwd_len] = [nx, ny]
                fwd_len += 1; cx, cy = nx, ny
            else: break
                
        cx, cy, bwd_len = sx, sy, 0
        for _ in range(max_steps):
            f_x1, f_y1 = sample_bilinear_numba(flow_x, cx, cy), sample_bilinear_numba(flow_y, cx, cy)
            tx, ty = cx - f_x1 * step_size, cy - f_y1 * step_size
            nx = cx - 0.5 * (f_x1 + sample_bilinear_numba(flow_x, tx, ty)) * step_size
            ny = cy - 0.5 * (f_y1 + sample_bilinear_numba(flow_y, tx, ty)) * step_size
            
            if nx < 0 or nx >= cols or ny < 0 or ny >= rows: break
            if is_valid_and_free(nx, ny, shadow_min, highlight_max, rows, cols, occupied, tone_map):
                if sample_bilinear_numba(slope_map, nx, ny) < slope_thresh: break
                bwd_path[bwd_len] = [nx, ny]
                bwd_len += 1; cx, cy = nx, ny
            else: break
                
        total_len = fwd_len + bwd_len
        if total_len >= min_len: 
            if p_idx + total_len >= max_points: break 
            path_segments[l_idx] = [p_idx, total_len]
            w_idx = p_idx
            for p in range(bwd_len - 1, -1, -1):
                all_points[w_idx] = bwd_path[p]
                mark_occupied_radius(bwd_path[p, 0], bwd_path[p, 1], rows, cols, occupied)
                w_idx += 1
            for p in range(fwd_len):
                all_points[w_idx] = fwd_path[p]
                mark_occupied_radius(fwd_path[p, 0], fwd_path[p, 1], rows, cols, occupied)
                w_idx += 1
            p_idx += total_len; l_idx += 1
    return p_idx, l_idx

global_points = np.zeros((15000000, 2), dtype=np.float64)
global_segments = np.zeros((300000, 2), dtype=np.int32)
global_point_idx, global_line_idx = 0, 0

for pass_num, spacing in enumerate(PASS_SPACINGS):
    ys = np.arange(0, rows, spacing)
    xs = np.arange(0, cols, spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    seeds_arr = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float64)
    np.random.shuffle(seeds_arr)
    
    global_point_idx, global_line_idx = trace_topography(
        seeds_arr, flow_x, flow_y, slope_map, shared_occupied_grid, MAX_STEPS, STEP_SIZE, 
        SHADOW_MIN_SPACING, HIGHLIGHT_MAX_SPACING, SLOPE_THRESHOLD, MIN_LINE_LENGTH, rows, cols, global_points, 
        global_segments, global_point_idx, global_line_idx, 15000000, tone_map
    )

raw_flow_segments = global_segments[:global_line_idx]


# ==========================================
# 7. VISIBILITY OCCLUSION & RAW GEOMETRY EXTRACTION
# ==========================================
print("Resolving valley depth buffers and projection arrays...")
flow_screen_list, flow_dem_list, flow_depths = [], [], []

for idx in range(len(raw_flow_segments)):
    start, length = raw_flow_segments[idx, 0], raw_flow_segments[idx, 1]
    if length < 2: continue
        
    px_array = global_points[start : start + length, 0]
    py_array = global_points[start : start + length, 1]
    
    z_feat = dem_feature[np.clip(py_array.astype(int), 0, rows-1), np.clip(px_array.astype(int), 0, cols-1)] 
    z_guide = dem_guide[np.clip(py_array.astype(int), 0, rows-1), np.clip(px_array.astype(int), 0, cols-1)]
    z_arr = (z_feat * 0.3) + (z_guide * 0.7)
    
    xc_line = (px_array - cols / 2.0) * scale_factor
    yc_line = (py_array - rows / 2.0) * scale_factor
    zc_line = ((z_arr / DEM_16BIT_MAX) / max_true_signal) * baseline_height_cap * EXAGGERATE_Z
    
    rot_x_line = xc_line * cos_y - yc_line * sin_y
    rot_y_line = xc_line * sin_y + yc_line * cos_y
    
    sx_vals = rot_x_line + (CANVAS_W / 2.0)
    sy_vals = (CANVAS_H * 0.50) + (rot_y_line * Y_COMPRESSION * sin_p - zc_line * cos_p)
    depths = -(rot_y_line * cos_p + zc_line * sin_p)
    
    terrain_depths = depth_buffer[np.clip(np.round(sy_vals).astype(np.int32), 0, CANVAS_H - 1), 
                                  np.clip(np.round(sx_vals).astype(np.int32), 0, CANVAS_W - 1)]
    
    current_segment, current_depths = [], []
    momentum = 3  
    seg_start = 0
    
    for i in range(length):
        is_visible = depths[i] <= (terrain_depths[i] + 2.5)
        momentum = min(momentum + 1, 4) if is_visible else max(momentum - 1, 0)
        
        if momentum > 0:
            if not current_segment: seg_start = i
            current_segment.append((sx_vals[i], sy_vals[i]))
            current_depths.append(depths[i])
        else:
            if len(current_segment) >= 6:  
                flow_screen_list.append(np.array(current_segment))
                flow_dem_list.append(np.column_stack([px_array[seg_start:i], py_array[seg_start:i]]))
                flow_depths.append(np.mean(current_depths))
            current_segment, current_depths = [], []
            
    if len(current_segment) >= 6:
        flow_screen_list.append(np.array(current_segment))
        flow_dem_list.append(np.column_stack([px_array[seg_start:length], py_array[seg_start:length]]))
        flow_depths.append(np.mean(current_depths))


# ==========================================
# 7.5 HIGH PERFORMANCE JIT PATH FRAGMENTATION
# ==========================================
print("Fragmenting paths by pure DEM-space vignette using optimized parallel Numba execution blocks...")
render_queue = []

def pack_and_flatten_geometries(screen_list, dem_list):
    lengths = np.array([len(x) for x in screen_list], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int32)
    flat_screen = np.vstack(screen_list) if screen_list else np.zeros((0, 2))
    flat_dem = np.vstack(dem_list) if dem_list else np.zeros((0, 2))
    return flat_screen, flat_dem, offsets, lengths

def unpack_fragmented_array(flat_screen, offsets, lengths):
    return [flat_screen[start : start + length] for start, length in zip(offsets, lengths)]

# --- Process Ridgelines ---
if len(ridge_screen_list) > 0:
    r_flat_scr, r_flat_dem, r_offsets, r_lengths = pack_and_flatten_geometries(ridge_screen_list, ridge_dem_list)
    
    frags_scr, f_offsets, f_lengths = batch_fragment_paths(
        r_flat_scr, r_flat_dem, r_offsets, r_lengths,
        VIGNETTE_DEM_OUTER_A, VIGNETTE_DEM_OUTER_B, VIGNETTE_DEM_INNER_RATIO,
        FRAGMENT_FREQUENCY * 0.5, MIN_FRAGMENT_VERTS, peak_r_dem, peak_c_dem
    )
    
    unpacked_frags = unpack_fragmented_array(frags_scr, f_offsets, f_lengths)
    
    frag_idx = 0
    for idx, orig_len in enumerate(r_lengths):
        mean_d = ridge_depths[idx]
        while frag_idx < len(f_offsets) and f_offsets[frag_idx] < r_offsets[idx] + orig_len:
            if frag_idx < len(unpacked_frags):
                render_queue.append({'type': 'ridge', 'depth': mean_d, 'path': unpacked_frags[frag_idx]})
            frag_idx += 1

# --- Process Flowlines ---
if len(flow_screen_list) > 0:
    f_flat_scr, f_flat_dem, f_offsets, f_lengths = pack_and_flatten_geometries(flow_screen_list, flow_dem_list)
    
    frags_scr, f_offsets, f_lengths = batch_fragment_paths(
        f_flat_scr, f_flat_dem, f_offsets, f_lengths,
        VIGNETTE_DEM_OUTER_A, VIGNETTE_DEM_OUTER_B, VIGNETTE_DEM_INNER_RATIO,
        FRAGMENT_FREQUENCY, MIN_FRAGMENT_VERTS, peak_r_dem, peak_c_dem
    )
    
    unpacked_frags = unpack_fragmented_array(frags_scr, f_offsets, f_lengths)
    
    frag_idx = 0
    for idx, orig_len in enumerate(f_lengths):
        mean_d = flow_depths[idx]
        while frag_idx < len(f_offsets) and f_offsets[frag_idx] < f_offsets[idx] + orig_len:
            if frag_idx < len(unpacked_frags):
                render_queue.append({'type': 'flow', 'depth': mean_d, 'path': unpacked_frags[frag_idx]})
            frag_idx += 1

# Sort elements using painter's depth order
render_queue.sort(key=lambda x: x['depth'], reverse=True)


# ==========================================
# 8. ASSEMBLE AND RENDER EXPORT (PUBLISHABLE PRESET)
# ==========================================
print("Exporting vector output to stylized SVG markup patterns...")

# We inject a native SVG texture pipeline using fine mathematical noise primitives
svg_elements = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
    f'width="100%" height="100%" style="background-color: {BG_COLOR};">\n',
    '  <defs>\n',
    '    \n',
    '    <filter id="paper-texture" x="0%" y="0%" width="100%" height="100%">\n',
    '      <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="4" result="noise"/>\n',
    '      <feDiffuseLighting in="noise" lighting-color="#ffffff" surfaceScale="1.2" result="light">\n',
    '        <feDistantLight azimuth="60" elevation="55"/>\n',
    '      </feDiffuseLighting>\n',
    '      <feComponentTransfer>\n',
    '        <feFuncR type="linear" slope="0.92"/>\n',
    '        <feFuncG type="linear" slope="0.91"/>\n',
    '        <feFuncB type="linear" slope="0.88"/>\n',
    '      </feComponentTransfer>\n',
    '      <feBlend mode="multiply" in="SourceGraphic" result="blend"/>\n',
    '    </filter>\n',
    '  </defs>\n\n',
    '  \n',
    '  <g filter="url(#paper-texture)">\n',
    f'    <rect width="{CANVAS_W}" height="{CANVAS_H}" fill="{BG_COLOR}"/>\n'
]

# Write out vector path configurations using uniform weight structures
for item in render_queue:
    path_data = "M " + " L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in item['path'])
    w = UNIFIED_THICKNESS 
    svg_elements.append(
        f'    <path d="{path_data}" stroke="{UNIFIED_COLOR}" stroke-width="{w:.2f}" '
        f'fill="none" stroke-linecap="round" stroke-linejoin="round" opacity="0.92"/>\n'
    )

svg_elements.append('  </g>\n')
svg_elements.append('</svg>\n')

output_file = "output_calibrated_topography.svg"
with open(output_file, 'w') as f:
    f.writelines(svg_elements)

print(f"Success! Output cleanly saved to {output_file}.")