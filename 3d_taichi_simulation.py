import taichi as ti
import numpy as np
import math
import csv

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None

"""
Shawn Nordstrom
02/21/2025
"""

ti.init(arch=ti.gpu)


#                       CONFIGURATION
# =========================================================
n = 12          # nxnxn grid | default: 12x12x12 (increase carefully for perf)
dx = 0.03       # at rest len between neighbor particles | default: 0.03

dt = 5e-4       # ts
substeps = 20   # physics steps per frame

gravity   = ti.Vector([0.0, 0.0, -9.8])
stiffness = 50.0 # (K) Trials
damping   = 50      # spring-direction damping (higher = less jiggle)
air_drag = 0.2      # global velocity damping in free-flight (1/s)
floor_friction = 1e6  # tangential damping when touching the floor (1/s)
floor_contact_eps = 0.003  # contact threshold above the surface
floor_restitution = 0.3    # bounce along surface normal .3
object_restitution = 0.65   # object material bounciness, independent from floor
floor_impact_damping = 0.2  # floor contact impact damping (0 = none, 1 = full stop)
object_impact_damping = 0.05  # object material impact damping (0 = none, 1 = full stop)

# spring family scaling for better material control
structural_stiffness_scale = 1.0
shear_stiffness_scale = 1.0
bending_stiffness_scale = 1.0
volume_stiffness = 0.0
volume_damping = 0.0
particle_mass = 1.0


# simple 3D -> 2D projection settings
view_azimuth = 2.6   # radians, rotate around z
view_pitch   = -7.2 #-7.4  # radians, rotate around x
view_scale   = 2   # larger = zoomed out
use_perspective = True
perspective = 1    # larger = weaker perspective (must be > max depth)
free_cam = True
cam_rotate_speed = 3.0   # radians per normalized screen drag
cam_zoom_speed = 2.0     # exponential zoom per normalized screen drag
cam_min_scale = 0.4
cam_max_scale = 6.0
view_center = np.array([0.5, 0.5, 0.0], dtype=np.float32)

# render quality controls
render_surface_only = True
render_filled_surface = True
render_surface_overlay = False  # draw points/wireframe on top of the filled shell
fill_stride = 1      # tessellation for the filled shell
wire_stride = 1      # draw every Nth surface edge
point_stride = 1     # draw every Nth surface point
line_radius = 1
point_radius = 1
gui_res = 1000

# depth/visibility controls
depth_shading = True
depth_bins = 50
depth_near_color = 0xFFFFFF
depth_far_color = 0x404040
surface_near_color = 0xD6D6D6
surface_far_color = 0x5C5C5C
surface_flat_color = 0x9A9A9A
line_radius_far_scale = 0.6
line_radius_near_scale = 1.6
floor_near_color = 0xa9a9a9
floor_far_color = 0x404040

# Floor shapes (set `floor_mode` before running)
if 1 == 1:
    floor_ax2 = 0.2
    floor_by2 = 0.2
    floor_cxy = 0.0
    floor_dx = 0.2
    floor_ey = 0.20
    floor_f = 0.00

    floor_cx = 0.5
    floor_cy = 0.5

    floor_ramp_h0 = 0.0
    floor_ramp_ax = 0.25
    floor_ramp_by = 0.0

    floor_bowl_h0 = 0.0
    floor_bowl_a = 0.6

    floor_saddle_h0 = 0.1
    floor_saddle_a = 0.6
    floor_saddle_b = 0.6

    floor_crater_h0 = 0.0
    floor_crater_a = 0.25
    floor_crater_r2 = 0.06

    floor_ring_h0 = 0.0
    floor_ring_a = 0.2
    floor_ring_r = 0.3
    floor_ring_w2 = 0.01

    floor_stair_h0 = 0.0
    floor_stair_h = 0.05
    floor_stair_w = 0.12
    floor_stair_origin = 0.0

    floor_ridge_h0 = 0.0
    floor_ridge_a = 0.06
    floor_ridge_f = 10.0
    floor_ridge_phase = 0.0

    floor_checker_h0 = 0.0
    floor_checker_a = 0.05
    floor_checker_fx = 10.0
    floor_checker_fy = 10.0

    floor_pillar_h0 = 0.0
    floor_pillar_a = 0.12
    floor_pillar_r2 = 0.01

    floor_samples = 80
    bumpy_amp = 0.05
    bumpy_fx = 10.0
    bumpy_fy = 8.0
    bumpy_phase1 = 0.7
    bumpy_phase2 = 1.1

FLOOR_FLAT = 0
FLOOR_BUMPY = 1
FLOOR_QUAD = 2
FLOOR_RAMP = 3
FLOOR_BOWL = 4
FLOOR_SADDLE = 5
FLOOR_CRATER = 6
FLOOR_RING = 7
FLOOR_STAIRS = 8
FLOOR_RIDGES = 9
FLOOR_CHECKER = 10
FLOOR_PILLARS = 11

"""
Floor 0 --> show two varying stiff/damp
Floor 1 --> show this floor
Floor 8 --> show this floor
Floor 10 --> show shading + lower stiffness where it collapses
"""

floor_mode = 0

# ---- PRESET SELECTOR (edit these before running) ----
# Choices: "bouncy_ball" | "bean_bag" | "pillow" | "jelly"
active_preset   = "jelly"
stiffness_scale = 1   # multiply preset stiffness (e.g. 0.5 = softer, 2.0 = stiffer)
damping_scale   = 1   # multiply preset damping   (e.g. 0.5 = bouncier, 2.0 = deader)
# -----------------------------------------------------

SHAPE_CUBE = 0
SHAPE_SPHERE = 1
SHAPE_BEAN_BAG = 2
SHAPE_PILLOW = 3

SHAPE_NAME_TO_MODE = {
    "cube": SHAPE_CUBE,
    "sphere": SHAPE_SPHERE,
    "bean_bag": SHAPE_BEAN_BAG,
    "pillow": SHAPE_PILLOW,
}

MATERIAL_PRESETS = {
    "bouncy_ball": {
        "shape": "sphere",
        "shape_scale": (1.0, 1.0, 1.0),
        "stiffness": 58.0,
        "damping": 12.0,
        "air_drag": 0.08,
        "structural_stiffness_scale": 1.0,
        "shear_stiffness_scale": 1.1,
        "bending_stiffness_scale": 0.8,
        "volume_stiffness": 0.0,
        "volume_damping": 0.0,
        "object_restitution": 0.95,
        "floor_restitution": 0.9,
        "floor_friction": 18.0,
        "floor_impact_damping": 0.03,
        "object_impact_damping": 0.02,
        "spawn_z": 0.78,
    },
    "bean_bag": {
        "shape": "bean_bag",
        "shape_scale": (1.15, 1.0, 0.78),
        "stiffness": 24.0,
        "damping": 130.0,
        "air_drag": 1.0,
        "structural_stiffness_scale": 0.95,
        "shear_stiffness_scale": 0.6,
        "bending_stiffness_scale": 0.55,
        "volume_stiffness": 0.0,
        "volume_damping": 0.0,
        "object_restitution": 0.15,
        "floor_restitution": 0.85,
        "floor_friction": 160.0,
        "floor_impact_damping": 0.28,
        "object_impact_damping": 0.35,
        "spawn_z": 0.70,
    },
    "pillow": {
        "shape": "pillow",
        "shape_scale": (1.2, 1.05, 0.62),
        "stiffness": 38.0,
        "damping": 88.0,
        "air_drag": 0.7,
        "structural_stiffness_scale": 1.0,
        "shear_stiffness_scale": 0.55,
        "bending_stiffness_scale": 0.85,
        "volume_stiffness": 0.0,
        "volume_damping": 0.0,
        "object_restitution": 0.2,
        "floor_restitution": 0.8,
        "floor_friction": 90.0,
        "floor_impact_damping": 0.08,
        "object_impact_damping": 0.08,
        "spawn_z": 0.72,
    },
    "jelly": {
        "shape": "cube",
        "shape_scale": (1.0, 1.0, 1.0),
        "stiffness": 13.0,
        "damping": 60.0,
        "air_drag": 0.45,
        "structural_stiffness_scale": 0.9,
        "shear_stiffness_scale": 0.45,
        "bending_stiffness_scale": 0.55,
        "volume_stiffness": 0.0,
        "volume_damping": 0.0,
        "object_restitution": 0.45,
        "floor_restitution": 0.85,
        "floor_friction": 70.0,
        "floor_impact_damping": 0.15,
        "object_impact_damping": 0.2,
        "spawn_z": 0.72,
    },
}

default_preset = "jelly"
masked_surface_hull_cache = {}

@ti.func
def z_floor(x, y):        # ground height (physics)
    if ti.static(floor_mode == FLOOR_FLAT):
        return floor_f
    if ti.static(floor_mode == FLOOR_BUMPY):
        return (
            floor_f
            + bumpy_amp
            * (
                ti.sin(bumpy_fx * x) * ti.cos(bumpy_fy * y)
                + 0.5 * ti.sin(2.1 * bumpy_fx * x + bumpy_phase1)
                * ti.cos(1.7 * bumpy_fy * y + bumpy_phase2)
            )
        )
    if ti.static(floor_mode == FLOOR_QUAD):
        return (
            floor_ax2 * x * x
            + floor_by2 * y * y
            + floor_cxy * x * y
            + floor_dx * x
            + floor_ey * y
            + floor_f
        )
    if ti.static(floor_mode == FLOOR_RAMP):
        return floor_ramp_h0 + floor_ramp_ax * x + floor_ramp_by * y
    if ti.static(floor_mode == FLOOR_BOWL):
        dx = x - floor_cx
        dy = y - floor_cy
        return floor_bowl_h0 + floor_bowl_a * (dx * dx + dy * dy)
    if ti.static(floor_mode == FLOOR_SADDLE):
        dx = x - floor_cx
        dy = y - floor_cy
        return floor_saddle_h0 + floor_saddle_a * (dx * dx) - floor_saddle_b * (dy * dy)
    if ti.static(floor_mode == FLOOR_CRATER):
        dx = x - floor_cx
        dy = y - floor_cy
        r2 = dx * dx + dy * dy
        return floor_crater_h0 - floor_crater_a * ti.exp(-r2 / floor_crater_r2)
    if ti.static(floor_mode == FLOOR_RING):
        dx = x - floor_cx
        dy = y - floor_cy
        r = ti.sqrt(dx * dx + dy * dy)
        return floor_ring_h0 + floor_ring_a * ti.exp(-((r - floor_ring_r) ** 2) / floor_ring_w2)
    if ti.static(floor_mode == FLOOR_STAIRS):
        step_idx = ti.floor((x - floor_stair_origin) / floor_stair_w)
        return floor_stair_h0 + floor_stair_h * step_idx
    if ti.static(floor_mode == FLOOR_RIDGES):
        return floor_ridge_h0 + floor_ridge_a * ti.sin(floor_ridge_f * x + floor_ridge_phase)
    if ti.static(floor_mode == FLOOR_CHECKER):
        v = ti.sin(floor_checker_fx * x) * ti.sin(floor_checker_fy * y)
        s = ti.select(v >= 0.0, 1.0, -1.0)
        return floor_checker_h0 + floor_checker_a * s
    if ti.static(floor_mode == FLOOR_PILLARS):
        z = floor_pillar_h0
        for cx, cy in ti.static(
            [
                (0.25, 0.25),
                (0.25, 0.5),
                (0.25, 0.75),
                (0.5, 0.25),
                (0.5, 0.5),
                (0.5, 0.75),
                (0.75, 0.25),
                (0.75, 0.5),
                (0.75, 0.75),
            ]
        ):
            dx = x - cx
            dy = y - cy
            z += floor_pillar_a * ti.exp(-(dx * dx + dy * dy) / floor_pillar_r2)
        return z
    return floor_f


@ti.func
def floor_normal(x, y):
    # finite-difference normal for z = f(x, y)
    eps = 1e-3
    zpx = z_floor(x + eps, y)
    zmx = z_floor(x - eps, y)
    zpy = z_floor(x, y + eps)
    zmy = z_floor(x, y - eps)
    dzdx = (zpx - zmx) / (2.0 * eps)
    dzdy = (zpy - zmy) / (2.0 * eps)
    n = ti.Vector([-dzdx, -dzdy, 1.0])
    return n / (n.norm() + 1e-6)

def z_floor_np(xs: np.ndarray, ys: np.ndarray):  # ground height (drawing)
    xs_arr = np.asarray(xs, dtype=np.float32)
    ys_arr = np.asarray(ys, dtype=np.float32)
    if floor_mode == FLOOR_FLAT:
        z = floor_f
    elif floor_mode == FLOOR_BUMPY:
        z = (
            floor_f
            + bumpy_amp
            * (
                np.sin(bumpy_fx * xs_arr) * np.cos(bumpy_fy * ys_arr)
                + 0.5 * np.sin(2.1 * bumpy_fx * xs_arr + bumpy_phase1)
                * np.cos(1.7 * bumpy_fy * ys_arr + bumpy_phase2)
            )
        )
    elif floor_mode == FLOOR_QUAD:
        z = (
            floor_ax2 * xs_arr * xs_arr
            + floor_by2 * ys_arr * ys_arr
            + floor_cxy * xs_arr * ys_arr
            + floor_dx * xs_arr
            + floor_ey * ys_arr
            + floor_f
        )
    elif floor_mode == FLOOR_RAMP:
        z = floor_ramp_h0 + floor_ramp_ax * xs_arr + floor_ramp_by * ys_arr
    elif floor_mode == FLOOR_BOWL:
        dx = xs_arr - floor_cx
        dy = ys_arr - floor_cy
        z = floor_bowl_h0 + floor_bowl_a * (dx * dx + dy * dy)
    elif floor_mode == FLOOR_SADDLE:
        dx = xs_arr - floor_cx
        dy = ys_arr - floor_cy
        z = floor_saddle_h0 + floor_saddle_a * (dx * dx) - floor_saddle_b * (dy * dy)
    elif floor_mode == FLOOR_CRATER:
        dx = xs_arr - floor_cx
        dy = ys_arr - floor_cy
        r2 = dx * dx + dy * dy
        z = floor_crater_h0 - floor_crater_a * np.exp(-r2 / floor_crater_r2)
    elif floor_mode == FLOOR_RING:
        dx = xs_arr - floor_cx
        dy = ys_arr - floor_cy
        r = np.sqrt(dx * dx + dy * dy)
        z = floor_ring_h0 + floor_ring_a * np.exp(-((r - floor_ring_r) ** 2) / floor_ring_w2)
    elif floor_mode == FLOOR_STAIRS:
        step_idx = np.floor((xs_arr - floor_stair_origin) / floor_stair_w)
        z = floor_stair_h0 + floor_stair_h * step_idx
    elif floor_mode == FLOOR_RIDGES:
        z = floor_ridge_h0 + floor_ridge_a * np.sin(floor_ridge_f * xs_arr + floor_ridge_phase)
    elif floor_mode == FLOOR_CHECKER:
        v = np.sin(floor_checker_fx * xs_arr) * np.sin(floor_checker_fy * ys_arr)
        s = np.where(v >= 0.0, 1.0, -1.0)
        z = floor_checker_h0 + floor_checker_a * s
    elif floor_mode == FLOOR_PILLARS:
        shape = np.broadcast(xs_arr, ys_arr).shape
        z = np.full(shape, floor_pillar_h0, dtype=np.float32)
        for cx, cy in [
            (0.25, 0.25),
            (0.25, 0.5),
            (0.25, 0.75),
            (0.5, 0.25),
            (0.5, 0.5),
            (0.5, 0.75),
            (0.75, 0.25),
            (0.75, 0.5),
            (0.75, 0.75),
        ]:
            dx = xs_arr - cx
            dy = ys_arr - cy
            z += floor_pillar_a * np.exp(-(dx * dx + dy * dy) / floor_pillar_r2)
    else:
        z = floor_f
    if np.ndim(z) == 0:
        if xs_arr.shape != ():
            return np.full_like(xs_arr, z, dtype=np.float32)
        if ys_arr.shape != ():
            return np.full_like(ys_arr, z, dtype=np.float32)
        return np.array(z, dtype=np.float32)
    return z.astype(np.float32)

# =========================================================
#
# ---------------------------------------------------------
# REMEMBER TO SAVE AFTER ANY CHANGE (otherwise hitting run
# will just run the prior runs's settings)
# ---------------------------------------------------------
pos   = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
vel   = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
force = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
rest_pos = ti.Vector.field(3, dtype=ti.f32, shape=(n, n, n))
active = ti.field(dtype=ti.i32, shape=(n, n, n))

sim_spring_stiffness = ti.field(dtype=ti.f32, shape=())
sim_spring_damping = ti.field(dtype=ti.f32, shape=())
sim_air_drag = ti.field(dtype=ti.f32, shape=())
sim_structural_stiffness_scale = ti.field(dtype=ti.f32, shape=())
sim_shear_stiffness_scale = ti.field(dtype=ti.f32, shape=())
sim_bending_stiffness_scale = ti.field(dtype=ti.f32, shape=())
sim_volume_stiffness = ti.field(dtype=ti.f32, shape=())
sim_volume_damping = ti.field(dtype=ti.f32, shape=())
sim_particle_mass = ti.field(dtype=ti.f32, shape=())
sim_floor_friction = ti.field(dtype=ti.f32, shape=())
sim_floor_contact_eps = ti.field(dtype=ti.f32, shape=())
sim_floor_restitution = ti.field(dtype=ti.f32, shape=())
sim_object_restitution = ti.field(dtype=ti.f32, shape=())
sim_floor_impact_damping = ti.field(dtype=ti.f32, shape=())
sim_object_impact_damping = ti.field(dtype=ti.f32, shape=())
object_shape_mode = ti.field(dtype=ti.i32, shape=())
object_center = ti.Vector.field(3, dtype=ti.f32, shape=())
object_scale = ti.Vector.field(3, dtype=ti.f32, shape=())
rest_radius = ti.field(dtype=ti.f32, shape=(n, n, n))
active_pos_sum = ti.Vector.field(3, dtype=ti.f32, shape=())
active_vel_sum = ti.Vector.field(3, dtype=ti.f32, shape=())
active_count = ti.field(dtype=ti.i32, shape=())


@ti.func
def cube_to_sphere(local):
    r = local.norm()
    m = ti.max(ti.abs(local.x), ti.max(ti.abs(local.y), ti.abs(local.z)))
    out = local
    if r > 1e-6:
        out = local / r * m
    return out


@ti.func
def shape_local_coords(u, v, w, mode):
    local = ti.Vector([u, v, w])
    if mode == SHAPE_BEAN_BAG:
        local = cube_to_sphere(local)
        local = ti.Vector([local.x, local.y, local.z * 0.78])
        if local.z < 0.0:
            local.z *= 0.58
        local.x += 0.08 * local.z * (1.0 - local.x * local.x)
    elif mode == SHAPE_PILLOW:
        bulge = ti.max(0.45, 1.0 - 0.35 * (u * u + v * v))
        local = ti.Vector([u * 1.1, v * 1.0, w * 0.5 * bulge])
    return local


@ti.func
def shape_active(u, v, w, mode):
    is_active = 1
    if mode == SHAPE_SPHERE or mode == SHAPE_BEAN_BAG:
        is_active = ti.cast(u * u + v * v + w * w <= 1.0, ti.i32)
    return is_active


@ti.kernel
def init():
    """Make initial object shape and state."""
    denom = ti.max(1.0, ti.cast(n - 1, ti.f32))
    half_span = 0.5 * ti.cast(n - 1, ti.f32) * dx
    center = object_center[None]
    scale = object_scale[None]
    mode = object_shape_mode[None]
    for i, j, k in pos:
        u = 2.0 * (ti.cast(i, ti.f32) / denom) - 1.0
        v = 2.0 * (ti.cast(j, ti.f32) / denom) - 1.0
        w = 2.0 * (ti.cast(k, ti.f32) / denom) - 1.0
        is_active = shape_active(u, v, w, mode)
        active[i, j, k] = is_active
        local = shape_local_coords(u, v, w, mode)
        if is_active == 1:
            pos[i, j, k] = center + half_span * ti.Vector(
                [local.x * scale.x, local.y * scale.y, local.z * scale.z]
            )
            rest_radius[i, j, k] = (pos[i, j, k] - center).norm()
            rest_pos[i, j, k] = pos[i, j, k]
        else:
            pos[i, j, k] = center
            rest_radius[i, j, k] = 0.0
            rest_pos[i, j, k] = center
        vel[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])


@ti.func
def add_spring(
    i1: int,
    j1: int,
    k1: int,
    i2: int,
    j2: int,
    k2: int,
    stiffness_scale: ti.f32,
):
    """Apply spring forces."""
    p1 = pos[i1, j1, k1]
    p2 = pos[i2, j2, k2]
    rp1 = rest_pos[i1, j1, k1]
    rp2 = rest_pos[i2, j2, k2]
    d = p2 - p1
    L = d.norm() + 1e-6
    rest_len = (rp2 - rp1).norm() + 1e-6
    dir = d / L
    extension = L - rest_len
    F = (sim_spring_stiffness[None] ** 3) * stiffness_scale * extension * dir
    # damping along the spring direction (relative motion)
    v1 = vel[i1, j1, k1]
    v2 = vel[i2, j2, k2]
    rel_v = v2 - v1
    Fd = sim_spring_damping[None] * rel_v.dot(dir) * dir
    force[i1, j1, k1] += F
    force[i2, j2, k2] -= F
    force[i1, j1, k1] += Fd
    force[i2, j2, k2] -= Fd


@ti.kernel
def clear_forces():
    mass = ti.max(sim_particle_mass[None], 1e-6)
    for i, j, k in force:
        if active[i, j, k] == 1:
            force[i, j, k] = gravity * mass
        else:
            force[i, j, k] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def apply_springs():
    structural_scale = sim_structural_stiffness_scale[None]
    shear_scale = sim_shear_stiffness_scale[None]
    bending_scale = sim_bending_stiffness_scale[None]
    for i, j, k in pos:
        if active[i, j, k] == 0:
            continue
        # Structural springs (6-neighbour)
        if i + 1 < n and active[i + 1, j, k] == 1:
            add_spring(i, j, k, i + 1, j, k, structural_scale)
        if j + 1 < n and active[i, j + 1, k] == 1:
            add_spring(i, j, k, i, j + 1, k, structural_scale)
        if k + 1 < n and active[i, j, k + 1] == 1:
            add_spring(i, j, k, i, j, k + 1, structural_scale)

        # Shear springs (diagonals in planes)
        if i + 1 < n and j + 1 < n and active[i + 1, j + 1, k] == 1:
            add_spring(i, j, k, i + 1, j + 1, k, shear_scale)
        if i + 1 < n and j - 1 >= 0 and active[i + 1, j - 1, k] == 1:
            add_spring(i, j, k, i + 1, j - 1, k, shear_scale)
        if i + 1 < n and k + 1 < n and active[i + 1, j, k + 1] == 1:
            add_spring(i, j, k, i + 1, j, k + 1, shear_scale)
        if i + 1 < n and k - 1 >= 0 and active[i + 1, j, k - 1] == 1:
            add_spring(i, j, k, i + 1, j, k - 1, shear_scale)
        if j + 1 < n and k + 1 < n and active[i, j + 1, k + 1] == 1:
            add_spring(i, j, k, i, j + 1, k + 1, shear_scale)
        if j + 1 < n and k - 1 >= 0 and active[i, j + 1, k - 1] == 1:
            add_spring(i, j, k, i, j + 1, k - 1, shear_scale)

        # Bending springs (two apart) to smooth kinks
        if i + 2 < n and active[i + 2, j, k] == 1:
            add_spring(i, j, k, i + 2, j, k, bending_scale)
        if j + 2 < n and active[i, j + 2, k] == 1:
            add_spring(i, j, k, i, j + 2, k, bending_scale)
        if k + 2 < n and active[i, j, k + 2] == 1:
            add_spring(i, j, k, i, j, k + 2, bending_scale)


@ti.kernel
def compute_active_center():
    active_pos_sum[None] = ti.Vector([0.0, 0.0, 0.0])
    active_vel_sum[None] = ti.Vector([0.0, 0.0, 0.0])
    active_count[None] = 0
    for i, j, k in pos:
        if active[i, j, k] == 1:
            ti.atomic_add(active_pos_sum[None].x, pos[i, j, k].x)
            ti.atomic_add(active_pos_sum[None].y, pos[i, j, k].y)
            ti.atomic_add(active_pos_sum[None].z, pos[i, j, k].z)
            ti.atomic_add(active_vel_sum[None].x, vel[i, j, k].x)
            ti.atomic_add(active_vel_sum[None].y, vel[i, j, k].y)
            ti.atomic_add(active_vel_sum[None].z, vel[i, j, k].z)
            ti.atomic_add(active_count[None], 1)


@ti.kernel
def apply_volume_preservation():
    k_vol = ti.max(0.0, sim_volume_stiffness[None])
    c_vol = ti.max(0.0, sim_volume_damping[None])
    cnt = ti.max(1, active_count[None])
    center = active_pos_sum[None] / ti.cast(cnt, ti.f32)
    center_v = active_vel_sum[None] / ti.cast(cnt, ti.f32)
    for i, j, k in pos:
        if active[i, j, k] == 1:
            d = pos[i, j, k] - center
            L = d.norm() + 1e-6
            extension = L - rest_radius[i, j, k]
            dir = d / L
            radial_rel_v = (vel[i, j, k] - center_v).dot(dir)
            force[i, j, k] += -(k_vol * extension + c_vol * radial_rel_v) * dir


@ti.kernel
def integrate(dt: ti.f32):
    mass = ti.max(sim_particle_mass[None], 1e-6)
    inv_mass = 1.0 / mass
    drag = ti.max(0.0, sim_air_drag[None])
    floor_eps = ti.max(0.0, sim_floor_contact_eps[None])
    floor_mu = ti.max(0.0, sim_floor_friction[None])
    restitution = ti.min(1.0, ti.max(0.0, sim_object_restitution[None] * sim_floor_restitution[None]))
    impact_damping = ti.min(
        1.0,
        ti.max(0.0, sim_floor_impact_damping[None] + sim_object_impact_damping[None]),
    )
    for i, j, k in pos:
        if active[i, j, k] == 0:
            continue
        # v ← v + a dt = v + (F/m) dt
        vel[i, j, k] += force[i, j, k] * inv_mass * dt
        vel[i, j, k] *= ti.exp(-drag * dt)
        pos[i, j, k] += vel[i, j, k] * dt

        # ground collision
        x = pos[i, j, k].x
        y = pos[i, j, k].y
        zf = z_floor(x, y)
        if pos[i, j, k].z <= zf + floor_eps:
            n = floor_normal(x, y)
            if pos[i, j, k].z < zf:
                pos[i, j, k].z = zf
            v = vel[i, j, k]
            vn = v.dot(n)
            if vn < 0.0:
                v = v - (1.0 + restitution) * vn * n
                # impact damping: reduce only the normal component on impact
                vn_post = v.dot(n)
                v = v - impact_damping * vn_post * n
            # floor friction: damp tangential motion in the contact plane
            vt = v - v.dot(n) * n
            v -= vt * ti.min(1.0, floor_mu * dt)
            vel[i, j, k] = v


@ti.kernel
def scale_object_height(center_z: ti.f32, scale: ti.f32):
    for i, j, k in pos:
        if active[i, j, k] == 1:
            pos[i, j, k].z = center_z + (pos[i, j, k].z - center_z) * scale
            vel[i, j, k] = ti.Vector([0.0, 0.0, 0.0])


# -------- projection + drawing helpers (NumPy only) --------

def project_points(points3: np.ndarray) -> np.ndarray:
    """
    Simple orthographic projection with two rotations.
    points3: shape (m, 3) in [0,1]^3
    returns: shape (m, 2) in roughly [0,1]^2
    """
    p = points3 - view_center
    ca, sa = np.cos(view_azimuth), np.sin(view_azimuth)
    cp, sp = np.cos(view_pitch), np.sin(view_pitch)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    p = p @ rz.T
    p = p @ rx.T
    if use_perspective:
        denom = np.clip(perspective - p[:, 2], 0.2, None)
        scale = (perspective / denom).astype(np.float32)
        p2 = (p[:, :2] * scale[:, None]) / view_scale + 0.5
    else:
        p2 = p[:, :2] / view_scale + 0.5
    return p2.astype(np.float32)


def project_points_with_depth(points3: np.ndarray):
    """
    Returns projected 2D points and view-space depth.
    """
    p = points3 - view_center
    ca, sa = np.cos(view_azimuth), np.sin(view_azimuth)
    cp, sp = np.cos(view_pitch), np.sin(view_pitch)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    p = p @ rz.T
    p = p @ rx.T
    depth = p[:, 2].astype(np.float32)
    if use_perspective:
        denom = np.clip(perspective - p[:, 2], 0.2, None)
        scale = (perspective / denom).astype(np.float32)
        p2 = (p[:, :2] * scale[:, None]) / view_scale + 0.5
    else:
        p2 = p[:, :2] / view_scale + 0.5
    return p2.astype(np.float32), depth


def lerp_color(c0: int, c1: int, t: float) -> int:
    t = float(np.clip(t, 0.0, 1.0))
    r0, g0, b0 = (c0 >> 16) & 0xFF, (c0 >> 8) & 0xFF, c0 & 0xFF
    r1, g1, b1 = (c1 >> 16) & 0xFF, (c1 >> 8) & 0xFF, c1 & 0xFF
    r = int(r0 + (r1 - r0) * t)
    g = int(g0 + (g1 - g0) * t)
    b = int(b0 + (b1 - b0) * t)
    return (r << 16) | (g << 8) | b


def on_surface(i: int, j: int, k: int) -> bool:
    return i == 0 or j == 0 or k == 0 or i == n - 1 or j == n - 1 or k == n - 1


def surface_indices(stride: int):
    idxs = list(range(0, n, stride))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return idxs


def is_active_surface_node(active_np: np.ndarray, i: int, j: int, k: int) -> bool:
    if not active_np[i, j, k]:
        return False
    for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        ni = i + di
        nj = j + dj
        nk = k + dk
        if ni < 0 or ni >= n or nj < 0 or nj >= n or nk < 0 or nk >= n:
            return True
        if not active_np[ni, nj, nk]:
            return True
    return False


def compute_active_surface_mask(active_np: np.ndarray):
    surf = np.zeros_like(active_np, dtype=bool)
    idx_active = np.argwhere(active_np)
    for i, j, k in idx_active:
        surf[i, j, k] = is_active_surface_node(active_np, int(i), int(j), int(k))
    return surf


def collect_surface_points(pos_np: np.ndarray, active_np: np.ndarray):
    if not bool(np.all(active_np)):
        surf = compute_active_surface_mask(active_np)
        pts = pos_np[surf]
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if point_stride > 1:
            pts = pts[::point_stride]
        return pts.astype(np.float32)

    pts = []
    idxs = surface_indices(point_stride)
    surf = compute_active_surface_mask(active_np)
    for i in idxs:
        for j in idxs:
            for k in idxs:
                if surf[i, j, k]:
                    pts.append(pos_np[i, j, k])
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def build_surface_wireframe(pos_np: np.ndarray, active_np: np.ndarray):
    if not bool(np.all(active_np)):
        surf = compute_active_surface_mask(active_np)
        begin = []
        end = []
        for i in range(0, n):
            for j in range(0, n):
                for k in range(0, n):
                    if not surf[i, j, k]:
                        continue
                    p = pos_np[i, j, k]
                    if i + 1 < n and surf[i + 1, j, k]:
                        begin.append(p)
                        end.append(pos_np[i + 1, j, k])
                    if j + 1 < n and surf[i, j + 1, k]:
                        begin.append(p)
                        end.append(pos_np[i, j + 1, k])
                    if k + 1 < n and surf[i, j, k + 1]:
                        begin.append(p)
                        end.append(pos_np[i, j, k + 1])
        if len(begin) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        begin = np.array(begin, dtype=np.float32)
        end = np.array(end, dtype=np.float32)
        if wire_stride > 1:
            begin = begin[::wire_stride]
            end = end[::wire_stride]
        return begin, end

    begin = []
    end = []
    idxs = surface_indices(wire_stride)
    surf = compute_active_surface_mask(active_np)
    for i_idx, i in enumerate(idxs):
        for j_idx, j in enumerate(idxs):
            for k_idx, k in enumerate(idxs):
                if not surf[i, j, k]:
                    continue
                p = pos_np[i, j, k]
                if i_idx + 1 < len(idxs):
                    ni = idxs[i_idx + 1]
                    if surf[ni, j, k]:
                        begin.append(p)
                        end.append(pos_np[ni, j, k])
                if j_idx + 1 < len(idxs):
                    nj = idxs[j_idx + 1]
                    if surf[i, nj, k]:
                        begin.append(p)
                        end.append(pos_np[i, nj, k])
                if k_idx + 1 < len(idxs):
                    nk = idxs[k_idx + 1]
                    if surf[i, j, nk]:
                        begin.append(p)
                        end.append(pos_np[i, j, nk])

    if len(begin) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    return np.array(begin, dtype=np.float32), np.array(end, dtype=np.float32)


def build_surface_triangles(pos_np: np.ndarray, active_np: np.ndarray):
    if not bool(np.all(active_np)):
        surf = compute_active_surface_mask(active_np)
        pts = pos_np[surf]
        if ConvexHull is None or len(pts) < 4:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty, empty
        if int(object_shape_mode[None]) == SHAPE_SPHERE:
            center = pos_np[active_np].mean(axis=0)
            dirs = pts - center
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            avg_r = float(np.median(norms))
            pts = center + dirs / np.maximum(norms, 1e-6) * avg_r
        key = surf.tobytes()
        simplices = masked_surface_hull_cache.get(key)
        if simplices is None:
            try:
                hull = ConvexHull(pts, qhull_options="QJ")
                simplices = hull.simplices.astype(np.int32)
                masked_surface_hull_cache[key] = simplices
            except Exception:
                empty = np.zeros((0, 3), dtype=np.float32)
                return empty, empty, empty
        tri = pts[simplices]
        return (
            tri[:, 0, :].astype(np.float32),
            tri[:, 1, :].astype(np.float32),
            tri[:, 2, :].astype(np.float32),
        )

    tri_a = []
    tri_b = []
    tri_c = []
    idxs = surface_indices(fill_stride)

    def add_face(index_a: int, index_b: int, fixed_axis: int, fixed_idx: int):
        for a_idx in range(len(idxs) - 1):
            for b_idx in range(len(idxs) - 1):
                corners = [[0, 0, 0] for _ in range(4)]
                ia0, ia1 = idxs[a_idx], idxs[a_idx + 1]
                ib0, ib1 = idxs[b_idx], idxs[b_idx + 1]
                samples = ((ia0, ib0), (ia1, ib0), (ia1, ib1), (ia0, ib1))
                for corner, (va, vb) in zip(corners, samples):
                    corner[index_a] = va
                    corner[index_b] = vb
                    corner[fixed_axis] = fixed_idx
                p00 = pos_np[tuple(corners[0])]
                p10 = pos_np[tuple(corners[1])]
                p11 = pos_np[tuple(corners[2])]
                p01 = pos_np[tuple(corners[3])]
                tri_a.extend((p00, p00))
                tri_b.extend((p10, p11))
                tri_c.extend((p11, p01))

    add_face(1, 2, 0, 0)
    add_face(1, 2, 0, n - 1)
    add_face(0, 2, 1, 0)
    add_face(0, 2, 1, n - 1)
    add_face(0, 1, 2, 0)
    add_face(0, 1, 2, n - 1)

    if len(tri_a) == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty, empty

    return (
        np.array(tri_a, dtype=np.float32),
        np.array(tri_b, dtype=np.float32),
        np.array(tri_c, dtype=np.float32),
    )


def build_floor_grid_3d():
    """Build projected line segments for the floor surface grid."""
    xs = np.linspace(-.5, 1.5, floor_samples, dtype=np.float32)
    ys = np.linspace(-.5, 1.5, floor_samples, dtype=np.float32)

    begin = []
    end = []

    # lines of constant y
    for y in ys:
        z = z_floor_np(xs, y)
        pts = np.stack([xs, np.full_like(xs, y), z], axis=1)
        begin.append(pts[:-1])
        end.append(pts[1:])

    # lines of constant x
    for x in xs:
        z = z_floor_np(x, ys)
        pts = np.stack([np.full_like(ys, x), ys, z], axis=1)
        begin.append(pts[:-1])
        end.append(pts[1:])

    begin = np.concatenate(begin, axis=0)
    end = np.concatenate(end, axis=0)
    return begin, end


def validate_render_settings():
    for name, value in [
        ("fill_stride", fill_stride),
        ("wire_stride", wire_stride),
        ("point_stride", point_stride),
    ]:
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")


def apply_material_preset(preset_name: str, stiffness_scale: float = 1.0, damping_scale: float = 1.0, spawn_z_override=None):
    global stiffness, damping, air_drag
    global floor_friction, floor_contact_eps, floor_restitution
    global object_restitution, floor_impact_damping, object_impact_damping
    global structural_stiffness_scale, shear_stiffness_scale, bending_stiffness_scale, volume_stiffness, volume_damping
    global particle_mass

    global masked_surface_hull_cache

    if preset_name not in MATERIAL_PRESETS:
        names = ", ".join(sorted(MATERIAL_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{preset_name}'. Expected one of: {names}")
    preset = MATERIAL_PRESETS[preset_name]
    masked_surface_hull_cache = {}

    stiffness = float(preset["stiffness"]) * float(stiffness_scale)
    damping = float(preset["damping"]) * float(damping_scale)
    air_drag = float(preset["air_drag"])
    structural_stiffness_scale = float(preset["structural_stiffness_scale"])
    shear_stiffness_scale = float(preset["shear_stiffness_scale"])
    bending_stiffness_scale = float(preset["bending_stiffness_scale"])
    volume_stiffness = float(preset.get("volume_stiffness", 0.0))
    volume_damping = float(preset.get("volume_damping", 0.0))
    floor_friction = float(preset["floor_friction"])
    floor_contact_eps = float(floor_contact_eps)
    floor_restitution = float(preset["floor_restitution"])
    object_restitution = float(preset["object_restitution"])
    floor_impact_damping = float(preset["floor_impact_damping"])
    object_impact_damping = float(preset["object_impact_damping"])
    particle_mass = float(max(1e-5, particle_mass))

    shape_name = str(preset["shape"])
    if shape_name not in SHAPE_NAME_TO_MODE:
        names = ", ".join(sorted(SHAPE_NAME_TO_MODE.keys()))
        raise ValueError(f"Unknown shape '{shape_name}' in preset '{preset_name}'. Expected one of: {names}")
    mode = int(SHAPE_NAME_TO_MODE[shape_name])
    shape_scale = preset.get("shape_scale", (1.0, 1.0, 1.0))
    spawn_z = float(preset.get("spawn_z", 0.6))
    if spawn_z_override is not None:
        spawn_z = float(spawn_z_override)

    sim_spring_stiffness[None] = stiffness
    sim_spring_damping[None] = damping
    sim_air_drag[None] = air_drag
    sim_structural_stiffness_scale[None] = structural_stiffness_scale
    sim_shear_stiffness_scale[None] = shear_stiffness_scale
    sim_bending_stiffness_scale[None] = bending_stiffness_scale
    sim_volume_stiffness[None] = volume_stiffness
    sim_volume_damping[None] = volume_damping
    sim_particle_mass[None] = particle_mass
    sim_floor_friction[None] = floor_friction
    sim_floor_contact_eps[None] = floor_contact_eps
    sim_floor_restitution[None] = floor_restitution
    sim_object_restitution[None] = object_restitution
    sim_floor_impact_damping[None] = floor_impact_damping
    sim_object_impact_damping[None] = object_impact_damping
    object_shape_mode[None] = mode
    object_center[None] = (0.5, 0.5, spawn_z)
    object_scale[None] = (
        float(shape_scale[0]),
        float(shape_scale[1]),
        float(shape_scale[2]),
    )
    print(f"Preset: {preset_name} | shape: {shape_name}")


def step_simulation(steps: int = 1):
    for _ in range(int(max(0, steps))):
        clear_forces()
        apply_springs()
        if sim_volume_stiffness[None] > 0.0 or sim_volume_damping[None] > 0.0:
            compute_active_center()
            apply_volume_preservation()
        integrate(dt)


def snapshot_state():
    pos_np = pos.to_numpy()
    active_np = active.to_numpy().astype(bool)
    p = pos_np[active_np]
    if len(p) == 0:
        return {
            "com_z": float("nan"),
            "z_min": float("nan"),
            "z_max": float("nan"),
            "height": float("nan"),
        }
    z_min = float(p[:, 2].min())
    z_max = float(p[:, 2].max())
    com_z = float(p[:, 2].mean())
    return {
        "com_z": com_z,
        "z_min": z_min,
        "z_max": z_max,
        "height": z_max - z_min,
    }


def simulate_and_record(total_steps: int, sample_every: int = 8):
    times = []
    com_z = []
    heights = []
    for step in range(int(max(0, total_steps))):
        step_simulation(1)
        if step % max(1, int(sample_every)) == 0:
            s = snapshot_state()
            times.append((step + 1) * dt)
            com_z.append(s["com_z"])
            heights.append(s["height"])
    return np.array(times, dtype=np.float32), np.array(com_z, dtype=np.float32), np.array(heights, dtype=np.float32)


def local_extrema_1d(values: np.ndarray):
    if len(values) < 3:
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty
    dv = np.diff(values)
    mins = np.where((dv[:-1] < 0.0) & (dv[1:] >= 0.0))[0] + 1
    maxs = np.where((dv[:-1] > 0.0) & (dv[1:] <= 0.0))[0] + 1
    return mins.astype(np.int32), maxs.astype(np.int32)


def compute_settling_time(times: np.ndarray, values: np.ndarray, start_idx: int):
    if len(times) == 0 or len(values) == 0:
        return float("nan")
    tail = max(5, len(values) // 8)
    target = float(np.mean(values[-tail:]))
    tol = max(0.003, 0.015 * max(abs(float(values[0]) - target), 1e-3))
    window = 5
    for i in range(int(max(0, start_idx)), max(0, len(values) - window)):
        if np.max(np.abs(values[i : i + window] - target)) < tol:
            return float(times[i] - times[start_idx])
    return float("nan")


def run_drop_test(preset_name: str, stiffness_scale: float, damping_scale: float):
    base_spawn = float(MATERIAL_PRESETS[preset_name].get("spawn_z", 0.6))
    apply_material_preset(
        preset_name,
        stiffness_scale=stiffness_scale,
        damping_scale=damping_scale,
        spawn_z_override=base_spawn + 0.1,
    )
    init()
    times, com_z, _ = simulate_and_record(total_steps=int(2.5 / dt), sample_every=8)

    floor_ref = float(np.asarray(z_floor_np(np.array([0.5], dtype=np.float32), np.array([0.5], dtype=np.float32))).reshape(-1)[0])
    if len(com_z) == 0:
        return {
            "drop_height": float("nan"),
            "rebound_ratio": float("nan"),
            "settling_time_s": float("nan"),
        }
    h0 = max(1e-6, float(com_z[0]) - floor_ref)
    mins, maxs = local_extrema_1d(com_z)
    rebound_ratio = 0.0
    settling_time_s = float("nan")
    impact_idx = 0
    if len(mins) > 0:
        impact_idx = int(mins[0])
        candidates = maxs[maxs > impact_idx]
        if len(candidates) > 0:
            h1 = max(0.0, float(com_z[int(candidates[0])]) - floor_ref)
            rebound_ratio = h1 / h0
        settling_time_s = compute_settling_time(times, com_z, impact_idx)

    return {
        "drop_height": h0,
        "rebound_ratio": rebound_ratio,
        "settling_time_s": settling_time_s,
    }


def run_compression_test(preset_name: str, stiffness_scale: float, damping_scale: float):
    apply_material_preset(
        preset_name,
        stiffness_scale=stiffness_scale,
        damping_scale=damping_scale,
    )
    init()
    step_simulation(int(0.5 / dt))
    s0 = snapshot_state()
    h_ref = max(1e-6, float(s0["height"]))
    center_z = float(s0["com_z"])

    scale_object_height(center_z, 0.62)
    times, _, heights = simulate_and_record(total_steps=int(1.8 / dt), sample_every=8)
    if len(heights) == 0:
        return {
            "height_ref": h_ref,
            "max_strain": float("nan"),
            "recovery_time_s": float("nan"),
        }

    h_min = float(np.min(heights))
    max_strain = max(0.0, (h_ref - h_min) / h_ref)
    recovery_threshold = h_ref * (1.0 - 0.05 * max_strain)
    recovery_time_s = float("nan")
    for t, h in zip(times, heights):
        if float(h) >= recovery_threshold:
            recovery_time_s = float(t)
            break

    return {
        "height_ref": h_ref,
        "max_strain": max_strain,
        "recovery_time_s": recovery_time_s,
    }


def evaluate_preset_metrics(preset_name: str, stiffness_scale: float = 1.0, damping_scale: float = 1.0):
    drop = run_drop_test(preset_name, stiffness_scale, damping_scale)
    compression = run_compression_test(preset_name, stiffness_scale, damping_scale)
    return {
        "preset": preset_name,
        "shape": MATERIAL_PRESETS[preset_name]["shape"],
        "stiffness": float(MATERIAL_PRESETS[preset_name]["stiffness"]) * float(stiffness_scale),
        "spring_damping": float(MATERIAL_PRESETS[preset_name]["damping"]) * float(damping_scale),
        "air_drag": float(MATERIAL_PRESETS[preset_name]["air_drag"]),
        "volume_stiffness": float(MATERIAL_PRESETS[preset_name].get("volume_stiffness", 0.0)),
        "volume_damping": float(MATERIAL_PRESETS[preset_name].get("volume_damping", 0.0)),
        "object_restitution": float(MATERIAL_PRESETS[preset_name]["object_restitution"]),
        "floor_restitution": float(MATERIAL_PRESETS[preset_name]["floor_restitution"]),
        "drop_height": drop["drop_height"],
        "rebound_ratio": drop["rebound_ratio"],
        "settling_time_s": drop["settling_time_s"],
        "max_strain": compression["max_strain"],
        "recovery_time_s": compression["recovery_time_s"],
    }


def write_metrics_csv(rows, csv_path: str):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_metrics_mode(preset_name: str, csv_path: str, stiffness_scale: float, damping_scale: float):
    if floor_mode != FLOOR_FLAT:
        print("Warning: metrics are easiest to interpret on FLOOR_FLAT (floor_mode = 0).")

    if preset_name == "all":
        presets = list(MATERIAL_PRESETS.keys())
    else:
        presets = [preset_name]

    rows = []
    for name in presets:
        row = evaluate_preset_metrics(name, stiffness_scale=stiffness_scale, damping_scale=damping_scale)
        rows.append(row)

    write_metrics_csv(rows, csv_path)
    print(f"Saved metrics to {csv_path}")
    for row in rows:
        print(
            f"{row['preset']:12s} shape={row['shape']:9s} rebound={row['rebound_ratio']:.3f} "
            f"settle_s={row['settling_time_s']:.3f} strain={row['max_strain']:.3f} "
            f"recover_s={row['recovery_time_s']:.3f}"
        )



def main(window_title: str = "Soft Blob 3D (Projected)"):
    global view_azimuth, view_pitch, view_scale
    validate_render_settings()
    gui = ti.GUI(window_title, res=(gui_res, gui_res))
    init()

    cam_azimuth = view_azimuth
    cam_pitch = view_pitch
    cam_scale = view_scale
    last_pos = None
    last_cam_azimuth = cam_azimuth
    last_cam_pitch = cam_pitch
    last_cam_scale = cam_scale
    floor_begin3, floor_end3 = build_floor_grid_3d()
    floor_proj_dirty = True
    floor_begin2 = floor_end2 = None
    floor_depth = None

    while gui.running:
        if free_cam:
            gui.get_event()
            cur_pos = gui.get_cursor_pos()
            lmb = gui.is_pressed(ti.GUI.LMB)
            rmb = gui.is_pressed(ti.GUI.RMB)
            if lmb or rmb:
                if last_pos is None:
                    last_pos = cur_pos
                dx = cur_pos[0] - last_pos[0]
                dy = cur_pos[1] - last_pos[1]
                if lmb:
                    cam_azimuth += dx * cam_rotate_speed
                    cam_pitch -= dy * cam_rotate_speed
                if rmb:
                    cam_scale *= math.exp(-dy * cam_zoom_speed)
                    cam_scale = float(np.clip(cam_scale, cam_min_scale, cam_max_scale))
                last_pos = cur_pos
            else:
                last_pos = None

            # apply camera to projection globals
            view_azimuth = cam_azimuth
            view_pitch = cam_pitch
            view_scale = cam_scale
            if (
                cam_azimuth != last_cam_azimuth
                or cam_pitch != last_cam_pitch
                or cam_scale != last_cam_scale
            ):
                floor_proj_dirty = True
                last_cam_azimuth = cam_azimuth
                last_cam_pitch = cam_pitch
                last_cam_scale = cam_scale

        for _ in range(substeps):
            clear_forces()
            apply_springs()
            if sim_volume_stiffness[None] > 0.0 or sim_volume_damping[None] > 0.0:
                compute_active_center()
                apply_volume_preservation()
            integrate(dt)

        pos_np = pos.to_numpy()
        active_np = active.to_numpy().astype(bool)
        if render_surface_only:
            draw_points = collect_surface_points(pos_np, active_np)
        else:
            draw_points = pos_np[active_np]
        if depth_shading:
            proj_points, depth = project_points_with_depth(draw_points)
        else:
            proj_points = project_points(draw_points)
            depth = None

        tri_a3, tri_b3, tri_c3 = build_surface_triangles(pos_np, active_np)
        if render_filled_surface and len(tri_a3) > 0:
            if depth_shading:
                tri_a2, tri_depth_a = project_points_with_depth(tri_a3)
                tri_b2, tri_depth_b = project_points_with_depth(tri_b3)
                tri_c2, tri_depth_c = project_points_with_depth(tri_c3)
                tri_depth = (tri_depth_a + tri_depth_b + tri_depth_c) / 3.0
            else:
                tri_a2 = project_points(tri_a3)
                tri_b2 = project_points(tri_b3)
                tri_c2 = project_points(tri_c3)
                tri_depth = None
        else:
            tri_a2 = tri_b2 = tri_c2 = np.zeros((0, 2), dtype=np.float32)
            tri_depth = None

        line_begin3, line_end3 = build_surface_wireframe(pos_np, active_np)
        if depth_shading:
            line_begin2, depth_b = project_points_with_depth(line_begin3)
            line_end2, depth_e = project_points_with_depth(line_end3)
            line_depth = 0.5 * (depth_b + depth_e)
        else:
            line_begin2 = project_points(line_begin3)
            line_end2 = project_points(line_end3)
            line_depth = None
        if floor_proj_dirty:
            if depth_shading:
                floor_begin2, floor_depth_b = project_points_with_depth(floor_begin3)
                floor_end2, floor_depth_e = project_points_with_depth(floor_end3)
                floor_depth = 0.5 * (floor_depth_b + floor_depth_e)
            else:
                floor_begin2 = project_points(floor_begin3)
                floor_end2 = project_points(floor_end3)
                floor_depth = None
            floor_proj_dirty = False

        gui.clear(0x000000)
        # draw floor first so cube sits on top in 2D projection
        if depth_shading:
            bin_count = max(1, int(depth_bins))
            if bin_count == 1:
                colors = [floor_near_color]
            else:
                colors = [
                    lerp_color(floor_far_color, floor_near_color, i / (bin_count - 1))
                    for i in range(bin_count)
                ]
            if floor_depth is not None and len(floor_depth) > 0:
                dmin = float(floor_depth.min())
                dmax = float(floor_depth.max())
                if dmax - dmin < 1e-6:
                    bin_ids = np.zeros_like(floor_depth, dtype=np.int32)
                else:
                    bins = np.linspace(dmin, dmax, bin_count + 1, dtype=np.float32)
                    bin_ids = np.clip(np.searchsorted(bins, floor_depth, side="right") - 1, 0, bin_count - 1)
                for i in range(bin_count):
                    idx = bin_ids == i
                    if np.any(idx):
                        gui.lines(floor_begin2[idx], floor_end2[idx], radius=1, color=colors[i])
        else:
            gui.lines(floor_begin2, floor_end2, radius=.5)
        if render_filled_surface and len(tri_a2) > 0:
            if depth_shading and tri_depth is not None:
                dmin = float(tri_depth.min())
                dmax = float(tri_depth.max())
                if dmax - dmin < 1e-6:
                    order = np.arange(len(tri_depth))
                    tri_colors = np.full(len(tri_depth), surface_near_color, dtype=np.int32)
                else:
                    t = np.clip((tri_depth - dmin) / (dmax - dmin), 0.0, 1.0)
                    order = np.argsort(tri_depth)
                    tri_colors = np.array(
                        [lerp_color(surface_far_color, surface_near_color, depth_t) for depth_t in t],
                        dtype=np.int32,
                    )
                for tri_idx in order:
                    gui.triangle(
                        tri_a2[tri_idx],
                        tri_b2[tri_idx],
                        tri_c2[tri_idx],
                        color=int(tri_colors[tri_idx]),
                    )
            else:
                gui.triangles(tri_a2, tri_b2, tri_c2, color=surface_flat_color)
        draw_surface_overlay = (not render_filled_surface) or render_surface_overlay or len(tri_a2) == 0
        if depth_shading and draw_surface_overlay:
            bin_count = max(1, int(depth_bins))
            if bin_count == 1:
                colors = [depth_near_color]
            else:
                colors = [
                    lerp_color(depth_far_color, depth_near_color, i / (bin_count - 1))
                    for i in range(bin_count)
                ]

            if line_depth is not None and len(line_depth) > 0:
                dmin = float(line_depth.min())
                dmax = float(line_depth.max())
                if dmax - dmin < 1e-6:
                    bin_ids = np.zeros_like(line_depth, dtype=np.int32)
                else:
                    bins = np.linspace(dmin, dmax, bin_count + 1, dtype=np.float32)
                    bin_ids = np.clip(np.searchsorted(bins, line_depth, side="right") - 1, 0, bin_count - 1)
                for i in range(bin_count):
                    idx = bin_ids == i
                    if np.any(idx):
                        if bin_count == 1:
                            t = 1.0
                        else:
                            t = i / (bin_count - 1)
                        radius = line_radius * (
                            line_radius_far_scale + (line_radius_near_scale - line_radius_far_scale) * t
                        )
                        gui.lines(line_begin2[idx], line_end2[idx], radius=radius, color=colors[i])

            if depth is not None and len(depth) > 0:
                dmin = float(depth.min())
                dmax = float(depth.max())
                if dmax - dmin < 1e-6:
                    bin_ids = np.zeros_like(depth, dtype=np.int32)
                else:
                    bins = np.linspace(dmin, dmax, bin_count + 1, dtype=np.float32)
                    bin_ids = np.clip(np.searchsorted(bins, depth, side="right") - 1, 0, bin_count - 1)
                for i in range(bin_count):
                    idx = bin_ids == i
                    if np.any(idx):
                        gui.circles(proj_points[idx], radius=point_radius, color=colors[i])
        elif draw_surface_overlay:
            if len(line_begin2) > 0:
                gui.lines(line_begin2, line_end2, radius=line_radius, color=0xFFFFFF)
            gui.circles(proj_points, radius=point_radius)
        gui.show()

if __name__ == "__main__":
    apply_material_preset(active_preset, stiffness_scale=stiffness_scale, damping_scale=damping_scale)
    main(window_title=f"Soft Blob 3D – {active_preset}")
