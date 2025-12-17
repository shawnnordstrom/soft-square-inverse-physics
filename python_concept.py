import taichi as ti
import numpy as np

"""
Shawn Nordstrom
12/01/2025
"""

ti.init(arch=ti.gpu)

# ----------------------------
# Global configurations
# ----------------------------
n = 20 # n x n grid
dx = 0.03 # rest length between neighbors
dt = 5e-4 # physics time step
substeps = 20 # physics steps per rendered frame

gravity = ti.Vector([0.0, -9.8])
y_floor = 0.1

laplacian_smoothing = 5

stiffness = ti.field(dtype=ti.f32, shape=())  # k
damping   = ti.field(dtype=ti.f32, shape=())  # c

pos   = ti.Vector.field(2, dtype=ti.f32, shape=(n, n))
vel   = ti.Vector.field(2, dtype=ti.f32, shape=(n, n))
force = ti.Vector.field(2, dtype=ti.f32, shape=(n, n))

def _print_progress(step: int, total: int, prefix: str = ""):
    if total <= 0:
        return
    pct = 100.0 * step / total
    print(f"\r{prefix}{step}/{total} ({pct:.1f}%)", end="", flush=True)
    if step >= total:
        print()

@ti.kernel
def init():
    for i, j in pos:
        x = 0.5 + (i - (n - 1) * 0.5) * dx
        y = 0.6 + (j - (n - 1) * 0.5) * dx
        pos[i, j] = ti.Vector([x, y])
        vel[i, j] = ti.Vector([0.0, 0.0])
        force[i, j] = ti.Vector([0.0, 0.0])


@ti.func
def add_spring(i1: int, j1: int, i2: int, j2: int, rest_len: ti.f32):
    p1 = pos[i1, j1]
    p2 = pos[i2, j2]
    d = p2 - p1
    L = d.norm() + 1e-6
    dir = d / L
    extension = L - rest_len
    k = stiffness[None]
    F = -k * extension * dir
    force[i1, j1] += F
    force[i2, j2] -= F


@ti.kernel
def clear_forces():
    for i, j in force:
        force[i, j] = gravity


@ti.kernel
def apply_springs():
    for i, j in pos:
        if i + 1 < n:
            add_spring(i, j, i + 1, j, dx)
        if j + 1 < n:
            add_spring(i, j, i, j + 1, dx)
        if i + 1 < n and j + 1 < n:
            add_spring(i, j, i + 1, j + 1, dx * ti.sqrt(2.0))
        if i + 1 < n and j - 1 >= 0:
            add_spring(i, j, i + 1, j - 1, dx * ti.sqrt(2.0))
        if i + 2 < n:
            add_spring(i, j, i + 2, j, 2.0 * dx)
        if j + 2 < n:
            add_spring(i, j, i, j + 2, 2.0 * dx)


@ti.kernel
def integrate(dt: ti.f32):
    c = damping[None]
    for i, j in pos:
        force[i, j] += -c * vel[i, j]

        acc = force[i, j]
        vel[i, j] += acc * dt
        pos[i, j] += vel[i, j] * dt

        if pos[i, j].y < y_floor:
            pos[i, j].y = y_floor
            if vel[i, j].y < 0.0:
                vel[i, j].y *= -0.3  # restitution

def build_outline_from_grid(pos_np: np.ndarray):
    boundary = []

    for i in range(n):
        boundary.append(pos_np[i, n - 1])
    for j in range(n - 2, -1, -1):
        boundary.append(pos_np[n - 1, j])
    for i in range(n - 2, -1, -1):
        boundary.append(pos_np[i, 0])
    for j in range(1, n - 1):
        boundary.append(pos_np[0, j])

    pts = np.array(boundary, dtype=np.float32)

    # Laplacian smoothing (visual only)
    for _ in range(laplacian_smoothing):
        pts = (
            0.25 * np.roll(pts, 1, axis=0)
            + 0.5 * pts
            + 0.25 * np.roll(pts, -1, axis=0)
        )

    begin = []
    end = []
    m = len(pts)
    for k in range(m):
        p = pts[k]
        q = pts[(k + 1) % m]
        begin.append([p[0], p[1]])
        end.append([q[0], q[1]])

    return np.array(begin, dtype=np.float32), np.array(end, dtype=np.float32)


def run_gui_demo():
    gui = ti.GUI("Soft Square", res=(800, 800))
    stiffness[None] = 24.0
    damping[None]   = 2.0
    init()

    while gui.running:
        for _ in range(substeps):
            clear_forces()
            apply_springs()
            integrate(dt)

        pos_np = pos.to_numpy()
        interior = pos_np[1:n-1, 1:n-1].reshape(-1, 2)
        line_begin, line_end = build_outline_from_grid(pos_np)

        gui.clear(0x000000)
        gui.lines(line_begin, line_end, radius=2)
        gui.circles(interior, radius=2)
        gui.show()

def simulate_trajectory(k_value: float,
                        c_value: float,
                        num_frames: int,
                        substeps_per_frame: int = substeps) -> np.ndarray:
    stiffness[None] = k_value
    damping[None]   = c_value
    init()

    frames = []
    for _ in range(num_frames):
        for _ in range(substeps_per_frame):
            clear_forces()
            apply_springs()
            integrate(dt)
        frames.append(pos.to_numpy().copy())  # (n, n, 2)

    return np.stack(frames, axis=0)  # (T, n, n, 2)

def compute_com(traj: np.ndarray) -> np.ndarray:
    T = traj.shape[0]
    com = traj.reshape(T, -1, 2).mean(axis=1)
    return com


def compute_vertical_extent(traj: np.ndarray) -> np.ndarray:
    T = traj.shape[0]
    y = traj[..., 1].reshape(T, -1)
    y_min = y.min(axis=1)
    y_max = y.max(axis=1)
    return y_max - y_min


def compute_features(traj: np.ndarray,
                     dt_frame: float) -> np.ndarray:
    T = traj.shape[0]
    com = compute_com(traj)              # (T, 2)
    extent_y = compute_vertical_extent(traj)  # (T,)

    # velocities from COM
    if T < 2:
        vel = np.zeros((1, 2), dtype=np.float32)
    else:
        vel = np.diff(com, axis=0) / dt_frame  # (T-1, 2)
    vy = vel[:, 1]
    speed = np.linalg.norm(vel, axis=1)

    # 1) max compression ratio: min extent / initial extent
    initial_extent = extent_y[0]
    compression_ratio = extent_y / (initial_extent + 1e-6)
    max_compression_ratio = compression_ratio.min()  # <= 1

    # 2) settling time: first time after which COM height & velocity stay near final
    com_y = com[:, 1]
    final_y = com_y[-1]
    final_vy = vy[-1] if len(vy) > 0 else 0.0

    eps_y = 0.01  # positional tolerance
    eps_v = 0.05  # velocity tolerance

    settling_index = T - 1
    for t in range(T):
        if (np.all(np.abs(com_y[t:] - final_y) < eps_y) and
                np.all(np.abs(vy[max(t - 1, 0):] - final_vy) < eps_v)):
            settling_index = t
            break
    settling_time = settling_index * dt_frame

    # 3) max COM speed
    max_speed = speed.max() if len(speed) > 0 else 0.0

    # 4) bounce count: number of times vy goes from negative to non-negative
    bounce_count = 0
    for t in range(1, len(vy)):
        if vy[t - 1] < 0 and vy[t] >= 0:
            bounce_count += 1

    features = np.array([
        max_compression_ratio,
        settling_time,
        max_speed,
        float(bounce_count),
    ], dtype=np.float32)

    return features

def inverse_physics_predict(traj_obs: np.ndarray,
                            k_grid: np.ndarray,
                            c_grid: np.ndarray,
                            lambda_scale: float,
                            num_frames: int,
                            dt_frame: float,
                            noise_scale: float = 0.0,
                            show_progress: bool = False) -> np.ndarray:
    phi_obs = compute_features(traj_obs, dt_frame)

    if noise_scale > 0.0:
        phi_obs = phi_obs + np.random.normal(
            0.0, noise_scale, size=phi_obs.shape
        )

    best_theta = None
    best_score = -np.inf

    total = len(k_grid) * len(c_grid)
    step = 0
    for k_val in k_grid:
        for c_val in c_grid:
            traj_hat = simulate_trajectory(k_val, c_val, num_frames)
            phi_hat = compute_features(traj_hat, dt_frame)
            d = np.sum((phi_obs - phi_hat) ** 2)
            score = -lambda_scale * d 
            if score > best_score:
                best_score = score
                best_theta = (k_val, c_val)
            step += 1
            if show_progress:
                _print_progress(step, total, prefix="Inverse search ")

    return np.array(best_theta, dtype=np.float32)

def generate_dataset(k_grid: np.ndarray,
                     c_grid: np.ndarray,
                     num_frames: int,
                     dt_frame: float,
                     show_progress: bool = False):
    X_list = []
    Y_list = []
    trajs = []
    total = len(k_grid) * len(c_grid)
    step = 0
    for k_val in k_grid:
        for c_val in c_grid:
            traj = simulate_trajectory(k_val, c_val, num_frames)
            feats = compute_features(traj, dt_frame)
            X_list.append(feats)
            Y_list.append([k_val, c_val])
            trajs.append(traj)
            step += 1
            if show_progress:
                _print_progress(step, total, prefix="Dataset sim ")
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y, trajs


def train_linear_regressor(X_train: np.ndarray,
                           Y_train: np.ndarray):
    N, d = X_train.shape
    X_aug = np.hstack([X_train, np.ones((N, 1), dtype=X_train.dtype)])  # (N, d+1)
    # W_full: (d+1, 2)
    W_full, _, _, _ = np.linalg.lstsq(X_aug, Y_train, rcond=None)
    W = W_full[:-1, :]  # (d, 2)
    b = W_full[-1, :]   # (2,)
    return W, b


def feature_model_predict(traj: np.ndarray,
                          W: np.ndarray,
                          b: np.ndarray,
                          dt_frame: float) -> np.ndarray:
    """
    Predict (k, c) for a single trajectory using linear regressor.
    """
    feats = compute_features(traj, dt_frame)  # (d,)
    return feats @ W + b

def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def example_training_and_eval():
    # Define parameter ranges
    k_min, k_max = 10.0, 40.0
    c_min, c_max = 1.0, 5.0

    num_k_data, num_c_data = 7, 7  # 7x7 = 49 trajectories total
    k_vals_data = np.linspace(k_min, k_max, num_k_data)
    c_vals_data = np.linspace(c_min, c_max, num_c_data)

    num_k_inf, num_c_inf = 7, 7  # 5x5 candidate parameters
    k_grid_inf = np.linspace(k_min, k_max, num_k_inf)
    c_grid_inf = np.linspace(c_min, c_max, num_c_inf)

    num_frames = 60  # frames per video
    dt_frame = substeps * dt

    print("Generating dataset...")
    X, Y, trajs = generate_dataset(
        k_vals_data,
        c_vals_data,
        num_frames,
        dt_frame,
        show_progress=True,
    )

    # Train/test split (80/20)
    N = X.shape[0]
    perm = np.random.permutation(N)
    split = int(0.8 * N)
    idx_train = perm[:split]
    idx_test = perm[split:]

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_test, Y_test = X[idx_test], Y[idx_test]

    print("Training feature-based linear regressor...")
    W, b = train_linear_regressor(X_train, Y_train)
    preds_feat = X_test @ W + b  # (N_test, 2)

    print("Evaluating inverse-physics model...")
    lambda_scale = 10.0
    noise_scale = 0.001  # small observation noise on features
    preds_phys = []
    for idx in idx_test:
        traj_obs = trajs[idx]  # reuse simulated traj as "observation"
        theta_hat = inverse_physics_predict(
            traj_obs,
            k_grid_inf, c_grid_inf,
            lambda_scale=lambda_scale,
            num_frames=num_frames,
            dt_frame=dt_frame,
            noise_scale=noise_scale,
            show_progress=True,
        )
        preds_phys.append(theta_hat)
    preds_phys = np.stack(preds_phys, axis=0)

    # Compute R^2 for each model and parameter
    r2_phys_k = r2_score(Y_test[:, 0], preds_phys[:, 0])
    r2_phys_c = r2_score(Y_test[:, 1], preds_phys[:, 1])
    r2_feat_k = r2_score(Y_test[:, 0], preds_feat[:, 0])
    r2_feat_c = r2_score(Y_test[:, 1], preds_feat[:, 1])

    print("Inverse-physics R^2 stiffness:", r2_phys_k)
    print("Inverse-physics R^2 damping :", r2_phys_c)
    print("Feature model R^2 stiffness :", r2_feat_k)
    print("Feature model R^2 damping  :", r2_feat_c)

if __name__ == "__main__":
    example_training_and_eval()
