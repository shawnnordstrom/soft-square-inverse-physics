import taichi as ti
import numpy as np

"""
Shawn Nordstrom
11/14/2025
"""

ti.init(arch=ti.gpu)


#                       CONFIGURATION  
# =========================================================
n = 20          # nxn grid | default: 20x20
dx = 0.03       # at rest len between particle i,j and particle E{(i+-1,j+-1)} | default: 0.03

dt = 5e-4       # ts
substeps = 20   # physics steps per frame

gravity   = ti.Vector([0.0, -9.8])
stiffness = 24.0 # 15000  # (K) Trials: 5000, 15000, 250000, 500000
damping   = 2.0      # (Velocity damping) Trials: 2, 5, 10 (for some reason > 100 makes it float?)

y_floor = 0.1        # ground height

laplacian_smoothing = 5 # Laplacian smoothing for outer line (doesn't change physics) Default = 3
# =========================================================
#
# ---------------------------------------------------------
# REMEMBER TO SAVE AFTER ANY CHANGE (otherwise hitting run 
# will just run the prior runs's settings)
# ---------------------------------------------------------
pos   = ti.Vector.field(2, dtype=ti.f32, shape=(n, n))
vel   = ti.Vector.field(2, dtype=ti.f32, shape=(n, n))
force = ti.Vector.field(2, dtype=ti.f32, shape=(n, n))


@ti.kernel
def init():
    """Make initial block"""
    for i, j in pos:
        x = 0.5 + (i - (n - 1) * 0.5) * dx
        y = 0.6 + (j - (n - 1) * 0.5) * dx
        pos[i, j] = ti.Vector([x, y])
        vel[i, j] = ti.Vector([0.0, 0.0])
        force[i, j] = ti.Vector([0.0, 0.0])


@ti.func
def add_spring(i1: int, j1: int, i2: int, j2: int, rest_len: ti.f32):
    """Apply spring forces."""
    p1 = pos[i1, j1]
    p2 = pos[i2, j2]
    d = p2 - p1
    L = d.norm() + 1e-6
    dir = d / L
    extension = L - rest_len
    F = stiffness**3 * extension * dir
    force[i1, j1] += F
    force[i2, j2] -= F


@ti.kernel
def clear_forces():
    for i, j in force:
        force[i, j] = gravity


@ti.kernel
def apply_springs():
    for i, j in pos:
        # Structural springs (4-neighbour)
        if i + 1 < n:
            add_spring(i, j, i + 1, j, dx)
        if j + 1 < n:
            add_spring(i, j, i, j + 1, dx)

        # Shear springs (diagonals)
        if i + 1 < n and j + 1 < n:
            add_spring(i, j, i + 1, j + 1, dx * ti.sqrt(2.0))
        if i + 1 < n and j - 1 >= 0:
            add_spring(i, j, i + 1, j - 1, dx * ti.sqrt(2.0))

        # Bending springs (two apart) to smooth kinks
        if i + 2 < n:
            add_spring(i, j, i + 2, j, 2.0 * dx)
        if j + 2 < n:
            add_spring(i, j, i, j + 2, 2.0 * dx)


@ti.kernel
def integrate(dt: ti.f32):
    for i, j in pos:
        # v ← v + a dt = v + (F/m) dt; m = 1
        vel[i, j] += force[i, j] * dt
        vel[i, j] *= ti.exp(-damping * dt)
        pos[i, j] += vel[i, j] * dt

        # ground collision
        if pos[i, j].y < y_floor:
            pos[i, j].y = y_floor
            if vel[i, j].y < 0.0:
                vel[i, j].y *= -0.3


# -------- outline construction + smoothing (NumPy only) --------

def build_outline_from_grid(pos_np: np.ndarray):
    """
    Build a smoothed polygonal outline from the outer grid points.

    pos_np: shape (n, n, 2) in [0,1] x [0,1].
    """
    boundary = []

    # top edge (left -> right)
    for i in range(n):
        boundary.append(pos_np[i, n - 1])
    # right edge (top -> bottom)
    for j in range(n - 2, -1, -1):
        boundary.append(pos_np[n - 1, j])
    # bottom edge (right -> left)
    for i in range(n - 2, -1, -1):
        boundary.append(pos_np[i, 0])
    # left edge (bottom -> top, skipping already-used corners)
    for j in range(1, n - 1):
        boundary.append(pos_np[0, j])

    pts = np.array(boundary, dtype=np.float32)

    # Laplacian smoothing -> only towards *drawn* outline (doesn't change the physics!!)
    for _ in range(laplacian_smoothing):
        pts = (
            0.25 * np.roll(pts, 1, axis=0)
            + 0.5 * pts
            + 0.25 * np.roll(pts, -1, axis=0)
        )

    # convert to line segment endpoints
    begin = []
    end = []
    m = len(pts)
    for k in range(m):
        p = pts[k]
        q = pts[(k + 1) % m]
        begin.append([p[0], p[1]])
        end.append([q[0], q[1]])

    return np.array(begin, dtype=np.float32), np.array(end, dtype=np.float32)


def main():
    gui = ti.GUI("Soft Blob with Smoothed Outline", res=(800, 800))
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

if __name__ == "__main__":
    main()