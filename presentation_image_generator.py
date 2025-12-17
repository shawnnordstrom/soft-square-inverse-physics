from PIL import Image, ImageDraw
import numpy as np

SCALE = 3
BASE_SIZE = 900
size = (BASE_SIZE * SCALE, BASE_SIZE * SCALE)
background = (0, 0, 0, 0) 
img = Image.new("RGBA", size, background)
draw = ImageDraw.Draw(img)

cx, cy = size[0] // 2, size[1] // 2
half = 180 * SCALE             
offset = (-120 * SCALE, -90 * SCALE)
line_width_outer = 4 * SCALE
line_width_inner = 2 * SCALE

outer_color = (0, 0, 0, 0)
inner_line_color = (0, 0, 0, 255)
corner_radius = 0  
corner_fill = (0, 0, 0, 255)
corner_outline = (0, 0, 0, 255)
grid_point_radius = 5 * SCALE
grid_point_fill = (0, 0, 0, 255)

front = [
    (cx - half, cy - half),
    (cx + half, cy - half),
    (cx + half, cy + half),
    (cx - half, cy + half),
]
back = [(x + offset[0], y + offset[1]) for (x, y) in front]

def draw_edges(points, color, width):
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        draw.line((x1, y1, x2, y2), fill=color, width=width)

draw_edges(front, outer_color, line_width_outer)
draw_edges(back, outer_color, line_width_outer)
for (fx, fy), (bx, by) in zip(front, back):
    draw.line((fx, fy, bx, by), fill=outer_color, width=line_width_outer)

def draw_corner(x, y):
    r = corner_radius
    draw.ellipse(
        (x - r, y - r, x + r, y + r),
        fill=corner_fill,
        outline=corner_outline,
        width=1
    )

for x, y in front + back:
    draw_corner(x, y)

divisions = 5  
coords = np.linspace(-1.0, 1.0, divisions) 

def project_point(u, v, w):
    x0 = cx + u * half
    y0 = cy + v * half

    z = (w + 1) / 2.0

    x = x0 + z * offset[0]
    y = y0 + z * offset[1]
    return x, y

def deform(u, v, w):
    r = (u**2 + v**2 + w**2) ** 0.5

    squash_strength = 0.05
    scale = 1.0 - squash_strength * (r**2)

    gravity_strength = 0.75
    v_sag = v - gravity_strength * (1.0 - r)

    u2 = u * scale
    v2 = v_sag * scale
    w2 = w * scale
    return u2, v2, w2

grid_points_3d = []
grid_points_2d = {}

for i, u in enumerate(coords):
    for j, v in enumerate(coords):
        for k, w in enumerate(coords):
            grid_points_3d.append((i, j, k))
            du, dv, dw = deform(u, v, w)
            x, y = project_point(du, dv, dw)
            grid_points_2d[(i, j, k)] = (x, y)

for j in range(divisions):
    for k in range(divisions):
        for i1 in range(divisions - 1):
            i2 = i1 + 1
            x1, y1 = grid_points_2d[(i1, j, k)]
            x2, y2 = grid_points_2d[(i2, j, k)]
            draw.line((x1, y1, x2, y2), fill=inner_line_color, width=line_width_inner)

for i in range(divisions):
    for k in range(divisions):
        for j1 in range(divisions - 1):
            j2 = j1 + 1
            x1, y1 = grid_points_2d[(i, j1, k)]
            x2, y2 = grid_points_2d[(i, j2, k)]
            draw.line((x1, y1, x2, y2), fill=inner_line_color, width=line_width_inner)

for i in range(divisions):
    for j in range(divisions):
        for k1 in range(divisions - 1):
            k2 = k1 + 1
            x1, y1 = grid_points_2d[(i, j, k1)]
            x2, y2 = grid_points_2d[(i, j, k2)]
            draw.line((x1, y1, x2, y2), fill=inner_line_color, width=line_width_inner)

for (i, j, k), (x, y) in grid_points_2d.items():
    r = grid_point_radius
    draw.ellipse(
        (x - r, y - r, x + r, y + r),
        fill=grid_point_fill
    )

final = img.resize((BASE_SIZE, BASE_SIZE), resample=Image.LANCZOS)
final.save("cube_density_grid.png", "PNG")
print("Saved cube_density_grid.png")