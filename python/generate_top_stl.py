from pathlib import Path
import math
import numpy as np
from stl import mesh

out_path = Path(__file__).with_name('top.stl')
segments = 16

facets = []

# Stem cylinder
for i in range(segments):
    a0 = 2 * math.pi * i / segments
    a1 = 2 * math.pi * (i + 1) / segments
    x0, y0 = math.cos(a0) * 0.12, math.sin(a0) * 0.12
    x1, y1 = math.cos(a1) * 0.12, math.sin(a1) * 0.12
    # bottom and top of stem
    p0 = (x0, y0, -0.35)
    p1 = (x1, y1, -0.35)
    p2 = (x0, y0, 0.35)
    p3 = (x1, y1, 0.35)
    facets.append((p0, p1, p3))
    facets.append((p0, p3, p2))

# Cone head
for i in range(segments):
    a0 = 2 * math.pi * i / segments
    a1 = 2 * math.pi * (i + 1) / segments
    x0, y0 = math.cos(a0) * 0.12, math.sin(a0) * 0.12
    x1, y1 = math.cos(a1) * 0.12, math.sin(a1) * 0.12
    p0 = (x0, y0, 0.35)
    p1 = (x1, y1, 0.35)
    p2 = (0.0, 0.0, 0.95)
    facets.append((p0, p1, p2))

# Small cap on top
def add_disc(z, radius, segs):
    for i in range(segs):
        a0 = 2 * math.pi * i / segs
        a1 = 2 * math.pi * (i + 1) / segs
        x0, y0 = math.cos(a0) * radius, math.sin(a0) * radius
        x1, y1 = math.cos(a1) * radius, math.sin(a1) * radius
        p0 = (0.0, 0.0, z)
        p1 = (x0, y0, z)
        p2 = (x1, y1, z)
        facets.append((p0, p2, p1))

add_disc(0.95, 0.03, 12)

stl_mesh = mesh.Mesh(np.zeros(len(facets), dtype=mesh.Mesh.dtype))
stl_mesh.vectors = np.array(facets, dtype=np.float32)
stl_mesh.save(out_path)
print(f'Created {out_path}')
