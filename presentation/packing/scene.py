"""Reproducción de los datos DEM, sin recalcular la física durante el render."""
from pathlib import Path
import numpy as np
from manim import (
    BLUE_D, DEGREES, GREY_B, GOLD, Line, Polygon, Sphere, ThreeDScene,
    ValueTracker, VGroup, linear,
)


def depth_edge(start, end, segments=32):
    """Cairo ordena por el centro: segmentar evita ordenar una arista entera."""
    points = np.linspace(start, end, segments + 1)
    return VGroup(*[
        Line(a, b, color=GREY_B, stroke_width=2).set_shade_in_3d()
        for a, b in zip(points[:-1], points[1:])
    ])


def depth_panel(corners, color, opacity, subdivisions=24):
    """Panel transparente con ordenación de profundidad por tesela."""
    origin, along_u, _, along_v = corners
    u, v = along_u - origin, along_v - origin
    tiles = VGroup()
    for i in range(subdivisions):
        for j in range(subdivisions):
            points = [origin + a / subdivisions * u + b / subdivisions * v
                      for a, b in [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]]
            tiles.add(Polygon(*points, fill_color=color, fill_opacity=opacity,
                              stroke_width=0).set_shade_in_3d())
    return tiles


def play_packing(scene, run_time=10):
    path = Path(__file__).with_name('trajectory.npz')
    if not path.exists():
        raise FileNotFoundError(f'Genera primero los datos: python {path.with_name("simulate.py")}')
    with np.load(path) as data:
        positions = data['positions'].copy()
        radii = data['radii'].copy()
        times = data['time'].copy()
        counts = data['active_count'].copy()
        lid_fraction = data['lid_fraction'].copy()
        size = data['initial_box_size'].copy()
        heights = data['box_height'].copy()
    offset = -size / 2
    scene.set_camera_orientation(phi=68 * DEGREES, theta=-55 * DEGREES, zoom=1.35)
    x, y, z = size
    corners = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
                        [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]]) + offset
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
             (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    box = VGroup(*[depth_edge(corners[a], corners[b])
                  for a, b in edges])
    floor = depth_panel(corners[:4], BLUE_D, 0.18)
    lid_panel = depth_panel(corners[4:], GREY_B, 0.25)
    lid_edges = VGroup(*[depth_edge(corners[a], corners[b])
                        for a, b in edges[4:8]])
    lid = VGroup(lid_panel, lid_edges)
    closed_center = lid.get_center().copy()
    spheres = VGroup(*[Sphere(radius=float(r), resolution=(8, 12))
                        .set_color(GOLD).set_stroke(width=0).set_opacity(0) for r in radii])
    tracker = ValueTracker(0)
    visible = np.zeros(len(radii), dtype=bool)
    last_height = float(z)

    def update(_=None):
        nonlocal last_height
        f = np.clip(tracker.get_value() * 60, 0, len(times) - 1)
        a = int(f)
        b = min(a + 1, len(times) - 1)
        fraction = f - a
        for i, sphere in enumerate(spheres):
            if i >= counts[a]:
                continue
            p = positions[a, i]
            if np.isfinite(positions[b, i]).all():
                p = (1 - fraction) * p + fraction * positions[b, i]
            sphere.move_to(p + offset)
            if not visible[i]:
                sphere.set_opacity(1)
                visible[i] = True
        closure = (1 - fraction) * lid_fraction[a] + fraction * lid_fraction[b]
        height = float((1 - fraction) * heights[a] + fraction * heights[b])
        if height != last_height:
            box.stretch(height / last_height, 2, about_point=offset)
            last_height = height
        lid.move_to(closed_center + np.array([(1 - closure) * (x + 0.15), 0, height - z]))
        showing_lid = closure > 0
        lid_panel.set_fill(opacity=0.25 if showing_lid else 0)
        lid_edges.set_stroke(opacity=1 if showing_lid else 0)

    update()
    # Toda la geometría participa en cada frame: Cairo no debe cachear la caja
    # como fondo estático mientras cambian su altura y las oclusiones.
    assembly = VGroup(floor, box, spheres, lid)
    scene.add(assembly)
    assembly.add_updater(update)
    scene.play(tracker.animate.set_value(float(times[-1])), run_time=run_time, rate_func=linear)
    assembly.remove_updater(update)
    update()
    return assembly


class PackingDemo(ThreeDScene):
    def construct(self):
        play_packing(self)
