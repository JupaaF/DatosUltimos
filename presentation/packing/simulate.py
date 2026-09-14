"""DEM translacional para una ilustración de packing; unidades SI."""
from pathlib import Path
import json
import argparse
import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent


def simulate(resume=False):
    rng = np.random.default_rng(42)
    n, fps, duration, substeps = 500, 60, 90, 60
    dt = 1 / (fps * substeps)
    radii = rng.uniform(0.18, 0.22, n)
    mass = (radii / 0.2) ** 3
    box = np.array([3.4, 3.4, 3.2])
    target_density = 0.585
    compression_density = 0.59
    solid_volume = float(np.sum(4 / 3 * np.pi * radii**3))
    final_height = solid_volume / (target_density * box[0] * box[1])
    compressed_height = solid_volume / (compression_density * box[0] * box[1])

    def lid_motion(t):
        u = np.clip((t - 28) / 14, 0, 1)
        height = box[2] + (compressed_height - box[2]) * (3 * u**2 - 2 * u**3)
        speed = (compressed_height - box[2]) * 6 * u * (1 - u) / 14
        # Alivio de presión: abrir verticalmente 2.1 cm después de compactar.
        w = np.clip((t - 55) / 5, 0, 1)
        height += (final_height - compressed_height) * (3 * w**2 - 2 * w**3)
        speed += (final_height - compressed_height) * 6 * w * (1 - w) / 5
        return height, speed
    pos = np.zeros((n, 3))
    vel = np.zeros_like(pos)
    frames = np.full((duration * fps + 1, n, 3), np.nan, np.float32)
    counts = np.zeros(len(frames), np.int32)
    release = np.linspace(0.05, 22, n)
    count = 0
    k, damping, friction, tangential_damping = 50000., 140., 0.10, 80.
    peak_overlap = 0.
    start_step = 0
    if resume:
        # Continuación de la corrida de este proyecto, cuya dinámica hasta 55 s
        # es idéntica. La ejecución normal no necesita este archivo auxiliar.
        with np.load(OUT / '_candidate.npz') as saved:
            assert int(saved['duration']) == 55
            assert np.array_equal(saved['radii'], radii)
            assert np.array_equal(saved['initial_box_size'], box)
            assert np.isclose(float(saved['final_height']), compressed_height)
            previous = saved['positions']
            frames[:len(previous)] = previous
            counts[:len(previous)] = saved['active_count']
            pos[:] = saved['final_position']
            vel[:] = saved['velocity']
            count = int(counts[len(previous) - 1])
            start_step = 55 * fps * substeps

    def contact_force(normal, relative_velocity, overlap):
        speed = np.sum(relative_velocity * normal, axis=1)
        fn = np.maximum(0., k * overlap - damping * speed)
        tangent = relative_velocity - speed[:, None] * normal
        norm = np.linalg.norm(tangent, axis=1)
        ft = np.minimum(tangential_damping, friction * fn / np.maximum(norm, 1e-12))
        return fn[:, None] * normal - ft[:, None] * tangent

    for step in range(start_step, duration * fps * substeps + 1):
        t = step * dt
        height, lid_speed = lid_motion(t)
        if count < n and t >= release[count]:
            # Un único nacimiento por paso, sin interpenetración en la entrada.
            for _ in range(100):
                candidate = np.array([rng.uniform(radii[count], box[0] - radii[count]),
                                      rng.uniform(radii[count], box[1] - radii[count]), 4.0])
                if not count or np.all(np.linalg.norm(pos[:count] - candidate, axis=1)
                                       > radii[:count] + radii[count] + 0.01):
                    pos[count] = candidate
                    count += 1
                    break
        if step % substeps == 0:
            frame = step // substeps
            frames[frame, :count] = pos[:count]
            counts[frame] = count
            if frame % 300 == 0:
                speed = float(np.linalg.norm(vel, axis=1).max())
                print(f't={t:.0f}s: {count} partículas; vmax={speed:.4f} m/s', flush=True)
                if t >= 65 and speed < 0.025:
                    duration = int(round(t))
                    frames = frames[:frame + 1]
                    counts = counts[:frame + 1]
                    break
        if step == duration * fps * substeps:
            break
        if not count:
            continue
        p, v, r = pos[:count], vel[:count], radii[:count]
        force = np.zeros_like(p)
        force[:, 2] = -9.81 * mass[:count]
        pairs = cKDTree(p).query_pairs(2 * radii.max(), output_type='ndarray')
        if len(pairs):
            i, j = pairs.T
            delta = p[i] - p[j]
            distance = np.linalg.norm(delta, axis=1)
            overlap = r[i] + r[j] - distance
            mask = overlap > 0
            i, j, delta, distance, overlap = i[mask], j[mask], delta[mask], distance[mask], overlap[mask]
            if len(i):
                peak_overlap = max(peak_overlap, float(overlap.max()))
                f = contact_force(delta / np.maximum(distance[:, None], 1e-12), v[i] - v[j], overlap)
                np.add.at(force, i, f)
                np.add.at(force, j, -f)
        # Suelo y paredes laterales de altura finita: la parte superior está abierta.
        for axis, boundary, direction in [(0, 0, 1), (0, box[0], -1),
                                          (1, 0, 1), (1, box[1], -1), (2, 0, 1)]:
            overlap = r - direction * (p[:, axis] - boundary)
            mask = overlap > 0
            if axis != 2:
                mask &= p[:, 2] <= height
            ids = np.flatnonzero(mask)
            normal = np.zeros((len(ids), 3))
            normal[:, axis] = direction
            force[ids] += contact_force(normal, v[ids], overlap[ids])
        if t >= 28:
            overlap = p[:, 2] + r - height
            ids = np.flatnonzero(overlap > 0)
            normal = np.zeros((len(ids), 3))
            normal[:, 2] = -1
            relative = v[ids] - np.array([0., 0., lid_speed])
            force[ids] += contact_force(normal, relative, overlap[ids])
        vel[:count] += force / mass[:count, None] * dt
        pos[:count] += vel[:count] * dt

    # Diagnóstico recuperable incluso si una comprobación física falla.
    np.savez_compressed(OUT / '_candidate.npz', positions=frames, radii=radii,
                        active_count=counts, velocity=vel, final_position=pos,
                        duration=duration, initial_box_size=box, final_height=final_height)
    print(f'Penetración máxima contra tapa: {np.max(pos[:, 2] + radii - final_height):.6f} m', flush=True)
    assert count == n
    assert np.isfinite(frames[-1]).all()
    final_speed = float(np.linalg.norm(vel, axis=1).max())
    assert final_speed < 0.05, f'Packing sin asentar: {final_speed}'
    # Cierre, compresión, alivio de presión y reposo adaptativo.
    times = np.arange(len(frames)) / fps
    lid = np.clip((times - 26) / 2, 0, 1).astype(np.float32)
    heights = lid_motion(times)[0].astype(np.float32)
    assert np.nanmax(frames[26 * fps:28 * fps + 1, :, 2] + radii) < box[2]
    assert np.max(pos[:, 2] + radii - final_height) < 0.01
    assert np.all(frames[-1, :, :2] >= radii[:, None] - 0.01)
    assert np.all(frames[-1, :, :2] <= box[:2] - radii[:, None] + 0.01)
    assert np.all(frames[-1, :, 2] >= radii - 0.01)
    final_box = np.array([box[0], box[1], final_height])
    final_pairs = cKDTree(pos).query_pairs(2 * radii.max(), output_type='ndarray')
    i, j = final_pairs.T
    final_overlap = float(np.max(radii[i] + radii[j] - np.linalg.norm(pos[i] - pos[j], axis=1)))
    assert final_overlap < 0.01, f'Penetración final excesiva: {final_overlap}'
    shared = dict(radii=radii.astype(np.float32), box_size=final_box.astype(np.float32),
                  initial_box_size=box.astype(np.float32))
    np.savez_compressed(OUT / 'trajectory_physics.npz', positions=frames,
                        time=times, active_count=counts, lid_fraction=lid, box_height=heights, **shared)
    indices = np.arange(0, len(frames), duration // 5)
    np.savez_compressed(OUT / 'trajectory.npz', positions=frames[indices],
                        time=np.arange(len(indices)) / fps, physical_time=times[indices],
                        active_count=counts[indices], lid_fraction=lid[indices],
                        box_height=heights[indices], **shared)
    np.savetxt(OUT / 'final_packing.csv', np.column_stack((np.arange(n), radii, pos)),
               delimiter=',', header='id,radius,x,y,z', comments='', fmt=['%d'] + ['%.7f'] * 4)
    metadata = dict(particles=n, fps=fps, animation_seconds=5, physical_seconds=duration,
                    seed=42, units='m, kg, s', time_step=dt, stiffness=k,
                    normal_damping=damping, friction=friction, tangential_damping=tangential_damping,
                    mass_formula='(radius / 0.2)**3 kg', box_size=final_box.tolist(),
                    initial_box_size=box.tolist(), target_density=target_density,
                    compression_density=compression_density,
                    max_pair_overlap_m=None if resume else peak_overlap,
                    resumed_from_seconds=55 if resume else None,
                    final_max_speed_m_s=final_speed,
                    final_max_lid_overlap_m=float(np.max(pos[:, 2] + radii - final_height)),
                    final_max_pair_overlap_m=final_overlap,
                    solid_volume_fraction=solid_volume / float(np.prod(final_box)),
                    final_bed_height_m=float(np.max(pos[:, 2] + radii)),
                    model='DEM translacional; sin rotación ni historia de fricción estática. Ilustrativo.',
                    lid=f'Cierre 26–28 s; compresión 28–42 s; alivio 55–60 s; reposo hasta {duration} s. Paredes telescópicas.')
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    (OUT / '_candidate.npz').unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-candidate', action='store_true',
                        help='Continuar el candidato compatible guardado a 55 s.')
    simulate(resume=parser.parse_args().resume_candidate)
