"""Build a unique swept-tube mesh for one culm.

Rings are placed along the axial parameter s:
- from -base_embed to detail_zone: plain rings every ring_spacing_detail plus 5 rings per node
  (cosine bulge sampled at -w/2, -w/4, 0, +w/4, +w/2), ring_segments_detail sides;
- above detail_zone: rings every ring_spacing_upper, ring_segments_upper sides;
- one stitch band of triangles joins the two resolutions.
Faces wind counter-clockwise seen from outside (USD rightHanded orientation, normals outward).
Points are in the culm's local frame (relative to base_position).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MeshConfig
from .culm import CulmSpec


@dataclass
class MeshData:
    points: np.ndarray  # (n, 3) float32, local frame
    face_vertex_counts: np.ndarray  # (f,) int32
    face_vertex_indices: np.ndarray  # (k,) int32

    @property
    def triangle_count(self) -> int:
        return int(np.sum(self.face_vertex_counts - 2))

    def extent(self) -> np.ndarray:
        return np.stack([self.points.min(axis=0), self.points.max(axis=0)])


def ring_stations(spec: CulmSpec, mcfg: MeshConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (s_detail, s_upper): sorted axial ring positions in each resolution band."""
    L = spec.top_s
    zone = min(mcfg.detail_zone_m, L)
    s_detail = [np.array([-mcfg.base_embed_m]), np.arange(0.0, zone, mcfg.ring_spacing_detail_m), [zone]]
    w = spec.node_ring_width
    for sn in spec.node_s[spec.node_s <= spec.node_ring_max_s]:
        s_detail.append(sn + w * np.array([-0.5, -0.25, 0.0, 0.25, 0.5]))
    s_detail = np.concatenate(s_detail)
    s_detail = s_detail[(s_detail >= -mcfg.base_embed_m) & (s_detail <= zone)]
    s_detail = _dedupe(np.sort(s_detail))

    if zone >= L:
        return s_detail, np.empty(0)
    n_up = max(1, int(np.ceil((L - zone) / mcfg.ring_spacing_upper_m)))
    s_upper = np.linspace(zone, L, n_up + 1)[1:]
    return s_detail, s_upper


def _dedupe(s: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    keep = np.concatenate([[True], np.diff(s) > tol])
    return s[keep]


def build_culm_mesh(spec: CulmSpec, mcfg: MeshConfig) -> MeshData:
    s_detail, s_upper = ring_stations(spec, mcfg)
    nd, nu = mcfg.ring_segments_detail, mcfg.ring_segments_upper

    pts: list[np.ndarray] = []
    ring_start: list[int] = []
    ring_n: list[int] = []
    count = 0
    for s_arr, n in ((s_detail, nd), (s_upper, nu)):
        if s_arr.size == 0:
            continue
        ang = 2 * np.pi * np.arange(n) / n
        N, B, _ = spec.frames(s_arr)
        c = spec.point(s_arr) - spec.base_position
        r = spec.radius(s_arr)
        ring = (
            c[:, None, :]
            + r[:, None, None] * (np.cos(ang)[None, :, None] * N[:, None, :] + np.sin(ang)[None, :, None] * B[:, None, :])
        )
        for k in range(len(s_arr)):
            pts.append(ring[k])
            ring_start.append(count)
            ring_n.append(n)
            count += n

    counts: list[int] = []
    idx: list[int] = []
    for k in range(len(ring_start) - 1):
        _connect(ring_start[k], ring_n[k], ring_start[k + 1], ring_n[k + 1], counts, idx)

    if mcfg.cap_ends:
        # bottom cap (normal points down the axis)
        c0 = count
        pts.append((spec.point(s_detail[0]) - spec.base_position)[None, :])
        count += 1
        a, n = ring_start[0], ring_n[0]
        for j in range(n):
            counts.append(3)
            idx += [c0, a + (j + 1) % n, a + j]
        # top cap
        s_top = s_upper[-1] if s_upper.size else s_detail[-1]
        c1 = count
        pts.append((spec.point(s_top) - spec.base_position)[None, :])
        count += 1
        a, n = ring_start[-1], ring_n[-1]
        for j in range(n):
            counts.append(3)
            idx += [c1, a + j, a + (j + 1) % n]

    return MeshData(
        points=np.concatenate(pts).astype(np.float32),
        face_vertex_counts=np.asarray(counts, dtype=np.int32),
        face_vertex_indices=np.asarray(idx, dtype=np.int32),
    )


def _connect(a: int, na: int, b: int, nb: int, counts: list, idx: list) -> None:
    """Join lower ring (a, na) to upper ring (b, nb); na must equal nb or be a multiple of it."""
    if na == nb:
        for j in range(na):
            j1 = (j + 1) % na
            counts.append(4)
            idx += [a + j, a + j1, b + j1, b + j]
        return
    if na % nb:
        raise ValueError("ring segment counts must divide")
    m = na // nb
    for j in range(nb):
        b0, b1 = b + j, b + (j + 1) % nb
        lower = [a + (m * j + i) % na for i in range(m + 1)]
        half = m // 2
        for i in range(m):
            counts.append(3)
            idx += [lower[i], lower[i + 1], b0 if i < half else b1]
        counts.append(3)
        idx += [lower[half], b1, b0]
