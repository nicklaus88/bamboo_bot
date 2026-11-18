"""Build one clump: sample culm slots, place bases, assign outward leans without intersections.

Algorithm (deterministic for a given rng state):
1. Draw slot count N and nominal footprint radius R (radius grows with N).
2. Sample every culm (diameter, length, nodes, ...); mark ~stump_fraction of slots as stumps.
3. Dart-throw base positions: r = R * u^(1/(2+edge_bias)), uniform angle, keeping
   surface gap >= min_gap. On repeated failure R grows by radius_growth (achieved R recorded).
4. All culms start upright with no bow. Visit culms outside-in; give each an outward lean
   (base draw + edge_gain * r/R, azimuth outward + jitter) and its bow, and check its capsule
   chain against the *current* geometry of every other culm. On collision redraw the lean,
   then straighten (scale lean and bow). Upright is always clear because every earlier
   culm was checked against this one while it was upright, so the result never intersects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import Config
from .culm import CulmSpec, make_stump, sample_culm

GroundFn = Callable[[float, float], float]


@dataclass
class ClumpSpec:
    clump_id: str
    center: np.ndarray  # (3,) center at ground level
    nominal_radius: float
    achieved_radius: float  # after any growth during placement
    culms: list[CulmSpec]
    extra: dict = field(default_factory=dict)

    def ground_truth(self) -> dict:
        offsets = [np.linalg.norm(c.base_position[:2] - self.center[:2]) + c.base_diameter / 2 for c in self.culms]
        return {
            "clump_id": self.clump_id,
            "center": self.center.tolist(),
            "footprint_radius_nominal": self.nominal_radius,
            "footprint_radius_achieved": self.achieved_radius,
            "footprint_radius_occupied": float(max(offsets)),  # outermost culm surface at ground
            "culm_count": sum(not c.is_stump for c in self.culms),
            "stump_count": sum(c.is_stump for c in self.culms),
            **self.extra,
        }


def build_clump(
    cfg: Config,
    rng: np.random.Generator,
    center_xy=(0.0, 0.0),
    clump_id: str = "clump",
    ground: GroundFn = lambda x, y: 0.0,
) -> ClumpSpec:
    cc, mc = cfg.clump, cfg.mesh
    cx, cy = map(float, center_xy)
    center = np.array([cx, cy, ground(cx, cy)])

    n = int(round(cc.culm_count.sample(rng)))
    fr = cc.footprint_radius_m
    R_nom = float(np.clip(fr.a + fr.b * n + rng.normal(0.0, fr.noise_sd), fr.min, fr.max))

    culms = [sample_culm(cfg.culm, rng, node_ring_max_s=mc.detail_zone_m) for _ in range(n)]
    for c in culms:
        c.extra["_sampled_lean_deg"] = c.lean_deg
        is_stump = rng.random() < cc.stump_fraction
        h = cc.stump_height_m.sample(rng)  # always drawn so the stream doesn't depend on the coin
        if is_stump:
            make_stump(c, float(h), cc.stump_cut_above_node_m)

    R = _place_bases(culms, rng, center, R_nom, cc, ground)

    # Start upright, then assign outward leans center-out.
    desired_bow = [c.bow_fraction for c in culms]
    for c in culms:
        c.lean_deg, c.bow_fraction = 0.0, 0.0
    chains = [_capsule_chain(c, cc.lean_check_segments) for c in culms]
    radial = [np.hypot(*(c.base_position[:2] - center[:2])) for c in culms]
    outward = [np.degrees(np.arctan2(*(c.base_position[:2] - center[:2])[::-1])) for c in culms]

    # Outside-in: outer culms lean away first, clearing room for inner culms to lean after them.
    for i in np.argsort(-np.asarray(radial), kind="stable"):
        c = culms[i]
        others = [chains[j] for j in range(n) if j != i]
        base_lean = c.extra.pop("_sampled_lean_deg")
        adjustment = "none"
        for attempt in range(cc.lean_redraw_attempts + 1):
            if attempt > 0:
                base_lean = float(cfg.culm.lean_deg.sample(rng))
            c.lean_deg = min(base_lean + cc.lean_edge_gain_deg * radial[i] / R, cc.lean_max_deg)
            c.lean_azimuth_deg = float((outward[i] + cc.lean_azimuth_jitter_deg.sample(rng)) % 360.0)
            c.bow_fraction = desired_bow[i]
            if _clear(_capsule_chain(c, cc.lean_check_segments), others, cc.min_gap_m):
                adjustment = "none" if attempt == 0 else f"redrawn:{attempt}"
                break
        else:
            steps = 0
            while True:
                steps += 1
                c.lean_deg *= cc.lean_straighten_factor
                c.bow_fraction *= cc.lean_straighten_factor
                if c.lean_deg < 0.1:
                    c.lean_deg, c.bow_fraction = 0.0, 0.0
                    break
                if _clear(_capsule_chain(c, cc.lean_check_segments), others, cc.min_gap_m):
                    break
            adjustment = f"straightened:{steps}"
        c.extra["lean_adjustment"] = adjustment
        chains[i] = _capsule_chain(c, cc.lean_check_segments)

    return ClumpSpec(clump_id=clump_id, center=center, nominal_radius=R_nom, achieved_radius=R, culms=culms)


def _place_bases(culms, rng, center, R, cc, ground) -> float:
    placed_xy: list[np.ndarray] = []
    placed_r: list[float] = []
    for c in culms:
        rc = 0.5 * c.base_diameter * (1.0 + c.node_ring_amplitude)
        while True:
            for _ in range(cc.placement_attempts):
                r = R * rng.random() ** (1.0 / (2.0 + cc.edge_bias))
                th = rng.uniform(0.0, 2 * np.pi)
                xy = center[:2] + r * np.array([np.cos(th), np.sin(th)])
                if all(np.linalg.norm(xy - p) >= rc + q + cc.min_gap_m for p, q in zip(placed_xy, placed_r)):
                    break
            else:
                R *= cc.radius_growth
                continue
            break
        placed_xy.append(xy)
        placed_r.append(rc)
        c.base_position = np.array([xy[0], xy[1], ground(float(xy[0]), float(xy[1]))])
    return float(R)


def _capsule_chain(c: CulmSpec, k: int):
    """Conservative capsule chain along the centerline: (P0 (k,3), P1 (k,3), radius (k,))."""
    s = np.linspace(0.0, c.top_s, k + 1)
    p = c.point(s)
    r = c.plain_radius(s[:-1]) * (1.0 + c.node_ring_amplitude)  # taper is non-increasing
    return p[:-1], p[1:], r


def _clear(chain, others, gap: float) -> bool:
    a0, a1, ra = chain
    for b0, b1, rb in others:
        d = _seg_seg_dist(a0[:, None], a1[:, None], b0[None], b1[None])
        if np.any(d < ra[:, None] + rb[None] + gap):
            return False
    return True


def _seg_seg_dist(p1, q1, p2, q2):
    """Vectorized minimum distance between segments p1-q1 and p2-q2 (broadcasting over leading dims)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1)
    b = np.sum(d1 * d2, -1)
    denom = a * e - b * b
    eps = 1e-12
    s = np.where(denom > eps, np.clip((b * f - c * e) / np.where(denom > eps, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / np.maximum(e, eps)
    t_c = np.clip(t, 0.0, 1.0)
    s = np.where(t < 0.0, np.clip(-c / np.maximum(a, eps), 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / np.maximum(a, eps), 0.0, 1.0), s))
    t = t_c
    c1 = p1 + s[..., None] * d1
    c2 = p2 + t[..., None] * d2
    return np.linalg.norm(c1 - c2, axis=-1)
