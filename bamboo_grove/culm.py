"""Single-culm parameter sampling and analytic geometry (centerline, frames, radius).

Conventions (asset frame: meters, Z-up):
- `s` is the axial parameter along the culm, 0 at the base point, `length` at the tip.
  With the small bows used here, s equals arc length to within ~0.5 %.
- Centerline: p(s) = base + s*u + bow*L*(s/L)^2 * h, where u is the lean direction and
  h the horizontal unit vector at the lean azimuth. The curve is planar, so the frame
  (N, B, T) with constant B = u x h is rotation-minimizing (no twist).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CulmConfig, Profile


@dataclass
class CulmSpec:
    base_position: np.ndarray  # (3,) world/asset position of the base point (ground contact)
    base_diameter: float
    length: float
    taper_profile: Profile
    lean_deg: float
    lean_azimuth_deg: float
    bow_fraction: float
    node_s: np.ndarray  # axial positions of nodes above the base
    node_ring_amplitude: float
    node_ring_width: float
    node_ring_max_s: float  # bulges only modeled below this s (mesh detail zone)
    wall_thickness_base: float
    wall_thickness_top: float
    cut_s: float | None
    stump_top_s: float | None = None  # set => harvested stump, geometry ends here (flat closed top)
    extra: dict = field(default_factory=dict)

    @property
    def is_stump(self) -> bool:
        return self.stump_top_s is not None

    @property
    def top_s(self) -> float:
        """Axial position where the geometry ends (tip, or stump cut)."""
        return self.length if self.stump_top_s is None else self.stump_top_s

    # ---- geometry -------------------------------------------------------
    @property
    def lean_dir(self) -> np.ndarray:
        th, ph = np.radians(self.lean_deg), np.radians(self.lean_azimuth_deg)
        return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])

    @property
    def bow_dir(self) -> np.ndarray:
        ph = np.radians(self.lean_azimuth_deg)
        return np.array([np.cos(ph), np.sin(ph), 0.0])

    @property
    def lean_vector(self) -> np.ndarray:
        """Unit vector from base toward tip (straight line, includes bow)."""
        v = self.point(self.length) - self.base_position
        return v / np.linalg.norm(v)

    def point(self, s):
        s = np.asarray(s, dtype=float)
        k = self.bow_fraction / self.length
        off = s[..., None] * self.lean_dir + (k * s**2)[..., None] * self.bow_dir
        return self.base_position + off

    def tangent(self, s):
        s = np.asarray(s, dtype=float)
        k = self.bow_fraction / self.length
        t = self.lean_dir + (2 * k * s)[..., None] * self.bow_dir
        return t / np.linalg.norm(t, axis=-1, keepdims=True)

    def frames(self, s):
        """Return (N, B, T) arrays of shape (..., 3); right-handed, N x B = T."""
        T = self.tangent(s)
        b = np.cross(self.lean_dir, self.bow_dir)
        if np.linalg.norm(b) < 1e-9:  # lean exactly along bow dir: impossible unless lean=90
            b = np.array([0.0, 1.0, 0.0])
        b = b / np.linalg.norm(b)
        B = np.broadcast_to(b, T.shape)
        N = np.cross(B, T)
        N /= np.linalg.norm(N, axis=-1, keepdims=True)
        return N, B, T

    def min_bend_radius(self) -> float:
        k = self.bow_fraction / self.length
        return np.inf if k == 0 else 1.0 / (2 * k)  # curvature is max at s=0 for this parabola

    def plain_radius(self, s):
        t = np.clip(np.asarray(s, dtype=float) / self.length, 0.0, 1.0)
        return 0.5 * self.base_diameter * self.taper_profile(t)

    def radius(self, s):
        """Outer radius including node-ring bulges (as realized in the mesh)."""
        s = np.asarray(s, dtype=float)
        r = self.plain_radius(s)
        w = self.node_ring_width
        bump = np.zeros_like(r)
        for sn in self.node_s[self.node_s <= self.node_ring_max_s]:
            d = s - sn
            inside = np.abs(d) < w / 2
            bump = np.where(inside, np.maximum(bump, np.cos(np.pi * d / w) ** 2), bump)
        return r * (1.0 + self.node_ring_amplitude * bump)

    def wall_thickness(self, s):
        t = np.clip(np.asarray(s, dtype=float) / self.length, 0.0, 1.0)
        return self.wall_thickness_base + t * (self.wall_thickness_top - self.wall_thickness_base)

    # ---- ground truth ---------------------------------------------------
    def ground_truth(self) -> dict:
        tip = self.point(self.top_s)
        node_s = self.node_s[self.node_s < self.top_s]
        z0 = self.base_position[2]
        if self.cut_s is None:
            cut_gt = {"cut_s": None, "cut_height": None, "cut_position": None, "cut_diameter": None}
        else:
            cut = self.point(self.cut_s)
            cut_gt = {
                "cut_s": self.cut_s,
                "cut_height": float(cut[2] - z0),
                "cut_position": cut.tolist(),
                "cut_diameter": float(2 * self.radius(self.cut_s)),
            }
        return {
            "is_stump": self.is_stump,
            "base_position": self.base_position.tolist(),
            "base_diameter": self.base_diameter,
            "length": float(self.top_s),
            "uncut_length": self.length,
            "height": float(tip[2] - z0),
            "tip_position": tip.tolist(),
            "lean_deg": self.lean_deg,
            "lean_azimuth_deg": self.lean_azimuth_deg,
            "lean_vector": self.lean_vector.tolist(),
            "bow_fraction": self.bow_fraction,
            "taper_profile": self.taper_profile.as_list(),
            "node_s": node_s.tolist(),
            "node_heights": (self.point(node_s)[:, 2] - z0).tolist() if node_s.size else [],
            "node_diameters": (2 * self.radius(node_s)).tolist(),
            "node_ring_amplitude": self.node_ring_amplitude,
            "wall_thickness_base": self.wall_thickness_base,
            "wall_thickness_top": self.wall_thickness_top,
            **cut_gt,
            **self.extra,
        }


def sample_culm(
    cfg: CulmConfig,
    rng: np.random.Generator,
    base_position=(0.0, 0.0, 0.0),
    *,
    node_ring_max_s: float,
    lean_azimuth_deg: float | None = None,
    lean_deg: float | None = None,
) -> CulmSpec:
    """Sample one culm. Draw order is fixed so results are deterministic per rng state.

    `lean_azimuth_deg` / `lean_deg` override the config draws (used by clump arrangement);
    the config draws are still consumed so the rng stream doesn't shift.
    """
    d = float(cfg.base_diameter_m.sample(rng))
    lm = cfg.length_m
    length = float(np.clip(lm.a + lm.b * d + rng.normal(0.0, lm.noise_sd), lm.min, lm.max))
    lean = float(cfg.lean_deg.sample(rng))
    az = float(cfg.lean_azimuth_deg.sample(rng))
    bow = float(cfg.bow_fraction.sample(rng))
    amp = float(cfg.node_ring_amplitude.sample(rng))
    wt_base = float(cfg.wall_thickness_base_m.sample(rng))
    wt_top = float(cfg.wall_thickness_top_m.sample(rng))
    max_inode = float(cfg.internode_max_length_m.sample(rng))
    node_s = _sample_nodes(cfg, rng, length, max_inode)

    spec = CulmSpec(
        base_position=np.asarray(base_position, dtype=float),
        base_diameter=d,
        length=length,
        taper_profile=cfg.taper_profile,
        lean_deg=lean if lean_deg is None else float(lean_deg),
        lean_azimuth_deg=az if lean_azimuth_deg is None else float(lean_azimuth_deg),
        bow_fraction=bow,
        node_s=node_s,
        node_ring_amplitude=amp,
        node_ring_width=cfg.node_ring_width_m,
        node_ring_max_s=node_ring_max_s,
        wall_thickness_base=wt_base,
        wall_thickness_top=wt_top,
        cut_s=_cut_s(cfg, node_s, length),
        extra={"internode_max_length": max_inode},
    )
    if spec.min_bend_radius() < 20 * spec.base_diameter:
        raise ValueError("bow too strong: bend radius < 20x base diameter (self-intersection risk)")
    return spec


def _sample_nodes(cfg: CulmConfig, rng, length: float, max_inode: float) -> np.ndarray:
    nodes = []
    s = 0.0
    sd = cfg.internode_relative_noise_sd
    while True:
        frac = cfg.internode_profile(min(s / length, 1.0))
        step = max_inode * frac * float(np.clip(1.0 + rng.normal(0.0, sd), 0.7, 1.3))
        s += step
        if s >= length - 0.5 * step:  # don't place a node right at the tip
            break
        nodes.append(s)
    return np.asarray(nodes)


def _cut_s(cfg: CulmConfig, node_s: np.ndarray, length: float) -> float:
    cp = cfg.cut_point
    if cp.mode == "above_base":
        s = cp.value
    else:
        candidates = node_s[node_s >= cp.min_height_m]
        if candidates.size == 0:
            raise ValueError("no node above cut_point.min_height_m")
        s = float(candidates[0]) + cp.value
    if not 0.0 < s < length:
        raise ValueError(f"cut point s={s:.3f} outside culm (length {length:.3f})")
    return float(s)


def make_stump(spec: CulmSpec, sampled_height: float, cut_above_node: float) -> CulmSpec:
    """Turn a sampled culm into a harvested stump: flat top cut_above_node above the highest
    node at or below sampled_height (first node if none is that low). Cut point is cleared."""
    below = spec.node_s[spec.node_s <= sampled_height]
    node = float(below[-1]) if below.size else float(spec.node_s[0])
    spec.stump_top_s = node + cut_above_node
    spec.cut_s = None
    return spec
