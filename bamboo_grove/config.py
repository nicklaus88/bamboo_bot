"""Load and validate the generator config (YAML) into typed objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .distributions import Dist, parse_dist

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Profile:
    """Piecewise-linear profile over normalized length t in [0, 1]."""

    t: np.ndarray
    v: np.ndarray

    def __call__(self, t):
        return np.interp(t, self.t, self.v)

    def as_list(self) -> list[list[float]]:
        return [[float(a), float(b)] for a, b in zip(self.t, self.v)]


@dataclass(frozen=True)
class LengthModel:
    a: float
    b: float
    noise_sd: float
    min: float
    max: float


@dataclass(frozen=True)
class CutPoint:
    mode: str
    value: float
    min_height_m: float


@dataclass(frozen=True)
class CulmConfig:
    base_diameter_m: Dist
    length_m: LengthModel
    taper_profile: Profile
    internode_max_length_m: Dist
    internode_profile: Profile
    internode_relative_noise_sd: float
    node_ring_amplitude: Dist
    node_ring_width_m: float
    lean_deg: Dist
    lean_azimuth_deg: Dist
    bow_fraction: Dist
    wall_thickness_base_m: Dist
    wall_thickness_top_m: Dist
    cut_point: CutPoint


@dataclass(frozen=True)
class MeshConfig:
    detail_zone_m: float
    ring_segments_detail: int
    ring_segments_upper: int
    ring_spacing_detail_m: float
    ring_spacing_upper_m: float
    base_embed_m: float
    cap_ends: bool


@dataclass(frozen=True)
class ClumpConfig:
    culm_count: Dist
    footprint_radius_m: LengthModel  # radius = a + b * count + noise, clamped
    edge_bias: float
    min_gap_m: float
    placement_attempts: int
    radius_growth: float
    lean_edge_gain_deg: float
    lean_azimuth_jitter_deg: Dist
    lean_max_deg: float
    lean_redraw_attempts: int
    lean_straighten_factor: float
    lean_check_segments: int
    stump_fraction: float
    stump_height_m: Dist
    stump_cut_above_node_m: float


@dataclass(frozen=True)
class Config:
    raw: dict  # exactly what was loaded, for embedding in the asset
    culm: CulmConfig
    mesh: MeshConfig
    clump: ClumpConfig


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return parse_config(raw)


def parse_config(raw: dict) -> Config:
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    return Config(
        raw=raw,
        culm=_parse_culm(_req(raw, "culm", "")),
        mesh=_parse_mesh(_req(raw, "mesh", "")),
        clump=_parse_clump(_req(raw, "clump", "")),
    )


def _req(d: dict, key: str, where: str):
    if key not in d:
        raise ValueError(f"missing required key {where}{key}")
    return d[key]


def _profile(points, where: str, *, start_one: bool = False, nonincreasing: bool = False) -> Profile:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
        raise ValueError(f"{where}: expected list of [t, value] pairs")
    t, v = arr[:, 0], arr[:, 1]
    if t[0] != 0.0 or t[-1] != 1.0 or np.any(np.diff(t) <= 0):
        raise ValueError(f"{where}: t must start at 0, end at 1 and strictly increase")
    if np.any(v <= 0):
        raise ValueError(f"{where}: values must be > 0")
    if start_one and v[0] != 1.0:
        raise ValueError(f"{where}: first value must be 1.0")
    if nonincreasing and np.any(np.diff(v) > 0):
        raise ValueError(f"{where}: values must be non-increasing")
    return Profile(t, v)


def _parse_culm(c: dict) -> CulmConfig:
    w = "culm."
    lm = _req(c, "length_m", w)
    inode = _req(c, "internode", w)
    ring = _req(c, "node_ring", w)
    cut = _req(c, "cut_point", w)
    if cut["mode"] not in ("above_base", "above_node"):
        raise ValueError("culm.cut_point.mode must be above_base or above_node")

    def d(key):
        return parse_dist(_req(c, key, w), w + key)

    return CulmConfig(
        base_diameter_m=d("base_diameter_m"),
        length_m=LengthModel(**{k: float(lm[k]) for k in ("a", "b", "noise_sd", "min", "max")}),
        taper_profile=_profile(_req(c, "taper_profile", w), w + "taper_profile", start_one=True, nonincreasing=True),
        internode_max_length_m=parse_dist(_req(inode, "max_length_m", w + "internode."), w + "internode.max_length_m"),
        internode_profile=_profile(_req(inode, "profile", w + "internode."), w + "internode.profile"),
        internode_relative_noise_sd=float(_req(inode, "relative_noise_sd", w + "internode.")),
        node_ring_amplitude=parse_dist(_req(ring, "amplitude", w + "node_ring."), w + "node_ring.amplitude"),
        node_ring_width_m=float(_req(ring, "width_m", w + "node_ring.")),
        lean_deg=d("lean_deg"),
        lean_azimuth_deg=d("lean_azimuth_deg"),
        bow_fraction=d("bow_fraction"),
        wall_thickness_base_m=d("wall_thickness_base_m"),
        wall_thickness_top_m=d("wall_thickness_top_m"),
        cut_point=CutPoint(str(cut["mode"]), float(cut["value"]), float(cut.get("min_height_m", 0.0))),
    )


def _parse_mesh(m: dict) -> MeshConfig:
    w = "mesh."
    cfg = MeshConfig(
        detail_zone_m=float(_req(m, "detail_zone_m", w)),
        ring_segments_detail=int(_req(m, "ring_segments_detail", w)),
        ring_segments_upper=int(_req(m, "ring_segments_upper", w)),
        ring_spacing_detail_m=float(_req(m, "ring_spacing_detail_m", w)),
        ring_spacing_upper_m=float(_req(m, "ring_spacing_upper_m", w)),
        base_embed_m=float(_req(m, "base_embed_m", w)),
        cap_ends=bool(_req(m, "cap_ends", w)),
    )
    if cfg.ring_segments_upper < 3 or cfg.ring_segments_detail % cfg.ring_segments_upper:
        raise ValueError("mesh.ring_segments_detail must be a multiple of ring_segments_upper (>= 3)")
    return cfg


def _parse_clump(c: dict) -> ClumpConfig:
    w = "clump."
    fr = _req(c, "footprint_radius_m", w)
    lean = _req(c, "lean", w)
    st = _req(c, "stumps", w)
    cfg = ClumpConfig(
        culm_count=parse_dist(_req(c, "culm_count", w), w + "culm_count"),
        footprint_radius_m=LengthModel(**{k: float(fr[k]) for k in ("a", "b", "noise_sd", "min", "max")}),
        edge_bias=float(_req(c, "edge_bias", w)),
        min_gap_m=float(_req(c, "min_gap_m", w)),
        placement_attempts=int(_req(c, "placement_attempts", w)),
        radius_growth=float(_req(c, "radius_growth", w)),
        lean_edge_gain_deg=float(_req(lean, "edge_gain_deg", w + "lean.")),
        lean_azimuth_jitter_deg=parse_dist(_req(lean, "azimuth_jitter_deg", w + "lean."), w + "lean.azimuth_jitter_deg"),
        lean_max_deg=float(_req(lean, "max_deg", w + "lean.")),
        lean_redraw_attempts=int(_req(lean, "redraw_attempts", w + "lean.")),
        lean_straighten_factor=float(_req(lean, "straighten_factor", w + "lean.")),
        lean_check_segments=int(_req(lean, "check_segments", w + "lean.")),
        stump_fraction=float(_req(st, "fraction", w + "stumps.")),
        stump_height_m=parse_dist(_req(st, "height_m", w + "stumps."), w + "stumps.height_m"),
        stump_cut_above_node_m=float(_req(st, "cut_above_node_m", w + "stumps.")),
    )
    if cfg.edge_bias <= -2:
        raise ValueError("clump.edge_bias must be > -2")
    if not 0.0 <= cfg.stump_fraction <= 1.0:
        raise ValueError("clump.stumps.fraction must be in [0, 1]")
    if not 0.0 < cfg.lean_straighten_factor < 1.0 or cfg.radius_growth <= 1.0:
        raise ValueError("clump.lean.straighten_factor must be in (0,1) and radius_growth > 1")
    return cfg
