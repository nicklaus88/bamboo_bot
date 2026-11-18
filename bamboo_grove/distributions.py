"""Config-driven random distributions.

A distribution spec in YAML is either a bare number (constant) or a mapping:

    {dist: constant,    value}
    {dist: uniform,     min, max}
    {dist: truncnormal, mean, sd, min, max}
    {dist: lognormal,   mean, sd, min?, max?}   # mean/sd of the variable itself, not of log
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MAX_REJECT = 10_000


@dataclass(frozen=True)
class Dist:
    kind: str
    params: dict

    def sample(self, rng: np.random.Generator, size: int | None = None):
        p = self.params
        if self.kind == "constant":
            v = float(p["value"])
            return v if size is None else np.full(size, v)
        if self.kind == "uniform":
            return rng.uniform(p["min"], p["max"], size)
        if self.kind == "truncnormal":
            return _reject(lambda n: rng.normal(p["mean"], p["sd"], n), p["min"], p["max"], size)
        if self.kind == "lognormal":
            m, s = p["mean"], p["sd"]
            sigma2 = np.log(1.0 + (s / m) ** 2)
            mu = np.log(m) - 0.5 * sigma2
            lo, hi = p.get("min", 0.0), p.get("max", np.inf)
            return _reject(lambda n: rng.lognormal(mu, np.sqrt(sigma2), n), lo, hi, size)
        raise AssertionError(self.kind)

    def bounds(self) -> tuple[float, float]:
        p = self.params
        if self.kind == "constant":
            return float(p["value"]), float(p["value"])
        return float(p.get("min", -np.inf)), float(p.get("max", np.inf))


_REQUIRED = {
    "constant": ("value",),
    "uniform": ("min", "max"),
    "truncnormal": ("mean", "sd", "min", "max"),
    "lognormal": ("mean", "sd"),
}


def parse_dist(spec, where: str = "?") -> Dist:
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return Dist("constant", {"value": float(spec)})
    if not isinstance(spec, dict) or "dist" not in spec:
        raise ValueError(f"{where}: expected a number or a mapping with 'dist', got {spec!r}")
    kind = spec["dist"]
    if kind not in _REQUIRED:
        raise ValueError(f"{where}: unknown dist {kind!r}; options: {sorted(_REQUIRED)}")
    missing = [k for k in _REQUIRED[kind] if k not in spec]
    if missing:
        raise ValueError(f"{where}: dist {kind} missing {missing}")
    params = {k: float(v) for k, v in spec.items() if k != "dist"}
    if "min" in params and "max" in params and params["min"] > params["max"]:
        raise ValueError(f"{where}: min > max")
    if kind in ("truncnormal", "lognormal") and params["sd"] < 0:
        raise ValueError(f"{where}: sd < 0")
    return Dist(kind, params)


def _reject(draw, lo, hi, size):
    n = 1 if size is None else int(size)
    out = np.empty(0)
    for _ in range(_MAX_REJECT):
        x = draw(max(n, 8))
        out = np.concatenate([out, x[(x >= lo) & (x <= hi)]])
        if out.size >= n:
            return float(out[0]) if size is None else out[:n]
    raise RuntimeError(f"rejection sampling failed for bounds [{lo}, {hi}]")
