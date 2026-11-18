"""Dev tool: write one sampled clump to a .usda for eyeballing (usdview / Isaac Sim).

    python -m bamboo_grove.preview_clump --config configs/grove_default.yaml --seed 1 --out out/clump.usda

Not the grove asset: no semantics/physics. Stumps are tinted brown so they are easy to spot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from .clump import build_clump
from .config import load_config
from .mesh import build_culm_mesh
from .preview_culm import write_culm


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    clump = build_clump(cfg, np.random.default_rng(np.random.SeedSequence(args.seed)), clump_id="Clump")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Preview")
    stage.SetDefaultPrim(root.GetPrim())

    tris = 0
    for i, c in enumerate(clump.culms):
        mesh = build_culm_mesh(c, cfg.mesh)
        tris += mesh.triangle_count
        color = (0.45, 0.35, 0.22) if c.is_stump else (0.55, 0.62, 0.45)
        write_culm(stage, f"/Preview/Culm_{i:03d}", c, mesh, color)
    stage.GetRootLayer().Save()

    gt = clump.ground_truth()
    print(
        f"clump: {gt['culm_count']} culms + {gt['stump_count']} stumps, "
        f"R nominal {gt['footprint_radius_nominal']:.2f} m, occupied {gt['footprint_radius_occupied']:.2f} m, "
        f"{tris} tris"
    )
    for i, c in enumerate(clump.culms):
        g = c.ground_truth()
        kind = "stump" if c.is_stump else "culm "
        print(
            f"  Culm_{i:03d} {kind} d={g['base_diameter']*100:4.1f} cm  h={g['height']:5.2f} m  "
            f"lean={g['lean_deg']:4.1f}°  r={np.hypot(*(c.base_position[:2]-clump.center[:2])):.2f} m  "
            f"{c.extra['lean_adjustment']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
