"""Dev tool: write one or a few sampled culms to a .usda for eyeballing (usdview / Isaac Sim).

    python -m bamboo_grove.preview_culm --config configs/grove_default.yaml --seed 1 --count 5 --out out/culms.usda

This is not the grove asset; culms are placed in a line 0.5 m apart, with no semantics/physics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from .config import load_config
from .culm import sample_culm
from .mesh import build_culm_mesh


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    rngs = [np.random.default_rng(ss) for ss in np.random.SeedSequence(args.seed).spawn(args.count)]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Preview")
    stage.SetDefaultPrim(root.GetPrim())

    for i, rng in enumerate(rngs):
        base = np.array([0.5 * i, 0.0, 0.0])
        spec = sample_culm(cfg.culm, rng, base, node_ring_max_s=cfg.mesh.detail_zone_m)
        mesh = build_culm_mesh(spec, cfg.mesh)

        write_culm(stage, f"/Preview/Culm_{i:03d}", spec, mesh)
        gt = spec.ground_truth()
        print(
            f"Culm_{i:03d}: d={gt['base_diameter']*100:.1f} cm  L={gt['length']:.2f} m  "
            f"h={gt['height']:.2f} m  lean={gt['lean_deg']:.1f}°  nodes={len(gt['node_s'])}  "
            f"cut@{gt['cut_height']:.2f} m (d={gt['cut_diameter']*100:.1f} cm)  "
            f"verts={len(mesh.points)} tris={mesh.triangle_count}"
        )

    stage.GetRootLayer().Save()
    print(f"wrote {args.out}")


def write_culm(stage, path: str, spec, mesh, color=(0.55, 0.62, 0.45)):
    """Author a culm Xform (translated to its base) with a visual Mesh child. Preview only."""
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*spec.base_position))
    m = UsdGeom.Mesh.Define(stage, xf.GetPath().AppendChild("Mesh"))
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.points))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(mesh.face_vertex_counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.face_vertex_indices))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(mesh.extent()))
    m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xf.GetPrim().CreateAttribute("bamboo:preview_gt_json", Sdf.ValueTypeNames.String).Set(json.dumps(spec.ground_truth()))
    return xf


if __name__ == "__main__":
    main()
