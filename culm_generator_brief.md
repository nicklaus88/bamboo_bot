# Agent Brief: Procedural Bamboo Clump Generator (v0)

> **Audience:** a coding agent building the first version of the bamboo clump generator for an NVIDIA Isaac Sim project.
> **How to use this brief:** it explains the *goal*, the *reasoning*, and the *constraints*. It deliberately does not prescribe an implementation. Investigate, make design decisions, and record why you made them. Where this brief gives numbers, treat them as starting assumptions, not truth.

---

## 1. Context: what this is for

A Clearpath Husky A200 will be deployed autonomously in a **commercial bamboo planting in Frostproof, Florida**. Its stack includes:
- VLP-16 3D lidar
- ZED 2i RGB-D camera, mounted high and facing sideways
- wheel encoders, a serial IMU, and a Garmin 18x GPS
- a nonlinear EKF, a global and a local planner, and a mission state machine

The robot's job is to navigate to bamboo culms and inspect them, and its behavior will be tuned *against the simulation*.

That makes the simulated environment the **fixed reference**. The robot adapts to it, never the other way around. If the bamboo is unrealistic, every downstream tuning decision is wrong in ways that only show up in the field. The bamboo is the core of that environment, and this generator creates it.

**Platform:**
- Isaac Sim 6.2 on an NVIDIA DGX Spark. This is **aarch64 (ARM)**, and GPU PhysX has known problems on it, so assume CPU physics.
- ROS 2 Jazzy on Ubuntu 24.04 / DGX OS 7.
- Units are meters; the stage is Z-up.

---

## 2. The goal of v0, and its scope

Write a **Python script that procedurally generates randomized bamboo clumps, arranged in a planting layout, as a single USD asset**. It is driven by a config file and a random seed. The output should be geometrically plausible for *Dendrocalamus asper* grown in rows, and carry full ground-truth metadata.

**In scope (this script only):**
- the culm geometry
- clumps built from culms
- placing clumps in the planting layout
- collision proxies, semantic labels and ground-truth metadata for all of the above

**Outputs:**
- `grove.usd`: a self-contained USD asset with every clump and culm
- a ground-truth sidecar file (JSON or CSV) with the same data

**Out of scope: these are built separately, by other work:**
- terrain and ground (including any ground plane)
- lighting and sky
- the top-level scene that composes everything
- the robot, sensors and ROS 2

The grove must therefore be a **clean, composable asset**. Another scene will reference or sublayer it, so it should make no assumptions about lights, ground or physics scene settings beyond the conventions in §1, and it should not author them.

v0 is about **correct geometry and structure**, not visual polish. Later versions will replace the placeholder statistics with field measurements, which is why everything has to be parametric from day one.

---

## 3. Why procedural (and not a downloaded model)

This decision is settled. The reasoning matters because it should shape your design:

- **There is no species-specific D. asper model online.** Store assets are generic or the wrong species, with no measured dimensions.
- **Hundreds of unique culms are needed**, each with its own diameter, height, lean and node spacing. Hand-editing or scaling a mesh distorts internode spacing and textures, and it doesn't scale.
- **Ground truth must come for free.** A procedural culm knows its own node heights, diameters and cut point because it built them. A downloaded mesh does not.
- **It must carry forward.** v1 swaps in field-measured distributions. With a parametric generator that is a config change, not rework.

---

## 4. Design principles (the reasoning to carry with you)

1. **Spend realism where the sensors look.**
   - The camera sits roughly 0.8–0.9 m above ground and looks sideways.
   - The lidar has a ±15° vertical field of view.
   - So the **lower ~3 m of each culm and the clump base** dominate what the robot perceives.
   - Prioritize geometric fidelity there: taper, node rings, lean, clump arrangement. Upper culm detail is low priority for v0.
2. **Geometry before appearance.** Navigation and lidar depend on shape, position and size. A plain placeholder material is fine.
3. **Separate visual and collision geometry.**
   - Isaac's RTX lidar ray-traces against **render geometry**, so the visual mesh shape is what the lidar sees.
   - PhysX should collide with **simple proxies**, such as a capsule or cylinder per culm, to stay cheap on CPU physics.
   - Verify both of these assumptions yourself in Isaac Sim 6.2. Don't take them on faith.
4. **Every number lives in the config, never in the code.** Distributions, not constants: means, ranges, spreads, correlations where they're meaningful (e.g. taller culms tend to be thicker).
5. **Deterministic given a seed.** The same config and seed must give an identical grove. This is essential for reproducible robot tuning.
6. **Bake in ground truth.** Every culm and clump carries its parameters as USD attributes and semantic labels (see §7).
7. **No repeated geometry: every culm and every clump is unique.**
   - Each culm gets its own geometry, built from its own sampled parameters.
   - Each clump gets its own culm count, radius and internal arrangement.
   - Reusing a small set of template culms or clumps (template instancing) is **not allowed**, even for performance.
   - Why: repeated geometry is a pattern a culm detector can learn in the sim and then fail on in the field. It also makes lidar scans unrealistically self-similar, which flatters lidar odometry and place recognition.
   - Budget for it: about 1,200 unique culms at a few thousand triangles each is a few million triangles, which RTX rendering handles comfortably.
8. **Ground-agnostic placement.**
   - v0 places clump bases at z = 0.
   - Design the placement step so a ground-height lookup can be plugged in later, e.g. a function from (x, y) to z supplied by the separate terrain work, without restructuring the generator.
9. **Structure for future data.** v1 will load real clump positions, for example from a CSV of surveyed coordinates, and real measured distributions. Design the layout stage so that "generated grid" and "loaded from file" are interchangeable sources.

---

## 5. What is being modeled

### Planting layout
- Florida commercial bamboo is about 95% *Dendrocalamus asper*, a **clumping** species.
- It is planted in **rows on a grid**, at spacing similar to citrus groves; many farms are converted citrus land.
- One Frostproof farm lists **8 ft × 12 ft** spacing (in-row × between-row). UF/IFAS recommends lower densities (around 160–200 plants per acre) for equipment access.
- **Spacing must be configurable**, and the real values will come from the field.
- Each grid point is a **clump**, not a single culm. Real plantings have imperfections: missing or dead plants, small offsets from the ideal grid point, and rows that may not be perfectly straight.

### Hierarchy

**Layout (rows / columns) → clump → culms**

Some parameters to think about (not exhaustive; investigate and add what matters):

| Level | Parameter ideas |
|---|---|
| Layout | rows, columns, in-row spacing, between-row spacing, row heading, missing-plant rate, grid jitter, georeference origin |
| Clump | center, footprint radius, culm count, age (which shifts the distributions), spatial arrangement of culms within the clump |
| Culm | base diameter, height, taper profile, internode-length profile, node ring geometry, lean angle and direction, curvature/bow |

### Botanical reference data (placeholders until field data exists)

| Property | Published value | Notes |
|---|---|---|
| Mature height | 15–30 m | Frostproof farm reports 25–30 ft (~7.5–9 m) for young plants |
| Mature diameter | 8–20 cm | Frostproof farm reports 3–4 in (~7.5–10 cm) for young plants. **Use this for v0.** |
| Internode length | 40–50 cm | Increases from base to mid-culm, then decreases toward the top |
| Taper | Diameter decreases with height | Model a taper profile, not a straight cone, if it's cheap |
| Wall thickness | 11–20 mm, thinning upward | Irrelevant for rendering (culms can be solid). Store it as metadata for future cutting simulation. |
| Surface | Pale/greyish green, fine short brown hairs on internodes; culm sheaths | Appearance cue for later; v0 can use a placeholder material |

**Unknown and must be assumed for now:** culms per clump, clump footprint radius, lean distribution. Choose sensible starting ranges, **state your assumptions explicitly** in the config and README, and make them trivially replaceable. Within a clump, outer culms typically lean outward. Also reason about how culms are distributed inside a clump; they are not uniform random points.

Foliage and branches are **out of scope for v0**, but leave a clear hook for attaching them to culms later.

---

## 6. Things to investigate and decide yourself

Record each decision and its reasoning in the README:

- **Generation toolchain:**
  - Generate with plain `pxr` (the `usd-core` package) so the generator runs *without* launching Isaac Sim, or use Isaac Sim's Python environment and APIs?
  - Consider aarch64 availability of `usd-core` wheels, which schemas each option can author (physics collision, semantics), and developer iteration speed.
- **Mesh construction for a culm:**
  - Polygon resolution around the circumference versus count along the length.
  - How to represent node rings: a slight flare or bulge at each node. Is that geometry or a normal-map concern?
  - Lean and bow without self-intersection.
- **Collision representation:** capsule per culm, a cylinder per segment, or a convex approximation. Weigh physical accuracy against CPU PhysX cost, given the robot only needs to *not drive through* culms.
- **Keeping unique geometry efficient:** every culm must be unique (principle 7). Within that constraint, decide:
  - how to organize the asset's prim hierarchy
  - whether to merge culm meshes per clump or keep one mesh per culm. Per-culm semantic labels and ground truth must survive either choice.
  - how to keep generation and load times reasonable
- **Avoiding intersections:** culms within a clump must not interpenetrate. Neither should clumps, once grid jitter is applied.
- **Asset structure for composition:** how the grove asset should be structured (e.g. a default prim, a stable root path) so a separate scene can reference or sublayer it cleanly, and regenerated groves drop in without edits elsewhere.
- **Georeferencing metadata:**
  - Store a lat/lon origin for the layout. Frostproof is roughly 27.7°N, 81.5°W; this is a placeholder until the real farm's coordinates are known.
  - Store the row heading, so other parts of the system can map grove coordinates to real-world coordinates.
- **Scale and performance:** target a realistic test block, on the order of 3–5 rows × 10–20 clumps. Measure generation time, load time and interactive frame rate on the target machine, or note what you could not measure.

---

## 7. Ground truth requirements (non-negotiable)

Each **culm** must carry at least:
- a unique instance ID and its parent clump ID
- a semantic class label usable by Isaac Sim / Replicator annotators, e.g. `culm`
- base position and lean vector, height, base diameter
- the taper profile, or enough parameters to reconstruct it
- node heights and diameters at each node
- a candidate cut point, e.g. a configurable height above the base or above a given node

Each **clump** must carry: ID, center, footprint radius, culm count, grid row/column index.

The **grove root** must carry: the config used, the seed, the generator version, and the georeference origin and row heading.

Also export the same ground truth to a **sidecar file** (JSON or CSV), so non-Isaac tools can use it. Examples: scoring a culm detector, or building a culm landmark map for navigation.

---

## 8. Out of scope for this script

Leave hooks where noted, but don't build these:
- terrain, ground planes, ground materials, irrigation lines, ground clutter (built separately; see principle 8 for the hook)
- lighting and sky (built separately)
- the top-level scene file (built separately)
- the robot, sensors and ROS 2 bridge
- foliage and branches (leave a hook)
- realistic photo-based materials (a placeholder is fine)
- Replicator randomization and synthetic data generation
- loading surveyed field data (but design for it; see principle 9)

---

## 9. Deliverables

1. **The generator script.** Run it headless from the command line with a config path, seed and output path. It writes `grove.usd` and the ground-truth sidecar.
2. **An example config** with every parameter documented, and with assumptions clearly marked as assumptions.
3. **A README** covering:
   - how to run it
   - the design decisions and their reasoning (§6)
   - known limitations
   - exactly which parameters will be replaced by field measurements
   - how a separate scene should reference or sublayer the grove
4. **A validation script or report** that checks the acceptance criteria below and prints summary statistics.

To test collision and lidar behavior, you will need a minimal throwaway test scene (a ground plane, a light, a test object or lidar). Keep it clearly separate from the deliverable, e.g. under a `tests/` directory. It is a test harness, not the project's ground or lighting.

---

## 10. Acceptance criteria

- **Determinism:** the same config and seed give byte-identical, or semantically identical, output.
- **Uniqueness:**
  - no two culms share the same parameter set
  - no two clumps share the same culm count, radius and arrangement
  - no culm or clump geometry is a template instance
- **Distributions:** the generated culm diameters, heights, culm counts and spacings match the configured distributions. Show summary statistics and, ideally, histograms.
- **Validity:**
  - no culm–culm or clump–clump interpenetration
  - clump bases at the configured base height (z = 0 in v0)
  - everything is in meters, Z-up
- **Clean asset:**
  - `grove.usd` authors no lights, ground or physics scene
  - it can be referenced or sublayered into another stage without edits
  - a grove regenerated with a different seed drops into the same parent scene unchanged
- **Isaac Sim compatibility (verified in the test harness):**
  - the grove opens in Isaac Sim 6.2
  - collision proxies work with PhysX; a test object can't pass through a culm
  - semantic labels are visible to Isaac's annotators
- **Lidar sanity:** an RTX lidar in the test harness returns points on culms, which confirms the render geometry is hit. Ideally include a quick check with a 16-channel rotating lidar profile.
- **Ground truth:** the USD attributes and the sidecar file agree, and every culm has every required field.
- **Performance:** loads and runs at interactive rates for the target block size on the target machine, or document what limited you.

---

## 11. Working style

- **Investigate before committing.** Isaac Sim's APIs changed substantially across 4.x → 5.x → 6.x, and older tutorials are often wrong for 6.2. Prefer current docs and verify behavior empirically.
- **Make the smallest thing that works, then grow it.** One culm → one clump → one row → the full layout.
- **When the brief is silent, decide, and write down why.** When a decision is genuinely the user's (e.g. a trade-off between realism and performance that changes what the robot will experience), flag it rather than silently picking.
- **Don't fake realism.** If something is a guess, label it as one. The whole point of this environment is that it can later be checked against the real field.
