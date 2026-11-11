"""Campus stair dimensions and the geometry shared by physics and task goals."""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput,
)

INCH = 0.0254
SPAWN_X = 0.65
TILE_SIZE = (7.4, 2.6)


@dataclass(frozen=True)
class CampusDesign:
  name: str
  riser_in: float
  tread_in: float
  source: str
  split_landing: bool = False


VIRGINIA_TECH = "https://www.facilities.vt.edu/content/dam/facilities_vt_edu/design-and-construction-standards/vt-design-and-construction-standards-manual-2021.pdf"
DESIGNS = (
  CampusDesign("georgia_tech_plaza", 5, 15, "https://facilities.gatech.edu/system/files/forms_files/greenbookfinal-20200810.pdf"),
  CampusDesign("lsu_entry", 6, 14, "https://www.lsu.edu/pdc/files/lsu_site_and_landscape_standards.pdf"),
  CampusDesign("wyoming_concrete", 6, 12, "https://www.uwyo.edu/uwops/_files/construction_standards_guidelines/design-guidelines-march-2023.pdf"),
  CampusDesign("virginia_tech_exterior", 7, 12, VIRGINIA_TECH),
  CampusDesign("uconn_exterior", 7, 15, "https://updc.media.uconn.edu/wp-content/uploads/sites/1525/2025/10/Volume-One-DESIGN-GUIDELINES-September-2025.pdf"),
  CampusDesign("virginia_tech_split", 7, 12, VIRGINIA_TECH, split_landing=True),
)
COURSES = tuple(f"{design.name}_{side}" for design in DESIGNS for side in ("up", "down"))


@dataclass(frozen=True)
class Difficulty:
  risers: int
  rise_scale: float
  added_tread_m: float
  width_m: float
  hold_s: float
  lift_m: float
  push_mps: float
  scan_dropout: float


DIFFICULTIES = (
  Difficulty(1, .30, .15, 2.15, .30, .045, .00, .00),
  Difficulty(2, .50, .11, 2.00, .55, .060, .04, .01),
  Difficulty(3, .72, .07, 1.85, .85, .075, .08, .02),
  Difficulty(4, .88, .03, 1.70, 1.20, .090, .12, .03),
  Difficulty(6, 1.00, .00, 1.60, 1.60, .105, .16, .04),
  Difficulty(6, 1.00, .00, 1.60, 2.10, .120, .22, .07),
)
MAX_RISERS = max(item.risers for item in DIFFICULTIES)


def course_layout(seed: int, level: int, column: int) -> tuple[np.ndarray, np.ndarray]:
  design = DESIGNS[column // 2]
  tier = DIFFICULTIES[level]
  # Same nominal geometry on levels 4 and 5. Level 5 adds harder disturbances.
  rng = np.random.default_rng(np.random.SeedSequence([seed, min(level, 4), column]))
  first_edge = 1.45 + rng.uniform(-.045, .045)
  lengths = np.full(tier.risers - 1, design.tread_in * INCH + tier.added_tread_m)
  if design.split_landing and len(lengths):
    lengths[(tier.risers - 1) // 2] += 1.0
  if column % 2:
    lengths = lengths[::-1]
  edges = np.r_[first_edge, first_edge + np.cumsum(lengths)]
  heights = np.arange(tier.risers + 1) * (design.riser_in * INCH * tier.rise_scale)
  return edges, heights[::-1].copy() if column % 2 else heights


@dataclass
class CampusTileCfg(SubTerrainCfg):
  column: int = 0
  seed: int = 41

  def layout(self, level):
    return course_layout(self.seed, level, self.column)

  def bounds(self, level):
    return self.size[0], DIFFICULTIES[level].width_m

  def function(self, difficulty, spec, rng):
    del rng
    level = min(len(DIFFICULTIES) - 1, int(difficulty * len(DIFFICULTIES)))
    edges, heights = self.layout(level)
    length, width = self.bounds(level)
    body = spec.body("terrain")
    boundaries = np.r_[0.0, edges, length]
    geometry = []
    for index, (left, right, height) in enumerate(zip(boundaries[:-1], boundaries[1:], heights, strict=True)):
      bottom = -.30
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=((left + right) / 2, self.size[1] / 2, (height + bottom) / 2),
        size=((right - left) / 2, width / 2, (height - bottom) / 2),
        group=0,
        rgba=(.48, .56, .63, 1) if index % 2 else (.62, .68, .73, 1),
      )
      geometry.append(TerrainGeometry(geom=geom))
    return TerrainOutput(np.array((SPAWN_X, self.size[1] / 2, heights[0])), geometry)


def campus_terrain(seed: int) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    seed=seed, curriculum=True, size=TILE_SIZE,
    num_rows=len(DIFFICULTIES), num_cols=len(COURSES),
    sub_terrains={name: CampusTileCfg(column=i, seed=seed) for i, name in enumerate(COURSES)},
  )


def terrain_manifest(seed: int, column_seeds: list[int] | None = None) -> dict:
  column_seeds = column_seeds or [seed] * len(COURSES)
  courses = []
  for level, tier in enumerate(DIFFICULTIES):
    for column, name in enumerate(COURSES):
      edges, heights = course_layout(column_seeds[column], level, column)
      courses.append({
        "course": name, "level": level, "settings": asdict(tier),
        "edge_x_m_from_tile_start": edges.tolist(), "surface_z_m": heights.tolist(),
      })
  return {"seed": seed, "column_seeds": column_seeds,
          "designs": [asdict(d) for d in DESIGNS], "courses": courses}
