"""Campus-inspired stair flights with a bounded landing for a U-turn."""

from dataclasses import asdict, dataclass

import numpy as np
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from src.tasks.campus.terrain import CampusTileCfg, DESIGNS, DIFFICULTIES, INCH

DEPTHS = {"short": (.25, .29), "medium": (.305, .35), "long": (.38, .44)}
PACE_BANDS = {"slow": (.20, .30), "normal": (.38, .48), "fast": (.57, .70)}
TURN_DEPTHS = (1.68, 1.57, 1.46, 1.36, 1.28, 1.22)
TREAD_EXTRA = (.060, .045, .030, .015, 0., 0.)
TILE_SIZE = (7.4, 2.6)
COURSES = tuple(f"{design.name}_{depth}" for design in DESIGNS for depth in DEPTHS)
RISERS = tuple(max(2, tier.risers) for tier in DIFFICULTIES)


@dataclass
class RoundTripTileCfg(CampusTileCfg):
  def layout(self, level):
    design = DESIGNS[self.column // len(DEPTHS)]
    low, high = tuple(DEPTHS.values())[self.column % len(DEPTHS)]
    rng = np.random.default_rng(np.random.SeedSequence([self.seed, min(level, 4), self.column]))
    depth = rng.uniform(low, high) + TREAD_EXTRA[level]
    jitter = .008 if level < 4 else .012
    treads = depth + rng.uniform(-jitter, jitter, RISERS[level] - 1)
    if design.split_landing:
      treads[len(treads) // 2] += 1.0
    edges = 1.45 + rng.uniform(-.045, .045) + np.r_[0., treads.cumsum()]
    height = design.riser_in * INCH * DIFFICULTIES[level].rise_scale
    return edges, np.arange(RISERS[level] + 1) * height

  def bounds(self, level):
    edges, _ = self.layout(level)
    return edges[-1] + TURN_DEPTHS[level], DIFFICULTIES[level].width_m


def round_trip_terrain(seed):
  return TerrainGeneratorCfg(
    seed=seed, curriculum=True, size=TILE_SIZE,
    num_rows=len(DIFFICULTIES), num_cols=len(COURSES),
    sub_terrains={name: RoundTripTileCfg(column=i, seed=seed) for i, name in enumerate(COURSES)},
  )


def terrain_manifest(generator):
  rows = []
  for level, tier in enumerate(DIFFICULTIES):
    for name, tile in generator.sub_terrains.items():
      edges, heights = tile.layout(level)
      end, width = tile.bounds(level)
      rows.append({
        "course": name, "level": level, "seed": tile.seed, "settings": asdict(tier),
        "risers": RISERS[level], "edge_x_m_from_tile_start": edges.tolist(),
        "surface_z_m": heights.tolist(), "tread_depths_m": np.diff(edges).tolist(),
        "upper_landing_depth_m": TURN_DEPTHS[level], "end_x_m": end, "lane_width_m": width,
      })
  return {
    "terrain_seed": generator.seed, "families": [asdict(design) for design in DESIGNS],
    "tread_depth_bands_m": DEPTHS, "pace_bands_mps": PACE_BANDS, "courses": rows,
  }
