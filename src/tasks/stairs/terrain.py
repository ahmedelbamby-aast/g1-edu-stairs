"""Straight stair flights on mjlab terrain tiles."""

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

TILE_SIZE = (5.8, 2.5)
SPAWN_X = 0.65
FINISH_X = 4.7 - SPAWN_X  # Relative to the spawn origin.
FLIGHTS = ("up_standard", "down_standard", "up_compact", "down_compact")
LEVELS = 6
TERRAIN_GROUP = 0


@dataclass
class StairFlightCfg(SubTerrainCfg):
  descending: bool = False
  compact: bool = False

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    level = min(LEVELS - 1, int(difficulty * LEVELS))
    # Rows progress from a flat approach to taller, longer flights. Compact
    # flights use shorter treads, narrower lanes and irregular step dimensions.
    count = (0, 2, 3, 4, 5, 6)[level]
    riser = (0.0, 0.045, 0.075, 0.105, 0.14, 0.17)[level]
    tread = (0.36, 0.36, 0.35, 0.34, 0.33, 0.32)[level]
    if self.compact:
      tread -= 0.045
    width = 1.9 if not self.compact else 1.80 - 0.11 * level
    variation = 0.08 if self.compact and level >= 2 else 0.0
    rises = riser * rng.uniform(1.0 - variation, 1.0 + variation, count)
    treads = tread * rng.uniform(1.0 - variation, 1.0 + variation, max(0, count - 1))
    heights = np.concatenate(([0.0], np.cumsum(rises)))
    if self.descending:
      heights = heights[::-1]

    first_edge = 1.45 + (rng.uniform(-0.08, 0.08) if level else 0.0)
    edges = [0.0, *(first_edge + np.concatenate(([0.0], np.cumsum(treads))))[:count], self.size[0]]
    body = spec.body("terrain")
    geoms = []
    for i, (start, end, height) in enumerate(zip(edges[:-1], edges[1:], heights, strict=True)):
      bottom = -0.30
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=((start + end) / 2, self.size[1] / 2, (height + bottom) / 2),
        size=((end - start) / 2, width / 2, (height - bottom) / 2),
        group=TERRAIN_GROUP,
        rgba=(0.37, 0.47, 0.56, 1.0) if i % 2 else (0.49, 0.60, 0.67, 1.0),
      )
      geoms.append(TerrainGeometry(geom=geom))
    return TerrainOutput(
      origin=np.array((SPAWN_X, self.size[1] / 2, heights[0])), geometries=geoms
    )


def stair_terrain(seed: int) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    seed=seed,
    curriculum=True,
    size=TILE_SIZE,
    num_rows=LEVELS,
    num_cols=len(FLIGHTS),
    sub_terrains={
      name: StairFlightCfg(descending=name.startswith("down"), compact=name.endswith("compact"))
      for name in FLIGHTS
    },
  )
