"""Transfer the validated walking policy and record reproducible campus runs."""

import json
import os
from pathlib import Path

import numpy as np
from mjlab.rl.exporter_utils import get_base_metadata

from src.tasks.stairs.runner import OBSERVATION_PREFIX, StairsOnPolicyRunner

from .skills import Drill, Phase, skill
from .terrain import terrain_manifest

RUNTIME_KEYS = (
  "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
  "NUMEXPR_NUM_THREADS", "NCCL_P2P_DISABLE", "MUJOCO_GL", "WARP_CACHE_PATH",
  "TORCHRUNX_LOG_DIR", "WANDB_MODE", "TZ", "LOCAL_RANK", "WORLD_SIZE",
)


class CampusRunner(StairsOnPolicyRunner):
  observation_layout = (*StairsOnPolicyRunner.observation_layout, "skill")
  teacher_layouts = (OBSERVATION_PREFIX, StairsOnPolicyRunner.observation_layout)

  def write_run_audit(self, log_dir: Path, rank: int) -> None:
    env = self.env.unwrapped
    state = skill(env)
    directory = log_dir / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    fields = sorted(env.event_manager.domain_randomization_fields)
    arrays = {field: getattr(env.sim.model, field).cpu().numpy() for field in fields}
    arrays.update(
      encoder_bias=state.robot.data.encoder_bias.cpu().numpy(),
      initial_qpos=env.sim.data.qpos.cpu().numpy(),
      initial_terrain_level=state.terrain.terrain_levels.cpu().numpy(),
      initial_terrain_column=state.terrain.terrain_types.cpu().numpy(),
      drill=state.drill.cpu().numpy(),
      support=state.support.cpu().numpy(),
    )
    np.savez_compressed(directory / f"startup-rank-{rank}.npz", **arrays)
    metadata = {
      "rank": rank, "seed": env.cfg.seed, "device": env.device, "worlds": env.num_envs,
      "control_step_s": env.step_dt, "physics_step_s": env.physics_dt,
      "process_settings": {key: os.environ.get(key) for key in RUNTIME_KEYS},
      "randomized_model_fields": fields,
      "robot_joint_names": list(state.robot.joint_names),
      "model_body_names": [env.sim.mj_model.body(i).name for i in range(env.sim.mj_model.nbody)],
      "model_geom_names": [env.sim.mj_model.geom(i).name for i in range(env.sim.mj_model.ngeom)],
      "model_actuator_names": [env.sim.mj_model.actuator(i).name for i in range(env.sim.mj_model.nu)],
      "phases": [phase.name.lower() for phase in Phase],
      "drills": [drill.name.lower() for drill in Drill],
    }
    (directory / f"runtime-rank-{rank}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if rank == 0:
      terrain = env.cfg.scene.terrain.terrain_generator
      (directory / "terrain.json").write_text(
        json.dumps(terrain_manifest(terrain.seed, [tile.seed for tile in terrain.sub_terrains.values()]), indent=2) + "\n",
        encoding="utf-8",
      )
      contract = {
        "version": 1,
        "control_period_s": env.step_dt,
        "actor_terms_in_order": dict(zip(self.observation_layout, (3, 3, 3, 2, 29, 29, 29, 63, 19), strict=True)),
        "skill_terms_in_order": {
          "phase_one_hot": metadata["phases"], "support_one_hot": ["left", "right"],
          "foothold_minus_swing_site_m": ["heading_x", "heading_y", "vertical_z"],
          "contact_flags": ["left", "right"], "drill_one_hot": metadata["drills"],
        },
        "terrain_scan": {"shape_yx": [7, 9], "spacing_m": .20, "height_offset_m": .80, "missing_value": 1.70},
        "raw_action_clip": self.env.clip_actions,
        "control": "default_joint_pos + action_scale * clipped_action",
        "unitree": get_base_metadata(env, "g1_edu_campus"),
      }
      (directory / "sim2real-contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
