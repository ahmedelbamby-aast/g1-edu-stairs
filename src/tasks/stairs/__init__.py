"""Unitree G1 EDU stair locomotion task."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, RayCastSensorCfg
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from . import mdp
from .terrain import LEVELS, TERRAIN_GROUP, stair_terrain

TASK_ID = "Unitree-G1-EDU-Stairs"
TERRAIN_SEED = 41


def stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_rough_env_cfg()
  cfg.episode_length_s = 14.0
  cfg.scene.terrain.terrain_generator = stair_terrain(TERRAIN_SEED)
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.sim.njmax = 600
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  for sensor in cfg.scene.sensors:
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.pattern = GridPatternCfg(size=(1.6, 1.2), resolution=0.2)
      sensor.include_geom_groups = (TERRAIN_GROUP,)
      sensor.max_distance = 2.5
      sensor.debug_vis = False
  cfg.scene.sensors += (
    ContactSensorCfg(
      name="body_ground_contact",
      primary=ContactMatch(mode="body", pattern=r"^(pelvis|torso_link)$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )
  for group in cfg.observations.values():
    scan = group.terms["height_scan"]
    scan.params = {"sensor_name": "terrain_scan", "offset": 0.8, "miss_value": 1.7}
    scan.scale = 1.0
  cfg.observations["critic"].terms["foot_height"].func = mdp.foot_height_from_pelvis

  command = cfg.commands["twist"]
  command.resampling_time_range = (cfg.episode_length_s + 1,) * 2
  command.rel_standing_envs = 0.0
  command.rel_heading_envs = 1.0
  command.heading_control_stiffness = 1.5
  command.ranges.lin_vel_x = (0.45, 0.70)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (-0.65, 0.65)
  command.ranges.heading = (0.0, 0.0)
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.08, 0.08), "y": (-0.08, 0.08), "yaw": (-0.12, 0.12),
  }
  cfg.events.pop("push_robot")
  cfg.events["foot_friction"].params["ranges"] = (0.55, 1.25)

  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_course_speed, weight=1.2, params={"command_name": "twist"}
  )
  cfg.rewards["progress"] = RewardTermCfg(
    func=mdp.signed_progress, weight=1.0, params={"command_name": "twist"}
  )
  cfg.rewards["centerline"] = RewardTermCfg(func=mdp.centerline_cost, weight=-2.0)
  cfg.rewards["landing"] = RewardTermCfg(func=mdp.success_bonus, weight=250.0)
  cfg.rewards["is_terminated"] = RewardTermCfg(func=mdp.failure_cost, weight=-150.0)
  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["foot_gait"].weight = 0.2
  # Unitree's flat-foot clearance target uses absolute world height.
  cfg.rewards.pop("foot_clearance")

  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(55.0)
  cfg.terminations["body_hit"] = TerminationTermCfg(func=mdp.body_hit)
  cfg.terminations["low_root"] = TerminationTermCfg(
    func=common.root_height_below_minimum, params={"minimum_height": 0.40}
  )
  cfg.terminations["out_of_lane"] = TerminationTermCfg(func=mdp.out_of_lane)
  cfg.terminations["landing"] = TerminationTermCfg(
    func=mdp.reached_landing,
    params={"asset_cfg": cfg.observations["critic"].terms["foot_height"].params["asset_cfg"]},
  )
  cfg.curriculum = {"stairs": CurriculumTermCfg(func=mdp.flight_curriculum)}
  if play:
    cfg.curriculum.clear()
    cfg.observations["actor"].enable_corruption = False
    cfg.scene.terrain.max_init_terrain_level = LEVELS - 1
  return cfg


def stairs_runner_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_edu_stairs"
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  return cfg


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=stairs_env_cfg(),
  play_env_cfg=stairs_env_cfg(play=True),
  rl_cfg=stairs_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
