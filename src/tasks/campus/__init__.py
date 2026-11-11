"""Unitree G1 EDU balance, isolated step and campus flight tasks."""

from copy import deepcopy
from dataclasses import replace

from mjlab.actuator.delayed_actuator import DelayedActuatorCfg
from mjlab.envs import mdp as common
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.stairs import stairs_env_cfg, stairs_runner_cfg

from . import mdp
from .runner import CampusRunner
from .skills import CampusSkillCfg, SkillMode
from .terrain import campus_terrain

TASK_ID = "Unitree-G1-EDU-Campus"
TERRAIN_SEED = 41
TASK_MODES = {
  TASK_ID: "mixed",
  "Unitree-G1-EDU-Balance-Left": "left",
  "Unitree-G1-EDU-Balance-Right": "right",
  "Unitree-G1-EDU-Step-Up": "up_step",
  "Unitree-G1-EDU-Step-Down": "down_step",
  "Unitree-G1-EDU-Campus-Flights": "flight",
}


def campus_env_cfg(
  mode: SkillMode = "mixed", *, play: bool = False, terrain_seed: int = TERRAIN_SEED,
  stress: bool = True,
):
  cfg = stairs_env_cfg()
  cfg.episode_length_s = 20.0 if mode in ("mixed", "flight") else 8.0
  cfg.scene.terrain.terrain_generator = campus_terrain(terrain_seed)
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.sim.nconmax = 72
  cfg.sim.njmax = 750
  cfg.commands["twist"].resampling_time_range = (cfg.episode_length_s + 1,) * 2
  cfg.commands["skill"] = CampusSkillCfg(
    mode=mode, fixed_level=0 if play else None,
    resampling_time_range=(cfg.episode_length_s + 1,) * 2,
  )
  for group in cfg.observations.values():
    group.terms["command"].func = mdp.command_for_skill
    group.terms["phase"].func = mdp.gait_clock
    group.terms["skill"] = ObservationTermCfg(func=common.generated_commands, params={"command_name": "skill"})
  # Actor and critic share several inherited term objects. Copy before modifying.
  cfg.observations["actor"].terms["height_scan"] = deepcopy(cfg.observations["actor"].terms["height_scan"])
  scan = cfg.observations["actor"].terms["height_scan"]
  scan.func = mdp.noisy_scan
  scan.params["dropout"] = stress
  scan.delay_max_lag = 2 if stress else 0
  scan.delay_update_period = 5
  cfg.observations["actor"].enable_corruption = stress

  for sensor in cfg.scene.sensors:
    if sensor.name == "body_ground_contact":
      sensor.primary = ContactMatch(mode="geom", pattern=r"^(?!.*foot).*_collision$", entity="robot")
  cfg.events["reset_base"].func = mdp.reset_near_course
  cfg.events["reset_base"].params["pose_range"] = {"x": (-.04, .04), "y": (-.045, .045), "yaw": (-.10, .10)}
  if stress:
    cfg.events["foot_friction"].params["ranges"] = (.38, 1.25)
    cfg.events["encoder_bias"].params["bias_range"] = (-.020, .020)
    cfg.events.pop("base_com")
    cfg.events["mass_and_inertia"] = EventTermCfg(
      mode="startup", func=dr.pseudo_inertia,
      params={"alpha_range": (-.045, .045), "t_range": (-.02, .02),
              "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    )
    cfg.events["motor_gains"] = EventTermCfg(
      mode="startup", func=dr.pd_gains, params={"kp_range": (.88, 1.12), "kd_range": (.85, 1.15)},
    )
    robot = cfg.scene.entities["robot"]
    robot.articulation = replace(robot.articulation, actuators=tuple(
      DelayedActuatorCfg(base_cfg=actuator, delay_max_lag=3, delay_update_period=200)
      for actuator in robot.articulation.actuators
    ))
    cfg.events["disturbance"] = EventTermCfg(mode="interval", func=mdp.course_push, interval_range_s=(2.0, 4.0))
  else:
    for name in ("foot_friction", "encoder_bias", "base_com"):
      cfg.events.pop(name)

  for name in ("track_linear_velocity", "progress", "centerline", "pose", "foot_gait", "stand_still"):
    cfg.rewards.pop(name)
  cfg.rewards["motion"] = RewardTermCfg(func=mdp.motion_reward, weight=1.2)
  cfg.rewards["lane_center"] = RewardTermCfg(func=mdp.lane_cost, weight=-1.5)
  cfg.rewards["practice"] = RewardTermCfg(func=mdp.practice_reward, weight=1.6)
  cfg.rewards["phase_bonus"] = RewardTermCfg(func=mdp.phase_bonus, weight=12.0)
  if mode in ("left", "right"):
    cfg.rewards["sustained_hold"] = RewardTermCfg(func=mdp.sustained_hold_reward, weight=2.0)
  cfg.terminations["time_out"] = TerminationTermCfg(func=mdp.timeout, time_out=True)
  cfg.terminations["out_of_lane"] = TerminationTermCfg(func=mdp.lane_exit)
  cfg.terminations["landing"] = TerminationTermCfg(func=mdp.advance)
  cfg.curriculum = {"campus": CurriculumTermCfg(func=mdp.course_curriculum)}
  return cfg


def campus_runner_cfg():
  cfg = stairs_runner_cfg()
  cfg.experiment_name = "g1_edu_campus"
  cfg.clip_actions = 3.0
  cfg.algorithm.learning_rate = .0003
  cfg.algorithm.schedule = "fixed"
  cfg.algorithm.entropy_coef = .003
  return cfg


for task_id, mode in TASK_MODES.items():
  register_mjlab_task(
    task_id=task_id, env_cfg=campus_env_cfg(mode),
    play_env_cfg=campus_env_cfg(mode, play=True, stress=False),
    rl_cfg=campus_runner_cfg(), runner_cls=CampusRunner,
  )
