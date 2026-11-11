"""G1 EDU task for climbing, turning, and descending the same flight."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.campus import campus_env_cfg, campus_runner_cfg

from . import mdp
from .runner import RoundTripRunner
from .skills import RoundTripSkillCfg
from .terrain import round_trip_terrain

TASK_ID = "Unitree-G1-EDU-Stair-Round-Trip"


def round_trip_env_cfg(*, play=False, stress=True, terrain_seed=41):
  cfg = campus_env_cfg("flight", play=play, stress=stress)
  cfg.episode_length_s = 52.
  cfg.scene.terrain.terrain_generator = round_trip_terrain(terrain_seed)
  resample = (cfg.episode_length_s + 1,) * 2
  cfg.commands["twist"].resampling_time_range = resample
  cfg.commands["skill"] = RoundTripSkillCfg(
    practice=not play, fixed_level=0 if play else None,
    resampling_time_range=resample, drill_timeout_s=22.,
  )
  for group in cfg.observations.values():
    group.terms["command"].func = mdp.velocity_command
    group.terms["phase"].func = mdp.gait_clock
    group.terms["route"] = ObservationTermCfg(func=mdp.route_features)
  cfg.events["reset_base"] = EventTermCfg(
    mode="reset", func=mdp.reset_for_route,
    params={"pose_range": {"x": (-.04, .04), "y": (-.045, .045), "yaw": (-.10, .10)}},
  )
  cfg.rewards.pop("track_angular_velocity")
  cfg.rewards["motion"] = RewardTermCfg(func=mdp.velocity_reward, weight=1.5)
  cfg.rewards["turn_progress"] = RewardTermCfg(func=mdp.turn_progress, weight=.6)
  cfg.terminations["out_of_lane"] = TerminationTermCfg(func=mdp.lane_exit)
  cfg.terminations["time_out"] = TerminationTermCfg(func=mdp.timeout, time_out=True)
  cfg.curriculum = {"route": CurriculumTermCfg(func=mdp.route_curriculum)}
  return cfg


def round_trip_runner_cfg():
  cfg = campus_runner_cfg()
  cfg.experiment_name = "g1_edu_stair_round_trip"
  return cfg


register_mjlab_task(
  task_id=TASK_ID, env_cfg=round_trip_env_cfg(),
  play_env_cfg=round_trip_env_cfg(play=True, stress=False),
  rl_cfg=round_trip_runner_cfg(), runner_cls=RoundTripRunner,
)
