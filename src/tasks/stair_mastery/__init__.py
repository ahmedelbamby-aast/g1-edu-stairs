"""Additive expert and student tasks for robust stair round trips."""

from copy import deepcopy

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import GridPatternCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.noise import UniformNoiseCfg

from src.tasks.round_trip import round_trip_env_cfg, round_trip_runner_cfg
from src.tasks.round_trip.improved import (
  ImprovedSkillCfg, adaptive_curriculum, controlled_transfer, descent_overspeed,
  estimated_skill, next_footholds, proprioception, speed_clock,
)
from src.tasks.campus.mdp import noisy_scan

from .runner import MasteryRunner

TASK_ID = "Unitree-G1-EDU-Stair-Mastery"
TASKS = {
  TASK_ID: ("round_trip", False),
  "Unitree-G1-EDU-Stair-Expert-Ascent": ("ascent", False),
  "Unitree-G1-EDU-Stair-Expert-Descent": ("descent", False),
  "Unitree-G1-EDU-Stair-Expert-Turn": ("turn", False),
  "Unitree-G1-EDU-Stair-Student": ("round_trip", True),
}


def mastery_env_cfg(expert="round_trip", *, student=False, play=False, stress=True, terrain_seed=41):
  cfg = round_trip_env_cfg(play=play, stress=stress, terrain_seed=terrain_seed)
  resample = (cfg.episode_length_s + 1,) * 2
  cfg.commands["skill"] = ImprovedSkillCfg(
    expert=expert, practice=not play, fixed_level=0 if play else None,
    resampling_time_range=resample, drill_timeout_s=30.,
    estimator_bias_m=.015 if student and stress else 0.,
    estimator_noise_m=.010 if student and stress else 0.,
    estimator_dropout=.02 if student and stress else 0.,
  )
  # Actor and critic have historically shared term objects. Isolate each group.
  cfg.observations = deepcopy(cfg.observations)
  for name, group in cfg.observations.items():
    group.terms = deepcopy(group.terms)
    group.terms["phase"].func = speed_clock
  actor = cfg.observations["actor"]
  actor.enable_corruption = student and stress
  actor.history_length = None
  actor.nan_policy = "error"
  actor.terms["skill"].func = estimated_skill
  actor.terms["skill"].params = {"estimated": student and stress}
  actor.terms["skill"].delay_max_lag = 1 if student and stress else 0
  actor.terms["skill"].delay_update_period = 5
  actor.terms["height_scan"].params["dropout"] = student and stress
  actor.terms["height_scan"].delay_max_lag = 2 if student and stress else 0
  fine = deepcopy(next(sensor for sensor in cfg.scene.sensors if sensor.name == "terrain_scan"))
  fine.name = "terrain_scan_fine"
  fine.pattern = GridPatternCfg(size=(1.2, .8), resolution=.10)
  cfg.scene.sensors += (fine,)
  actor.terms["footholds"] = ObservationTermCfg(
    func=next_footholds, params={"estimated": student and stress},
    delay_max_lag=1 if student and stress else 0, delay_update_period=5,
  )
  actor.terms["fine_scan"] = ObservationTermCfg(
    func=noisy_scan, params={"sensor_name": fine.name, "offset": .8, "miss_value": 1.7,
                            "dropout": student and stress},
    noise=UniformNoiseCfg(n_min=-.025, n_max=.025) if student and stress else None,
    delay_max_lag=2 if student and stress else 0, delay_update_period=5,
  )
  actor.terms["proprioception_history"] = ObservationTermCfg(
    func=proprioception, history_length=5,
    noise=UniformNoiseCfg(n_min=-.01, n_max=.01) if student and stress else None,
    delay_max_lag=1 if student and stress else 0, delay_update_period=5,
  )
  # Preserve a separate accurate input stream for expert imitation.
  teacher = deepcopy(actor)
  teacher.enable_corruption = False
  for name, term in teacher.terms.items():
    term.delay_max_lag = term.delay_min_lag = 0
    if name == "footholds":
      term.params["estimated"] = False
    if name == "skill":
      term.params["estimated"] = False
    if name in ("height_scan", "fine_scan"):
      term.params["dropout"] = False
      term.func = noisy_scan
  cfg.observations["teacher"] = teacher
  cfg.rewards["descent_overspeed"] = RewardTermCfg(func=descent_overspeed, weight=-1.4)
  cfg.rewards["controlled_transfer"] = RewardTermCfg(func=controlled_transfer, weight=.8)
  cfg.curriculum["route"] = CurriculumTermCfg(func=adaptive_curriculum)
  return cfg


def mastery_runner_cfg():
  cfg = round_trip_runner_cfg()
  cfg.experiment_name = "g1_edu_stair_mastery"
  cfg.max_iterations = 500
  cfg.save_interval = 100
  return cfg


for task_id, (expert, student) in TASKS.items():
  register_mjlab_task(
    task_id=task_id, env_cfg=mastery_env_cfg(expert, student=student),
    play_env_cfg=mastery_env_cfg(expert, student=student, play=True, stress=False),
    rl_cfg=mastery_runner_cfg(), runner_cls=MasteryRunner,
  )
