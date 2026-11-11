"""Capture the configured stair courses and optional checkpoint rollouts."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scene import Scene
from PIL import Image

from src.tasks.stairs import TERRAIN_SEED, stairs_env_cfg, stairs_runner_cfg
from src.tasks.stairs.terrain import FLIGHTS, LEVELS, SPAWN_X, TILE_SIZE

COURSES = ((1, 0), (3, 1), (5, 2), (5, 3))
WIDTH, HEIGHT = 1280, 720
FRAME_STRIDE = 4  # 50 Hz policy / 4 = 12.5 frames/s.


@torch.inference_mode()
def rollout(args, cfg, courses):
  count = len(courses) * args.attempts
  cfg.scene.num_envs = count
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    agent = stairs_runner_cfg()
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    terrain = env.scene.terrain
    levels, columns = (
      torch.tensor(values, device=args.device).repeat_interleave(args.attempts)
      for values in zip(*courses, strict=True)
    )
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = columns
    terrain.env_origins[:] = terrain.terrain_origins[levels, columns]
    obs, _ = wrapper.reset()
    runner = MjlabOnPolicyRunner(wrapper, asdict(agent), device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    origins = terrain.env_origins.cpu().numpy().copy()
    qpos = env.sim.data.qpos
    frames = [qpos.cpu().numpy().copy()]
    done_once = torch.zeros(count, device=args.device, dtype=torch.bool)
    lengths = torch.zeros(count, device=args.device, dtype=torch.long)
    outcomes = torch.zeros_like(lengths)
    for step in range(1, env.max_episode_length + 1):
      obs, _, done, _ = wrapper.step(policy(obs))
      first_done = done.bool() & ~done_once
      manager = env.termination_manager
      fall = manager.get_term("fell_over") | manager.get_term("body_hit") | manager.get_term("low_root")
      codes = torch.where(fall, 2, torch.where(manager.get_term("out_of_lane"), 3, 4))
      codes = torch.where(manager.get_term("landing"), 1, codes)
      outcomes = torch.where(first_done, codes, outcomes)
      lengths = torch.where(first_done, (step - 1) // FRAME_STRIDE + 1, lengths)
      done_once |= done.bool()
      if step % FRAME_STRIDE == 0:
        frames.append(qpos.cpu().numpy().copy())
      if done_once.all():
        break

    frame_data = np.stack(frames)
    names = ("unfinished", "landed", "fell", "left_lane", "timed_out")
    results = []
    for course in range(len(courses)):
      candidates = []
      for index in range(course * args.attempts, (course + 1) * args.attempts):
        length = int(lengths[index]) or len(frames)
        poses = frame_data[:length, index]
        outcome = int(outcomes[index])
        progress = float(poses[:, 0].max() - origins[index, 0])
        candidates.append((outcome == 1, progress, index, poses, names[outcome]))
      _, progress, index, poses, outcome = max(candidates, key=lambda item: item[:2])
      results.append((origins[index], poses, outcome, progress))
    return env.sim.mj_model, results
  finally:
    env.close()


def course_views(cfg, courses):
  # Native MuJoCo compiles the same scene without allocating rollout workers.
  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()
  origins = scene.terrain.terrain_origins.cpu().numpy()
  pose = model.key("init_state").qpos
  views = []
  for level, column in courses:
    origin = origins[level, column]
    start = pose.copy()
    start[:3] += origin
    views.append((origin, start[None, :], "course_preview", 0.0))
  return model, views


def render(model, results, courses, output, screenshot_only):
  model.vis.global_.offwidth = WIDTH
  model.vis.global_.offheight = HEIGHT
  model.vis.global_.fovy = 42
  model.stat.extent = 5.0
  model.light_castshadow[:] = False
  data = mujoco.MjData(model)
  terrain_id = model.body("terrain").id
  terrain_mask = model.geom_bodyid == terrain_id
  original_alpha = model.geom_rgba[:, 3].copy()
  manifest = []
  with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
    for (level, column), (origin, poses, outcome, progress) in zip(courses, results, strict=True):
      name = f"level-{level}-{FLIGHTS[column].replace('_', '-')}"
      lower_x = origin[0] - SPAWN_X
      x, y = model.geom_pos[:, :2].T
      tile = (x > lower_x) & (x < lower_x + TILE_SIZE[0]) & (abs(y - origin[1]) < 0.01)
      model.geom_rgba[:, 3] = np.where(terrain_mask & ~tile, 0, original_alpha)
      camera = mujoco.MjvCamera()
      camera.lookat[:] = (lower_x + 2.65, origin[1], 0.48)
      camera.distance = 6.1
      camera.azimuth = 92
      camera.elevation = -17

      def frame(pose):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        return renderer.render()

      near_stairs = np.argmin(np.abs(poses[:, 0] - (origin[0] + 1.30)))
      Image.fromarray(frame(poses[near_stairs])).save(output / f"{name}.png", optimize=True)
      if not screenshot_only:
        video = output / f"{name}.mp4"
        with subprocess.Popen(
          ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
           "-framerate", "12.5", "-i", "-", "-c:v", "libx264", "-threads", "2",
           "-preset", "fast", "-crf", "28", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(video)],
          stdin=subprocess.PIPE,
        ) as encoder:
          for pose in poses:
            encoder.stdin.write(frame(pose).tobytes())
          encoder.stdin.close()
          if encoder.wait() != 0:
            raise RuntimeError(f"FFmpeg could not encode {video}")
      manifest.append({
        "course": name, "outcome": outcome, "furthest_distance_m": round(progress, 3),
        "frames": len(poses), "png": f"{name}.png",
        "mp4": None if screenshot_only else f"{name}.mp4",
      })
  return manifest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--screenshots-only", action="store_true")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--attempts", type=int, default=4)
  parser.add_argument("--seed", type=int, default=913)
  parser.add_argument("--terrain-seed", type=int, default=TERRAIN_SEED)
  parser.add_argument("--course", choices=FLIGHTS)
  parser.add_argument("--level", type=int, choices=range(LEVELS))
  args = parser.parse_args()
  if args.attempts < 1:
    parser.error("--attempts must be positive")
  if (args.course is None) != (args.level is None):
    parser.error("--course and --level must be used together")
  courses = COURSES if args.course is None else ((args.level, FLIGHTS.index(args.course)),)
  args.output.mkdir(parents=True, exist_ok=True)
  cfg = stairs_env_cfg(play=True)
  cfg.seed = args.seed
  cfg.scene.terrain.terrain_generator.seed = args.terrain_seed
  if args.checkpoint:
    model, results = rollout(args, cfg, courses)
  else:
    model, results = course_views(cfg, courses)
  manifest = {
    "checkpoint": args.checkpoint.name if args.checkpoint else None,
    "terrain_seed": args.terrain_seed,
    "resolution": [WIDTH, HEIGHT],
    "courses": render(model, results, courses, args.output, args.screenshots_only or not args.checkpoint),
  }
  (args.output / "media.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  print(f"Saved {len(courses)} courses to {args.output}")


if __name__ == "__main__":
  main()
