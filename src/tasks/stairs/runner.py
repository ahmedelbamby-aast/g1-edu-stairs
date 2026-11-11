"""Initialize stair training from the bundled Unitree flat-walking actor."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from mjlab.rl.exporter_utils import get_base_metadata

from src.tasks.velocity.rl import VelocityOnPolicyRunner

OBSERVATION_PREFIX = (
  "base_ang_vel", "projected_gravity", "command", "phase", "joint_pos", "joint_vel", "actions",
)
EXPLORATION_STD = 0.28
PRIOR_SAMPLES = 262_144


class StairsOnPolicyRunner(VelocityOnPolicyRunner):
  def save(self, path: str, infos=None) -> None:
    # Only expose checkpoint files after the write and matching export finish.
    destination = Path(path)
    temporary = destination.with_suffix(".pt.tmp")
    super().save(str(temporary), infos)
    temporary.replace(destination)

  @torch.no_grad()
  def initialize_actor(self, path: Path) -> dict[str, str | float | int]:
    import onnx
    import onnxruntime as ort

    model = onnx.load(path)
    source = {entry.key: entry.value for entry in model.metadata_props}
    current = get_base_metadata(self.env.unwrapped, "stairs")
    if tuple(source["observation_names"].split(",")) != OBSERVATION_PREFIX:
      raise ValueError("The source ONNX file must use Unitree's flat G1 observations")
    if tuple(current["observation_names"]) != (*OBSERVATION_PREFIX, "height_scan"):
      raise ValueError("The stair actor's observation layout is incompatible with the flat policy")
    if source["joint_names"].split(",") != current["joint_names"]:
      raise ValueError("The source ONNX file uses a different joint order")
    for key in ("action_scale", "default_joint_pos", "joint_stiffness", "joint_damping"):
      np.testing.assert_allclose(
        np.fromstring(source[key], sep=","), current[key], rtol=0.0, atol=5.1e-4
      )

    arrays = {item.name: onnx.numpy_helper.to_array(item) for item in model.graph.initializer}
    denominator, = (node.input[1] for node in model.graph.node if node.op_type == "Div")
    actor = self.alg.get_policy()
    state = actor.state_dict()
    source_mean = arrays["obs_normalizer._mean"]
    prefix_size = 11 + 3 * self.env.num_actions
    if source_mean.shape != (1, prefix_size):
      raise ValueError("Unexpected Unitree flat actor input size")

    def tensor(name: str) -> torch.Tensor:
      return torch.as_tensor(arrays[name].copy(), device=self.device)

    for name, weights in state.items():
      if name == "mlp.0.weight":
        weights.zero_()
        weights[:, :prefix_size].copy_(tensor(name))
      elif name.startswith("mlp."):
        weights.copy_(tensor(name))

    normalizer = actor.obs_normalizer
    state["obs_normalizer._mean"][:, :prefix_size].copy_(tensor("obs_normalizer._mean"))
    std = (tensor(denominator) - normalizer.eps).clamp_min(0)
    state["obs_normalizer._std"][:, :prefix_size].copy_(std)
    state["obs_normalizer._var"][:, :prefix_size].copy_(std.square())
    state["obs_normalizer.count"].fill_(PRIOR_SAMPLES)
    state["distribution.std_param"].fill_(EXPLORATION_STD)

    # Verify the transfer against the original ONNX on observations from this scene.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    obs = self.env.get_observations()[:8]
    flat_obs = obs["actor"][:, :prefix_size].contiguous().cpu().numpy()
    input_name = session.get_inputs()[0].name
    expected = np.concatenate([session.run(None, {input_name: row[None, :]})[0] for row in flat_obs])
    actual = actor(obs).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
    error = float(np.max(np.abs(actual - expected)))
    print(f"[INFO] Initialized from {path.name}. ONNX maximum action error: {error:.2e}")
    return {
      "source": path.name,
      "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      "onnx_max_abs_error": error,
      "exploration_std": EXPLORATION_STD,
      "normalization_prior_samples": PRIOR_SAMPLES,
    }
