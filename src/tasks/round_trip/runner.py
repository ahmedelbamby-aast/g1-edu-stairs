"""Add route-specific records to the existing campus training audit."""

from src.tasks.campus.runner import CampusRunner

from .skills import Route
from .terrain import PACE_BANDS, terrain_manifest


class RoundTripRunner(CampusRunner):
  observation_layout = (*CampusRunner.observation_layout, "route")
  observation_sizes = (*CampusRunner.observation_sizes, 5)
  teacher_sizes = {**CampusRunner.teacher_sizes, CampusRunner.observation_layout: 180}

  def terrain_audit(self, terrain):
    return terrain_manifest(terrain)

  def extra_audit_arrays(self, state):
    return {
      name: getattr(state, name).cpu().numpy()
      for name in ("start_mode", "rehearsal_level", "speed_band", "speed_target", "turn_side")
    }

  def extra_contract(self):
    return {
      "route_terms_in_order": {
        "stage_one_hot": [stage.name.lower() for stage in Route],
        "remaining_m_clipped_to_1": 1, "signed_heading_error_over_pi": 1,
      },
      "pace_bands_mps": PACE_BANDS,
    }
