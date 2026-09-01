"""Phase-II research modules for GeoRectNetPlus."""

from .edl import kl_dirichlet, edl_loss
from .sdr import SignedDistanceHead, compute_sdf_target
from .gates import _geometry_gate, gradient_orientation_gate, cafcg_gate
from .pseudo_label import (
    progressive_threshold,
    TemporalVotingBuffer,
    multi_uncertainty_gate,
    save_tvr_buffer,
    load_tvr_buffer,
)

__all__ = [
    "kl_dirichlet", "edl_loss",
    "SignedDistanceHead", "compute_sdf_target",
    "_geometry_gate", "gradient_orientation_gate", "cafcg_gate",
    "progressive_threshold", "TemporalVotingBuffer",
    "multi_uncertainty_gate", "save_tvr_buffer", "load_tvr_buffer",
]
