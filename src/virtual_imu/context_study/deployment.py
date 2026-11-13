"""Finite-window causal deployment without waiting for the offline trailing context."""
from __future__ import annotations
import torch

def available_prediction(model,prefix,lookahead=0):
    """Return the latest target whose declared future is available.

    The caller supplies only arrived samples from the current fixed anchor start.
    For the study's center40 schedule use prefix lengths141..180 (+lookahead)
    and reset the anchor start every40 samples. No trailing140 points are needed.
    This is window replay rather than persistent RNN-state streaming.
    """
    if not isinstance(lookahead,int) or lookahead<0 or prefix.shape[-1]<=lookahead:
        raise ValueError('Need a nonempty arrived prefix beyond finite lookahead')
    return model(prefix)[...,-lookahead-1]
