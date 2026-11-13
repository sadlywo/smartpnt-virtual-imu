"""Evaluation input gradients with cuDNN's gradient-capable recurrent forward."""
from __future__ import annotations
from torch import nn

def eval_forward_for_input_gradients(model,inputs):
    """Keep eval behavior and restore all flags after constructing the graph.

    cuDNN RNN backward needs its training forward. For zero-dropout RNNs the
    training flag only requests the backward workspace. Other modules stay in
    eval mode, including normalization and head dropout. Fitting, metrics,
    SHAP forward calls and latency use the normal eval path.
    """
    if model.training:raise ValueError('Explanations require an eval-mode model')
    recurrent=[m for m in model.modules() if isinstance(m,(nn.RNN,nn.GRU,nn.LSTM))]
    if any(m.dropout and m.num_layers>1 for m in recurrent):
        raise ValueError('Recurrent dropout must be zero for eval-equivalent input gradients')
    original=[m.training for m in recurrent]
    try:
        for m in recurrent:m.training=True
        return model(inputs)
    finally:
        for m,flag in zip(recurrent,original):m.training=flag
