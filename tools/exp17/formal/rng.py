"""Serializable snapshots of every RNG used by the frozen native path.

Capture never initializes CUDA. CUDA training must already have initialized it;
the CPU protocol tests deliberately have no CUDA state or model operations.
"""
import copy
import random

import numpy as np
import torch


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": state[0], "keys": state[1].tolist(),
                      "position": state[2], "has_gauss": state[3], "cached_gaussian": state[4]},
            "torch_cpu": torch.get_rng_state().clone(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "torch_cuda": [v.cpu().clone() for v in torch.cuda.get_rng_state_all()]
                          if torch.cuda.is_initialized() else []}


def rng_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(rng_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(rng_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def restore_rng(state):
    if state["cuda_initialized"] != torch.cuda.is_initialized():
        raise ValueError("CUDA initialization topology changed at recovery")
    if state["cuda_initialized"] and len(state["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("CUDA device count changed at recovery")
    np_state = state["numpy"]
    random.setstate(copy.deepcopy(state["python"]))
    np.random.set_state((np_state["algorithm"], np.asarray(np_state["keys"], dtype=np.uint32),
                         np_state["position"], np_state["has_gauss"], np_state["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["cuda_initialized"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if not rng_equal(capture_rng(), state):
        raise ValueError("RNG did not restore exactly")
