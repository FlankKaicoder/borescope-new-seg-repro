"""Three native AdamW groups and head-aware EMA with separate active clocks."""
import copy
import math

import torch
from torch import nn

HEAD_PREFIX = "state_head."


def require(value, message):
    if not value:
        raise ValueError(message)


def build_adamw_groups(model, *, lr=.01, weight_decay=.0005):
    groups = {"weight": [], "bn": [], "bias": []}
    names = {"weight": [], "bn": [], "bias": []}
    seen = set()
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            require(id(parameter) not in seen, "Optimizer parameter is shared or repeated")
            seen.add(id(parameter))
            group = "bias" if name == "bias" else "bn" if isinstance(module, nn.modules.batchnorm._BatchNorm) else "weight"
            groups[group].append(parameter)
            names[group].append((module_name + "." if module_name else "") + name)
    required = {id(p) for p in model.parameters() if p.requires_grad}
    require(seen == required, "Optimizer omitted a trainable parameter")
    result = [{"params": groups[name], "param_group": name, "lr": lr, "initial_lr": lr,
               "weight_decay": weight_decay if name == "weight" else 0.}
              for name in ("weight", "bn", "bias")]
    return result, [names[name] for name in ("weight", "bn", "bias")]


def build_adamw(model, *, lr=.01, weight_decay=.0005):
    groups, names = build_adamw_groups(model, lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(.937, .999), eps=1e-8, amsgrad=False)
    return optimizer, names


class CPUModelEMA:
    """Synthetic reference for the native global EMA formula, without CUDA APIs.

    Future native runtime supplies its pinned ModelEMA object to HeadAwareEMA.
    """
    def __init__(self, model, *, decay=.9999, tau=2000.):
        self.ema = copy.deepcopy(model).eval()
        self.updates, self.decay_limit, self.tau = 0, decay, tau
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    def decay(self, updates):
        return self.decay_limit * (1. - math.exp(-updates / self.tau))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay(self.updates)
        source = model.state_dict()
        target_values, source_values = [], []
        for name, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                target_values.append(value)
                source_values.append(source[name].detach())
        if target_values:
            # Pinned 8.4.117/Torch2.8 CPU path, including its rounding order.
            torch._foreach_lerp_(target_values, source_values, 1. - decay)


class HeadAwareEMA:
    """Call wrapped native EMA exactly once; freeze its head on empty batches."""
    def __init__(self, native_ema, *, head_prefix=HEAD_PREFIX):
        self.native = native_ema
        self.head_prefix = head_prefix
        self.head_active_updates = 0
        self._pending_active = None

    @property
    def ema(self):
        return self.native.ema

    @property
    def updates(self):
        return self.native.updates

    @updates.setter
    def updates(self, value):
        self.native.updates = value

    def set_head_active(self, active):
        require(type(active) is bool and self._pending_active is None, "EMA head activity must be set once per update")
        self._pending_active = active

    @torch.no_grad()
    def update(self, model):
        require(type(self._pending_active) is bool, "EMA head activity was not explicitly set")
        active = self._pending_active
        self._pending_active = None
        head_state = {k: v.detach().clone() for k, v in self.ema.state_dict().items()
                      if k.startswith(self.head_prefix)} if not active else {}
        before = self.updates
        self.native.update(model)
        require(self.updates == before + 1, "Native EMA must advance exactly once")
        for name, value in head_state.items():
            self.ema.state_dict()[name].copy_(value)
        if active:
            self.head_active_updates += 1

    def state_dict(self):
        require(self._pending_active is None, "Cannot checkpoint pending EMA activity")
        return {"updates": self.updates, "head_active_updates": self.head_active_updates,
                "state": copy.deepcopy(self.ema.state_dict())}

    def load_state_dict(self, state):
        require(self._pending_active is None, "Cannot restore pending EMA activity")
        self.ema.load_state_dict(state["state"], strict=True)
        self.updates = state["updates"]
        self.head_active_updates = state["head_active_updates"]


def validate_gradients(model, *, head_active):
    head, gradients = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(HEAD_PREFIX):
            head.append(parameter)
            require((parameter.grad is not None) == head_active, "State-head gradient disagrees with matching activity")
        else:
            require(parameter.grad is not None, "Native/projection gradient missing: " + name)
        if parameter.grad is not None:
            gradients.append(parameter.grad.detach())
    require(not head_active or bool(head), "Active state loss has no state head")
    if gradients:
        # One aggregate device observation, without synchronizing each tensor.
        norms = torch.stack(torch._foreach_norm(gradients, 2))
        require(bool(torch.isfinite(norms).all()), "A native/projection/state gradient is nonfinite")


@torch.no_grad()
def joint_gradient_stats(model):
    """Observe existing gradients without backward, RNG draws or state updates.

    Native parameters and the unchanged J1 projection form the main domain;
    the raw64 scalar head is reported separately. Empty head grads remain None.
    Device scalars are reduced in groups; there is no per-parameter .item().
    """
    domains = {"main": [], "state_head": []}
    for name, parameter in model.named_parameters():
        domains["state_head" if name.startswith(HEAD_PREFIX) else "main"].append(parameter)
    report = {}
    for name, parameters in domains.items():
        trainable = [p for p in parameters if p.requires_grad]
        gradients = [p.grad.detach() for p in trainable if p.grad is not None]
        require(all(not gradient.is_sparse for gradient in gradients), "Gradient observer requires dense native tensors")
        elements = sum(g.numel() for g in gradients)
        base = {"trainable_tensors": len(trainable), "frozen_tensors": len(parameters) - len(trainable),
                "present_tensors": len(gradients), "missing_tensors": len(trainable) - len(gradients),
                "elements": elements}
        if not gradients:
            report[name] = {**base, "norm": 0., "max_abs": 0., "zero_elements": 0, "nonfinite_elements": 0}
            continue
        norms = torch.stack(torch._foreach_norm(gradients, 2))
        maxima = torch.stack(torch._foreach_norm(gradients, float("inf")))
        nonzero = torch.stack([g.count_nonzero() for g in gradients]).sum()
        nonfinite = torch.stack([(~torch.isfinite(g)).count_nonzero() for g in gradients]).sum()
        # A single grouped host observation for this domain's summary.
        observed = torch.stack((torch.linalg.vector_norm(norms).double(), maxima.max().double(),
                                nonzero.double(), nonfinite.double())).tolist()
        report[name] = {**base, "norm": observed[0], "max_abs": observed[1],
                        "zero_elements": elements - int(observed[2]), "nonfinite_elements": int(observed[3])}
    report["joint_norm"] = math.hypot(report["main"]["norm"], report["state_head"]["norm"])
    report["main_scope"] = "native_model_and_unchanged_J1_SupCon_projection"
    return report
