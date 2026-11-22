"""Fixed-weight gradient diagnostics on disposable model/batch copies only."""
import math

import torch

from tools.exp17.engineering.calibration import gradient_basis, gradient_report, norm, parameter_layout
from tools.exp17.engineering.model import forward_joint
from tools.exp17.formal.validation import require


def analyze_gradients(yolo, supcon8, pair, domains, class_components=None):
    """Explain the frozen lambda4 objective, without selecting or changing lambda."""
    require(all(torch.isfinite(v).all().item() for v in (yolo, supcon8, pair)), "Nonfinite isolated gradient")
    joint, reference = yolo + supcon8 + 4 * pair, yolo + supcon8
    coefficient = min(1.0, 10.0 / (norm(joint) + 1e-6))
    old_coefficient = min(1.0, 10.0 / (norm(reference) + 1e-6))
    components = class_components or {}
    denominator = sum(norm(v[domains["backbone"]]) for v in components.values())
    return {"lambda_pair": 4.0, "gradient_domains": gradient_report(yolo, supcon8, pair, domains),
            "joint_all_trainable_norm": norm(joint), "clip_coefficient": coefficient,
            "J1_reference_clip_coefficient": old_coefficient,
            "clip_coefficient_relative_to_J1": coefficient / old_coefficient,
            "clipped_joint_norm": norm(joint) * coefficient,
            "per_class": {str(c): {"backbone_component_norm": norm(v[domains["backbone"]]),
                "backbone_component_norm_share": norm(v[domains["backbone"]]) / denominator if denominator else 0.0}
                for c, v in components.items()},
            "class_share_definition": "component norm / sum of component norms, not norm of sum",
            "optimizer_updates": 0, "dynamic_lambda": False}


def fixed_gradient_diagnostic(model_clone, batch_factory, *, amp, gradient_scale):
    """Call inside validation.run_isolated(allow_grad=True), never on the live model.

    batch_factory is a deterministic frozen CPU-cache -> prepared-batch adapter.
    Native replay and prefix/BN/ROI/loss implementations are imported unchanged.
    """
    require(math.isfinite(gradient_scale) and gradient_scale > 0, "Invalid frozen AMP scale")
    model_clone.train()
    prepared = batch_factory(model_clone)
    names, _, domains = parameter_layout(model_clone)
    parameters = [dict(model_clone.named_parameters())[n] for n in names]
    device_type = next(model_clone.parameters()).device.type
    with torch.autocast(device_type, enabled=amp, dtype=torch.float16 if device_type == "cuda" else torch.bfloat16):
        output = forward_joint(model_clone, prepared.batch, prepared.a_regions, prepared.normal_images,
            prepared.normal_masks, prepared.normal_regions, prepared.bindings,
            bn_mode="replay_a", lambda_pair=4.0, checkpoint_prefix=False)
    y = gradient_basis(output.yolo, parameters, gradient_scale)
    s = gradient_basis(8 * output.supcon, parameters, gradient_scale)
    p = gradient_basis(output.pair, parameters, gradient_scale)
    represented = len(output.per_class_pair_tensors)
    class_components = {c: gradient_basis(loss / represented, parameters, gradient_scale)
                        for c, loss in output.per_class_pair_tensors.items()}
    result = analyze_gradients(y, s, p, domains, class_components)
    result["losses"] = {name: float(getattr(output, name).detach()) for name in ("yolo", "supcon", "pair", "total")}
    result["bn_mode"] = "replay_a"
    result["assumption"] = "DETACHED_WITHIN_STEP_A_STATISTICS_ON_N"
    result["read_only_scope"] = "DISPOSABLE_MODEL_COPY_AND_FROZEN_BATCH_COPY"
    return result
