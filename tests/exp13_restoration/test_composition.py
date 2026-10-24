import numpy as np

from tools.exp13_restoration.roi_composition import CompositionItem, RGBROIEdit, compose_support, compose_full_image
from tools.exp13_restoration.protocol import BudgetProtocol


def _items():
    return [
        CompositionItem("z", "source", "Burn", ((2, 2), (8, 2), (8, 8), (2, 8))),
        CompositionItem("a", "source", "Crack", ((5, 5), (12, 5), (12, 12), (5, 12))),
        CompositionItem("m", "source", "Dent", ((20, 20), (23, 20), (23, 23), (20, 23))),
    ]


def test_composition_is_order_independent_and_preserves_background():
    a = compose_support(_items(), 32, 32)
    b = compose_support(list(reversed(_items())), 32, 32)
    assert np.array_equal(a.owner_map, b.owner_map)
    assert np.array_equal(a.edit_support, b.edit_support)
    assert a.provenance["composition_status"] == "PASS"
    assert a.provenance["overlap_pixels"] > 0
    assert not a.edit_support[31, 31]


def test_hold_policy_does_not_turn_risk_into_pass():
    result = compose_support(_items()[:2], 32, 32, unresolved_overlap_policy="hold_on_overlap")
    assert result.hold
    assert result.provenance["composition_status"] == "ROI_COMPOSITION_HOLD"


def test_risk_flag_is_auditable():
    result = compose_support([CompositionItem("a", "s", "Burn", ((1, 1), (4, 1), (4, 4), (1, 4)), ("FOV_UNCERTAIN",))], 8, 8)
    assert result.provenance["risk_sources"] == ["s"]


def test_budget_caps_block_execution():
    budget = BudgetProtocol()
    budget.reserve(120, kind="pilot_planned")
    try:
        budget.reserve(1, kind="revision_reserve")
    except RuntimeError:
        pass
    else:
        raise AssertionError("smoke/pilot/revision cap did not block")


def test_full_rgb_composition_preserves_background_and_shape():
    native = np.zeros((12, 16, 3), dtype=np.uint8)
    native[:, :, :] = [3, 4, 5]
    polygon = ((2, 2), (8, 2), (8, 7), (2, 7))
    edit = RGBROIEdit("r", "source", polygon, native[2:8, 2:9].copy())
    result = compose_full_image(native, "source", [edit], context=0, halo_radius=0)
    assert result.image.shape == native.shape and result.image.dtype == native.dtype
    assert np.array_equal(result.image[~result.edit_support], native[~result.edit_support])
    assert result.provenance["background_immutable"]
