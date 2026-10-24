import numpy as np

from tools.exp13_restoration.geometry import build_geometry, canvas_transform, group_polygons, inverse_canvas_mask, map_mask_to_canvas, Polygon


def test_non_square_and_inverse_geometry():
    g = build_geometry([(2, 2), (11, 2), (11, 6), (2, 6)], 16, 10, context=3, canvas_size=1024)
    assert g.polygon_mask.any()
    assert g.polygon_mask.sum() == g.edit_core.sum()
    assert g.roi_box.width != g.roi_box.height
    assert g.transform.pad_right >= 0 and g.transform.pad_bottom >= 0
    round_trip = inverse_canvas_mask(map_mask_to_canvas(g.edit_core[g.roi_box.y0:g.roi_box.y1, g.roi_box.x0:g.roi_box.x1], g.transform), g.transform)
    assert round_trip.shape == (g.roi_box.height, g.roi_box.width)
    assert round_trip.any() and round_trip.sum() <= g.edit_core[g.roi_box.y0:g.roi_box.y1, g.roi_box.x0:g.roi_box.x1].size


def test_boundary_and_thin_crack_are_not_dropped():
    g = build_geometry([(0.1, 4.1), (14.9, 4.2), (14.9, 4.8), (0.1, 4.7)], 15, 9, context=4)
    assert g.polygon_mask.sum() > 0
    assert g.provenance["roi_clamped"]
    assert g.provenance["fov_status"] == "FOV_UNCERTAIN"


def test_gray_border_is_uncertain_not_edit_support():
    g = build_geometry([(3, 3), (7, 3), (7, 7), (3, 7)], 10, 10, gray_border_width=1)
    assert g.fov_uncertain.any()
    assert not np.any(g.edit_support & g.fov_uncertain)


def test_group_order_independent():
    p = [Polygon("b", ((4, 4), (7, 4), (7, 7), (4, 7)), source_id="s"), Polygon("a", ((1, 1), (5, 1), (5, 5), (1, 5)), source_id="s"), Polygon("c", ((20, 20), (22, 20), (22, 22), (20, 22)), source_id="s")]
    assert [[x.roi_id for x in group] for group in group_polygons(p)] == [["a", "b"], ["c"]]
    assert [[x.roi_id for x in group] for group in group_polygons(list(reversed(p)))] == [["a", "b"], ["c"]]
