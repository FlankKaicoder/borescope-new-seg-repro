"""Select reproducible A and B proposals without promoting human or geometry status."""
from collections import Counter, defaultdict

from tools.exp13_cloud.g3.verify_train_export import CLASSES

CANDIDATE = "CANDIDATE_ONLY_NOT_TRAINING_APPROVED"
GEOMETRY_PENDING = "PAIR_GEOMETRY_PENDING_GATE"
LABEL_HOLD = "A_LABEL_PLAN_HOLD"
SEMANTIC = "HUMAN_SEMANTIC_CANDIDATE"
ALIGNMENT_PENDING = "FINAL_ALIGNMENT_UNRESOLVED"

REASONS = {
    "缺陷残留": "RESIDUAL_DEFECT", "缺材未补全": "MATERIAL_MISSING_NOT_RESTORED",
    "机械结构错误": "MECHANICAL_STRUCTURE_ERROR", "新增异常内容": "NEW_VISUAL_ARTIFACT",
    "配准错位": "REGISTRATION_UNRELIABLE", "黑边截断": "VISIBLE_EDGE_TRUNCATION",
    "相机 UI 变化": "CAMERA_UI_CHANGED", "其他": "HUMAN_REVIEW_HOLD_OR_FAILURE",
}


def mention_risks(item, pid=None):
    """Flag evidence for human adjudication without replacing the CSV decision."""
    ai = item.get("ai", {})
    human = item["human"]
    notes = human["notes"] or ai.get("ai_notes", "")
    risks = set()
    for detail, terms in (
        ("VISIBLE_RESIDUAL_OR_GEOMETRY_EVIDENCE", ("仍有原有", "仍可见", "仍明显", "仍残留", "未修平", "黑边")),
        ("MECHANICAL_STRUCTURE_OR_OCCLUSION", ("填黑", "机械结构错误", "真实结构", "遮挡", "正常件", "待确认开口")),
    ):
        # Record negative notes as conflict evidence only when the same AI region
        # disagrees or the current human text explicitly asks for another check.
        explicit_risk = any(term in human["notes"] for term in ("仍有原有", "仍待", "仍需", "整图建议保留人工核对", "填黑"))
        ai_disagrees = ai.get("ai_polygon_decisions", {}).get(pid) in {"HOLD", "RETRY_REPAIR", "RETRY_ALIGNMENT"} if pid else (
            ai.get("ai_decision") in {"HOLD", "RETRY_REPAIR", "RETRY_ALIGNMENT"})
        if (explicit_risk or ai_disagrees) and any(term in notes for term in terms):
            risks.add(detail)
    if pid is None and any(term in human["notes"] for term in ("原有小划痕", "原有面上划痕", "原有划痕", "其它粗糙点", "其他粗糙点")):
        risks.add("RESIDUAL_MARKS_NEED_SEVEN_CLASS_REVIEW")
    if pid is None and any(term in human["notes"] for term in ("整图建议保留人工核对", "整图需复核", "整图改", "整体需核")):
        risks.add("WHOLE_IMAGE_REVIEW_REQUESTED_IN_NOTES")
    if pid is None and human["reason"] in {"机械结构错误", "新增异常内容", "相机 UI 变化"}:
        risks.add(REASONS[human["reason"]])
    if human["reason"] in {"机械结构错误", "新增异常内容"} and pid is not None:
        risks.add(REASONS[human["reason"]])
    if ai.get("ai_reason") == "机械结构错误" and (pid is None or ai.get("ai_polygon_decisions", {}).get(pid) in {"HOLD", "RETRY_REPAIR"}):
        risks.add("MECHANICAL_STRUCTURE_CONFLICT")
    return risks


def vref(item):
    record = item["record"]
    return {k: record[k] for k in ("variant_id", "version", "review_id")}


def review_risks(item, pid=None):
    ai = item.get("ai", {})
    risks = mention_risks(item, pid)
    if pid is None and ai.get("ai_decision") in {"HOLD", "RETRY_ALIGNMENT", "RETRY_REPAIR"}:
        risks.add("HUMAN_AI_WHOLE_DECISION_CONFLICT")
    if pid is not None and ai.get("ai_polygon_decisions", {}).get(pid) in {"HOLD", "RETRY_ALIGNMENT", "RETRY_REPAIR"}:
        risks.add("HUMAN_AI_POLYGON_DECISION_CONFLICT")
    return sorted(risks)


def a_proposals(source, source_info, entries, quarantined):
    all_ids = set(source["all_polygon_ids"])
    selected = set(source["selected_polygon_ids"])
    proposals = []
    for item in entries:
        h, rec = item["human"], item["record"]
        if h["decision"] != "PASS" or h["whole_image_normal"] != "YES":
            continue
        audit = item.get("audit")
        problems = []
        if quarantined or audit is None or "resolved" not in item:
            problems.append("IDENTITY_QUARANTINE")
        elif not rec["aligned_sha256"] or rec["geometry_status"] != "GEOMETRY_CANDIDATE":
            problems.append("ALIGNMENT_UNVERIFIED")
        else:
            if audit["visible_loss"]:
                problems.append("VISIBLE_SOURCE_PIXELS_UNSUPPORTED")
            if audit["invalid_pixels"]:
                problems.append("NONFULL_FRAME_VALID_MASK")
            if any(p["unsupported_pixels"] != 0 for p in audit["support"].values()):
                problems.append("MASK_SUPPORT_INSUFFICIENT")
            if not audit["mapping_ok"]:
                problems.append("STORED_MATRIX_UNAVAILABLE")
        if all_ids != selected:
            problems.append("UNSELECTED_ANNOTATED_DEFECT")
        if set(item.get("resolved", {}).values()) != {"PASS"}:
            problems.append("POLYGON_PASS_INCOMPLETE")
        if source.get("hold"):
            problems.append(source["hold"])
        problems.extend(review_risks(item))
        view_required = item["batch"] == 5 or bool(audit and "bounded.png" in audit["assets"])
        proposals.append({**vref(item), "decision": h["decision"], "whole_image_normal": h["whole_image_normal"],
            "asset_type": "ALIGNED", "asset": audit["assets"].get("aligned.png") if audit else None,
            "bounded_alternative": audit["assets"].get("bounded.png") if audit else None,
            "pixel_asset_state": "NEEDS_HUMAN_ASSET_CONFIRMATION" if view_required else "ALIGNED_REVIEW_ID_BOUND",
            "unselected_polygon_ids": sorted(all_ids - selected), "original_label_counts": dict(Counter(source_info.get("labels", {}).values())),
            "label_plan": LABEL_HOLD if problems else "EMPTY_LABEL_ELIGIBLE_PENDING_ASSET_CONFIRMATION" if view_required
            else "EMPTY_LABEL_ELIGIBLE_PENDING_DATA_GATE",
            "reasons": sorted(set(problems)), "candidate_status": CANDIDATE})
    return proposals


def b_proposals(source, entries, pid, quarantined):
    proposals = []
    for item in entries:
        h = item["human"]
        if item.get("resolved", {}).get(pid) != "PASS" or h["b_region_review"] != "CONFIRMED":
            continue
        audit = item.get("audit")
        support = audit["support"][pid] if audit else None
        risks = review_risks(item, pid)
        problems = []
        if quarantined or audit is None:
            problems.append("IDENTITY_QUARANTINE")
        else:
            if not audit["mapping_ok"]:
                problems.append("STORED_MATRIX_UNAVAILABLE")
            if support["unsupported_pixels"] != 0:
                problems.append("MASK_SUPPORT_INSUFFICIENT")
            if audit["geometry_risks"]:
                risks = sorted(set(risks) | set(audit["geometry_risks"]))
        if source.get("hold"):
            problems.append(source["hold"])
        problems.extend(r for r in risks if r in {"HUMAN_AI_POLYGON_DECISION_CONFLICT", "MECHANICAL_STRUCTURE_CONFLICT",
                                                 "MECHANICAL_STRUCTURE_ERROR", "NEW_VISUAL_ARTIFACT"})
        proposals.append({**vref(item), "semantic_status": SEMANTIC, "human_polygon_state": "PASS",
            "human_b_state": h["b_region_review"], "polygon_state_basis": "EXPLICIT" if pid in item["explicit"] else "OVERALL_DEFAULT",
            "asset_type": "ALIGNED", "aligned": audit["assets"].get("aligned.png") if audit else None,
            "valid_mask": audit["assets"].get("valid.png") if audit else None,
            "geometry": audit["assets"].get("geometry.json") if audit else None,
            "matrix_output_to_source": audit["matrix"] if audit else None, "support": support,
            "geometry_status": "INELIGIBLE" if problems else GEOMETRY_PENDING,
            "pixel_asset_state": "NEEDS_HUMAN_ASSET_CONFIRMATION" if item["batch"] == 5 or
            bool(audit and "bounded.png" in audit["assets"]) else "ALIGNED_REVIEW_ID_BOUND",
            "risk_evidence": risks, "reasons": sorted(set(problems)), "candidate_status": CANDIDATE})
    return proposals


def priority(proposal):
    return (proposal["pixel_asset_state"] != "ALIGNED_REVIEW_ID_BOUND",
            len(proposal.get("risk_evidence", ())), proposal.get("polygon_state_basis") == "OVERALL_DEFAULT",
            proposal["version"], proposal["variant_id"])


def route_stats(manifest, original_labels, route):
    classes, versions = Counter(), Counter()
    for row in manifest:
        main = row["recommended"]
        versions[main["version"]] += 1
        classes.update(set(original_labels[row["source_id"]]) if route == "a" else [row["class"]])
    return {"classes": {k: classes[k] for k in CLASSES}, "versions": {f"V{i}": versions[f"V{i}"] for i in range(1, 5)}}


def exclusion_reasons(item, source):
    human = item["human"]
    reasons = set()
    if human["reason"] in REASONS and human["decision"] != "PASS":
        reasons.add(REASONS[human["reason"]])
    if human["decision"] == "UNREVIEWED":
        reasons.add("HUMAN_UNREVIEWED")
    elif human["decision"] == "HOLD":
        reasons.add("HUMAN_HOLD")
    elif human["decision"] == "RETRY_ALIGNMENT":
        reasons.add("REGISTRATION_UNRELIABLE")
    elif human["decision"] == "RETRY_REPAIR" and not reasons:
        reasons.add("REPAIR_UNRESOLVED")
    if human["decision"] == "HOLD" and "填黑" in human["notes"]:
        reasons.add("MECHANICAL_STRUCTURE_ERROR")
    if source.get("hold"):
        reasons.add(source["hold"])
    return sorted(reasons)


def consolidate(evidence):
    sources, source_info = evidence["sources"], evidence["source_info"]
    prior = evidence["prior_registry"]
    a8_routes = {r["source_id"]: r for r in evidence["a8_routes"]}
    by_source, issues, excluded, queue = defaultdict(list), evidence["issues"], [], []
    quarantined = {r["source_id"] for r in issues}
    for item in evidence["reviews"]:
        by_source[item["record"]["source_id"]].append(item)
    a_candidates, b_candidates, polygon_rows, source_rows = [], [], [], []
    a_audit, b_audit = [], []
    original_labels = {}
    stats = Counter(A_ONLY=0, B_ONLY=0, A_AND_B=0, NEITHER=0)
    for sid, source in sorted(sources.items(), key=lambda pair: int(pair[0])):
        entries = by_source[sid]
        info = source_info.get(sid, {})
        a8_route = a8_routes[sid]
        original_labels[sid] = list(info.get("labels", {}).values())
        a_all = a_proposals(source, info, entries, sid in quarantined)
        a_audit.extend({"source_id": sid, **r} for r in a_all)
        a_pass = sorted((r for r in a_all if not r["reasons"]), key=priority)
        alignment_risks = set(prior[sid]["remaining_alignment_risk"])
        b_selected, pending_b = {}, []
        v4 = next((item for item in entries if item["batch"] == 5), None)
        actual_choice = v4["human"].get("final_version_choice", "") if v4 else ""
        if a_pass and actual_choice != "EXCLUDE":
            main = next((r for r in a_pass if r["variant_id"] == actual_choice), a_pass[0])
            a_candidates.append({"source_id": sid, "recommended": main, "alternates": [r for r in a_pass if r != main],
                "user_final_version_choice": actual_choice, "source_group_id": source["group_id"],
                "original_annotation": source["annotation"], "original_annotation_sha256": source["annotation_sha256"]})
        else:
            main = None
        by_pid = {}
        for pid in source["selected_polygon_ids"]:
            b_all = b_proposals(source, entries, pid, sid in quarantined)
            stats["human_semantic_b_polygons"] += bool(b_all)
            stats["b_polygons_full_valid_support"] += any(p["support"] and p["support"]["unsupported_pixels"] == 0 for p in b_all)
            stats["human_semantic_b_version_pairs"] += len(b_all)
            stats["unusable_b_version_pairs"] += sum(p["geometry_status"] == "INELIGIBLE" for p in b_all)
            b_pass = sorted((p for p in b_all if p["geometry_status"] == GEOMETRY_PENDING), key=priority)
            recommended = b_pass[0] if b_pass and actual_choice != "EXCLUDE" else None
            stats["unusable_b_polygons"] += not recommended
            hist = [{**vref(item), "state": item.get("resolved", {}).get(pid), "explicit": pid in item.get("explicit", {}),
                     "b_region_review": item["human"]["b_region_review"], "batch": item["batch"]} for item in entries]
            if recommended:
                b_selected[pid] = recommended["variant_id"]
                if recommended["pixel_asset_state"] == "NEEDS_HUMAN_ASSET_CONFIRMATION":
                    pending_b.append(pid)
                b_candidates.append({"source_id": sid, "polygon_id": pid, "class": source["polygon_labels"][pid],
                    "recommended": recommended, "alternates": b_pass[1:], "original_path": source["image"],
                    "original_sha256": source["image_sha256"], "annotation_path": source["annotation"],
                    "annotation_sha256": source["annotation_sha256"],
                    "polygon_coordinates": info["shapes"][source["all_polygon_ids"].index(pid)]["points"]})
            b_audit.append({"source_id": sid, "polygon_id": pid, "class": source["polygon_labels"][pid],
                            "semantic_candidates": b_all, "recommended": recommended, "formal_tre": "NOT_ESTABLISHED"})
            item = {"source_id": sid, "polygon_id": pid, "class": source["polygon_labels"][pid], "human_history": hist,
                    "best_b_candidate": recommended["variant_id"] if recommended else "",
                    "other_b_candidates": [p["variant_id"] for p in b_pass[1:]],
                    "geometry_status": GEOMETRY_PENDING if recommended else "UNAVAILABLE",
                    "prior_risk": a8_route["deferred_targets"].get(pid) or
                    (ALIGNMENT_PENDING if pid in alignment_risks else "")}
            by_pid[pid] = item
            polygon_rows.append(item)
        queue_reasons = []
        if v4 and not actual_choice:
            queue_reasons.append("V4_FINAL_VERSION_CHOICE_EMPTY")
        if len(a_pass) > 1:
            queue_reasons.append("MULTIPLE_A_CANDIDATE_VERSIONS")
        if any(r["pixel_asset_state"] == "NEEDS_HUMAN_ASSET_CONFIRMATION" for r in a_pass):
            queue_reasons.append("ALIGNED_BOUNDED_ASSET_CHOICE")
        if pending_b:
            queue_reasons.append("REGIONAL_ASSET_CONFIRMATION")
        if any(r["reasons"] for r in a_all) and (not main or any(r["version"] == "V4" and r["reasons"] for r in a_all)):
            queue_reasons.append(LABEL_HOLD)
        if v4 and v4["human"]["decision"] == "UNREVIEWED":
            queue_reasons.append("V4_HUMAN_UNREVIEWED")
        if v4 and v4["human"]["decision"] == "HOLD":
            queue_reasons.append("V4_HUMAN_HOLD")
        if sid in quarantined:
            queue_reasons.append("IDENTITY_QUARANTINE")
        variant_conflict = len(a_pass) > 1 or any(reason in {
            "HUMAN_AI_WHOLE_DECISION_CONFLICT", "HUMAN_AI_POLYGON_DECISION_CONFLICT", "MECHANICAL_STRUCTURE_CONFLICT"
        } for r in a_all for reason in r["reasons"])
        if variant_conflict:
            queue_reasons.append("FINAL_VERSION_CONFLICT")
        if queue_reasons:
            queue.append({"source_id": sid, "reasons": sorted(set(queue_reasons)),
                "suggested_a_version": main["variant_id"] if main else "",
                "suggested_a_asset": main["asset_type"] if main else "",
                "other_a_options": [r["variant_id"] for r in a_pass if r != main],
                "suggested_b_polygon_versions": b_selected, "b_asset_confirmation_polygon_ids": pending_b,
                "v4_variant": v4["record"]["variant_id"] if v4 else "",
                "v4_human_decision": v4["human"]["decision"] if v4 else "", "current_user_choice": actual_choice,
                "user_selected_version": "", "user_selected_asset_type": "", "user_notes": ""})
        any_human_a = bool(a_all)
        any_human_b = any(e["human"]["b_region_review"] == "CONFIRMED" for e in entries)
        status = "A_AND_B" if main and b_selected else "A_ONLY" if main else "B_ONLY" if b_selected else "NEITHER"
        problems = set()
        for item in entries:
            variant_reasons = exclusion_reasons(item, source)
            if variant_reasons:
                excluded.append({"source_id": sid, "scope": "VARIANT", "variant_id": item["record"]["variant_id"],
                    "review_id": item["record"]["review_id"], "reasons": variant_reasons, "disposition": "PRESERVE_HISTORY"})
            problems.update(variant_reasons)
        if status == "NEITHER":
            problems.add("NO_GATE_REVIEWABLE_A_OR_B_PROPOSAL")
        if any_human_a and main is None:
            problems.add(LABEL_HOLD)
        if alignment_risks:
            problems.add(ALIGNMENT_PENDING)
        if prior[sid]["delivery"]:
            problems.add("HISTORICAL_API_UNDELIVERED")
        if sid in quarantined:
            problems.add("IDENTITY_QUARANTINE")
        if variant_conflict:
            problems.add("FINAL_VERSION_CONFLICT")
        if status == "NEITHER" or problems & {LABEL_HOLD, ALIGNMENT_PENDING, "HISTORICAL_API_UNDELIVERED", "IDENTITY_QUARANTINE", "FINAL_VERSION_CONFLICT"}:
            excluded.append({"source_id": sid, "scope": "SOURCE", "variant_id": "", "review_id": "",
                "reasons": sorted(problems), "disposition": "EXCLUDED_FROM_CANDIDATES" if status == "NEITHER" else "PARTIAL_SCOPE_HOLD"})
        source_rows.append({"source_id": sid, "original_path": source["image"], "original_sha256": source["image_sha256"],
            "annotation_sha256": source["annotation_sha256"], "annotation_path": source["annotation"], "group_id": source["group_id"],
            "classes": sorted(set(original_labels[sid])), "selected_polygon_ids": source["selected_polygon_ids"],
            "all_gt_polygon_ids": source["all_polygon_ids"], "review_ids": [e["record"]["review_id"] for e in entries],
            "variants": [e["record"]["variant_id"] for e in entries], "recommended_a": main["variant_id"] if main else "",
            "recommended_b_polygon_count": len(b_selected), "ab_status": status, "a_label_hold": any_human_a and main is None,
            "b_geometry_risk": bool(b_selected or alignment_risks or (any_human_b and not b_selected)),
            "version_conflict": variant_conflict, "version_selection_pending": bool(queue_reasons),
            "human_a_evidence": any_human_a, "human_b_evidence": any_human_b,
            "risk_reasons": sorted(problems), "final_user_choice": actual_choice, "candidate_status": CANDIDATE})
        stats[status] += 1
        stats["known_human_a_sources"] += any_human_a
        stats["known_human_b_sources"] += any_human_b
        stats["a_label_hold_sources"] += any_human_a and main is None
        stats["a_known_pixel_file_sources"] += bool(main and main["pixel_asset_state"] == "ALIGNED_REVIEW_ID_BOUND")
        stats["a_pending_asset_choice_sources"] += bool(main and main["pixel_asset_state"] == "NEEDS_HUMAN_ASSET_CONFIRMATION")
        stats["version_conflict_sources"] += variant_conflict
        stats["v4_unselected_sources"] += bool(v4 and not actual_choice)
    stats["a_candidate_sources"] = len(a_candidates)
    stats["a_empty_label_plan_eligible"] = stats["a_known_pixel_file_sources"]
    stats["a_empty_label_plan_conditional_on_asset"] = stats["a_pending_asset_choice_sources"]
    stats["b_candidate_polygons"] = len(b_candidates)
    stats["b_candidate_sources"] = len({r["source_id"] for r in b_candidates})
    stats["b_geometry_gate_pending_polygons"] = len(b_candidates)
    stats["b_pending_asset_choice_polygons"] = sum(r["recommended"]["pixel_asset_state"] == "NEEDS_HUMAN_ASSET_CONFIRMATION"
                                                  for r in b_candidates)
    stats["consistent_candidate_sources"] = len(sources) - stats["NEITHER"]
    stats["version_selection_queue"] = len(queue)
    stats["identity_quarantined_sources"] = len(quarantined)
    return {"sources": source_rows, "polygons": polygon_rows, "a_candidates": a_candidates,
            "b_candidates": b_candidates, "a_audit": a_audit, "b_audit": b_audit, "queue": queue,
            "excluded": excluded, "counts": dict(stats),
            "a_coverage": route_stats(a_candidates, original_labels, "a"),
            "b_coverage": route_stats(b_candidates, original_labels, "b")}
