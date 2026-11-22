"""Conservative retained-byte budget; no deletion or model execution."""
import math


def budget(available_bytes, *, measured_checkpoint_bytes, measured_full_batch_cache_bytes):
    # Bound a recovery snapshot more generously than the measured smoke file;
    # Best needs master+EMA, not AdamW. Both roles retain all selection winners.
    recovery = max(60 * 1024**2, math.ceil(measured_checkpoint_bytes * 1.25))
    best = 32 * 1024**2
    epoch_cache = math.ceil(measured_full_batch_cache_bytes * 21 * 1.25)
    entries = {"periodic_recovery_20": 20*recovery, "active_recovery_two_slots": 2*recovery,
               "best_master_and_ema_four_atomic_slots": 4*best,
               "two_retained_epoch_caches": 2*epoch_cache,
               "new_epoch_cache_before_retiring_oldest": epoch_cache,
               "final_native_master_and_EMA": 2*16*1024**2,
               "logs_VAL_gradient_receipts": 640*1024**2,
               "atomic_checkpoint_orphan_and_temp": 2*recovery,
               "free_reserve": 1024**3}
    total = sum(entries.values())
    return {"schema":"exp17_g1_storage_budget_v1", "available_bytes":available_bytes,
            "measured_checkpoint_bytes":measured_checkpoint_bytes,
            "measured_full_batch_cache_bytes":measured_full_batch_cache_bytes,
            "estimated_max_recovery_bytes":recovery,"estimated_max_epoch_cache_bytes":epoch_cache,
            "categories":entries,"required_bytes":total,"remaining_margin_bytes":available_bytes-total,
            "status":"PASS_ESTIMATED_CAPACITY" if available_bytes>=total else "HOLD_CAPACITY_REQUIRES_NEW_GATE",
            "retention":"Latest per update uses two active atomic slots; latest at every epoch, 20 immutable periodic archives; current+previous epoch caches with third-cache construction peak; Best master/EMA each use two active atomic slots and immutable lightweight selection history. Retirement of only this new run's ephemeral cache requires the explicit future Gate.",
            "scope":"ESTIMATE_FROM_EXISTING_FILE_SIZES_ONLY_NO_NEW_MODEL_OR_CACHE_BUILD",
            "runtime_rule":"Recheck before cache generation/checkpoint writes; insufficient space hard-stops without reducing scientific settings or deleting history",
            "historical_cleanup_authorized":False,"formal_training_authorized":False}
