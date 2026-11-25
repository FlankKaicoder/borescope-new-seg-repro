"""CPU data boundary tests; only synthetic tensors and newly created artifacts."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import json
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid

from tools.exp18.data import epoch_seed, project_reference_rows


def attempt_directory():
    root = Path(os.environ.get("EXP18_TEST_OUTPUT", Path(__file__).resolve().parents[3] / "private_assets/exp18_tests"))
    path = root / ("data_" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    return path


class EpochSeedTests(unittest.TestCase):
    def test_first_100_epochs_preserve_original_seed_identity(self):
        for epoch in range(100):
            old = int.from_bytes(hashlib.sha256(f"exp14-a11-full:v1:42:{epoch}".encode()).digest()[:8], "little")
            self.assertEqual(epoch_seed(42, epoch), old)

    def test_new_horizon_seed_domain_and_bad_inputs(self):
        values = [epoch_seed(42, epoch) for epoch in range(200)]
        self.assertEqual(len(values), len(set(values)))
        for args in ((42, 200), (42, -1), (True, 1), (42, True)):
            with self.assertRaises(ValueError):
                epoch_seed(*args)
        with self.assertRaises(ValueError):
            epoch_seed(42, 100, horizon=100)

    def test_reference_projection_preserves_every_compared_value(self):
        native = {"step": 1, "epoch": 1, "images": 32, "raw_targets": 62,
                  "augmented_targets": 62, "ordered_sources_sha256": "a" * 64,
                  "augmented_tensors_sha256": "b" * 64, "lr_bias": .1}
        traces = ["c" * 64, "d" * 64]
        complete = [{"native": native, "traces": traces, "coverage": {"unused": list(range(100))}}]
        result = project_reference_rows(complete)
        self.assertEqual(result, [{"native": native, "traces": traces}])
        self.assertEqual(set(result[0]), {"native", "traces"})
        self.assertIn("coverage", complete[0])
        for value in ({"native": {}, "traces": None}, {"native": [], "traces": []}, {}):
            with self.assertRaises(ValueError):
                project_reference_rows([value])


class PermanentGenerationTests(unittest.TestCase):
    def test_frozen_old_cycles_are_protected_but_new_batch_cycles_collect(self):
        # Exercise the actual interpreter mechanism in a tiny isolated process;
        # never freeze the test runner, its guards, or its later fixtures.
        code = r'''
import gc,json,weakref
class Node: pass
gc.collect()
old=Node();old.link=old;old_ref=weakref.ref(old)
gc.freeze();frozen_count=gc.get_freeze_count()
del old;gc.collect()
old_protected=old_ref() is not None
new=Node();new.link=new;new_ref=weakref.ref(new)
del new;gc.collect()
new_collected=new_ref() is None
gc.unfreeze();gc.collect()
print(json.dumps({'frozen_count':frozen_count,'old_protected':old_protected,
                  'new_collected':new_collected,'old_collected_after_unfreeze':old_ref() is None}))
'''
        result = json.loads(subprocess.check_output([sys.executable, "-B", "-c", code], text=True))
        self.assertGreater(result["frozen_count"], 0)
        self.assertTrue(result["old_protected"])
        self.assertTrue(result["new_collected"])
        self.assertTrue(result["old_collected_after_unfreeze"])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is required for native cache tests")
class CacheTests(unittest.TestCase):
    def test_cache_is_exclusive_and_native_dictionary_keeps_version(self):
        import numpy as np
        from tools.exp18.data import exclusive_label_cache
        directory = attempt_directory()
        target = directory / "labels.cache"
        cache = {"labels": [], "hash": "synthetic"}
        exclusive_label_cache("", target, cache, "v1")
        original = target.read_bytes()
        self.assertEqual(cache["version"], "v1")
        with target.open("rb") as stream:
            self.assertEqual(np.load(stream, allow_pickle=True).item()["version"], "v1")
        with self.assertRaises(FileExistsError):
            exclusive_label_cache("", target, {"labels": ["changed"]}, "v2")
        self.assertEqual(target.read_bytes(), original)

    def test_failed_cache_write_preserves_partial_evidence(self):
        from tools.exp18.data import exclusive_label_cache
        target = attempt_directory() / "interrupted.cache"
        def interrupted(stream, value):
            stream.write(b"partial")
            raise OSError("synthetic disk failure")
        with patch("numpy.save", side_effect=interrupted):
            with self.assertRaises(OSError):
                exclusive_label_cache("", target, {}, "v1")
        self.assertEqual(target.read_bytes(), b"partial")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Use the existing server Torch CPU environment")
class TensorTransportTests(unittest.TestCase):
    def test_roi_only_epoch_view_preserves_regions_and_closed_matching(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from tools.exp15.regions import inspect_regions
        from tools.exp18.data import retain_native_roi_view, check_retained_roi_view, tensor_summary
        from tools.exp18.geometry import array_sha256
        from tools.exp18.pairing import CandidateExposure, MatchingConfig, build_matching
        directory = attempt_directory()
        names = []
        for name in ("source-a.meta", "source-b.meta"):
            path = directory / name
            with path.open("x") as stream:
                stream.write("SYNTHETIC_SOURCE_ID_ONLY")
            names.append(str(path.resolve()))
        groups = dict(zip(names, ("group-a", "group-b"), strict=True))
        masks = torch.zeros((2, 4, 4), dtype=torch.float32)
        masks[:, 1, 1] = 1
        batch = {"img": torch.arange(2 * 3 * 16 * 16, dtype=torch.int32).to(torch.uint8).reshape(2, 3, 16, 16),
                 "cls": torch.tensor([[0.], [0.]]), "bboxes": torch.tensor([[.5, .5, .5, .5]] * 2),
                 "masks": masks, "sem_masks": masks.clone(), "batch_idx": torch.tensor([0., 1.]),
                 "im_file": names, "j2_replay": (SimpleNamespace(sha256="a" * 64), SimpleNamespace(sha256="b" * 64))}
        original_regions = inspect_regions(batch, groups)
        view = retain_native_roi_view(batch, groups)
        self.assertNotIn("img", view.batch)
        self.assertNotIn("sem_masks", view.batch)
        self.assertEqual(view.six_tensors["img"], tensor_summary(batch["img"]))
        self.assertEqual(view.a_image_summaries[0], tensor_summary(batch["img"][0]))
        check_retained_roi_view(view, groups)
        def candidates(regions):
            return [CandidateExposure(a_row=row, a_slot=int(slot), source_id=f"source-{slot}",
                        group_id=str(regions.groups[row]), polygon_id=f"source-{slot}_000",
                        class_id=int(regions.classes[row]), repair_version="V1", n_sha256="c" * 64,
                        source_area_fraction=.1, p3_cells=int(regions.cells[row]),
                        augmentation_sha256=batch["j2_replay"][int(slot)].sha256,
                        mask_sha256=array_sha256(batch["masks"][int(slot)].numpy() == 1))
                    for row, slot in enumerate(regions.slots)]
        old = build_matching(candidates(original_regions), MatchingConfig(), epoch=0, batch_index=0)
        new = build_matching(candidates(view.regions), MatchingConfig(), epoch=0, batch_index=0)
        self.assertEqual(old.to_dict(), new.to_dict())
        self.assertEqual(len(new.a_rows), 2)
        view.batch["masks"][0, 1, 1] = 0
        with self.assertRaises(ValueError):
            check_retained_roi_view(view, groups)

    def test_native_tensor_transport_preserves_existing_storage(self):
        import torch
        from tools.exp18.data import SIX_TENSORS, restore_native_batch, native_batch_digest
        batch = {name: torch.arange(6, dtype=torch.int32).reshape(2, 3) for name in SIX_TENSORS}
        result = restore_native_batch(batch)
        self.assertEqual(native_batch_digest(batch), native_batch_digest(result))
        for name in SIX_TENSORS:
            self.assertIs(result[name], batch[name])
            self.assertEqual(result[name].data_ptr(), batch[name].data_ptr())

    def test_numpy_transport_preserves_six_native_bytes_and_metadata(self):
        import torch
        from tools.exp18.data import SIX_TENSORS, _numpy_transport, restore_native_batch, native_batch_digest
        batch = {name: torch.arange(6, dtype=torch.int32).reshape(2, 3) for name in SIX_TENSORS}
        batch["metadata"] = ("TRAIN", {"source": "synthetic"})
        before = native_batch_digest(batch)
        result = restore_native_batch(_numpy_transport(batch))
        self.assertEqual(before, native_batch_digest(result))
        self.assertEqual(result["metadata"], batch["metadata"])
        self.assertTrue(all(value.device.type == "cpu" for name, value in result.items() if name in SIX_TENSORS))

    def test_cpu_rng_snapshot_never_fetches_cuda_rng(self):
        import torch
        from tools.exp18.data import cpu_rng_snapshot, cpu_rng_equal
        with patch.object(torch.cuda, "get_rng_state_all", side_effect=AssertionError("CUDA RNG access")), \
             patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA availability probe")):
            first = cpu_rng_snapshot()
            self.assertTrue(cpu_rng_equal(first, cpu_rng_snapshot()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_checkpoint_is_rejected_before_any_deserialization(self):
        import torch
        from tools.exp18.data import load_fixed_cpu_packet
        with patch.object(torch, "load", side_effect=AssertionError("Must not deserialize")):
            with self.assertRaises(ValueError):
                load_fixed_cpu_packet({"batch_position": 2, "path": "real_model.pt", "sha256": "0" * 64})

    def test_memory_release_does_not_consume_native_cpu_rng(self):
        from tools.exp18.data import release_cpu_memory, cpu_rng_snapshot, cpu_rng_equal
        before = cpu_rng_snapshot()
        report = release_cpu_memory()
        self.assertTrue(cpu_rng_equal(before, cpu_rng_snapshot()))
        self.assertIn("malloc_trim_supported", report)


if __name__ == "__main__":
    unittest.main()
