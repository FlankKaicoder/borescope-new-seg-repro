"""CPU synthetic protocol tests: never open research images or initialize CUDA."""
import copy
import random
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tools.exp17.formal.checkpoint import (AtomicCheckpointStore, capture_training_state,
    restore_training_state, sha256_file, state_hash)
from tools.exp17.formal.rng import capture_rng, restore_rng, rng_equal
from tools.exp17.formal.sampler import SealedEpochCache

BINDINGS = {"protocol_sha256": "p" * 64, "manifest_sha256": "m" * 64,
            "code_sha256": "c" * 64, "coco_sha256": "w" * 64,
            "TRAIN_manifest_sha256": "t" * 64, "runtime": {"device": "CPU_SYNTHETIC_ONLY"}}


class SyntheticWorkerDataset(Dataset):
    def __len__(self): return 68
    def __getitem__(self, index):
        # Real worker-local Python / NumPy / Torch draws, no model or image IO.
        values = torch.tensor([random.random(), float(np.random.random()), float(torch.rand(()))])
        return {"idx": index, "features": values, "pair_binding": f"synthetic-class{index%2}-group{index//2}"}


def synthetic_seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def seed_all():
    random.seed(10); np.random.seed(11); torch.manual_seed(12)


def toy_trainer():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4),
                                torch.nn.ReLU(), torch.nn.Dropout(.1), torch.nn.Linear(4, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 1.0 - epoch / 100 * .99)
    return SimpleNamespace(model=model, optimizer=opt, scheduler=scheduler,
        scaler=torch.amp.GradScaler("cpu", init_scale=8, growth_interval=4),
        ema=SimpleNamespace(ema=copy.deepcopy(model).eval(), updates=0),
        audit=SimpleNamespace(optimizer_attempts=0, native_steps=0, effective_updates=0, amp_skips=0), counters=Counter())


def synthetic_epoch(epoch):
    # Stateless immutable input creation; it cannot advance main training RNG.
    g = torch.Generator().manual_seed(3000 + epoch)
    order = torch.randperm(668, generator=g)
    result = []
    for batch in range(21):
        idx = order[batch*32:(batch+1)*32]
        result.append({"idx": idx, "features": torch.randn(len(idx),3,generator=g),
            "masks": idx % 2, "trace": {"epoch":epoch, "batch":batch,"actual_affine":torch.eye(3)},
            "pair_binding": [f"synthetic-{epoch}-{int(i)}" for i in idx]})
    return result


def make_caches(root, epochs=4):
    return [SealedEpochCache.build(root / f"epoch_{epoch:03d}", synthetic_epoch(epoch), epoch=epoch,
        bindings=BINDINGS, batch_fingerprint=lambda n,b:{"index":n,"identity":state_hash(b)},expected_batches=21) for epoch in range(epochs)]


def step(trainer, batch, global_update):
    t = trainer
    t.audit.optimizer_attempts += 1
    # Uses all main RNG streams, exposing an incorrect restoration immediately.
    noise = random.random() + float(np.random.random()) + float(torch.rand(()))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = t.model(batch["features"])
        loss = (output.float() - batch["masks"].float().unsqueeze(1) + noise*.001).square().mean()
    t.scaler.scale(loss).backward(); t.scaler.unscale_(t.optimizer)
    torch.nn.utils.clip_grad_norm_(t.model.parameters(), 10)
    t.scaler.step(t.optimizer); t.scaler.update(); t.optimizer.zero_grad(set_to_none=True)
    t.audit.native_steps += 1; t.audit.effective_updates += 1
    t.ema.updates += 1
    with torch.no_grad():
        for n, value in t.ema.ema.state_dict().items():
            source=t.model.state_dict()[n]
            if value.is_floating_point(): value.mul_(.9).add_(source,alpha=.1)
            else: value.copy_(source)
    t.counters.update(updates=1, examples=len(batch["idx"]))
    return float(loss.detach())


def run(trainer, caches, *, start=None, interrupt=None, store=None):
    epoch = 0 if start is None else start[0]["epoch"]
    batch_pos = 0 if start is None else start[0]["batch_position"]
    update = 0 if start is None else start[0]["global_update"]
    losses, identities = [], []
    for e in range(epoch,len(caches)):
        cursor=caches[e].cursor(batch_pos if e==epoch else 0)
        if start is not None and e==epoch:cursor.load_state_dict(start[1])
        if cursor.next_index==0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore");trainer.scheduler.step()
        while cursor.next_index<21:
            batch=cursor.next_batch(); identities.append(state_hash(batch))
            # Synthetic warmup exercises resume inside and past all 63 positions.
            if update<63:
                trainer.optimizer.param_groups[0]["lr"]=(update+1)/63*.001
            update+=1; losses.append(step(trainer,batch,update))
            position={"epoch":e,"batch_position":cursor.next_index,"global_update":update}
            if update==interrupt:
                payload=capture_training_state(trainer,bindings=BINDINGS,position=position,sampler_state=cursor.state_dict(),extra={"binding":state_hash(batch)})
                receipt=store.save_latest(payload)
                return losses,identities,payload,receipt
        batch_pos=0;start=None
    position={"epoch":len(caches)-1,"batch_position":21,"global_update":update}
    return losses,identities,capture_training_state(trainer,bindings=BINDINGS,position=position,sampler_state=cursor.state_dict()),None


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_rng_snapshot_restores_python_numpy_torch_without_cuda(self):
        seed_all();s=capture_rng()
        expected=(random.random(),np.random.random(),torch.rand(4))
        restore_rng(s);actual=(random.random(),np.random.random(),torch.rand(4))
        self.assertEqual(expected[0],actual[0]);self.assertEqual(expected[1],actual[1]);self.assertTrue(torch.equal(expected[2],actual[2]))
        self.assertFalse(s["cuda_initialized"])

    def test_two_worker_native_style_stream_is_sealed_once_and_main_rng_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            def loader():return DataLoader(SyntheticWorkerDataset(),batch_size=32,shuffle=True,num_workers=2,
                generator=torch.Generator().manual_seed(2394),worker_init_fn=synthetic_seed_worker,multiprocessing_context="spawn")
            seed_all();before=capture_rng()
            direct=list(loader());self.assertTrue(rng_equal(before,capture_rng()))
            cache=SealedEpochCache.build(Path(temporary)/"cache",loader(),epoch=0,bindings=BINDINGS,
                batch_fingerprint=lambda n,b:state_hash(b),expected_batches=3)
            self.assertTrue(rng_equal(before,capture_rng()))
            cursor=cache.cursor();first=cursor.next_batch();saved=cursor.state_dict()
            resumed=cache.cursor();resumed.load_state_dict(saved)
            self.assertEqual(state_hash(first),state_hash(direct[0]))
            self.assertEqual([state_hash(b) for b in resumed],[state_hash(b) for b in direct[1:]])

    def test_mid_epoch_warmup_epoch_boundary_ema_bn_amp_exact_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);caches=make_caches(root/"cache")
            seed_all();reference=toy_trainer();losses,identities,expected,_=run(reference,caches)
            for boundary in (1,21,51,63,64):
                with self.subTest(boundary=boundary):
                    seed_all();original=toy_trainer();store=AtomicCheckpointStore(root/f"recovery_{boundary}",BINDINGS)
                    l0,i0,_,receipt=run(original,caches,interrupt=boundary,store=store)
                    loaded,proof=store.load_latest(expected_sha256=receipt['sha256'])
                    resumed=toy_trainer() # Deliberately consumes RNG before exact restoration.
                    position,sampler,extra=restore_training_state(resumed,loaded,bindings=BINDINGS)
                    l1,i1,actual,_=run(resumed,caches,start=(position,sampler))
                    self.assertEqual(l0+l1,losses);self.assertEqual(i0+i1,identities)
                    self.assertEqual(state_hash(actual),state_hash(expected))
                    self.assertEqual(resumed.audit.effective_updates,84);self.assertEqual(resumed.ema.updates,84)
                    self.assertEqual(sum(m.num_batches_tracked.item() for m in resumed.model.modules() if isinstance(m,torch.nn.BatchNorm1d)),84)

    def test_sampler_rejects_manifest_cursor_and_pair_bytes_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache=make_caches(Path(temporary),epochs=1)[0];cursor=cache.cursor();cursor.next_batch();state=cursor.state_dict()
            broken={**state,"cache_manifest_sha256":"0"*64}
            with self.assertRaisesRegex(ValueError,'binding'):cursor.load_state_dict(broken)
            path=cache.directory/cache.manifest['batches'][1]['file']
            with path.open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'corruption'):cursor.next_batch()

    def test_retirement_only_exact_owned_old_synthetic_cache_after_explicit_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            caches=[SealedEpochCache.build(root/f'epoch_{epoch:03d}',synthetic_epoch(epoch)[:1],epoch=epoch,
                bindings=BINDINGS,batch_fingerprint=lambda n,b:state_hash(b),expected_batches=1) for epoch in range(3)]
            old=caches[0];latest=caches[2].cursor().state_dict()
            with self.assertRaisesRegex(ValueError,'not explicitly authorized'):
                old.retire(owned_cache_root=root,committed_sampler_state=latest)
            with self.assertRaisesRegex(ValueError,'still needs'):
                old.retire(owned_cache_root=root,committed_sampler_state=caches[1].cursor().state_dict(),retention_authorized=True)
            unexpected=old.directory/'UNIQUE_ASSET_KEEP.txt';unexpected.write_text('synthetic-only unexpected file')
            with self.assertRaisesRegex(ValueError,'Unexpected'):
                old.retire(owned_cache_root=root,committed_sampler_state=latest,retention_authorized=True)
            self.assertTrue((old.directory/'batch_000.pt').exists())
            unexpected.unlink() # Remove only this test's explicitly created sentinel.
            old.retire(owned_cache_root=root,committed_sampler_state=latest,retention_authorized=True)
            self.assertFalse(old.directory.exists())
            self.assertTrue((caches[1].directory/'manifest.json').exists());self.assertTrue((caches[2].directory/'manifest.json').exists())


if __name__=='__main__':unittest.main()
