"""Atomic serialization and identity failures in tiny CPU synthetic states."""
import copy
import tempfile
import unittest
from pathlib import Path

from tools.exp17.formal.checkpoint import (AtomicCheckpointStore,capture_training_state,
    restore_training_state,state_hash)
from tools.exp17.formal.tests.test_recovery import BINDINGS,seed_all,toy_trainer,step,synthetic_epoch


def payload(update=0):
    seed_all();t=toy_trainer()
    for number in range(update):step(t,synthetic_epoch(0)[number],number+1)
    return capture_training_state(t,bindings=BINDINGS,position={'epoch':0,'batch_position':update,'global_update':update},
                                  sampler_state={'synthetic':True,'next_batch':update})


class CheckpointTests(unittest.TestCase):
    def test_all_precommit_failure_points_preserve_prior_valid_last(self):
        for stage in ('after_fsync','before_blob_rename','after_blob_rename','before_pointer_commit'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as temporary:
                store=AtomicCheckpointStore(temporary,BINDINGS);original=payload();receipt=store.save_latest(original)
                def fail(at):
                    if at==stage:raise OSError('injected interrupted save')
                failed=AtomicCheckpointStore(temporary,BINDINGS,failpoint=fail)
                with self.assertRaisesRegex(OSError,'interrupted'):failed.save_latest(payload(1))
                value,latest=store.load_latest(expected_sha256=receipt['sha256'])
                self.assertEqual(state_hash(value),state_hash(original));self.assertEqual(latest,receipt)

    def test_after_pointer_failure_is_committed_new_state_not_replayed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=AtomicCheckpointStore(temporary,BINDINGS);store.save_latest(payload())
            def fail(at):
                if at=='after_pointer_commit':raise OSError('postcommit interruption')
            with self.assertRaises(OSError):AtomicCheckpointStore(temporary,BINDINGS,failpoint=fail).save_latest(payload(1))
            value,_=store.load_latest();self.assertEqual(value['position']['global_update'],1)
            with self.assertRaisesRegex(ValueError,'skipped or repeated'):store.save_latest(payload(1))

    def test_corruption_wrong_sha_manifest_code_and_protocol_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=AtomicCheckpointStore(temporary,BINDINGS);receipt=store.save_latest(payload())
            with self.assertRaisesRegex(ValueError,'conflicts'):store.load_latest(expected_sha256='0'*64)
            for key in ('protocol_sha256','manifest_sha256','code_sha256','TRAIN_manifest_sha256','coco_sha256'):
                with self.subTest(key=key):
                    changed={**BINDINGS,key:'0'*64}
                    with self.assertRaisesRegex(ValueError,'binding'):AtomicCheckpointStore(temporary,changed).load_latest()
            p=Path(temporary)/receipt['file']
            with p.open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'corruption'):store.load_latest()

    def test_best_periodic_final_independent_immutable_and_master_ema_not_mixed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=AtomicCheckpointStore(temporary,BINDINGS);state=payload(1);last=store.save_latest(state)
            self.assertNotEqual(state['master_state_sha256'],state['ema_state_sha256'])
            for role in ('best','periodic','final','ema','epoch_last'):
                proof=store.save_artifact(role+'.pt',state,role=role)
                loaded,_=store.load_artifact(role+'.pt',expected_sha256=proof['sha256'])
                self.assertEqual(state_hash(loaded),state_hash(state))
                with self.assertRaisesRegex(ValueError,'already exists'):store.save_artifact(role+'.pt',state,role=role)
            self.assertEqual(store.load_latest()[1],last)

    def test_update_skip_same_update_state_change_and_unsafe_names_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=AtomicCheckpointStore(temporary,BINDINGS);state=payload();store.save_latest(state)
            with self.assertRaisesRegex(ValueError,'skipped or repeated'):store.save_latest(payload(2))
            changed=copy.deepcopy(state);changed['optimizer']['param_groups'][0]['lr']*=2
            with self.assertRaisesRegex(ValueError,'training state'):store.save_latest(changed,allow_same_update=True)
            for name in ('../outside.pt','nested/path.pt','..pt','/absolute.pt'):
                with self.assertRaises(ValueError):store.save_artifact(name,state,role='final')

    def test_corrupt_counters_and_fp16_master_rejected_before_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            state=payload();state['audit']['effective_updates']=3
            with self.assertRaisesRegex(ValueError,'counters'):restore_training_state(toy_trainer(),state,bindings=BINDINGS)
            state=payload();state['state']={k:v.half() if v.is_floating_point() else v for k,v in state['state'].items()}
            state['master_state_sha256']=state_hash(state['state'])
            with self.assertRaisesRegex(ValueError,'FP32'):restore_training_state(toy_trainer(),state,bindings=BINDINGS)

    def test_adam_state_and_bn_fingerprint_corruption_rejected(self):
        state=payload(1);next(iter(state['optimizer']['state'].values()))['step']+=1
        with self.assertRaisesRegex(ValueError,'AdamW step'):restore_training_state(toy_trainer(),state,bindings=BINDINGS)
        state=payload();state['bn_state_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'BN identity'):restore_training_state(toy_trainer(),state,bindings=BINDINGS)



if __name__=='__main__':unittest.main()
