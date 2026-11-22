"""Same executor core, exclusively tiny CPU synthetic inputs and metric fixtures.

The 2,100 optimizer calls here belong to a small synthetic Linear/BN network,
not YOLO or J2. Scheduled metrics are fixture dictionaries: there is no VAL
image read, VAL model forward, CUDA initialization, or research training.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch

from tools.exp17.formal.authorization import HardGate, RunJournal
from tools.exp17.formal.checkpoint import AtomicCheckpointStore, sha256_file, state_hash
from tools.exp17.formal.executor import run_session
from tools.exp17.formal.rng import capture_rng, rng_equal
from tools.exp17.formal.sampler import SealedEpochCache
from tools.exp17.formal.tests.test_recovery import BINDINGS, seed_all, step, synthetic_epoch, toy_trainer
from tools.exp17.formal.validation import METRIC_KEYS

PROTOCOL = {"training":{"epochs":100},"effective_updates":2100,
            "resources":{"before_epoch_min_free_bytes":0,"before_checkpoint_min_free_bytes":0}}


class SyntheticAdapter:
    synthetic_only = True
    def __init__(self):
        self.trainer=toy_trainer()
        # The fixture's short recovery tests grow every four calls; a complete
        # 2,100-call simulation instead uses the historical 2,000-call cadence.
        self.trainer.scaler=torch.amp.GradScaler("cpu",init_scale=1,growth_interval=2000)
        self.val_events=[];self.diagnostic_epochs=[];self.cache_epochs=[];self.cursor=None
    def start_epoch(self,epoch):
        self.trainer.epoch=epoch;self.trainer.model.train()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore");self.trainer.scheduler.step()
    def build_cache(self,epoch,directory):
        self.cache_epochs.append(epoch)
        return SealedEpochCache.build(directory,synthetic_epoch(epoch),epoch=epoch,bindings=BINDINGS,
            batch_fingerprint=lambda n,b:{"index":n,"tensor_and_pair_identity":state_hash(b)},expected_batches=21)
    def set_sampler(self,cursor):self.cursor=cursor
    def train_batch(self,batch,number):
        if number<=63:self.trainer.optimizer.param_groups[0]['lr']=number/63*.001
        loss=step(self.trainer,batch,number)
        return {"step":number,"epoch":self.trainer.epoch+1,"batch_index":self.cursor.next_index,
            "effective_updates":self.trainer.audit.effective_updates,"optimizer_attempts":self.trainer.audit.optimizer_attempts,
            "amp_skipped":False,"images":len(batch['idx']),"raw_targets":66 if self.cursor.next_index==21 else 60,
            "yolo_loss":loss,"supcon_loss":0.,"pair_loss":0.,"total_loss":loss,"lambda_pair":4.,
            "gradient_norm":1.,"clip_coefficient":1.,"lr":[g['lr'] for g in self.trainer.optimizer.param_groups],
            "pair_by_class":{"0":1},"A_groups_by_class":{"0":["CPU_SYNTHETIC_GROUP"]},"pair_loss_by_class":{"0":0.},
            "pair_skips":{},"bn_state_sha256":state_hash({n:v for n,v in self.trainer.model.state_dict().items() if 'running_' in n}),
            "parameter_state_sha256":state_hash(dict(self.trainer.model.named_parameters())),
            "actual_synthetic_batch_sha256":state_hash(batch),"CPU_SYNTHETIC_EXECUTOR_ONLY":True}
    def gradient_diagnostics(self):
        self.diagnostic_epochs.append(self.trainer.epoch+1)
        return {"status":"SYNTHETIC_DIAGNOSTIC_CALLBACK_FIXTURE","optimizer_updates":0,"model_forwards":0}
    def evaluate(self,epoch,role):
        self.val_events.append((epoch,role))
        # Deliberately fixed native fitness tests latest-equality selection.
        values={key:.2 for key in METRIC_KEYS};values['fitness']=.4
        return {"standard":{"overall":values,"CPU_SYNTHETIC_METRICS_ONLY":True},
                "fixed":{"CPU_SYNTHETIC_METRICS_ONLY":True,"model_forwards":0},"simulated_native_passes":2}


def synthetic_final(trainer,output,bindings,*,role):
    """Only the architecture-specific endpoint writer is replaced in this test.

    Actual standalone seven-class export/loader behavior is covered by the
    separate exports test suite; this fixture never constructs YOLO.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/f"CPU_SYNTHETIC_FINAL_{role}.pt"
    source=trainer.model if role=='master' else trainer.ema.ema
    with path.open('xb') as f:torch.save({'CPU_SYNTHETIC_ONLY':True,'role':role,'state':source.state_dict()},f)
    return {'sha256':sha256_file(path),'role':role,'synthetic_only':True,'updates':2100}


def session(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    output=root/'output';output.mkdir();control=root/'control';control.mkdir()
    return output,RunJournal(control/'updates.jsonl',fresh=True),AtomicCheckpointStore(output/'recovery',BINDINGS)


def rows(output):
    return [json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]


class ExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old_threads)

    def test_full_same_core_2100_synthetic_updates_20_events_80_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            seed_all();adapter=SyntheticAdapter();output,journal,store=session(Path(temporary)/'full')
            with patch('tools.exp17.formal.exports.write_final_native',side_effect=synthetic_final):
                paused=run_session(adapter,PROTOCOL,BINDINGS,output,journal,store,synthetic_stop_after=2100)
                self.assertEqual(paused['status'],'SYNTHETIC_PAUSE_AT_COMMITTED_UPDATE')
                self.assertEqual(journal.committed,2100)
                self.assertEqual(adapter.diagnostic_epochs,list(range(5,100,5)))
                self.assertEqual(len(adapter.val_events),38)
                committed,_=store.load_latest()
                self.assertFalse(committed['extra']['epoch_events_complete'])
                result=run_session(adapter,PROTOCOL,BINDINGS,output,journal,store,recovered=committed)
                finished,_=store.load_latest()
                for field in ('state','ema','optimizer','scheduler','scaler','rng','audit','counters'):
                    self.assertEqual(state_hash(committed[field]),state_hash(finished[field]),field)
            self.assertEqual(result['updates'],2100);self.assertEqual(journal.committed,2100)
            self.assertEqual(adapter.trainer.ema.updates,2100);self.assertEqual(len(rows(output)),2100)
            self.assertEqual(adapter.diagnostic_epochs,list(range(5,101,5)))
            self.assertEqual(adapter.val_events,[(e,r) for e in range(5,101,5) for r in ('master','ema')])
            self.assertEqual(len(adapter.val_events)*2,80)
            self.assertEqual(adapter.cache_epochs,list(range(100)))
            self.assertEqual(len(list((output/'recovery').glob('epoch*_full_recovery.pt'))),20)
            self.assertEqual(len(list((output/'recovery').glob('recovery_slot_*.pt'))),2)
            self.assertEqual(sorted(p.name for p in (output/'epoch_cache').iterdir()),['epoch_098','epoch_099'])
            self.assertEqual(result['best']['master']['epoch'],100);self.assertEqual(result['best']['ema']['epoch'],100)
            self.assertEqual(len(list((output/'best').glob('best_*_slot*.pt'))),4)
            self.assertEqual(len(list((output/'best').glob('best_*_epoch*.selection.json'))),40)
            for role in ('master','ema'):
                pointer=json.loads((output/'best'/f'best_{role}.json').read_text())
                self.assertEqual(pointer['epoch'],100)
                self.assertEqual(pointer['role'],'best_'+role)
            payload,receipt=store.load_latest();self.assertTrue(payload['extra']['epoch_events_complete'])
            self.assertEqual(payload['position'],{'epoch':99,'batch_position':21,'global_update':2100})
            self.assertEqual(sum(r['images'] for r in rows(output)),66800)
            self.assertEqual(sum(r['raw_targets'] for r in rows(output)),126600)
            self.assertFalse(torch.cuda.is_initialized())

    def test_same_core_committed_resume_51_to_74_equals_direct_74(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            seed_all();direct=SyntheticAdapter();do,dj,ds=session(root/'direct')
            run_session(direct,PROTOCOL,BINDINGS,do,dj,ds,synthetic_stop_after=74)
            expected,_=ds.load_latest()
            seed_all();first=SyntheticAdapter();out,journal,store=session(root/'resume')
            run_session(first,PROTOCOL,BINDINGS,out,journal,store,synthetic_stop_after=51)
            saved,_=store.load_latest();continued=SyntheticAdapter()
            reopened=RunJournal(journal.path)
            run_session(continued,PROTOCOL,BINDINGS,out,reopened,store,recovered=saved,synthetic_stop_after=74)
            actual,_=store.load_latest()
            for field in ('state','ema','optimizer','scheduler','scaler','rng','position','audit','counters'):
                self.assertEqual(state_hash(actual[field]),state_hash(expected[field]),field)
            self.assertEqual(rows(out),rows(do));self.assertEqual(continued.cache_epochs,[3])
            self.assertEqual(actual['sampler']['next_batch'],11)
            self.assertEqual(actual['sampler']['epoch'],3)

    def test_complete_cli_arguments_without_authorization_reject_before_any_runtime_import_or_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'FORBIDDEN_OUTPUT';settings=root/'FORBIDDEN_SETTINGS'
            code="""import builtins,runpy,sys
original=builtins.__import__
def guard(name,*a,**k):
 if name.split('.')[0] in ('torch','ultralytics') or name.endswith('native_adapter'):raise AssertionError('FORBIDDEN_RUNTIME_IMPORT')
 return original(name,*a,**k)
builtins.__import__=guard
sys.argv=['executor','--protocol',sys.argv[1],'--runtime-config',sys.argv[2],'--output',sys.argv[3]]
runpy.run_module('tools.exp17.formal.executor',run_name='__main__')
"""
            environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',YOLO_CONFIG_DIR=str(settings))
            result=subprocess.run([sys.executable,'-c',code,str(root/'ABSENT_PROTOCOL.json'),str(root/'ABSENT_CONFIG.json'),str(output)],
                cwd=Path(__file__).resolve().parents[4],env=environment,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0);self.assertIn('formal-training authorization is absent',result.stderr)
            self.assertNotIn('FORBIDDEN_RUNTIME_IMPORT',result.stderr);self.assertFalse(output.exists());self.assertFalse(settings.exists())

    def test_resume_checkpoint_journal_disagreement_and_truncated_metrics_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            seed_all();a=SyntheticAdapter();ao,aj,ast=session(root/'fifty')
            run_session(a,PROTOCOL,BINDINGS,ao,aj,ast,synthetic_stop_after=50);older,_=ast.load_latest()
            seed_all();b=SyntheticAdapter();bo,bj,bst=session(root/'fiftyone')
            run_session(b,PROTOCOL,BINDINGS,bo,bj,bst,synthetic_stop_after=51)
            with self.assertRaisesRegex(HardGate,'Checkpoint and permanent journal disagree'):
                run_session(SyntheticAdapter(),PROTOCOL,BINDINGS,bo,bj,bst,recovered=older,synthetic_stop_after=74)
            metrics=bo/'steps.jsonl'
            with metrics.open('ab') as f:f.write(b'{"partial":')
            saved,_=bst.load_latest()
            with self.assertRaisesRegex(HardGate,'Truncated metrics'):
                run_session(SyntheticAdapter(),PROTOCOL,BINDINGS,bo,bj,bst,recovered=saved,synthetic_stop_after=74)


if __name__=='__main__':unittest.main()
