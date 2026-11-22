import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.exp17.formal.authorization import (ACTIONS,AUTH_STAGE,EXPERIMENT,HardGate,RunJournal,RunLease,
    claim_fresh,digest,exclusive_json,file_sha,protocol_bindings,validate_authorization,verify_existing_claim)


class AuthorizationTests(unittest.TestCase):
    def test_absent_gate_rejects_before_protocol_or_output_access(self):
        with TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaises(HardGate): validate_authorization(p/'missing',None,p/'output',p)
            self.assertFalse((p/'output').exists())

    def fixture(self,d):
        root=Path(d);anchor=root/'control';out=root/'run';sources={}
        p={'status':'PENDING_CHATGPT_FORMAL_TRAINING_GATE','formal_training_authorized':False,'pending_fields':[],
           'code':{'files':sources,'sha256':digest(sources)},'identity':{'trusted_manifest_sha256':'a'*64,'native_train_identity_sha256':'b'*64,'coco_sha256':'c'*64},'environment':{'torch':'test'}}
        path=root/'protocol.json';exclusive_json(path,p)
        a={'stage':AUTH_STAGE,'gate':'CHATGPT_FORMAL_TRAINING_GATE_PASS','authorized':True,'experiment':EXPERIMENT,
           'authorized_actions':ACTIONS,'bindings':protocol_bindings(p,file_sha(path)),
           'output':str(out.resolve()),'control_anchor':str(anchor.resolve()),'execution_id':'SYNTHETIC_ONLY_123456'}
        gate=root/'auth.json';exclusive_json(gate,a)
        return root,anchor,out,p,path,a,gate

    def test_every_identity_change_requires_new_gate(self):
        with TemporaryDirectory() as d:
            root,anchor,out,p,path,a,gate=self.fixture(d)
            with patch('tools.exp17.formal.authorization.control_anchor',return_value=anchor):
                validate_authorization(path,gate,out,root)
                for key in ['protocol_sha256','code_sha256','manifest_sha256','native_identity_sha256','coco_sha256','seed','epochs','effective_updates','environment']:
                    changed=json.loads(json.dumps(a));changed['bindings'][key]='CHANGED';gate.write_text(json.dumps(changed))
                    with self.assertRaises(HardGate,msg=key):validate_authorization(path,gate,out,root)
                gate.write_text(json.dumps(a))
                with self.assertRaises(HardGate):validate_authorization(path,gate,root/'other',root)

    def test_pending_protocol_and_unapproved_val_scope_rejected(self):
        with TemporaryDirectory() as d:
            root,anchor,out,p,path,a,gate=self.fixture(d)
            with patch('tools.exp17.formal.authorization.control_anchor',return_value=anchor):
                p['pending_fields']=['fitness'];path.write_text(json.dumps(p))
                with self.assertRaises(HardGate):validate_authorization(path,gate,out,root)

    def test_claim_cannot_reset_by_output_or_copy_or_ledger_deletion(self):
        with TemporaryDirectory() as d:
            root,anchor,out,p,path,a,gate=self.fixture(d)
            with patch('tools.exp17.formal.authorization.control_anchor',return_value=anchor):
                claim_fresh(a,out);j=RunJournal(anchor/'updates.jsonl',fresh=True);j.begin(1);j.commit(1,'d'*64)
                verify_existing_claim(a,out)
                for newout in [out,root/'another_output']:
                    with self.assertRaises(HardGate):claim_fresh(a,newout)
                (anchor/'updates.jsonl').unlink()
                with self.assertRaises(HardGate):verify_existing_claim(a,out)
                with self.assertRaises(HardGate):claim_fresh(a,out)

    def test_no_duplicate_skipped_or_excessive_updates(self):
        with TemporaryDirectory() as d:
            j=RunJournal(Path(d)/'ledger',fresh=True)
            for step in range(1,2101):j.begin(step);j.commit(step,'e'*64)
            self.assertEqual(RunJournal(j.path).committed,2100)
            for step in [2100,2101,2102]:
                with self.assertRaises(HardGate):j.begin(step)

    def test_ambiguous_attempt_never_replayed(self):
        with TemporaryDirectory() as d:
            j=RunJournal(Path(d)/'ledger',fresh=True);j.begin(1)
            with self.assertRaisesRegex(HardGate,'AMBIGUOUS'):RunJournal(j.path)

    def test_auxiliary_epoch_event_cannot_overlap_update_or_retry(self):
        with TemporaryDirectory() as d:
            j=RunJournal(Path(d)/'ledger',fresh=True);j.append('AUX_BEGIN',epoch=5)
            with self.assertRaises(HardGate):j.begin(1)
            with self.assertRaisesRegex(HardGate,'EPOCH_EVENT'):RunJournal(j.path)

    def test_tampered_or_truncated_journal_rejected(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'ledger';j=RunJournal(p,fresh=True);j.begin(1);j.commit(1,'e'*64)
            old=p.read_text();p.write_text(old.replace('UPDATE_COMMIT','UPDATE_CHANGED'))
            with self.assertRaises(HardGate):RunJournal(p)
            p.write_text(old.rstrip('\n'))
            with self.assertRaises(HardGate):RunJournal(p)

    def test_process_lease_is_exclusive(self):
        with TemporaryDirectory() as d:
            with RunLease(d):
                with self.assertRaises(HardGate):
                    with RunLease(d):pass
            with RunLease(d):pass


if __name__=='__main__':unittest.main()
