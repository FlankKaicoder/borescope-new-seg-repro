#!/usr/bin/env python3
"""CPU simulation: inherited history, cross-process atomic pair race, hard cap.

Simulation ledger is temporary, never the M1 or S5 authority. No model imports.
"""
import json
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.exp13_restoration.protocol import BudgetProtocol


def race(path, sid, barrier, q):
    budget=BudgetProtocol(ledger_path=Path(path),smoke_pilot_revision_cap=6)
    barrier.wait(timeout=15)
    try:
        records=budget.reserve_source_group(source_id=sid,roi_ids=['M1_DS','M1_XL'],stage='pilot',strength=.8,model_config_id='CPU_SIMULATION')
        q.put(('ok',sid,len(records)))
    except RuntimeError:
        q.put(('blocked',sid,0))


def main():
    baseline=Path(sys.argv[1]); initial=baseline.read_bytes()
    with tempfile.TemporaryDirectory(prefix='m1_budget_simulation_') as td:
        path=Path(td)/'SIMULATION_ONLY.json'; path.write_bytes(initial)
        barrier=mp.Barrier(2); q=mp.Queue()
        ps=[mp.Process(target=race,args=(str(path),sid,barrier,q)) for sid in ['simulation_a','simulation_b']]
        for p in ps:p.start()
        for p in ps:p.join(timeout=25); assert p.exitcode==0
        results=[q.get(timeout=3) for _ in ps]
        assert sorted(r[0] for r in results)==['blocked','ok'],results
        budget=BudgetProtocol(ledger_path=path,smoke_pilot_revision_cap=6)
        snap=budget.snapshot(); old=json.loads(initial)
        assert snap['occupied_calls']==6 and snap['actual_calls']==4 and snap['remaining']['smoke']==0
        assert all(snap['calls'][k]==v for k,v in old['calls'].items())
        new=[c for k,c in snap['calls'].items() if k not in old['calls']]
        assert len(new)==2 and len({c['source_id'] for c in new})==1
        cid=new[0]['call_id']; budget.start_call(cid,real_model_call=True)
        assert json.loads(path.read_text())['calls'][cid]['status']=='STARTED'
        budget.finish_call(cid,success=False,error='CPU_SIMULATION_NO_REAL_INFERENCE')
        final=budget.snapshot()
        assert final['occupied_calls']==6 and final['actual_calls']==5 and final['calls'][cid]['status']=='FAILED'
        try:budget.start_call(cid)
        except RuntimeError:pass
        else:raise AssertionError('terminal call restarted')
        assert baseline.read_bytes()==initial
        print(json.dumps({'status':'PASS_CPU_SIMULATION_ONLY','atomic_competing_pair_cap_race':results,
                          'history_inherited':True,'started_durable':True,'failure_charged':True,
                          'terminal_restart_blocked':True,'authoritative_ledgers_modified':False}))


if __name__=='__main__':main()
