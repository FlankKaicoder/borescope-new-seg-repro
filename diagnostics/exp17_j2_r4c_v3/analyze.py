"""Offline NumPy audit of sealed results; never imports Torch or runs a model."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def compare(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));nd=float(np.linalg.norm(b-a))
    return {'A_L2':na,'B_L2':nb,'delta_L2':nd,'relative_delta':nd/na if na else None,
            'cosine':float(a@b/(na*nb)) if na and nb else None,'max_abs_delta':float(np.max(np.abs(b-a))) if a.size else 0.}


def main(evidence,destination):
    load=lambda n:json.loads((evidence/n).read_text(encoding='utf8'))
    c=load('completion.json');assert c['status']=='COMPLETE_PENDING_REVIEW'
    assert c['counts']['model_autograd']==6 and c['counts']['cpu_autograd']==0
    assert c['pending'] is None and not c['changed_original_files']
    assert load('process_exit.json')['exit_code']==0
    rows=[json.loads(x) for x in (evidence/'calls.jsonl').read_text(encoding='utf8').splitlines()]
    previous='0'*64
    for row in rows:
        h=row['sha256'];assert row['previous']==previous
        assert digest({k:v for k,v in row.items() if k!='sha256'})==h;previous=h
    starts=[x['label'] for x in rows if x['event']=='ATTEMPT_BEGIN']
    commits=[x['label'] for x in rows if x['event']=='COMMIT']
    assert starts==commits==c['used'] and len(commits)==10 and len(set(starts))==10
    a=load('A_forward.json');b=load('B_forward.json');assert a['identity']==b['identity']
    layout=load('parameter_layout.json');names=[n for n,_ in layout['names_shapes']]
    assert len(names)==294
    arrays={label:dict(np.load(evidence/(label+'_gradients.npz'))) for label in starts[4:]}
    for label,items in arrays.items():
        assert list(items)==names
        for n,shape in layout['names_shapes']:
            assert list(items[n].shape)==shape and items[n].dtype==np.float32 and np.isfinite(items[n]).all()
        receipt=next(x for x in rows if x['event']=='COMMIT' and x['label']==label)
        assert hashlib.sha256((evidence/(label+'_gradients.npz')).read_bytes()).hexdigest()==receipt['file_sha256']
    flat=lambda label,selected:np.concatenate([arrays[label][n].ravel() for n in selected])
    controls={}
    for component in ('yolo','supcon8'):
        va=flat('A_'+component,names);vb=flat('B_'+component,names)
        np.testing.assert_allclose(va,vb,atol=1e-6,rtol=1e-5)
        controls[component]=compare(va,vb)
    summary=load('gradient_summary.json')
    def group(n):
        if n.startswith('region_projection.'):return 'projection'
        i=int(n.split('.')[1])
        return 'backbone_0_10' if i<=10 else 'neck13' if i==13 else 'neck16' if i==16 else 'head23' if i==23 else 'other_neck'
    for domain,entry in summary.items():
        selected=[n for n in names if domain=='all' or group(n)==domain]
        for component in ('yolo','supcon8','pair4'):
            actual=compare(flat('A_'+component,selected),flat('B_'+component,selected))
            for k in ('A_L2','B_L2','delta_L2','relative_delta','cosine'):
                if actual[k] is None:assert entry['components'][component][k] is None
                else:np.testing.assert_allclose(entry['components'][component][k],actual[k],rtol=1e-12,atol=1e-12)
    bn_a=load('A_bn_gradients.json');bn_b=load('B_bn_gradients.json')
    assert not bn_a['statistics'] and len(bn_b['statistics'])==48
    trace={};n_checks=0
    for key,entry in bn_a['inputs'].items():
        if ':N:' not in key:continue
        n_checks+=1
        for field,values in entry.items():np.testing.assert_allclose(values,bn_b['inputs'][key][field],rtol=1e-5,atol=1e-6)
        name=key.split(':',2)[2]
        gm=np.asarray(bn_b['statistics'][name]['mean']);negative_sum=-np.asarray(bn_b['inputs'][key]['sum'])
        # Two separately reduced FP32 expressions need not cancel bitwise.
        trace[name]=compare(negative_sum,gm)
        np.testing.assert_allclose(gm,negative_sum,rtol=2e-4,atol=2e-6)
    per_layer={}
    layers=sorted(set(n.rsplit('.',1)[0] for n in names))
    for layer in layers:
        selected=[n for n in names if n.rsplit('.',1)[0]==layer]
        per_layer[layer]=compare(flat('A_pair4',selected),flat('B_pair4',selected))
    actual_parameters={}
    for category,predicate in [('conv_weights',lambda n:n.endswith('.conv.weight')),
                               ('BN_gamma',lambda n:n.endswith('.bn.weight')),
                               ('BN_beta',lambda n:n.endswith('.bn.bias'))]:
        selected=[n for n in names if predicate(n)]
        actual_parameters[category]=compare(flat('A_pair4',selected),flat('B_pair4',selected))
    zero_pair=[]
    for n in names:
        if n.startswith('region_projection.'):
            np.testing.assert_allclose(arrays['A_pair4'][n],arrays['B_pair4'][n],atol=1e-6,rtol=1e-5)
        elif int(n.split('.')[1])>16:
            assert not np.any(arrays['A_pair4'][n]) and not np.any(arrays['B_pair4'][n]);zero_pair.append(n)
    result={'passed':True,'ledger_rows':len(rows),'unique_attempts_and_commits':len(commits),
            'bitwise_forward_equal':True,'gradient_parameters':len(names),'controls':controls,
            'N_input_gradient_comparisons':n_checks,'mean_trace':trace,'actual_parameter_categories':actual_parameters,
            'Pair_zero_outside_prefix_parameters':len(zero_pair),'per_layer_pair':per_layer,
            'original_files_unchanged':c['original_files_checked'],'model_calls':0,'autograd_calls':0,
            'cumulative_consumption':{'A_attempts':5,'A_complete_returns':4,'N_attempts_and_returns':4,
                                      'model_autograd':6,'CPU_autograd':12,'optimizer':0,'VAL':0,'TEST':0}}
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('mean_trace','per_layer_pair')},indent=2))


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
