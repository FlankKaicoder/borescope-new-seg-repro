"""Read-only saved-tensor algebra; no model, new gradients or optimizer updates."""
import os,sys
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.dont_write_bytecode=True
from pathlib import Path
import gc,hashlib,json,time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import numpy.ma


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def measure(x,y):
    x=np.asarray(x,dtype=np.float64).ravel();y=np.asarray(y,dtype=np.float64).ravel()
    assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    nx=float(np.linalg.norm(x));ny=float(np.linalg.norm(y));dot=float(x@y)
    return {'x_L2':nx,'y_L2':ny,'dot':dot,'cosine':dot/(nx*ny) if nx and ny else None,
            'projection_on_y_unit':dot/ny if ny else None}


def tests():
    assert measure([1,0],[0,1])['cosine']==0
    assert measure([1,0],[-2,0])['cosine']==-1
    assert measure([0,0],[1,0])['cosine'] is None
    assert measure([2,0],[1,0])['projection_on_y_unit']==2
    assert measure([1,1],[2,2])['cosine']>1-1e-15
    for x,y in (([1],[1,2]),([float('nan')],[1]),([float('inf')],[1])):
        try:measure(x,y)
        except AssertionError:pass
        else:raise AssertionError('Invalid fixture accepted')
    return 8


def main(root):
    start=time.monotonic();root=root.resolve();q=json.loads((root/'request.json').read_text())
    assert tests()==8
    sources={x['path']:x['sha256'] for x in q['checkpoints']}
    sources.update(q['gradient_files'])
    sources[str(root/'request.json')]=sha(root/'request.json');sources[str(Path(__file__).resolve())]=sha(__file__)
    for p,s in sources.items():assert sha(p)==s,('Input identity mismatch',p)
    def write(name,data):
        with (root/name).open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
    def audit(event,a):
        if event=='open' and isinstance(a[0],(str,bytes,os.PathLike)):
            p,m,flags=a;writing=(isinstance(m,str) and any(c in m for c in 'wax+')) or ((flags or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and not Path(os.fsdecode(p)).resolve().is_relative_to(root):raise RuntimeError('Write outside R4-D')
        if event in ('os.remove','os.rmdir','os.rename'):raise RuntimeError('R4-D prohibits deletion/rename')
    import torch
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    sys.addaudithook(audit)
    assert not torch.cuda.is_initialized()
    counts={k:0 for k in ('model_constructor','model_call','functional_model','autograd','optimizer','save','cuda')}
    def forbid(kind):
        def reject(*a,**kw):counts[kind]+=1;raise RuntimeError('Forbidden '+kind)
        return reject
    rng=torch.get_rng_state().clone()
    refs={n:dict(np.load(p)) for n,p in q['gradient_paths'].items()}
    names=q['names'];assert all(set(v)==set(names) for v in refs.values())
    refs['omitted_pair4']={n:refs['A_pair4'][n]-refs['B_pair4'][n] for n in names}
    domains={'all':names,'backbone_0_10':[n for n in names if n.startswith('model.') and int(n.split('.')[1])<=10],
             'neck13':[n for n in names if n.startswith('model.13.')],
             'neck16':[n for n in names if n.startswith('model.16.')],
             'conv_weights':[n for n in names if n.endswith('.conv.weight')],
             'BN_gamma':[n for n in names if n.endswith('.bn.weight')],
             'BN_beta':[n for n in names if n.endswith('.bn.bias')],
             'projection':[n for n in names if n.startswith('region_projection.')]}
    flat=lambda d,ns:np.concatenate([d[n].ravel() for n in ns]).astype(np.float64)
    reference={key:{d:flat(v,ns) for d,ns in domains.items()} for key,v in refs.items()}
    table=[];radial=[];previous=None;last_state=None;baseline=None
    with ExitStack() as stack:
        for obj,attr,kind in ((torch.nn.Module,'__init__','model_constructor'),(torch.nn.Module,'_call_impl','model_call'),
                (torch.nn.functional,'batch_norm','functional_model'),(torch.nn.functional,'conv2d','functional_model'),
                (torch.autograd,'grad','autograd'),(torch.autograd,'backward','autograd'),(torch.Tensor,'backward','autograd'),
                (torch.optim.Optimizer,'__init__','optimizer'),(torch.optim.AdamW,'step','optimizer'),
                (torch,'save','save'),(torch.cuda,'_lazy_init','cuda')):
            stack.enter_context(patch.object(obj,attr,forbid(kind)))
        for row in q['checkpoints']:
            ck=torch.load(row['path'],map_location='cpu',weights_only=True)
            assert ck['position']['global_update']==row['update']
            state={n:ck['state'][n].detach().numpy().astype(np.float64) for n in names}
            assert all(v.shape==refs['A_pair4'][n].shape for n,v in state.items())
            opt={}
            for group,ns in zip(ck['optimizer']['param_groups'],ck['optimizer_names'],strict=True):
                for idx,n in zip(group['params'],ns,strict=True):
                    if n in names:opt[n]=(group,ck['optimizer']['state'][idx])
            # A saved-moment algebraic direction, not an optimizer update or the actual next step.
            adaptive={};moments={};decay={}
            for n in names:
                group,os_=opt[n];step=float(os_['step']);assert step==row['update']
                b1,b2=group['betas'];m=os_['exp_avg'].detach().numpy().astype(np.float64)
                v=os_['exp_avg_sq'].detach().numpy().astype(np.float64)
                moments[n]=-m
                adaptive[n]=-group['lr']*(m/(1-b1**step))/(np.sqrt(v/(1-b2**step))+group['eps'])
                decay[n]=-group['lr']*group['weight_decay']*state[n]
            for domain,ns in domains.items():
                theta=flat(state,ns)
                vectors={'saved_negative_moment':flat(moments,ns),'saved_adaptive_direction_proxy':flat(adaptive,ns),
                         'saved_decay_direction_proxy':flat(decay,ns)}
                if previous is not None and previous[0]['update']<row['update']:
                    vectors['actual_interval_displacement']=theta-flat(previous[1],ns)
                if baseline is not None and row['update']>105:vectors['actual_cumulative_displacement_since_105']=theta-flat(baseline,ns)
                for kind,x in vectors.items():
                    for key in ('A_pair4','B_pair4','omitted_pair4','A_yolo','A_supcon8'):
                        table.append({'label':row['label'],'update':row['update'],'epoch':row['update']/21,
                             'previous_update':previous[0]['update'] if previous is not None else None,
                             'domain':domain,'kind':kind,'endpoint_reference':key,
                             'reference_semantics':'NEGATIVE_ENDPOINT_GRADIENT_AT_2100_NOT_HISTORICAL_GRADIENT',
                             **measure(x,-reference[key][domain])})
            for n in names:
                if not n.endswith('.conv.weight'):continue
                # Actual parameter-space radial direction, distinct from virtual BN-input translation.
                theta=state[n]
                for key in ('A_pair4','B_pair4','omitted_pair4','A_yolo','A_supcon8'):
                    radial.append({'epoch':row['update']/21,'update':row['update'],'parameter':n,'reference':key,
                                   **measure(-refs[key][n],theta)})
            if row['label'].startswith('J2_epoch005'):baseline=state
            previous=(row,state);last_state=state
            del ck,adaptive,moments,decay;gc.collect()
            print('READ',row['label'],flush=True)
        assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized() and not any(counts.values())
    for p,s in sources.items():assert sha(p)==s,('Input changed',p)
    write('direction_alignment.json',table);write('parameter_radial_alignment.json',radial)
    write('completion.json',{'status':'R4_D_CPU_SAVED_TENSOR_ALGEBRA_COMPLETE','input_SHA_unchanged':len(sources),
          'checkpoints':len(q['checkpoints']),'direction_rows':len(table),'radial_rows':len(radial),
          'forbidden_calls':counts,'CUDA_initialized':False,'CPU_RNG_unchanged':True,'elapsed_seconds':time.monotonic()-start,
          'arithmetic_fixture_tests':8,'warning':'Endpoint direction alignment is descriptive and does not identify historical gradient contribution.'})


if __name__=='__main__':main(Path(sys.argv[1]))
