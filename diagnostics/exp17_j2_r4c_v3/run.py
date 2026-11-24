"""One-shot R4-C execution. No production changes, optimizer or evaluation."""
import os
import sys
sys.dont_write_bytecode = True
from core import *
from disk_saved import DiskSaved
PREVIOUS = Path("/root/autodl-tmp/exp17_j2_r4c_detach_mechanism_v2")
from guards import attach_method_counter, validate_dispatch, audit_policy, seal_before_check
ORIGIN = Path("/root/autodl-tmp/exp17_j2_r4c_detach_mechanism_v1")
os.environ.update(PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  YOLO_CONFIG_DIR=str(ROOT/'settings'), YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false',
                  CUDA_CACHE_PATH=str(ROOT/'settings/cuda_cache'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
os.environ['TMPDIR']=str(ROOT/'settings/tmp')
sys.path.insert(0, str(REPO))
import copy
import contextlib
import gc
import signal
import traceback
from types import SimpleNamespace
from unittest.mock import patch

def forbid(*a, **kw): raise RuntimeError('Forbidden extra model/backward/optimizer/save operation')

def tensors(value, prefix=''):
    import torch
    if torch.is_tensor(value): yield prefix, value
    elif isinstance(value, dict):
        for k,v in value.items(): yield from tensors(v, prefix+'/'+str(k))
    elif isinstance(value, (list,tuple)):
        for i,v in enumerate(value): yield from tensors(v, prefix+'/'+str(i))

def identity(value):
    import torch
    rows = {}
    for k,v in tensors(value):
        v=v.detach().cpu().contiguous()
        assert not v.is_floating_point() or bool(torch.isfinite(v).all()), ('Nonfinite tensor',k)
        rows[k]={'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()}
    return rows

def protected():
    prior=json.loads((R4B/'preflight.json').read_text())['original_artifacts']
    for row in json.loads((R4B/'artifact_manifest.json').read_text()):
        p=R4B/row['path']; assert p.resolve().is_relative_to(R4B.resolve())
        prior[str(p)]=row['sha256']
    prior[str(R4B/'artifact_manifest.json')]=sha(R4B/'artifact_manifest.json')
    for p,s in BOUND.items():
        if p in prior: assert prior[p]==s
        prior[p]=s
    for row in json.loads((ORIGIN/'artifact_manifest.json').read_text()):
        path=ORIGIN/row['path'];assert path.resolve().is_relative_to(ORIGIN)
        prior[str(path)]=row['sha256']
    prior[str(ORIGIN/'artifact_manifest.json')]=sha(ORIGIN/'artifact_manifest.json')
    for row in json.loads((PREVIOUS/'artifact_manifest.json').read_text()):
        path=PREVIOUS/row['path'];assert path.resolve().is_relative_to(PREVIOUS)
        prior[str(path)]=row['sha256']
    prior[str(PREVIOUS/'artifact_manifest.json')]=sha(PREVIOUS/'artifact_manifest.json')
    for p,s in prior.items(): assert sha(p)==s, ('Protected evidence changed',p)
    return prior

def factory(payload, args, torch):
    from ultralytics.nn.tasks import SegmentationModel, _initialize_yolo_model
    from ultralytics.nn.modules import Detect, Segment
    from ultralytics.utils.torch_utils import initialize_weights
    from tools.exp15.regions import add_projection
    from tools.exp17.engineering.model import _native_activation_scope
    from tools.exp17.formal.checkpoint import state_hash
    classes=(Detect,Segment)
    old=[(c,'legacy' in c.__dict__,c.__dict__.get('legacy')) for c in classes]
    with torch.random.fork_rng(devices=[]), _native_activation_scope(), patch.object(torch.nn.Module,'_call_impl',forbid):
        try:
            model=SegmentationModel.__new__(SegmentationModel);torch.nn.Module.__init__(model)
            _initialize_yolo_model(model,copy.deepcopy(payload['model_config']),3,7,False)
            head=model.model[-1]
            assert len(model.model)==24 and type(head) is Segment and not head.end2end
            head.inplace=model.inplace;head.stride=torch.tensor([8.,16.,32.]);model.stride=head.stride
            head.legacy=False;head.dynamic=False;head.export=False;head.format=None;head.xyxy=False
            head.shape=None;head.anchors=torch.empty(0);head.strides=torch.empty(0);head.max_det=300
            initialize_weights(model);add_projection(model)
            model.load_state_dict(payload['state'],strict=True)
            model.names=copy.deepcopy(payload['class_names']);model.nc=7;model.args=SimpleNamespace(**args)
            named=dict(model.named_parameters()); expected=set(payload['trainable_parameter_names'])
            assert expected <= named.keys() and len(expected)==294
            for n,p in named.items():p.requires_grad_(n in expected)
            modules=dict(model.named_modules());assert modules.keys()==payload['training_modes'].keys()
            for n,mode in payload['training_modes'].items():modules[n].training=mode
            assert model.training and all(m.training for m in modules.values() if isinstance(m,torch.nn.BatchNorm2d))
            assert state_hash(model.state_dict())==payload['master_state_sha256']
            assert all(m.eps==.001 and m.momentum==.03 for m in modules.values() if isinstance(m,torch.nn.BatchNorm2d))
            return model
        finally:
            for c,owned,value in old:
                if owned:setattr(c,'legacy',value)
                else:delattr(c,'legacy')

class Graph:
    def __init__(self, label, model, budget, torch):
        self.label,self.model,self.budget,self.torch=label,model,budget,torch
        self.phase=None;self.handles=[];self.moments={};self.native={};self.inputs={};self.input_grads={}
        self.stat_grads={};self.counts={'A':[],'N':[],'projection':0,'root_dispatch':[],'method_dispatch':[]};self.component=None;self.raw={};self.bridge_count=0
        self.Bridge=bridge_type(torch)
        self.restore_forward=attach_method_counter(model,self.counts,lambda:self.phase=='A' and budget.active==label+'_full')
        self.focus=lambda n:int(n.split('.')[1]) in (0,4,10,13,16)
        self.base_parameter_identity=identity(dict(model.named_parameters()))
        self.modes={n:m.training for n,m in model.named_modules()}
        self.buffer_identity=lambda:identity(dict(model.named_buffers()))
        self.before_buffers=self.buffer_identity()
        for n,m in model.named_modules():
            if isinstance(m,torch.nn.BatchNorm2d):
                self.handles.append(m.register_forward_pre_hook(self.pre(n)))
                self.handles.append(m.register_forward_hook(self.post(n)))
        self.handles.append(model.register_forward_pre_hook(self.root_pre))
        self.handles.append(model.register_forward_hook(self.root_post))
        self.handles.append(model.model[-1].register_forward_hook(lambda m,i,o:self.raw.update(identity(o))))
        def projection_call(m,i):self.counts['projection']+=1
        self.handles.append(model.region_projection.register_forward_pre_hook(projection_call))
        for i,m in enumerate(model.model):self.handles.append(m.register_forward_pre_hook(self.layer(i)))

    def layer(self,i):
        def check(m,args):
            assert self.phase in ('A','N') and (self.phase=='A' or i<=16)
            if self.phase=='N' and i==0:
                assert self.budget.active is None
                self.budget.begin(self.label+'_N_prefix')
            self.counts[self.phase].append(i)
        return check

    def root_pre(self,m,args):
        self.counts['root_dispatch'].append('dict' if isinstance(args[0],dict) else 'tensor')
        if isinstance(args[0],dict):
            assert self.phase is None;self.budget.begin(self.label+'_full');self.phase='A'
        else:assert self.phase=='A' and self.budget.active==self.label+'_full'

    def root_post(self,m,args,out):
        if isinstance(args[0],dict):
            assert self.counts['A']==list(range(24))
            self.after_a_buffers=self.buffer_identity()
            self.budget.commit(output_digest=digest(self.raw));self.phase='N'

    def pre(self,name):
        def hook(m,args):
            torch=self.torch;x=args[0];phase=self.phase
            assert phase in ('A','N') and x.dtype==torch.float32
            if phase=='N':assert self.budget.active==self.label+'_N_prefix'
            if phase=='A':
                connected=self.label=='B' and int(name.split('.')[1])<=16
                v,mu=torch.var_mean(x.float() if connected else x.detach().float(),dim=(0,2,3),unbiased=False)
                assert name not in self.moments;self.moments[name]=(mu,v)
            key=phase+':'+name
            assert key not in self.inputs
            # Store only fixed focus inputs, losslessly on CPU; sums for all BN inputs.
            self.inputs[key]=x.detach().cpu() if self.focus(name) else None
            if x.requires_grad:
                def gradient_hook(g):
                    if not self.focus(name):return
                    assert self.component is not None
                    cpu=g.detach().cpu();dims=(0,2,3);source=self.inputs[key]
                    mean=source.mean(dims).reshape(1,-1,1,1)
                    entry={'sum':cpu.sum(dims).double().tolist(),
                           'centered_dot':(cpu*(source-mean)).sum(dims).double().tolist(),
                           'l2':cpu.square().sum(dims).sqrt().double().tolist()}
                    output_key=self.component+':'+key
                    assert output_key not in self.input_grads
                    self.input_grads[output_key]=entry
                x.register_hook(gradient_hook)
        return hook

    def post(self,name):
        def hook(m,args,out):
            if self.phase!='N' or self.label!='B':return None
            mu,v=self.moments[name];assert mu.requires_grad and v.requires_grad
            def observer(gm,gv):
                assert self.component=='pair4' and name not in self.stat_grads
                assert bool(self.torch.isfinite(gm).all() and self.torch.isfinite(gv).all())
                self.stat_grads[name]={'mean':gm.detach().cpu().double().tolist(),'variance':gv.detach().cpu().double().tolist()}
            self.bridge_count+=1
            return self.Bridge.apply(out,mu,v,args[0].detach(),m.weight.detach(),m.eps,observer)
        return hook

    def finish_forward(self,out):
        observed={'counts':copy.deepcopy(self.counts),'bridge_count':self.bridge_count,
                  'raw':self.raw,'values':identity({'yolo':out.yolo,'parts':out.parts,'supcon':out.supcon,
                      'pair':out.pair,'total':out.total,'a_p3':out.a_p3,'n_p3':out.n_p3,
                      'a_embedding':out.a_embedding,'n_embedding':out.n_embedding}),
                  'buffers_before':self.before_buffers,'buffers_after_A':self.after_a_buffers,
                  'buffers_after_N':self.buffer_identity(),'resources':self.budget.check()}
        seal_before_check(lambda record:write(ROOT/(self.label+'_forward_observed.json'),record),
                          lambda:validate_dispatch(self.counts),observed)
        assert self.after_a_buffers==self.buffer_identity()
        assert self.base_parameter_identity==identity(dict(self.model.named_parameters()))
        assert self.modes=={n:m.training for n,m in self.model.named_modules()}
        assert len(out.bn_statistics)==90
        for n,items in out.bn_statistics.items():
            assert len(items)==1
            mu,v=self.moments[n];original=items[0]
            assert self.torch.equal(mu.detach(),original.mean) and self.torch.equal(v.detach(),original.biased_variance)
        assert self.bridge_count==(48 if self.label=='B' else 0)
        values={'yolo':out.yolo,'parts':out.parts,'supcon':out.supcon,'pair':out.pair,'total':out.total,
                'a_p3':out.a_p3,'n_p3':out.n_p3,'a_embedding':out.a_embedding,'n_embedding':out.n_embedding}
        self.forward_identity={'raw':self.raw,'values':identity(values),'moments':identity(self.moments),
                               'buffers':self.after_a_buffers,'parameters':self.base_parameter_identity,'modes':self.modes}
        self.budget.commit(output_digest=digest(self.forward_identity));self.phase=None
        self.out=out
        for h in self.handles:h.remove()
        self.handles=[]
        self.restore_forward()
        return self.forward_identity

def metrics(a,b,np):
    a=a.astype(np.float64);b=b.astype(np.float64);na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    delta=float(np.linalg.norm(b-a))
    return {'A_L2':na,'B_L2':nb,'A_RMS':na/(len(a)**.5) if len(a) else 0.,
            'B_RMS':nb/(len(b)**.5) if len(b) else 0.,'delta_L2':delta,
            'relative_delta':delta/na if na else None,'cosine':float(a@b/(na*nb)) if na and nb else None}

def execute():
    import numpy as np
    import torch
    import yaml
    from tools.exp17.formal.authorization import source_catalog
    from tools.exp17.formal.checkpoint import validate_payload,state_hash
    from tools.exp17.formal.rng import capture_rng,rng_equal,restore_rng
    from tools.exp17.engineering.runtime import prepare_batch,DecodedAssetCache,runtime_asset_map
    from tools.exp17.engineering.data import TrustedPairIndex
    from tools.exp15.ja import checked_train,batch_digest
    from tools.exp17.engineering.model import forward_joint
    import ast
    # Prior derivative validation is reused only for byte-identical bridge/fixtures.
    assert not torch.cuda.is_initialized()
    auth=json.loads((ROOT/'authorization.json').read_text())
    scripts={p.name:sha(p) for p in ROOT.glob('*.py')}
    assert auth['scripts']==scripts and auth['authorized'] is True and auth['labels']==LABELS
    for p,s in auth['framework'].items():assert sha(p)==s
    for p,s in auth['origin_artifacts'].items():assert sha(p)==s,('Origin evidence conflict',p)
    assert json.loads((ORIGIN/'cpu_tests.json').read_text())['passed'] is True
    assert sha(ROOT/'fixtures.py')==auth['CPU_reuse']['fixtures_sha256']
    core_text=(ROOT/'core.py').read_text()
    bridge=next(n for n in ast.parse(core_text).body if isinstance(n,ast.FunctionDef) and n.name=='bridge_type')
    bridge_source=ast.get_source_segment(core_text,bridge)
    assert hashlib.sha256(bridge_source.encode()).hexdigest()==auth['CPU_reuse']['bridge_function_sha256']
    for kind in ('guard','storage'):
        assert sha(ROOT/(kind+'_tests.json'))==auth[kind+'_tests_sha256']
        assert json.loads((ROOT/(kind+'_tests.json')).read_text())['passed'] is True
    def audit(event,arguments):audit_policy(ROOT,event,arguments)
    sys.addaudithook(audit)
    assert digest(source_catalog(REPO))==CODE_SHA
    originals=protected();write(ROOT/'protected_before.json',originals)
    budget=Budget(ROOT,auth);start=time.monotonic();original_grad=torch.autograd.grad
    counts={'cpu_autograd':0,'reused_CPU_autograd_evidence':12,'model_autograd':0,'optimizer':0,'VAL':0,'TEST':0}
    result={'status':'HOLD','counts':counts};graphs=[];disk=None;offload=None
    def timeout(*_):raise RuntimeError('R4-C 1200-second watchdog')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
    def counted_grad(*a,**kw):
        assert budget.active is not None
        assert budget.active in LABELS[4:],'Autograd outside an authorized component'
        key='model_autograd';limit=6
        assert counts[key]<limit
        counts[key]+=1
        return original_grad(*a,**kw)
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(torch.autograd,'grad',counted_grad))
            stack.enter_context(patch.object(torch.autograd,'backward',forbid))
            stack.enter_context(patch.object(torch.Tensor,'backward',forbid))
            stack.enter_context(patch.object(torch.optim.Optimizer,'__init__',forbid))
            stack.enter_context(patch.object(torch.optim.AdamW,'step',forbid))
            stack.enter_context(patch.object(torch,'save',forbid))
            budget.event('CPU_EVIDENCE_REUSED',**auth['CPU_reuse'])
            assert not torch.cuda.is_initialized()
            torch.set_num_threads(4)
            cfg=json.loads(CONFIG.read_text());args=yaml.safe_load(ARGS.read_text())
            assert (args['box'],args['cls'],args['dfl'],args['overlap_mask'],args['mask_ratio'])==(7.5,.5,1.5,True,4)
            checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
            validate_payload(checkpoint,auth['checkpoint_bindings'])
            assert checkpoint['position']=={'epoch':99,'batch_position':21,'global_update':2100}
            assert checkpoint['master_state_sha256']=='609f1ae560829ca9c5bad715a68f1022a36be85b7ce45d003a0464dc21e1ea85'
            source=factory(checkpoint,args,torch);source_hash=state_hash(source.state_dict())
            assert tuple(source.names[k] for k in sorted(source.names))==('Burn','Crack','Dent','Material missing','Tears','Tip curl','corrosion')
            packet=torch.load(PACKET,map_location='cpu',weights_only=False)
            assert packet['epoch']==0 and packet['batch_position']==1 and packet['schema']=='EXP17_J2_NATIVE_CPU_BATCH_CACHE_V1'
            assert batch_digest(packet['batch'])=='cd68f03b38dd29874322e54a22052ff78401dc042f66fd0313dd4dd03443b63a'
            assert packet['batch']['img'].shape==(32,3,640,640) and len(packet['batch']['cls'])==62
            index=TrustedPairIndex.from_files(Path(cfg['manifest']),Path(cfg['native_identity']),
                expected_manifest_sha256=cfg['manifest_sha256'],runtime_assets=runtime_asset_map(cfg),native_dataset_root=Path(cfg['release'])/'dataset')
            groups=checked_train(Path(cfg['release']))
            prepared=prepare_batch(packet['batch'],index,groups,decode_N=True,device='cpu',asset_cache=DecodedAssetCache(max_bytes=0))
            assert prepared.bindings and prepared.coverage['actual_valid_ABN']>0
            write(ROOT/'prepared.json',{'coverage':prepared.coverage,'bindings':[vars(x) for x in prepared.bindings],
                  'A_identity':identity(prepared.batch),'N_identity':identity(prepared.normal_images)})
            assert torch.cuda.device_count()==1 and '3080 Ti' in torch.cuda.get_device_name(0)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
            torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(0).total_memory)
            torch.cuda.reset_peak_memory_stats()
            prepared.batch={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in prepared.batch.items()}
            prepared.normal_images=prepared.normal_images.cuda();prepared.normal_masks=prepared.normal_masks.cuda()
            restore_rng(checkpoint['rng'])
            offload=Offload(torch,budget);disk=DiskSaved(torch,budget,ROOT/'activation_cache_A')
            rng_before=capture_rng();prepared_before=identity((prepared.batch,prepared.normal_images,prepared.normal_masks))
            # Guard every module call outside the two explicit feature phases, including during backward.
            original_call=torch.nn.Module._call_impl
            def module_call(m,*a,**kw):
                active=next((g for g in graphs if g.phase is not None),None)
                if active is None and not (graphs and m is graphs[-1].model and isinstance(a[0],dict)):
                    raise RuntimeError('Unbudgeted module call or backward recomputation')
                return original_call(m,*a,**kw)
            stack.enter_context(patch.object(torch.nn.Module,'_call_impl',module_call))
            for label in ('A','B'):
                model=copy.deepcopy(source).cuda()
                model.criterion=model.init_criterion()
                graph=Graph(label,model,budget,torch);graphs.append(graph)
                storage=disk if label=='A' else offload
                with torch.autograd.graph.saved_tensors_hooks(storage.pack,storage.unpack):
                    out=forward_joint(model,prepared.batch,prepared.a_regions,prepared.normal_images,prepared.normal_masks,
                                      prepared.normal_regions,prepared.bindings,bn_mode='replay_a',lambda_pair=4.,checkpoint_prefix=False)
                graph.finish_forward(out)
                assert rng_equal(rng_before,capture_rng())
                assert float(out.pair.detach())>0,'No active Pair hinge; stop without replacement'
                write(ROOT/f'{label}_forward.json',{'identity':graph.forward_identity,
                      'losses':{k:float(getattr(out,k).detach()) for k in ('yolo','supcon','pair','total')},
                      'production_base_report':out.report,'diagnostic_graph':label,'counts':graph.counts,
                      'bridge_count':graph.bridge_count,'resources':budget.check()})
            assert graphs[0].forward_identity==graphs[1].forward_identity,'Forward equivalence failed'
            write(ROOT/'forward_gate.json',{'passed':True,'bitwise_equal':True,'A_traversals':2,'N_traversals':2})
            names=[n for n,p in graphs[0].model.named_parameters() if p.requires_grad]
            assert names==[n for n,p in graphs[1].model.named_parameters() if p.requires_grad]
            gradients={};unused={};names_shapes=[(n,list(dict(graphs[0].model.named_parameters())[n].shape)) for n in names]
            for component,weight in (('yolo',1.),('supcon8',8.),('pair4',4.)):
                for graph in graphs:
                    label=graph.label+'_'+component;graph.component=component
                    loss=getattr(graph.out,{'yolo':'yolo','supcon8':'supcon','pair4':'pair'}[component])*weight
                    parameters=[dict(graph.model.named_parameters())[n] for n in names]
                    budget.begin(label)
                    values=torch.autograd.grad(loss,parameters,retain_graph=component!='pair4',create_graph=False,allow_unused=True)
                    arrays={n:np.zeros(tuple(p.shape),dtype=np.float32) if v is None else v.detach().cpu().numpy().copy()
                            for n,p,v in zip(names,parameters,values,strict=True)}
                    assert all(np.isfinite(a).all() for a in arrays.values())
                    unused[label]=[n for n,v in zip(names,values,strict=True) if v is None]
                    np.savez(ROOT/(label+'_gradients.npz'),**arrays)
                    gradients[label]=arrays
                    budget.commit(file_sha256=sha(ROOT/(label+'_gradients.npz')),unused=unused[label],resources=budget.check())
                    assert all(p.grad is None for p in parameters)
                    assert graph.after_a_buffers==graph.buffer_identity() and graph.base_parameter_identity==identity(dict(graph.model.named_parameters()))
                    assert rng_equal(rng_before,capture_rng())
                    del values
                if component!='pair4':
                    assert unused['A_'+component]==unused['B_'+component]
                    a=np.concatenate([gradients['A_'+component][n].ravel() for n in names]);b=np.concatenate([gradients['B_'+component][n].ravel() for n in names])
                    np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
                    assert np.linalg.norm((b-a).astype(np.float64))<=1e-5*np.linalg.norm(a.astype(np.float64)) if np.any(a) else np.array_equal(a,b)
            assert len(graphs[1].stat_grads)==48 and not graphs[0].stat_grads
            for n in names:
                if n.startswith('region_projection.'):
                    np.testing.assert_allclose(gradients['A_pair4'][n],gradients['B_pair4'][n],atol=1e-6,rtol=1e-5)
                elif int(n.split('.')[1])>16:
                    assert not np.any(gradients['A_pair4'][n]) and not np.any(gradients['B_pair4'][n])
            for key,row in graphs[0].input_grads.items():
                if ':N:' in key:
                    for field in row:np.testing.assert_allclose(row[field],graphs[1].input_grads[key][field],atol=1e-6,rtol=1e-5)
            def domain(n):
                if n.startswith('region_projection.'):return 'projection'
                i=int(n.split('.')[1])
                return 'backbone_0_10' if i<=10 else ('neck13' if i==13 else ('neck16' if i==16 else ('head23' if i==23 else 'other_neck')))
            summary={}
            for group in ['all','backbone_0_10','neck13','neck16','other_neck','projection','head23']:
                selected=[n for n in names if group=='all' or domain(n)==group]
                flat={label:np.concatenate([a[n].ravel() for n in selected]).astype(np.float64) for label,a in gradients.items()}
                summary[group]={'components':{c:metrics(flat['A_'+c],flat['B_'+c],np) for c in ('yolo','supcon8','pair4')}}
                for role in ('A','B'):
                    y,s,p=(flat[role+'_'+c] for c in ('yolo','supcon8','pair4'))
                    summary[group][role]={'pair_vs_yolo':metrics(y,p,np),'pair_vs_supcon8':metrics(s,p,np),
                                         'yolo_vs_supcon8':metrics(y,s,np),'pair_vs_main':metrics(y+s,p,np),
                                         'joint_norm':float(np.linalg.norm(y+s+p))}
            per_parameter={n:{c:metrics(gradients['A_'+c][n].ravel(),gradients['B_'+c][n].ravel(),np)
                              for c in ('yolo','supcon8','pair4')} for n in names}
            for graph in graphs:
                write(ROOT/(graph.label+'_bn_gradients.json'),{'statistics':graph.stat_grads,'inputs':graph.input_grads})
            write(ROOT/'parameter_gradients.json',per_parameter)
            write(ROOT/'gradient_summary.json',summary)
            write(ROOT/'parameter_layout.json',{'names_shapes':names_shapes,'unused':unused})
            assert prepared_before==identity((prepared.batch,prepared.normal_images,prepared.normal_masks))
            assert state_hash(source.state_dict())==source_hash and state_hash(checkpoint['state'])==source_hash
            assert counts['cpu_autograd']==0 and counts['model_autograd']==6 and budget.used==LABELS
            result.update(status='COMPLETE_PENDING_REVIEW',summary=summary,
                          GPU_peak_allocated=torch.cuda.max_memory_allocated(),GPU_peak_reserved=torch.cuda.max_memory_reserved(),
                          offload={'pack_calls':offload.pack_calls,'unpack_calls':offload.unpack_calls,'cumulative_packed_bytes':offload.packed_bytes})
            assert result['GPU_peak_allocated']<=10*1024**3
            budget.event('COMPLETE',counts=counts)
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc(),
                      graph_observations=[{'label':g.label,'counts':g.counts,'bridge_count':g.bridge_count,
                                           'component':g.component,'phase':g.phase} for g in graphs])
        budget.event('HOLD',error=repr(exc),pending=budget.active,counts=counts)
    finally:
        signal.alarm(0)
        result.update(elapsed_seconds=time.monotonic()-start,used=budget.used,pending=budget.active,
                      RSS_peak_bytes=budget.RSS_peak,cgroup_peak_bytes=budget.cgroup_peak,
                      inherited_consumption={'CPU_autograd':12,'A_attempts':3,'N_attempts':2,'model_autograd':0})
        if disk is not None:result['disk_saved']=disk.report()
        if offload is not None:result['memory_saved']={'pack_calls':offload.pack_calls,'unpack_calls':offload.unpack_calls,'cumulative_packed_bytes':offload.packed_bytes}
        changed=[p for p,s in originals.items() if sha(p)!=s]
        result['original_files_checked']=len(originals);result['changed_original_files']=changed
        if changed:result['status']='HOLD_ORIGINAL_IDENTITY_CONFLICT'
        result['current_production_catalog_sha256']=digest(source_catalog(REPO))
        if torch.cuda.is_initialized():
            result['GPU_peak_allocated']=torch.cuda.max_memory_allocated();result['GPU_peak_reserved']=torch.cuda.max_memory_reserved()
        write(ROOT/'completion.json',result);budget.file.close()
        print(json.dumps({k:v for k,v in result.items() if k not in ('summary','traceback')},sort_keys=True))
    return 0 if result['status']=='COMPLETE_PENDING_REVIEW' else 2

if __name__=='__main__':sys.exit(execute())
