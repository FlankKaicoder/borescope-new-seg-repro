# Exp16 TRAIN 梯度回放源码审查副本

实际执行脚本 SHA256：7718fcc89c98aa15a2bbccc5dbfec3d39c0a44ed80ec2253fb2faab0c8880ab9。

本页仅将运行根目录的单个绝对路径替换为环境变量，其余算法和检查顺序保持实际执行字节中的内容。它是脱敏审查视图，不能冒充上述哈希对应的执行文件；实际文件保存在已离机核验的私有证据归档。核心训练及评价快照仍为 e9b44a65f456b8b64569b39a484823bd777efdbc。

```python
"""Optional private TRAIN-only replay; invoke only after three accepted J3 arms."""
import argparse,copy,hashlib,json,os,sys,traceback
from pathlib import Path

BASE=Path(os.environ["EXP16_PRIVATE_RUN_ROOT"]).resolve(strict=True)
PLAN_SHA='ea477e24922c7c3f1a609bae0024a4c57fa5882c30f29446676e5b2d004aa90c'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-after-training-audit',action='store_true',required=True)
    parser.parse_args()
    identity=load(BASE/'private/post_gradient_script_identity_v2.json')
    assert sha(__file__)==identity['script_sha256'] and identity['plan_sha256']==PLAN_SHA,'Private helper identity changed'
    plan_path=BASE/'private/post_gradient_plan.json'
    assert sha(plan_path)==PLAN_SHA,'Registered diagnostic plan changed'
    plan=load(plan_path)
    assert plan['lambda_ce']==160.0 and plan['probe_count']==12
    assert sha(BASE/'formal_protocol.json')==plan['protocol_sha256']
    assert sha(BASE/'preparation_r2/preflight.json')==plan['preflight_sha256']
    completed=load(BASE/'private/formal_sequence_complete.json')
    assert completed['status']=='PASS_THREE_FORMAL_AND_INDEPENDENT_AUDITS'
    assert completed['completed_seeds']==[42,142,242] and completed['protocol_sha256']==plan['protocol_sha256']
    for seed in plan['seeds']:
        accepted=load(BASE/'formal'/str(seed)/'accepted.json')
        assert accepted['status']=='PASS_EXP16_J3_FORMAL_ARM' and accepted['matched_native_batches']==2100
        assert accepted['protocol_sha256']==plan['protocol_sha256']
    out=BASE/'gradient_replay'
    out.mkdir()
    os.environ.setdefault('YOLO_CONFIG_DIR',str(out/'settings'))
    sys.path.insert(0,str(BASE/'code_r2'))
    import gc
    import torch
    from ultralytics.utils.torch_utils import autocast
    from tools.exp14.a11_full_numerics import capture_rng_state,restore_rng_state
    from tools.exp14.a11_smoke_numerics import finite_tree,exact_tree
    from tools.exp14.formal_e0_e1 import hash_state
    from tools.exp15 import ja,jb
    from tools.exp15.regions import inspect_regions
    from tools.exp16 import ce,train,experiment
    torch.set_num_threads(4)
    protocol=load(BASE/'formal_protocol.json')
    assert experiment.sources()==protocol['code_source_sha256']
    env=experiment.environment(load(BASE/'private/paths.json'))
    preflight=load(BASE/'preparation_r2/preflight.json')
    records=[]
    trainer=None
    try:
        for seed in plan['seeds']:
            root=out/str(seed);root.mkdir()
            trainer_dir=root/'trainer';trainer_dir.mkdir()
            prior=preflight['priors'][str(seed)]
            old=BASE/'formal'/str(seed)
            acceptance=load(old/'accepted.json')
            summary=load(old/'j3/summary.json')
            assert sha(old/'j3/summary.json')==acceptance['summary_sha256']
            trainer=train.make_trainer(env['release'],env['weight'],trainer_dir,env['groups'],
                {'seed':seed,'phase':'post_training_gradient_replay','plan_sha256':PLAN_SHA},seed)
            trainer.setup()
            assert trainer.initial_native==prior['native_initial_sha256']
            assert trainer.initial_bn==prior['initial_bn_sha256']
            assert [[n for n in group if not n.startswith(ce.HEAD_PREFIX)] for group in trainer.optimizer_names]==prior['native_groups']
            selected={}
            for step,batch in jb.matched_batches(trainer,21):
                if step in plan['fixed_train_epoch_zero_batches']:
                    assert ja.batch_digest(batch)==prior['batch_tensor_sha256'][step-1]
                    assert inspect_regions(batch,trainer.groups).report()==prior['per_batch_regions'][step-1]
                    selected[step]=copy.deepcopy(batch)
            assert sorted(selected)==[1,21]
            for checkpoint in plan['checkpoints']:
                step=checkpoint['step']
                path=old/'j3'/checkpoint['file']
                file_before=sha(path)
                assert file_before==summary['checkpoints'][str(step)]['sha256']
                declared=next(item for item in acceptance['checkpoints'] if item['step']==step)
                assert declared['sha256']==file_before
                strict=train.strict_checkpoint(path,step,summary['identity'])
                checkpoint_state=torch.load(path,map_location='cpu',weights_only=True)
                assert checkpoint_state['model_state_sha256']==declared['master_state_sha256']
                assert checkpoint_state['native_state_sha256']==declared['master_native_sha256']
                for batch_step,batch in selected.items():
                    trainer.model.load_state_dict(checkpoint_state['state'],strict=True)
                    state_before=hash_state(trainer.model.state_dict())
                    bn_before=ja.bn_fingerprint(trainer.model)
                    assert state_before==checkpoint_state['model_state_sha256']
                    rng=capture_rng_state()
                    try:
                        region=inspect_regions(batch,trainer.groups)
                        batch_sha=ja.batch_digest(batch)
                        payload=trainer.preprocess_batch(dict(batch))
                        with autocast(trainer.amp,device='cuda'):
                            native,parts,auxiliary=ce.forward(trainer.model,payload,region,lambda:ja.bn_fingerprint(trainer.model))
                            total=native+plan['lambda_ce']*auxiliary
                        assert finite_tree((native,parts,auxiliary,total))
                        named=[(name,p) for name,p in trainer.model.named_parameters() if p.requires_grad]
                        names,params=zip(*named)
                        gy=torch.autograd.grad(native,params,retain_graph=True,allow_unused=True)
                        gj=torch.autograd.grad(total,params,allow_unused=True)
                        assert finite_tree((gy,gj))
                        def scope(indexes):
                            native_vectors=[gy[i] for i in indexes]
                            joint_vectors=[gj[i] for i in indexes]
                            n=jb.grad_norm(native_vectors);j=jb.grad_norm(joint_vectors)
                            delta=[(torch.zeros_like(params[i]) if gj[i] is None else gj[i])-
                                   (torch.zeros_like(params[i]) if gy[i] is None else gy[i]) for i in indexes]
                            d=jb.grad_norm(delta)
                            return {'native_norm':n,'joint_norm':j,'actual_joint_minus_native_norm':d,
                                'actual_joint_minus_native_ratio':d/n if n else None,
                                'native_joint_cosine':jb.dot(native_vectors,joint_vectors)/(n*j) if n and j else None,
                                'native_clip_coef':min(1.0,10/(n+1e-6)),
                                'joint_clip_coef':min(1.0,10/(j+1e-6)),
                                'joint_nonzero_tensors':sum(gj[i] is not None and bool(gj[i].count_nonzero()) for i in indexes)}
                        backbone=[i for i,n in enumerate(names) if n.startswith('model.') and int(jb.native_layer(n))<=10]
                        metrics={'backbone':scope(backbone),'global':scope(list(range(len(names)))),
                            'layers':{layer:scope([i for i,n in enumerate(names) if jb.native_layer(n)==layer])
                                      for layer in dict.fromkeys(map(jb.native_layer,names))}}
                        row={'seed':seed,'epoch':checkpoint['epoch'],'checkpoint_step':step,'train_epoch':0,
                            'batch':batch_step,'batch_sha256':batch_sha,'lambda_ce':plan['lambda_ce'],
                            'native_loss':float(native.detach()),'ce_loss':float(auxiliary.detach()),'total_loss':float(total.detach()),
                            'ce_coverage':ce.coverage(region),'j1_geometry_coverage':region.report(),
                            'gradient_definition':plan['gradient_definition'],'clip_definition':'Actual clipping is global; per-layer coefficients are hypothetical scope-only values and are not applied', 'metrics':metrics,
                            'checkpoint_sha256':file_before,'native_master_sha256':checkpoint_state['native_state_sha256'],
                            'strict_checkpoint':strict,'state_before_sha256':state_before,'bn_before_sha256':bn_before,
                            'optimizer_attempts':0,'effective_updates':0}
                        del gy,gj,native,parts,auxiliary,total,payload
                    finally:
                        trainer.model.load_state_dict(checkpoint_state['state'],strict=True)
                        restore_rng_state(rng)
                    row['state_after_sha256']=hash_state(trainer.model.state_dict())
                    row['bn_after_sha256']=ja.bn_fingerprint(trainer.model)
                    row['rng_restored_exact']=exact_tree(rng,capture_rng_state())
                    assert row['state_after_sha256']==state_before and row['bn_after_sha256']==bn_before and row['rng_restored_exact']
                    assert trainer.audit.optimizer_attempts==trainer.audit.native_steps==trainer.audit.effective_updates==trainer.ema.updates==0
                    row['checkpoint_after_sha256']=sha(path)
                    assert row['checkpoint_after_sha256']==file_before
                    write(root/f"epoch{checkpoint['epoch']:03d}_batch{batch_step:02d}.json",row)
                    records.append(row)
                assert sha(path)==file_before
            trainer.close();trainer=None;gc.collect();torch.cuda.empty_cache()
        assert len(records)==plan['probe_count']
        write(out/'summary.json',{'status':'PASS_POST_TRAIN_GRADIENT_REPLAY','plan_sha256':PLAN_SHA,
            'script_sha256':sha(__file__),'protocol_sha256':plan['protocol_sha256'],'lambda_ce':plan['lambda_ce'],
            'optimizer_attempts':0,'optimizer_updates':0,'val_inference':0,'test_inference':0,
            'probe_count':len(records),'gradient_definition':plan['gradient_definition'],'probes':records})
        print('PASS_POST_TRAIN_GRADIENT_REPLAY: 12 probes; zero optimizer/VAL/TEST')
    except Exception:
        write(out/'failure.json',{'status':'POST_TRAIN_DIAGNOSTIC_STOPPED','completed_probes':len(records),
            'optimizer_attempts':0,'optimizer_updates':0,'error':traceback.format_exc()})
        raise
    finally:
        if trainer:trainer.close()
if __name__=='__main__':main()


```
