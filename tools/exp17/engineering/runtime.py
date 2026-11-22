"""J1-compatible setup and sparse, explicitly replayed 521-Pair batch assembly.

No formal-training entry is provided. Root orchestration owns all GPU runs,
calibration gates, durable update budgets and final checkpoint auditing.
"""
import copy
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.exp14.formal_e0_e1 import hash_state
from tools.exp15.ja import JATrainer, checked_train
from tools.exp15.regions import Regions, inspect_regions
from tools.exp17.pair_loss import tensor_identity
from tools.exp17.engineering.augmentation import IndexIdentityResolver, install_native_replay, replay_normal, support_from_trace
from tools.exp17.engineering.batch_pairing import PairPlan, bind_plan, plan_pairs
from tools.exp17.engineering.data import TrustedPairIndex, load, require, sha

INITIAL_NATIVE='8e0479ecdae6dc60558ed44d5d62c01f8b61075514bc34c109295950603fc886'
INITIAL_PROJECTION='83664f93724017b43a2861826141fc41074a0ae58729d170829f787395d1fc6e'


def value(config,key,default=None):
    return config.get(key,default) if isinstance(config,dict) else getattr(config,key,default)


def runtime_asset_map(config):
    supplied=value(config,'runtime_assets',{})
    return load(supplied) if isinstance(supplied,(str,Path)) else dict(supplied)


class EngineeringTrainer(JATrainer):
    def __init__(self,config,output):
        self.engineering_config=config
        require(value(config,'formal_training_authorized',False) is False,'This runtime cannot authorize formal training')
        self.engineering_limit=int(value(config,'smoke_update_limit',50))
        require(0<self.engineering_limit<=50,'Engineering optimizer update limit exceeds authorization')
        release=Path(value(config,'release')).resolve(strict=True)
        self.index=TrustedPairIndex.from_files(Path(value(config,'manifest')),Path(value(config,'native_identity')),
            expected_manifest_sha256=value(config,'manifest_sha256'),runtime_assets=runtime_asset_map(config),
            native_dataset_root=release/'dataset')
        groups=checked_train(release)
        identity={'scope':'EXP17_J2_ENGINEERING_ONLY','trusted_manifest_sha256':value(config,'manifest_sha256'),
            'runtime_source_sha256':sha(__file__),'formal_training_authorized':False}
        super().__init__(release,Path(value(config,'weight')).resolve(strict=True),Path(output),'j1',groups,identity)

    def setup(self):
        super().setup()
        reference_path=value(self.engineering_config,'j1_reference_summary')
        reference=load(reference_path) if reference_path else load(Path(__file__).resolve().parents[3]/'results/exp15_jc_formal_seed42/j1_summary.json')
        require(reference['initial_native_sha256']==INITIAL_NATIVE and reference['projection_before']==INITIAL_PROJECTION,
                'J1 reference initial identity differs from accepted seed42 experiment')
        self.projection_initial=hash_state({n:p for n,p in self.model.named_parameters() if n.startswith('region_projection.')})
        require(self.initial_native==INITIAL_NATIVE and self.projection_initial==INITIAL_PROJECTION,
                'Fresh COCO main model or shared projection initialization differs from J1 seed42')
        require(len(self.train_loader)==21 and self._get_warmup_iterations(21)==63,'Native schedule changed')
        install_native_replay(self.train_loader.dataset,IndexIdentityResolver(self.index))
        from tools.exp14.a11_smoke_numerics import NumericalAudit
        self.audit.close();self.audit=NumericalAudit(self,update_limit=self.engineering_limit)
        self.event('exp17_engineering_setup',manifest_sha256=self.index.manifest_sha256,
            initial_native_sha256=self.initial_native,projection_initial_sha256=self.projection_initial,
            original_train_images=668,original_instances=1266,formal_training_authorized=False)
        return self

    def run_formal(self,*_args,**_kwargs):
        raise RuntimeError('Exp17 engineering cannot run formal training; independent Training Protocol Gate required')

    def train(self,*_args,**_kwargs):
        raise RuntimeError('Use independently gated bounded engineering orchestration, never trainer.train()')


def setup_trainer(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    trainer=EngineeringTrainer(config,output)
    try:return trainer.setup()
    except Exception:
        trainer.close();raise


class DecodedAssetCache:
    """Content-bound bounded CPU cache; no RNG and no model-image exposure."""
    def __init__(self,max_bytes=512*1024*1024):
        self.max_bytes=max_bytes;self.items=OrderedDict();self.bytes=0;self.hits=0;self.decodes=Counter()

    def read(self,path,expected_sha,kind):
        path=Path(path).resolve(strict=True);stat=path.stat();key=(str(path),expected_sha,kind)
        stamp=(stat.st_size,stat.st_mtime_ns)
        if key in self.items:
            saved_stamp,array=self.items.pop(key)
            if saved_stamp==stamp:
                self.items[key]=(saved_stamp,array);self.hits+=1;return array
            self.bytes-=array.nbytes
        require(sha(path)==expected_sha,'A frozen image/Valid changed before replay decode')
        mode=cv2.IMREAD_COLOR if kind=='N' else cv2.IMREAD_GRAYSCALE
        array=cv2.imread(str(path),mode)
        require(array is not None and array.dtype==np.uint8,'Cannot decode frozen replay asset')
        if kind!='N':require(np.isin(array,[0,1,255]).all(),'Frozen replay validity is not binary')
        self.decodes[kind]+=1
        if array.nbytes<=self.max_bytes:
            while self.items and self.bytes+array.nbytes>self.max_bytes:
                _,(_,removed)=self.items.popitem(last=False);self.bytes-=removed.nbytes
            self.items[key]=(stamp,array);self.bytes+=array.nbytes
        return array

    def report(self):
        return {'cache_bytes':self.bytes,'cache_hits':self.hits,'asset_decodes':dict(self.decodes),'cache_limit_bytes':self.max_bytes}


DEFAULT_ASSET_CACHE=DecodedAssetCache()


@dataclass
class PreparedBatch:
    batch:dict
    a_regions:object
    plan:PairPlan
    coverage:dict
    normal_images:object=None
    normal_masks:object=None
    normal_regions:object=None
    bindings:tuple=()
    trace:object=None


def native_region_rows(batch,index,regions):
    traces=batch.get('j2_replay')
    require(traces is not None and len(traces)==len(batch['im_file']),'Native ReplayTrace is missing from the batch')
    rows=[];full_indexes=[]
    for original_index in np.flatnonzero(regions.cells>0):
        slot=int(regions.slots[original_index]);ordinal=int(regions.ordinals[original_index]);trace=traces[slot]
        require(index.source_for_image(batch['im_file'][slot])==trace.native_target_rows[ordinal]['source_id'],
                'Replay source does not match its exact original image slot')
        native=trace.native_target_rows[ordinal]
        require(int(native['native_ordinal'])==ordinal and int(native['mask_id'])==ordinal+1,
                'Post-Format native ordinal or overlap mask identity differs')
        require(int(native['class_id'])==int(regions.classes[original_index]) and native['group_id']==str(regions.groups[original_index]),
                'Original J1 region class/group differs from ReplayTrace identity')
        rows.append({'row_index':len(rows),'slot':slot,'source_id':native['source_id'],'group_id':native['group_id'],
            'polygon_id':native['polygon_id'],'class_id':int(native['class_id']),
            'annotation_ordinal':int(native['annotation_ordinal']),'native_label_ordinal':int(native['native_label_ordinal']),
            'native_post_format_ordinal':ordinal,'p3_cells':int(regions.cells[original_index]),'augmentation_sha256':trace.sha256})
        full_indexes.append(int(original_index))
    return rows,full_indexes


def normalize_batch(batch,device):
    # The inherited native preprocessing is deterministic /255 with multi-scale
    # disabled. Preserve every original field; added replay metadata stays CPU.
    result={k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
    image=result['img']
    if image.dtype==torch.uint8:result['img']=image.float()/255.
    else:require(image.dtype==torch.float32 and bool(torch.isfinite(image).all()) and float(image.min())>=0 and float(image.max())<=1,
                 'Already normalized native image dtype/range differs')
    return result


def prepare_batch(batch,index,groups,*,decode_N=False,device=None,asset_cache=None):
    """Native CPU batch→audited sparse Pair pack, or metadata-only coverage.

    decode_N=False never decodes N pixels: only Valid masks are replayed. This
    lets coverage preserve the six native tensors and all RNG streams exactly.
    True returns normalized float32 A/N tensors and actual tensor byte bindings.
    """
    cache=asset_cache if asset_cache is not None else DEFAULT_ASSET_CACHE
    n_decodes_before=int(getattr(cache,'decodes',{}).get('N',0))
    require(batch['img'].device.type=='cpu' and batch['masks'].device.type=='cpu',
            'Prepare from the unmodified native CPU batch before preprocessing')
    regions=inspect_regions(batch,groups)
    rows,full_indexes=native_region_rows(batch,index,regions)
    raw_plan=plan_pairs(rows,index);traces=batch['j2_replay'];support_by_slot={}
    for slot,slot_plan in raw_plan.n_slots.items():
        trusted=slot_plan['trusted_record']
        require(tuple(traces[slot].original_hw)==tuple(trusted['source_size'][::-1]),
                'A native source frame differs from its admitted Aligned coordinate system')
        sv=cache.read(index.asset_path(trusted,'source_valid'),trusted['source_valid_mask_sha256'],'source_valid')
        nv=cache.read(index.asset_path(trusted,'n_valid'),trusted['n_valid_mask_sha256'],'n_valid')
        require(sv.shape==nv.shape==tuple(traces[slot].original_hw),'A/N original-frame Valid dimensions differ')
        combined=np.ascontiguousarray(((sv>0)&(nv>0)).astype(np.uint8)*255)
        support=support_from_trace(combined,traces[slot])
        support_by_slot[slot]={'combined_valid':combined,'support':{r['native_ordinal']:r for r in support['support']}}
    valid=[];rejected=[]
    for anchor in raw_plan.anchors:
        native_ordinal=rows[anchor.a_row]['native_post_format_ordinal']
        support=support_by_slot[anchor.a_slot]['support'][native_ordinal]
        require(support['polygon_id']==anchor.polygon_id,'Valid support attached to another augmented Polygon')
        (valid if support['support_complete'] else rejected).append(anchor)
    active_slots=sorted({a.a_slot for a in valid})
    classes=copy.deepcopy(raw_plan.coverage['by_class'])
    for a in rejected:
        cls=classes[str(a.class_id)];cls['planned_ABN']-=1;cls['skip_augmented_valid_support']=cls.get('skip_augmented_valid_support',0)+1
    skip=dict(raw_plan.coverage['skip']);skip['augmented_valid_support_incomplete']=len(rejected)
    coverage={**raw_plan.coverage,'pre_valid_support_planned_ABN':len(raw_plan.anchors),'planned_ABN':len(valid),
        'actual_valid_ABN':len(valid),'valid_A_rows':[a.a_row for a in valid],'valid_N_slots':active_slots,
        'validArows':[a.a_row for a in valid],'validNslots':active_slots,
        'distinct_B_region_rows':len({i for a in valid for g in a.b_rows_by_group.values() for i in g}),
        'N_image_slots':len(active_slots),'independent_A_groups':len({a.group for a in valid}),
        'skip':skip,'by_class':classes,'N_images_decoded_this_call':0,'N_image_tensors_prepared_this_call':0,
        'native_region_rows':[{'a_row':r['row_index'],'source_id':r['source_id'],'polygon_id':r['polygon_id'],
            'group_id':r['group_id'],'class_id':r['class_id'],'slot':r['slot']} for r in rows]}
    plan=replace(raw_plan,anchors=tuple(valid),n_slots={s:raw_plan.n_slots[s] for s in active_slots},coverage=coverage)
    if not decode_N:
        coverage['asset_cache']=cache.report()
        return PreparedBatch(batch,regions,plan,coverage,trace=traces)
    target_device=torch.device(device) if device is not None else torch.device('cpu')
    native=normalize_batch(batch,target_device)
    if not valid:return PreparedBatch(native,regions,plan,coverage,trace=traces)
    compact={a_slot:n_slot for n_slot,a_slot in enumerate(active_slots)}
    n_images=[];n_masks=[]
    for slot in active_slots:
        trusted=plan.n_slots[slot]['trusted_record']
        image=cache.read(index.asset_path(trusted,'n'),trusted['n_sha256'],'N')
        replayed=replay_normal(image,support_by_slot[slot]['combined_valid'],traces[slot])
        require(torch.equal(replayed['masks'][0],batch['masks'][slot]),'Sparse N replay changed the original instance mask IDs')
        n_images.append(replayed['img']);n_masks.append(replayed['masks'][0])
    normal_images=torch.stack(n_images).to(target_device,non_blocking=True).float()/255.
    normal_masks=torch.stack(n_masks).to(target_device,non_blocking=True)
    selected=np.array([i for i,s in enumerate(regions.slots) if int(s) in compact],dtype=np.int64)
    normal_regions=Regions(np.array([compact[int(regions.slots[i])] for i in selected],dtype=np.int64),
        regions.ordinals[selected].copy(),regions.classes[selected].copy(),regions.groups[selected].copy(),
        regions.cells[selected].copy(),regions.foreground_fraction[selected].copy(),
        max(int(np.count_nonzero(regions.slots==s)) for s in active_slots))
    n_row_lookup={int(original):nrow for nrow,original in enumerate(selected[normal_regions.cells>0])}
    a_hashes={s:tensor_identity(native['img'][s]) for s in active_slots}
    n_hashes={n:tensor_identity(normal_images[n]) for n in range(len(active_slots))}
    replay_records={}
    for anchor in valid:
        t=anchor.trusted_record;n_slot=compact[anchor.a_slot]
        replay_records[anchor.a_row]={'n_row':n_row_lookup[full_indexes[anchor.a_row]],'n_slot':n_slot,
            'source_id':anchor.source_id,'polygon_id':anchor.polygon_id,'source_sha256':t['a_sha256'],
            'repair_sha256':t['n_sha256'],'matrix_sha256':t['matrix_identity_sha256'],
            'valid_mask_sha256':t['source_valid_mask_sha256'],'augmentation_sha256':anchor.augmentation_sha256,
            'a_tensor_sha256':a_hashes[anchor.a_slot],'n_tensor_sha256':n_hashes[n_slot],'support_complete':True}
    bindings,bound_coverage=bind_plan(plan,replay_records)
    coverage={**coverage,**bound_coverage,
        'N_images_decoded_this_call':int(getattr(cache,'decodes',{}).get('N',0))-n_decodes_before,
        'N_image_tensors_prepared_this_call':len(active_slots),'asset_cache':cache.report()}
    return PreparedBatch(native,regions,plan,coverage,normal_images,normal_masks,normal_regions,tuple(bindings),traces)
