"""Strict native-row A/B/N planning and real, sparse-N provenance bindings.

The unchanged tools.exp17.pair_loss.source_ranking computes the actual loss.
This module never samples images, creates fake B, changes losses or runs models.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from tools.exp17.engineering.data import digest, require


def hash64(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class EngineeringBinding:
    a_row:int
    n_row:int
    group:str
    class_id:int
    polygon_id:str
    source_id:str
    a_slot:int
    n_slot:int
    manifest_record_sha256:str
    source_sha256:str
    repair_sha256:str
    matrix_sha256:str
    valid_mask_sha256:str
    augmentation_sha256:str
    a_tensor_sha256:str
    n_tensor_sha256:str
    repair_version:str
    support_complete:bool=True

    @property
    def identity(self):return digest(asdict(self))


@dataclass(frozen=True)
class PlannedAnchor:
    a_row:int
    a_slot:int
    source_id:str
    polygon_id:str
    group:str
    class_id:int
    augmentation_sha256:str
    trusted_record:dict
    b_rows_by_group:dict


@dataclass(frozen=True)
class PairPlan:
    anchors:tuple
    native_rows:tuple
    n_slots:dict
    coverage:dict


def plan_pairs(native_rows,index):
    """Rows are exactly J1's surviving cells>0 embedding rows, in that order.

    Required row fields: row_index,slot,source_id,group_id,polygon_id,class_id,
    annotation_ordinal (or original native_label_ordinal), p3_cells. The slot's
    post-Format mask ordinal is separate and must not replace annotation_ordinal.
    Optional augmentation_sha256/trace_sha256 becomes mandatory before binding.
    """
    rows=[dict(r) for r in native_rows]
    require([r['row_index'] for r in rows]==list(range(len(rows))),'A embedding-row order is not native cells>0 order')
    by_slot={};seen=set();by_class=defaultdict(list)
    for row in rows:
        require(type(row['slot']) is int and row['slot']>=0 and type(row['class_id']) is int
                and int(row['p3_cells'])>0,'Expected valid original native region rows only')
        index.validate_native_region(row)
        require((row['slot'],row['polygon_id']) not in seen,'Duplicate native Polygon row within a slot')
        seen.add((row['slot'],row['polygon_id']))
        require(row['slot'] not in by_slot or by_slot[row['slot']]==row['source_id'],'One native image slot contains different source images')
        by_slot[row['slot']]=row['source_id'];by_class[row['class_id']].append(row)
    anchors=[];n_slots={};skip=Counter();classes=defaultdict(Counter)
    for row in rows:
        cls=row['class_id'];classes[cls]['native_regions']+=1
        trusted=index.by_polygon.get(row['polygon_id'])
        if trusted is None:
            skip['no_admitted_N']+=1;classes[cls]['skip_no_N']+=1;continue
        require(trusted['record_sha256']==digest({k:v for k,v in trusted.items() if k!='record_sha256'}),
                'Admitted manifest record mutated after loading')
        require((trusted['source_id'],trusted['group_id'],trusted['class_id'])==(row['source_id'],row['group_id'],cls),
                'Admitted N is bound to another native Polygon identity')
        classes[cls]['admitted_A']+=1
        b_groups=defaultdict(list)
        for other in by_class[cls]:
            if other['group_id']!=row['group_id']:b_groups[other['group_id']].append(other['row_index'])
        if not b_groups:
            skip['no_cross_source_same_class_B']+=1;classes[cls]['skip_no_B']+=1;continue
        trace=row.get('augmentation_sha256',row.get('trace_sha256',''))
        anchor=PlannedAnchor(row['row_index'],row['slot'],row['source_id'],row['polygon_id'],row['group_id'],
            cls,trace,trusted,{k:tuple(v) for k,v in sorted(b_groups.items())})
        anchors.append(anchor);classes[cls]['planned_ABN']+=1
        slot_key=(trusted['source_id'],trusted['n_sha256'],trusted['n_review_id'],trusted['n_variant_id'])
        require(row['slot'] not in n_slots or n_slots[row['slot']]['identity']==slot_key,
                'One N slot acquired different repaired versions')
        n_slots[row['slot']]={'identity':slot_key,'source_id':row['source_id'],'n_sha256':trusted['n_sha256'],
                              'trusted_record':trusted}
    coverage={'native_regions':len(rows),'admitted_A':sum(v['admitted_A'] for v in classes.values()),
        'planned_ABN':len(anchors),'distinct_B_region_rows':len({i for a in anchors for g in a.b_rows_by_group.values() for i in g}),
        'independent_A_groups':len({a.group for a in anchors}),'N_image_slots':len(n_slots),
        'skip':dict(skip),'by_class':{str(k):dict(v) for k,v in sorted(classes.items())},
        'main_sampler_modified':False,'B_scope':'CURRENT_NATIVE_BATCH_ALL_REAL_ORIGINAL_TRAIN_REGIONS',
        'B_distance_reduction':'mean_independent_B_group(mean_regions_distance)',
        'anchor_reduction':'mean_class(mean_A_group(mean_polygon(hinge)))'}
    return PairPlan(tuple(anchors),tuple(rows),n_slots,coverage)


def bind_plan(plan,replay_records):
    """Bind actual replayed, full-supported sparse N rows; model checks tensor bytes.

    replay_records is keyed by A embedding-row index. Every admitted record must
    contain n_row (sparse N cells>0 embedding order), n_slot (sparse image slot),
    frozen source/repair/matrix/valid hashes, trace hash, actual A/N tensor hashes,
    source_id,polygon_id and support_complete. False support is skipped; wrong
    identities are hard errors. No extra N is created for skipped/no-B anchors.
    """
    require(isinstance(replay_records,dict),'Replay provenance must be keyed by native A embedding row')
    planned={a.a_row:a for a in plan.anchors}
    require(set(replay_records)<=set(planned),'Replay introduced N for an unplanned/unadmitted native Anchor')
    bindings=[];skipped=Counter();n_rows=set();slot_map={}
    for a in plan.anchors:
        record=replay_records.get(a.a_row)
        if record is None:
            skipped['missing_replayed_N']+=1;continue
        t=a.trusted_record
        require(record['source_id']==a.source_id and record['polygon_id']==a.polygon_id,'Replayed N Polygon/source identity differs')
        expected={'source_sha256':t['a_sha256'],'repair_sha256':t['n_sha256'],
                  'matrix_sha256':t['matrix_identity_sha256'],'valid_mask_sha256':t['source_valid_mask_sha256']}
        require(all(record.get(k)==v for k,v in expected.items()),'Replayed source/N/matrix/Valid identity differs from Data Gate')
        require(hash64(a.augmentation_sha256) and record.get('augmentation_sha256')==a.augmentation_sha256,
                'N did not replay the actual A augmentation trace')
        require(all(hash64(record.get(k)) for k in ['a_tensor_sha256','n_tensor_sha256']),
                'Actual augmented A/N tensor identities are missing')
        require(type(record['support_complete']) is bool,'Unknown replay support cannot be accepted')
        if not record['support_complete']:
            skipped['incomplete_augmented_target_valid_support']+=1;continue
        require(type(record['n_row']) is type(record['n_slot']) is int and min(record['n_row'],record['n_slot'])>=0,
                'Sparse N embedding/image indexes differ')
        require(record['n_row'] not in n_rows,'Two A Polygons were incorrectly bound to one sparse N region row')
        n_rows.add(record['n_row'])
        nslot=record['n_slot'];signature=(a.a_slot,a.source_id,t['n_sha256'])
        require(nslot not in slot_map or slot_map[nslot]==signature,'A sparse N image slot mixed original sources or repair versions')
        slot_map[nslot]=signature
        bindings.append(EngineeringBinding(a.a_row,record['n_row'],a.group,a.class_id,a.polygon_id,a.source_id,
            a.a_slot,nslot,t['record_sha256'],t['a_sha256'],t['n_sha256'],t['matrix_identity_sha256'],
            t['source_valid_mask_sha256'],a.augmentation_sha256,record['a_tensor_sha256'],record['n_tensor_sha256'],t['repair_version']))
    by_class=Counter(b.class_id for b in bindings)
    return bindings,{**plan.coverage,'bound_ABN':len(bindings),'bound_N_image_slots':len(slot_map),
        'bound_independent_A_groups':len({b.group for b in bindings}),'bound_by_class':{str(k):v for k,v in sorted(by_class.items())},
        'replay_skip':dict(skipped),'binding_identity_sha256':digest([b.identity for b in bindings]),
        'binding_row_domains':{'a_row':'MAIN_J1_CELLS_GT0_EMBEDDING_ORDER','n_row':'SPARSE_N_CELLS_GT0_EMBEDDING_ORDER'}}


class CoverageAccumulator:
    """Online metadata-only coverage over an unchanged native batch trajectory."""
    def __init__(self):
        self.batches=0;self.totals=Counter();self.by_class=defaultdict(Counter);self.group_frequency=Counter()

    def add(self,plan,bindings=None):
        self.batches+=1
        for key in ['native_regions','admitted_A','planned_ABN','N_image_slots']:
            self.totals[key]+=plan.coverage[key]
        for key,value in plan.coverage['skip'].items():self.totals['skip_'+key]+=value
        for cls,counts in plan.coverage['by_class'].items():self.by_class[cls].update(counts)
        participants=bindings if bindings is not None else plan.anchors
        self.group_frequency.update(b.group for b in participants)
        self.totals['actual_bound_ABN' if bindings is not None else 'planned_without_forward']+=len(participants)

    def report(self):
        return {'batches':self.batches,'totals':dict(self.totals),'by_class':{k:dict(v) for k,v in sorted(self.by_class.items())},
            'group_participation_frequency':dict(sorted(self.group_frequency.items())),
            'group_frequency_counts_are_anchor_exposures_not_independent_samples':True,
            'sampler_modified':False,'model_updates':0}
