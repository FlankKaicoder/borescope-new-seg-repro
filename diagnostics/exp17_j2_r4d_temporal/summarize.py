"""Join saved tensor geometry with existing R3 tables. NumPy only, no Torch."""
from pathlib import Path
import csv,hashlib,json,os
import numpy as np

REPO=Path(__file__).resolve().parents[2]
PRIVATE=REPO/'private_assets/exp17_j2_r4d_20261124'
OUT=REPO/'results/exp17_j2_r4d_temporal'


def main():
    OUT.mkdir(exist_ok=False)
    sources={}
    def load(p):
        p=Path(p);sources[str(p.relative_to(REPO))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text(encoding='utf8'))
    def read(p):
        p=Path(p);sources[str(p.relative_to(REPO))]=hashlib.sha256(p.read_bytes()).hexdigest()
        with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
    def save(n,d):(OUT/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf8')
    e=PRIVATE/'evidence';r=load(e/'direction_alignment.json');rad=load(e/'parameter_radial_alignment.json')
    completion=load(e/'completion.json');meta=load(e/'request.json')
    assert completion['checkpoints']==22 and not any(completion['forbidden_calls'].values())
    opt=read(REPO/'private_assets/exp17_j2_r3_20261124/analysis_v2/optimizer_bn_associations.csv')
    opt=[x for x in opt if x['label'].startswith('J2_epoch')]
    conv=read(REPO/'results/exp17_j2_r3_bn_drift/conv_layer_changes.csv')
    checks=0
    def domain(n,d):
        if d=='all':return True
        if d=='projection':return n.startswith('region_projection.')
        if d=='conv_weights':return n.endswith('.conv.weight')
        if d=='BN_gamma':return n.endswith('.bn.weight')
        if d=='BN_beta':return n.endswith('.bn.bias')
        if d=='backbone_0_10':return n.startswith('model.') and int(n.split('.')[1])<=10
        return n.startswith('model.'+d.removeprefix('neck')+'.')
    for row in r:
        if not row['label'].startswith('J2_epoch'):continue
        if row['kind']=='saved_adaptive_direction_proxy':
            selected=[x for x in opt if int(float(x['epoch']))==row['epoch'] and domain(x['parameter'],row['domain'])]
            expected=np.linalg.norm([float(x['adaptive_l2_proxy']) for x in selected])
            np.testing.assert_allclose(row['x_L2'],expected,rtol=2e-12,atol=1e-12);checks+=1
        if row['kind']=='actual_interval_displacement' and row['domain']=='conv_weights':
            selected=[x for x in conv if int(x['epoch'])==row['epoch']]
            expected=np.linalg.norm([float(x['weight_delta_l2_l2_across_channels']) for x in selected])
            np.testing.assert_allclose(row['x_L2'],expected,rtol=2e-12,atol=1e-12);checks+=1
    aggregates=[]
    for d in ('all','backbone_0_10','neck13','neck16','conv_weights','BN_gamma','BN_beta','projection'):
        for kind in ('actual_interval_displacement','saved_adaptive_direction_proxy','saved_negative_moment'):
            for key in ('A_pair4','B_pair4','omitted_pair4','A_yolo','A_supcon8'):
                selected=[x for x in r if x['label'].startswith('J2_') and x['domain']==d and x['kind']==kind and x['endpoint_reference']==key]
                values=[x['cosine'] for x in selected if x['cosine'] is not None]
                aggregates.append({'domain':d,'kind':kind,'endpoint_reference':'detached_excess' if key=='omitted_pair4' else key,
                    'n':len(values),'positive':sum(x>0 for x in values),'median':float(np.median(values)) if values else None,
                    'min':min(values) if values else None,'max':max(values) if values else None,
                    'values':[{'epoch':x['epoch'],'previous_update':x['previous_update'],'cosine':x['cosine']} for x in selected]})
    selected_names=['model.0.conv.weight','model.4.cv1.conv.weight','model.13.cv1.conv.weight',
                    'model.16.cv1.conv.weight','model.16.cv2.conv.weight']
    endpoint=[{**x,'reference':'detached_excess' if x['reference']=='omitted_pair4' else x['reference']}
              for x in rad if x['update']==2100 and x['parameter'] in selected_names]
    radial_proxy=[]
    for n in selected_names:
        rows=[x for x in opt if x['parameter']==n]
        for x in rows:
            # Positive means outward in the parameter direction, from the saved-state proxy only.
            a=float(x['adaptive_l2_proxy']);cos=float(x['adaptive_theta_cosine']);dec=float(x['current_theta_decay_l2_proxy'])
            radial_proxy.append({'epoch':int(float(x['epoch'])),'parameter':n,'theta_L2':float(x['theta_l2']),
                'negative_adaptive_outward_component':-a*cos,'decay_inward_component':dec,
                'net_proxy_outward_component':-a*cos-dec,'lr':float(x['lr']),
                'not_actual_next_update':True})
    definitions={'Graph_A':'production detached replay_a at final checkpoint, fixed TRAIN batch1, FP32',
                 'Graph_B':'same forward with connected A-statistics surrogate at final checkpoint',
                 'added_statistic_gradient':'s = gB - gA',
                 'detached_excess':'d = gA - gB = -s; private raw key omitted_pair4',
                 'direction_reference':'-gA, -gB, -d; compare with real historical parameter displacement',
                 'cosine_interpretation':'descriptive alignment, not historical gradients, causal attribution or drift-explained fraction',
                 'saved_moment_proxy':'post-step bias-corrected AdamW adaptive direction at saved LR; mixed clipped history, not loss-decomposable',
                 'radial_proxy':'positive outward projection onto current saved parameter; not BN input translation',
                 'batch_mismatch':'final optimizer state ends at batch21; R4-C gradient uses position1'}
    save('direction_summary.json',{'definitions':definitions,'aggregates':aggregates})
    save('endpoint_radial.json',{'definitions':definitions,'rows':endpoint})
    save('saved_optimizer_radial.json',radial_proxy)
    save('offline_verification.json',{'passed':True,'R3_independent_magnitude_matches':checks,
        'reader_completion':completion,'8_arithmetic_fixtures':True,'new_forward':0,'new_Autograd':0,
        'optimizer_update':0,'VAL':0,'TEST':0,'first_reader_failure':
        'Frozen DFL parameter lacks AdamW state; first reader stopped before outputs. Second reader reads only the exact294 trainable names. Both versions/logs preserved.'})
    for p,s in sources.items():assert hashlib.sha256((REPO/p).read_bytes()).hexdigest()==s
    save('input_manifest.json',{'inputs':sources,'source_commit':'93ece1d9aed59c8428a40f3894d4c6b8f2448b77',
        'checkpoint_bindings':meta['checkpoints'],'all_inputs_unchanged':True,
        'private_remote_archive':json.loads((PRIVATE/'archive_receipt.json').read_text(encoding='utf8'))})
    # Scientific figures derived only from the saved tables above.
    os.environ['MPLCONFIGDIR']=str(PRIVATE/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    colors={'A_pair4':'#b35232','B_pair4':'#2977a0','detached_excess':'#715f9f'}
    for ax,d in zip(axes,('all','neck16')):
        for key in colors:
            row=next(x for x in aggregates if x['domain']==d and x['kind']=='actual_interval_displacement' and x['endpoint_reference']==key)
            ax.plot([x['epoch'] for x in row['values']],[x['cosine'] for x in row['values']],label=key,color=colors[key])
        ax.axhline(0,color='gray',linewidth=.7);ax.set(title=d,xlabel='End epoch of saved 5-epoch interval',ylabel='Cosine with negative endpoint reference')
    axes[0].legend(fontsize=8);fig.savefig(OUT/'01_historical_direction_alignment.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for n in ('model.16.cv1.conv.weight','model.16.cv2.conv.weight'):
        rows=[x for x in radial_proxy if x['parameter']==n]
        axes[0].plot([x['epoch'] for x in rows],[x['theta_L2'] for x in rows],label=n)
        axes[1].plot([x['epoch'] for x in rows],[x['net_proxy_outward_component'] for x in rows],label=n)
    axes[0].set(xlabel='Saved epoch',ylabel='Conv weight L2',title='Observed weights')
    axes[1].set(xlabel='Saved epoch',ylabel='Signed radial proxy',title='Saved AdamW direction minus decay')
    axes[0].legend(fontsize=8);axes[1].axhline(0,color='gray',linewidth=.7)
    fig.savefig(OUT/'02_neck16_scale_and_optimizer.png',dpi=160);plt.close(fig)
    print(json.dumps({'status':'R4_D_DIRECTION_SUMMARIES_COMPLETE','checks':checks}))


if __name__=='__main__':main()
