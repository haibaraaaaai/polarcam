"""Controlled October comparisons and a synchronized detector-overlay viewer.

Offline scientific processing only. Run with repository Python; --render-only
reuses numerical results. Original recordings and previous analyses are read-only.
"""
from pathlib import Path
import hashlib
import json
import sys
import warnings
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from first_hinges_theta import load_module, cliques, clean_json
from local_spot_check import filtering, local_metrics
from spot_detection_example import display20
from pipeline_audit import import_file, radius, theta, extract_mean, OLD_ABC, GRID, PACKAGE

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]
OUT=BASE/'october_pipeline_check'
EXAMPLE='154730'
METHODS={
    'old5': 'Supplied detector · 5 frames',
    'old150': 'Supplied detector · 150-frame mean',
    'native5': 'Native detector · 5 frames',
    'native150': 'Native detector · 150-frame mean',
    'native_local': 'Native 150 + local contrast ≥8',
}
COLORS=['#0072b2','#be522b','#252a2d']
PANELS=[
 ('A · Averaging', ['old5_5med7','old5_5mean7','old5_150mean7'], ['Median radius · 5 frames','Mean channels · 5 frames','Mean channels · 150 frames'], True),
 ('B · Extraction window', ['old5_150mean7','old5_150mean8','old5_150mean18'], ['7 × 7 pixels','8 × 8 pixels','18 × 18 pixels'], True),
 ('C · Detector / local cutoff', ['old150_150mean18','native150_150mean18','native150_local_150mean18'], ['Supplied detector · 150 frames','Native detector · 150 frames','Native + local contrast ≥8'], False),
 ('D · Optical model', ['old5_150mean7','old5_150mean7_newmodel'], ['Supplied hole + Fresnel','Recent manuscript annular'], True),
 ('E · Legacy brightness gate', ['old5_5med7','old5_gate_5med7'], ['No legacy brightness gate','Legacy gate applied'], False),
 ('F · Complete pipelines', ['old5_gate_5med7','current'], ['Supplied 5-frame recipe','Recent 150-frame recipe'], False),
]


def density(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    if not len(values): return np.zeros_like(GRID)
    return np.exp(-.5*((GRID[:,None]-values[None,:])/4)**2).sum(axis=1)/(len(values)*4*np.sqrt(2*np.pi))


def describe(values):
    from scipy.signal import find_peaks
    values=np.asarray(values);values=values[np.isfinite(values)]
    if not len(values): return dict(n=0,median=None,q25=None,q75=None,modes=[])
    d=density(values);peaks,_=find_peaks(d,prominence=.02*d.max(),distance=40)
    return dict(n=len(values),median=float(np.median(values)),q25=float(np.percentile(values,25)),
                q75=float(np.percentile(values,75)),modes=GRID[peaks].round(1).tolist())


def extract_first5(raw, pts):
    channels=[];background=[]
    for y,x in pts:
        crop=np.asarray(raw[:5,y-3:y+4,x-3:x+4],dtype=np.float32)
        yy,xx=np.ogrid[y-3:y+4,x-3:x+4]
        channels.append(np.stack([crop[:,(yy%2==pr)&(xx%2==pc)].mean(axis=1)
                                   for pr,pc in ((0,0),(0,1),(1,0),(1,1))],axis=-1))
        corner_mask=((yy-y)**2+(xx-x)**2>=16)&((yy-y)**2+(xx-x)**2<=64)
        assert corner_mask.sum()==4
        background.append(np.median(crop.mean(axis=0)[corner_mask]))
    channels=np.asarray(channels);background=np.asarray(background,dtype=np.float32)
    legacy=channels-background[:,None,None]
    common=(legacy[:,:,0]+legacy[:,:,3]>50)&(legacy[:,:,1]+legacy[:,:,2]>50)
    r=radius(channels)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        gated=np.nanmedian(np.where(common,r,np.nan),axis=1)
    return channels,background,common,np.median(r,axis=1),gated


def compute():
    OUT.mkdir(exist_ok=True)
    prior=json.loads((BASE/'theta_comparison/results.json').read_text(encoding='utf-8'))
    current_abc=prior['methods']['ABC']
    detector,detector_path=load_module('october_native','src/polarcam/live/Detection_alg_offline.py')
    old=import_file('october_supplied_detector',PACKAGE/'detector_intensity_reference.py')
    stacks=[];arrays={};example={}
    for previous in [s for s in prior['stacks'] if s['group'] in ('hinges','gold')]:
        stamp=previous['stamp'];group=previous['group']
        path=REPO/'data/hinges/polarcam/2026-10-09'/previous['file']
        before=path.stat();meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8-sig'))
        raw=np.load(path,mmap_mode='r',allow_pickle=False)
        assert len(raw)==151 and meta['actual']['frames']==150 and meta['actual']['phase_marker_appended']
        assert meta['background']['background_subtracted'] and meta['roi']['OffsetX']==meta['roi']['OffsetY']==0
        old5,info5=old.detect_candidates(raw[:5])
        assert len(old5)>0
        mean=np.zeros(raw.shape[1:],float)
        for frame in raw[:150]:mean+=frame
        mean/=150
        # Use an already-computed mean as one detector input image. This avoids
        # a 3 GB float32 stack allocation while changing only detection duration.
        old150,info150=old.detect_candidates(mean.astype(np.float32)[None])
        mean5=np.mean(raw[:5],axis=0)
        f5=filtering(detector.intensity_detection_map(mean5),detector)
        f150=filtering(detector.intensity_detection_map(mean),detector)
        pts_native=f150['centres'][:,::-1]
        np.testing.assert_allclose(pts_native,[[r['y'],r['x']] for r in previous['rows']],rtol=0,atol=1e-12)
        scores=np.array([local_metrics(f150,f150['centres'],j)['local_score'] for j in range(len(pts_native))])
        local_pass=np.isfinite(scores)&(scores>=8)
        np.testing.assert_array_equal(local_pass,[r['local_pass'] for r in previous['rows']])
        ch,bg,common,r_med,r_gated=extract_first5(raw,old5)
        r7=radius(extract_mean(mean,old5,7))
        r8=radius(extract_mean(mean,old5,8))
        r18=radius(extract_mean(mean,old5,18))
        old150_r18=radius(extract_mean(mean,old150,18))
        native18=radius(extract_mean(mean,pts_native,18))
        native7=radius(extract_mean(mean,pts_native,7))
        np.testing.assert_allclose(native18,[r['r'] for r in previous['rows']],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(theta(native18,current_abc),[r['theta_deg'] if r['theta_deg'] is not None else np.nan for r in previous['rows']],atol=1e-11)
        a=dict(old5_points=old5,old150_points=old150,native5_points=f5['centres'][:,::-1],native150_points=pts_native,
               local_pass=local_pass,local_score=scores,legacy_frame_mask=common,legacy_corner_bg=bg,
               old5_legacy_gate=common.any(axis=1),old5_r_median5=r_med,
               old5_r_mean150_7=r7,old5_r_mean150_18=r18,native150_r_mean150_7=native7,native150_r_mean150_18=native18)
        curve_r=dict(old5_5med7=r_med,old5_5mean7=radius(ch.astype(float).mean(axis=1)),
            old5_150mean7=r7,old5_150mean8=r8,old5_150mean18=r18,
            old150_150mean18=old150_r18,native150_150mean18=native18,
            native150_local_150mean18=np.where(local_pass,native18,np.nan),
            old5_150mean7_newmodel=r7,old5_gate_5med7=r_gated,
            current=np.where(local_pass,native18,np.nan))
        for key,r in curve_r.items():
            a['pre_'+key]=theta(r,current_abc if key in ('current','old5_150mean7_newmodel') else OLD_ABC)
        s=dict(stamp=stamp,time=previous['time'],group=group,file=path.name,
            source_size=before.st_size,source_mtime_ns=before.st_mtime_ns,
            source_json_sha256=hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest(),
            old5_info=info5,old150_info=info150,
            counts=dict(old5=len(old5),old150=len(old150),native5=len(f5['centres']),native150=len(pts_native),
                native_local=int(local_pass.sum()),legacy_gate_pass=int(common.any(axis=1).sum())),
            thresholds=dict(native5=float(f5['threshold']),native150=float(f150['threshold'])))
        arrays[stamp]=a;stacks.append(s)
        if stamp==EXAMPLE:
            for name,img in [('first',detector.intensity_detection_map(raw[0])),('mean5',f5['image']),('mean150',f150['image'])]:
                Image.fromarray(display20(img)).save(OUT/f'example_{name}.png')
            example=dict(stamp=stamp,width=raw.shape[2],height=raw.shape[1])
        assert path.stat().st_size==before.st_size and path.stat().st_mtime_ns==before.st_mtime_ns
        print(f"{group} {stamp}: old5={len(old5)}, old150={len(old150)}, native5={len(f5['centres'])}, native150={len(pts_native)}, local={local_pass.sum()}, legacy gate={common.any(axis=1).sum()}",flush=True)
        del raw,mean,mean5,f5,f150
    matches={}
    for group in ('hinges','gold'):
        group_stacks=[s for s in stacks if s['group']==group]
        for method in ('old5','old150','native5','native150'):
            points=[arrays[s['stamp']][method+'_points'] for s in group_stacks]
            found=cliques(points,tuple(range(5)),6)
            matches[f'{group}_{method}']=found
            for s,pts in zip(group_stacks,points):arrays[s['stamp']][method+'_recurring']=np.zeros(len(pts),bool)
            for clique in found:
                for k,j in clique:arrays[group_stacks[k]['stamp']][method+'_recurring'][j]=True
    for s in stacks:
        a=arrays[s['stamp']]
        s['recurring']={key:int(a[key+'_recurring'].sum()) for key in ('old5','old150','native5','native150')}
        for pre_key in [key for key in a if key.startswith('pre_')]:
            key=pre_key[4:]
            method='old5' if key.startswith('old5_') else ('old150' if key.startswith('old150_') else 'native150')
            a['t_'+key]=np.where(a[method+'_recurring'],np.nan,a[pre_key])
        s['distributions']={k[2:]:describe(v) for k,v in a.items() if k.startswith('t_')}
        s['pre_recurrence']={k[4:]:describe(v) for k,v in a.items() if k.startswith('pre_')}
        for name,k1,k2 in [('window7to18','old5_150mean7','old5_150mean18'),('model','old5_150mean7','old5_150mean7_newmodel')]:
            t1,t2=a['t_'+k1],a['t_'+k2];valid=np.isfinite(t1)&np.isfinite(t2)
            s[name]=dict(n=int(valid.sum()),median_delta=float(np.median(t2[valid]-t1[valid])) if valid.any() else None)
        previous=next(p for p in prior['stacks'] if p['stamp']==s['stamp'])
        expected=np.array([r['theta_deg'] if r['included'] else np.nan for r in previous['rows']])
        np.testing.assert_allclose(a['t_current'],expected,atol=1e-11,equal_nan=True)
        np.savez_compressed(OUT/f"stack_{s['stamp']}.npz",**a)
    current=arrays[EXAMPLE];example['methods']={}
    for method in METHODS:
        base_method='native150' if method=='native_local' else method
        pts=current[base_method+'_points'];rows=[]
        for j,(y,x) in enumerate(pts):
            if method=='native_local' and not current['local_pass'][j]:continue
            row=dict(id=j+1,x=float(x),y=float(y),recurring=bool(current[base_method+'_recurring'][j]))
            if base_method=='old5':
                row.update(legacy_gate=bool(current['old5_legacy_gate'][j]),
                    theta7=float(theta(current['old5_r_mean150_7'][j:j+1])[0]),
                    theta18=float(theta(current['old5_r_mean150_18'][j:j+1])[0]))
            elif base_method=='native150':
                row.update(local_score=float(current['local_score'][j]),local_pass=bool(current['local_pass'][j]),
                    theta7=float(theta(current['native150_r_mean150_7'][j:j+1])[0]),
                    theta18=float(theta(current['native150_r_mean150_18'][j:j+1])[0]))
            rows.append(row)
        example['methods'][method]=dict(label=METHODS[method],rows=rows)
    # Review presets are geometric examples, not a representative truth set.
    oldpts=current['old150_points'];native=current['native150_points']
    distance=cKDTree(native).query(oldpts)[0]
    only=np.flatnonzero(distance>6);both=np.flatnonzero(distance<=6)
    example['presets']=[dict(label='Questioned native #15',x=float(native[14,1]),y=float(native[14,0]))]
    for label,inds in [('Supplied-only example',only),('Nearby detections in both',both)]:
        if len(inds):
            j=int(inds[len(inds)//2]);y,x=oldpts[j]
            example['presets'].append(dict(label=label,x=float(x),y=float(y)))
    state=clean_json(dict(stacks=stacks,example=example,matches=matches,old_ABC=OLD_ABC,new_ABC=current_abc,
        scope='Five first-round hinge stacks and five gold-only stacks, 9 October 2026',
        recurrence='Exclude candidates recurring in all five lists within each sample group and detector/duration; every pair within 6 sensor pixels',
        source_hashes={str(p.relative_to(REPO)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),PACKAGE/'detector_intensity_reference.py',detector_path,BASE/'local_spot_check.py')}))
    (OUT/'results.json').write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8')
    return state


def panel_values(stacks,keys,common):
    chunks=[[] for _ in keys]
    for s in stacks:
        with np.load(OUT/f"stack_{s['stamp']}.npz",allow_pickle=False) as a:
            vals=[a['t_'+k] for k in keys]
            mask=np.logical_and.reduce([np.isfinite(v) for v in vals]) if common else None
            for j,v in enumerate(vals):chunks[j].append(v[mask] if common else v[np.isfinite(v)])
    return [np.concatenate(chunk) for chunk in chunks]


def figures(state):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    choices=[(g,[s for s in state['stacks'] if s['group']==g],f"9 October · {'first-round hinges' if g=='hinges' else 'gold-only'} · five stacks pooled") for g in ('hinges','gold')]
    choices += [(s['stamp'],[s],f"9 October {s['time']} · {s['group']}") for s in state['stacks']]
    for name,stacks,title in choices:
        fig,axs=plt.subplots(2,3,figsize=(14,7.8),layout='constrained')
        for ax,(heading,keys,labels,common) in zip(axs.flat,PANELS):
            vals=panel_values(stacks,keys,common)
            ymax=max(float(density(v).max()) for v in vals)
            for v,color,label in zip(vals,COLORS,labels):
                ax.plot(GRID,density(v),color=color,lw=1.8,label=f'{label} · n={len(v)}')
            ax.axvspan(35,45,color='#6c7a80',alpha=.07,lw=0)
            ax.set(title=heading,xlim=(0,90),ylim=(0,max(ymax*1.4,.01)),xticks=[0,30,60,90],xlabel='Effective optical θ (degrees)',ylabel='Density (per degree)')
            ax.legend(fontsize=7.6,frameon=False,loc='upper left')
        fig.suptitle(title+'\n4° smoothing · all-five recurrence veto applied · n = candidates used in each curve',fontsize=13)
        fig.savefig(OUT/f'controls_{name}.png',dpi=155);plt.close(fig)
    fig,axs=plt.subplots(2,5,figsize=(16,6.5),sharex=True,sharey=True,layout='constrained')
    for ax,s in zip(axs.flat,state['stacks']):
        vals=panel_values([s],['old5_gate_5med7','current'],False)
        for v,color,label in zip(vals,COLORS,['Supplied recipe','Recent recipe']):
            ax.plot(GRID,density(v),color=color,lw=1.7,label=f'{label} · n={len(v)}')
        ax.axvspan(35,45,color='#6c7a80',alpha=.07,lw=0)
        ax.set(title=f"{s['group']} · {s['time']}",xlim=(0,90),xticks=[0,30,60,90]);ax.legend(frameon=False,fontsize=7,loc='upper left');ax.margins(y=.28)
    for ax in axs[-1]:ax.set_xlabel('Effective optical θ (degrees)')
    for ax in axs[:,0]:ax.set_ylabel('Density (per degree)')
    fig.suptitle('Complete recipes on each October stack · small blue samples reflect the legacy brightness gate',fontsize=13)
    fig.savefig(OUT/'per_stack.png',dpi=165);plt.close(fig)
    im=np.asarray(Image.open(OUT/'example_mean150.png'))
    fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,method,color in zip(axs,['old150','native150','native_local'],['#19d3eb','#ff9d45','#6feb9a']):
        m=state['example']['methods'][method]
        ax.imshow(im,cmap='gray',vmin=0,vmax=255)
        for r in m['rows']:ax.add_patch(Circle((r['x'],r['y']),14,fill=False,lw=.75,color=color,ls='--' if r['recurring'] else '-'))
        ax.set_title(f"{m['label']}\n{len(m['rows'])} detections",fontsize=10)
        ax.set_xlabel('Sensor x (pixels)');ax.set_ylabel('Sensor y (pixels)')
    fig.suptitle('15:47:30 hinges · same 150-frame detection input · fixed 20× display',fontsize=12)
    fig.savefig(OUT/'spots_overview.png',dpi=180);plt.close(fig)


def page(state):
    example=state['example']
    old_rows=example['methods']['old150']['rows']
    native_rows=example['methods']['native150']['rows']
    tree=cKDTree([[r['x'],r['y']] for r in native_rows])
    unique=[r for r in old_rows if not r['recurring'] and tree.query([r['x'],r['y']])[0]>6]
    recurring=[r for r in old_rows if r['recurring']]
    example['presets']=[p for p in example['presets'] if p['label'] not in ('Supplied-only example','Supplied-only, non-recurring','Recurring in all five')]
    for label,rows in [('Supplied-only, non-recurring',unique),('Recurring in all five',recurring)]:
        if rows:
            r=rows[len(rows)//2];example['presets'].append(dict(label=label,x=r['x'],y=r['y']))
    options=''.join(f'<option value="{g}">{label}</option>' for g,label in [('hinges','First-round hinges · five stacks pooled'),('gold','Gold-only · five stacks pooled')])
    options+=''.join(f'<option value="{s["stamp"]}">{s["group"]} · {s["time"]}</option>' for s in state['stacks'])
    tables=[]
    for s in state['stacks']:
        c=s['counts'];d=s['distributions'];rec=s['recurring']
        items=[s['group'],s['time'],c['old5'],c['old150'],c['native5'],c['native150'],c['native_local'],c['legacy_gate_pass'],
               f"{rec['old5']} / {rec['old150']} / {rec['native150']}",d['old5_gate_5med7']['n'],d['current']['n']]
        tables.append('<tr>'+''.join(f'<td>{v}</td>' for v in items)+'</tr>')
    pools={}
    pool_rows=[]
    for group in ('hinges','gold'):
        stacks=[s for s in state['stacks'] if s['group']==group]
        vals=panel_values(stacks,['old5_gate_5med7','current','old5_5med7'],False)
        pools[group]={k:describe(v) for k,v in zip(['supplied','recent','supplied_without_gate'],vals)}
        for label,desc in pools[group].items():
            pool_rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in (group,label.replace('_',' '),desc['n'],f"{desc['median']:.1f}°" if desc['median'] is not None else '—',', '.join(f'{v:.1f}°' for v in desc['modes'])))+'</tr>')
    html=HTML.replace('__OPTIONS__',options).replace('__ROWS__',''.join(tables)).replace('__POOL_ROWS__',''.join(pool_rows))
    html=html.replace('__EXAMPLE__',json.dumps(state['example'],allow_nan=False))
    (OUT/'index.html').write_text(html,encoding='utf-8')
    (OUT/'pooled_summary.json').write_text(json.dumps(clean_json(pools),indent=2),encoding='utf-8')
    print(json.dumps(pools),flush=True)


HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>October theta: one change at a time</title><style>
body{margin:0;background:#f6f7f5;color:#23313c;font:16px/1.55 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:24px}h1{font-size:29px}h2{font-size:22px;margin-top:32px}p{max-width:1120px}a{color:#0072b2}img{display:block;width:100%;height:auto}select,button{font:inherit;padding:7px;max-width:100%;background:white;border:1px solid #aab8bd;border-radius:4px}button{cursor:pointer}.scroll{overflow:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:8px 12px;text-align:left;border-bottom:1px solid #ccd5d6;white-space:nowrap}th{background:#e9eeee}.note{font-size:14px;color:#536773}.viewgrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.viewgrid section{min-width:0}.viewgrid canvas{display:block;width:100%;height:auto;background:#000;cursor:crosshair}.viewgrid .info{font-size:13px;min-height:65px}.viewgrid select{width:100%;font-size:14px;margin-bottom:8px}.crop{margin-top:12px;image-rendering:pixelated}.buttons{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px}.box{border-left:4px solid #0072b2;background:#eaf1f4;padding:8px 18px}.caption{font-size:13px;color:#566a72}details{margin-top:20px}code{overflow-wrap:anywhere}@media(max-width:800px){.viewgrid{grid-template-columns:1fr}main{padding:15px}}
</style><main><h1>9 October · change one part at a time</h1>
<p>Five first-round hinge stacks and five gold-only stacks. The supplied September algorithm has now been run on these October recordings. Each curve uses one effective θ per candidate; all distributions use the same 4° smoothing.</p>
<p><b>The strongest October effect is the old brightness gate.</b> For the hinges, the supplied recipe has a pooled mode near 69°; removing just its brightness gate gives modes near 33° and 59°. The recent recipe gives modes near 29° and 56°. Changing the optical coefficients alone has very little effect. These are descriptive modes under each selection, not established physical populations.</p>
<div class="box"><p><b>Keep detection separate from θ reconstruction.</b> The detector chooses positions. Averaging, window size and the optical model change the values calculated at those positions. The legacy brightness gate and local-contrast cutoff can additionally remove candidates from the statistics.</p></div>
<h2>Controlled distributions</h2>
<label for="distribution">Show: </label><select id="distribution">__OPTIONS__</select>
<img id="distributionImage" src="controls_hinges.png" alt="Six controlled comparisons for the first-round hinges">
<p class="note"><b>A, B and D:</b> fixed centres from the supplied detector on the first five frames, with the legacy brightness gate switched off to isolate the named change; each panel uses the same model-valid candidates in all its curves. <b>C:</b> both detectors run on the same 150-frame mean; extraction is fixed at 18 × 18 pixels. <b>E:</b> isolate the old brightness gate, keeping the original 7 × 7 / five-frame radius-median calculation. <b>F:</b> the full supplied recipe versus the recent recipe. All panels except the explicit model comparison and recent endpoint use the supplied hole + Fresnel coefficients.</p>
<p class="note">The previous fixed-position rule is applied separately within each five-stack group and each detector/detection-duration combination: detections present in all five, with every pair within 6 sensor pixels, are excluded. Dashed circles in the viewer identify these recurring candidates. Independent detections can still represent the same physical object; pooled counts are candidate observations, not confirmed unique rods.</p>
<p class="note">The more permissive detector finds recurring candidates that the native detector missed: about 31 per first-round hinge stack using five frames, and 41–42 using 150 frames. The corresponding gold lists contain 13–14 and 29–31 recurring candidates. The native lists have none. This identifies fixed-position candidates under your exclusion rule; it does not establish their physical origin.</p>
<h2>Each stack and the full recipes</h2><img src="per_stack.png" alt="Supplied and recent recipes for all ten October stacks">
<p>The old brightness rule requires both channel-pair sums to exceed 50 after subtracting a four-corner scalar estimate, in at least one of the first five frames. The displayed old θ still uses saved channel values without that additional subtraction. This gate removes many October candidates, so its complete-recipe curves can be based on small samples. Panel E shows the resulting selection effect.</p>
<div class="scroll"><table><thead><tr><th>Sample</th><th>Recipe</th><th>θ count</th><th>Median θ</th><th>Smoothed modes</th></tr></thead><tbody>__POOL_ROWS__</tbody></table></div>
<details><summary>Candidate counts and recurrence</summary><div class="scroll"><table><thead><tr><th>Sample</th><th>Time</th><th>Supplied 5</th><th>Supplied 150</th><th>Native 5</th><th>Native 150</th><th>Native + local</th><th>Old gate pass</th><th>Recurring: old5 / old150 / native150</th><th>Supplied θ</th><th>Recent θ</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="note">Detector/local/gate counts precede recurrence and θ-validity exclusions. The final two columns include both. “Supplied 150” applies the same detector to a float32 representation of the 150-frame mean; the original supplied recipe uses five frames.</p></details>
<h2>Which spots change? · 15:47:30 hinges</h2>
<p>Compare three methods on the same image. Click any overview or enlarged crop to move all three inspection windows together. The default compares the detectors on identical 150-frame input; the menus also expose five-frame detection.</p>
<p>Your previously questioned native candidate <b>#15</b> is close to a position found in all five hinge stacks by the supplied detector. It is omitted from the native + local list, and the corresponding recurring supplied detections are also excluded from θ statistics. This supports keeping that location flagged, without establishing whether the structure is background or a real object.</p>
<div class="controls"><label>Display image: <select id="imageMode"><option value="mean150">150-frame mean</option><option value="mean5">First-five mean</option><option value="first">First frame</option></select></label><label><input id="showIds" type="checkbox"> Show IDs in overview</label><label><input id="showBoxes" type="checkbox" checked> 7 / 18-pixel windows in crop</label></div>
<div id="presets" class="buttons"></div><p id="location" class="caption"></p>
<div class="viewgrid">''' + ''.join(f'''<section><select id="method{i}" aria-label="Detector for panel {i+1}"></select><canvas id="overview{i}" width="1232" height="1028" aria-label="Full-frame spot map {i+1}"></canvas><canvas id="crop{i}" class="crop" width="440" height="440" aria-label="Synchronized enlarged spot map {i+1}"></canvas><p id="info{i}" class="info"></p></section>''' for i in range(3)) + r'''
</div><p class="note">Fixed 20× display: black = 0, white = 204.75 saved intensity units; display clipping never enters detection or extraction. Images are the four-polarizer-cell intensity map. Changing the display image leaves the selected candidate lists unchanged. Solid circles mark found candidates; dashed circles mark all-five recurrence exclusions. The enlarged view covers 110 × 110 sensor pixels. Small white / magenta squares show 7 × 7 / 18 × 18 extraction windows at the nearest candidate. IDs belong to each detector list and are not cross-method object identities.</p>
<details><summary>Method details and reproducibility</summary>
<p>The supplied detector uses a raw-mosaic mean, hot-pixel replacement for detection only, DoG median + 8 × robust MAD, components of 10–250 pixels, and a broad-object pass. The native camera detector uses a polarization-cell mean, DoG mean + 8 × SD, and the same area range; it is called directly offline. The local cutoff is the previously used neighbourhood DoG contrast ≥8. All measurement channels are extracted from the saved background-subtracted data. The appended marker frame is excluded.</p>
<p>Common model: A=0.097840, B=0.011915, C=0.101852. Recent model: A=0.1141765217, B=0.01494632045, C=0.1188995691. Out-of-range radii are omitted, never forced to 90°. A 7 × 7 crop has unequal spatial sampling among polarizer parities; the 8 × 8 comparison probes that sensitivity. Peaks are descriptive KDE modes, not fitted physical populations or validated hinge opening angles.</p>
<p>Original data and camera code are unchanged. Current-recipe radii, local-filter decisions and θ values were checked against the previous ten-stack report.</p></details>
<p class="note"><a href="results.json">Settings, counts and detector coordinates</a> · <a href="pooled_summary.json">Pooled summaries</a> · <a href="../october_pipeline_check.py">Analysis script</a> · <a href="../pipeline_audit/index.html">September audit</a></p>
</main><script>
const E=__EXAMPLE__;const methods=E.methods;const colors=['#24d7eb','#ffab54','#7deb9b'];let centre={x:E.presets[0].x,y:E.presets[0].y};let loaded=false;const im=new Image();
function imageLoad(){loaded=false;im.onload=()=>{loaded=true;drawAll()};im.src='example_'+document.getElementById('imageMode').value+'.png'}
function selected(i){return methods[document.getElementById('method'+i).value]}
function roundEven(x){const f=Math.floor(x);return x-f===.5?(f%2===0?f:f+1):Math.round(x)}
function nearest(rows){let best=null,dist=Infinity;for(const p of rows){const d=Math.hypot(p.x-centre.x,p.y-centre.y);if(d<dist){best=p;dist=d}}return {p:best,dist}}
function mark(ctx,p,i,tx,ty,s,labels){let x=(p.x-tx)*s,y=(p.y-ty)*s;ctx.strokeStyle=colors[i];ctx.lineWidth=1.1;ctx.setLineDash(p.recurring?[3,2]:[]);ctx.beginPath();ctx.arc(x,y,6*s,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);if(labels){ctx.fillStyle=colors[i];ctx.font='12px system-ui';ctx.fillText(p.id,x+7*s,y+3)}}
function draw(i){if(!loaded)return;const m=selected(i),c=document.getElementById('overview'+i),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(im,0,0,c.width,c.height);let s=c.width/E.width;for(const p of m.rows)mark(ctx,p,i,0,0,s,document.getElementById('showIds').checked);ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.strokeRect((centre.x-55)*s,(centre.y-55)*s,110*s,110*s);
 const cc=document.getElementById('crop'+i),cx=cc.getContext('2d');const left=centre.x-55,top=centre.y-55;cx.imageSmoothingEnabled=false;cx.fillStyle='#000';cx.fillRect(0,0,440,440);cx.drawImage(im,left,top,110,110,0,0,440,440);for(const p of m.rows)if(Math.abs(p.x-centre.x)<63&&Math.abs(p.y-centre.y)<63)mark(cx,p,i,left,top,4,true);
 const n=nearest(m.rows);if(n.p&&n.dist<55&&document.getElementById('showBoxes').checked){for(const [width,col] of [[18,'#f099e6'],[7,'#fff']]){cx.strokeStyle=col;cx.lineWidth=1;let x=roundEven(n.p.x)-Math.floor(width/2),y=roundEven(n.p.y)-Math.floor(width/2);cx.strokeRect((x-left)*4,(y-top)*4,width*4,width*4)}}
 let info=m.rows.length+' detections; '+m.rows.filter(p=>p.recurring).length+' recur in all five.';if(n.p&&n.dist<55){const p=n.p;info+='<br>Nearest #'+p.id+' · ('+p.x.toFixed(1)+', '+p.y.toFixed(1)+')'+(p.recurring?' · RECURRING: excluded from θ':'');if(p.local_score!==undefined)info+=' · local '+p.local_score.toFixed(1)+(p.local_pass?' (pass)':' (fail)');if(p.legacy_gate!==undefined)info+=' · legacy gate '+(p.legacy_gate?'pass':'fail');if(p.theta7!==undefined)info+='<br>150-frame θ: 7px '+(p.theta7===null?'invalid':p.theta7.toFixed(1)+'°')+'; 18px '+(p.theta18===null?'invalid':p.theta18.toFixed(1)+'°')};document.getElementById('info'+i).innerHTML=info;
}
function drawAll(){centre.x=Math.max(55,Math.min(E.width-55,centre.x));centre.y=Math.max(55,Math.min(E.height-55,centre.y));document.getElementById('location').textContent='Shared crop centre: x='+centre.x.toFixed(1)+', y='+centre.y.toFixed(1)+' sensor pixels. θ readouts use the common supplied optical model.';for(let i=0;i<3;i++)draw(i)}
for(let i=0;i<3;i++){const sel=document.getElementById('method'+i);for(const [key,m]of Object.entries(methods)){const op=document.createElement('option');op.value=key;op.textContent=m.label+' ('+m.rows.length+')';sel.appendChild(op)}sel.value=['old150','native150','native_local'][i];sel.addEventListener('change',drawAll);for(const typ of ['overview','crop'])document.getElementById(typ+i).addEventListener('click',ev=>{let r=ev.currentTarget.getBoundingClientRect(),x=(ev.clientX-r.left)/r.width,y=(ev.clientY-r.top)/r.height;if(typ==='overview')centre={x:x*E.width,y:y*E.height};else centre={x:centre.x-55+x*110,y:centre.y-55+y*110};drawAll()})}
for(const p of E.presets){let b=document.createElement('button');b.textContent=p.label;b.addEventListener('click',()=>{centre={x:p.x,y:p.y};drawAll()});document.getElementById('presets').appendChild(b)}
document.getElementById('imageMode').addEventListener('change',imageLoad);document.getElementById('showIds').addEventListener('change',drawAll);document.getElementById('showBoxes').addEventListener('change',drawAll);document.getElementById('distribution').addEventListener('change',ev=>{document.getElementById('distributionImage').src='controls_'+ev.target.value+'.png';document.getElementById('distributionImage').alt=ev.target.selectedOptions[0].textContent+' controlled comparisons'});imageLoad();
</script></html>'''


if __name__=='__main__':
    state=json.loads((OUT/'results.json').read_text(encoding='utf-8')) if '--render-only' in sys.argv else compute()
    figures(state);page(state)
