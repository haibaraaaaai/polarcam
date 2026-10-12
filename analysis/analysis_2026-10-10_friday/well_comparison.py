"""Wide-field return-to-well analysis and conditional, stack-level comparisons.

Two wells on one surface, not biological replicates. Predefined primary recipe:
native intensity candidates + local contrast >=8, 18px, 150-frame channel mean,
manuscript annular model. Alternative recipes are a full reported sensitivity
grid, not a search for a small p-value. Original recordings are read-only.
"""
from pathlib import Path
import hashlib
import html
import itertools
import json
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from first_hinges_theta import load_module, clean_json, cliques
from local_spot_check import filtering, local_metrics
from pipeline_audit import import_file, extract_mean, radius, theta, PACKAGE, OLD_ABC
from october_pipeline_check import extract_first5, density, describe, GRID
from spot_detection_example import display20

B=Path(__file__).resolve().parent
REPO=B.parents[1]
DATA=REPO/'data/hinges/polarcam/2026-10-09'
OUT=B/'well_comparison'
H1=('154551','154730','154844','154929','155041')
GOLD=('161931','162234','162337','162429','162516')
H2=('171848','172245','173432','174132')
NARROW=('170404','170733','171421','172011','172407','173649','174356')
PRIMARY='local_18'
METHODS={f'{det}_{width}':f'{label} · {width}×{width}' for det,label in [('supplied','Supplied'),('native','Native'),('local','Native + local')] for width in (7,8,18)}
METHODS['legacy']='Supplied 5-frame + brightness gate · 7×7'
NAMES={'hinges1':'Well 1 · first visit','hinges2':'Well 1 · return visit','gold':'Well 2 · gold-only label','hinges_all':'Well 1 · visits combined'}
COLORS={'hinges1':'#147fa2','hinges2':'#8b4a9c','gold':'#bc801c','hinges_all':'#227b75'}
COMPARISONS=[('hinges1','gold'),('hinges2','gold'),('hinges_all','gold'),('hinges1','hinges2')]


def all_values(a,key):
    v=np.asarray(a['theta_'+key]);return v[np.isfinite(v)]


def compute():
    OUT.mkdir(exist_ok=True);(OUT/'previews').mkdir(exist_ok=True)
    previous=json.loads((B/'theta_comparison/results.json').read_text(encoding='utf-8'))
    abc=previous['methods']['ABC']
    detector,detector_path=load_module('well_native','src/polarcam/live/Detection_alg_offline.py')
    supplied=import_file('well_supplied',PACKAGE/'detector_intensity_reference.py')
    stacks=[];classification=[]
    for stamp in [*H2,*H1,*GOLD,*NARROW]:
        p=DATA/f'frame_stack_20261009-{stamp}.npy';before=p.stat();mb=p.with_suffix('.json').read_bytes();m=json.loads(mb.decode('utf-8-sig'))
        raw=np.load(p,mmap_mode='r',allow_pickle=False)
        assert len(raw)==151 and m['actual']['frames']==150 and m['actual']['phase_marker_appended']
        assert raw.shape[1:]==(2056,2464) and m['roi']['OffsetX']==m['roi']['OffsetY']==0
        assert m['background']['background_subtracted']
        mean=np.zeros(raw.shape[1:],float)
        for frame in raw[:150]:mean+=frame
        mean/=150
        f=filtering(detector.intensity_detection_map(mean),detector);points=f['centres'][:,::-1]
        native=np.asarray(detector.find_spot_centers_dog(f['image'][10:-10,10:-10])).reshape(-1,2)+10
        np.testing.assert_allclose(points[:,::-1],native,atol=1e-12,rtol=0)
        scores=[]
        for j in range(len(points)):
            try:score=local_metrics(f,f['centres'],j)['local_score']
            except (ValueError,ZeroDivisionError,FloatingPointError):score=np.nan
            scores.append(score)
        scores=np.asarray(scores);passed=np.isfinite(scores)&(scores>=8)
        stats=dict(image_mean=float(f['image'].mean()),image_median=float(np.median(f['image'])),
            p99=float(np.percentile(f['image'],99)),fraction_above20=float(np.mean(f['image']>20)),
            native_count=len(points),local_count=int(passed.sum()),threshold=float(f['threshold']))
        group='hinges2' if stamp in H2 else ('hinges1' if stamp in H1 else ('gold' if stamp in GOLD else 'narrow'))
        settings=dict(fps=m['actual']['fps'],exposure_us=m['actual']['timing']['exposure_us'],
            gain_analog=m['actual']['gains']['analog']['val'],gain_digital=m['actual']['gains']['digital']['val'],
            sound_checkbox=m.get('sound',{}).get('sound_on'),drive_declaration=m.get('conditions',{}).get('interference_drive'),
            note=m.get('conditions',{}).get('condition_note',''),background_subtracted=True)
        s=dict(stamp=stamp,time=f'{stamp[:2]}:{stamp[2:4]}:{stamp[4:]}',file=p.name,group=group,settings=settings,image_stats=stats,
            source_size=before.st_size,source_mtime_ns=before.st_mtime_ns,source_json_sha256=hashlib.sha256(mb).hexdigest())
        if stamp in H2 or stamp in NARROW:
            preview=Image.fromarray(display20(f['image']));preview.resize((986,822),Image.Resampling.BOX).save(OUT/'previews'/f'{stamp}.png')
            classification.append(dict(**s,field='wide' if stamp in H2 else 'narrow',basis='Previously operator-reviewed gallery, checked against illumination/background timeline and image metrics'))
        if stamp not in NARROW:
            old_cache=B/'october_pipeline_check'/f'stack_{stamp}.npz'
            if old_cache.exists():
                with np.load(old_cache,allow_pickle=False) as c:
                    oldpts=c['old150_points'].copy();old5=c['old5_points'].copy()
                    legacy=c['pre_old5_gate_5med7'].copy()
                    np.testing.assert_allclose(points,c['native150_points'],atol=1e-12,rtol=0)
                    np.testing.assert_array_equal(passed,c['local_pass'])
                    old_recurring=c['old150_recurring'].copy();native_recurring=c['native150_recurring'].copy()
                    old5_recurring=c['old5_recurring'].copy()
            else:
                oldpts,_=supplied.detect_candidates(mean.astype(np.float32)[None]);old5,_=supplied.detect_candidates(raw[:5])
                _,_,_,_,rgated=extract_first5(raw,old5);legacy=theta(rgated,OLD_ABC)
                old_recurring=np.zeros(len(oldpts),bool);native_recurring=np.zeros(len(points),bool);old5_recurring=np.zeros(len(old5),bool)
            a=dict(supplied_points=oldpts,native_points=points,old5_points=old5,local_score=scores,local_pass=passed,
                supplied_recurring=old_recurring,native_recurring=native_recurring,old5_recurring=old5_recurring,
                theta_legacy=legacy,source_file=np.array(p.name))
            for width in (7,8,18):
                for method,pts in [('supplied',oldpts),('native',points)]:
                    channels=extract_mean(mean,pts,width);r=radius(channels);t=theta(r,abc)
                    a[f'channels_{method}_{width}']=channels;a[f'r_{method}_{width}']=r;a[f'theta_{method}_{width}']=t
                    if method=='native':a[f'theta_local_{width}']=np.where(passed,t,np.nan)
            if stamp in H1 or stamp in GOLD:
                old=next(z for z in previous['stacks'] if z['stamp']==stamp)
                np.testing.assert_allclose(a['theta_local_18'],[r['theta_deg'] if r['included'] else np.nan for r in old['rows']],atol=1e-11,equal_nan=True)
            s['supplied_count']=len(oldpts);s['method_summaries']={key:describe(all_values(a,key)) for key in METHODS}
            np.savez_compressed(OUT/f'stack_{stamp}.npz',**a);stacks.append(s)
            if stamp in H2:
                # A small image supports between-stack repeated-field checks.
                small=f['image'].reshape(514,4,616,4).mean(axis=(1,3))
                np.savez_compressed(OUT/f'field_check_{stamp}.npz',image=small)
            print(f"{group} {stamp}: native {len(points)}, local {passed.sum()}, supplied {len(oldpts)}, primary n={s['method_summaries'][PRIMARY]['n']} median={s['method_summaries'][PRIMARY]['median']:.2f}",flush=True)
        else:print(f"classify {stamp}: narrow; mean {stats['image_mean']:.2f}, native {len(points)}",flush=True)
        assert before.st_size==p.stat().st_size and before.st_mtime_ns==p.stat().st_mtime_ns and p.with_suffix('.json').read_bytes()==mb
        del raw,mean,f
        # Persist numerical progress, useful even if later diagnostics are interrupted.
        (OUT/'computed.json').write_text(json.dumps(clean_json(dict(stacks=stacks,classification=classification,ABC=abc)),indent=2),encoding='utf-8')
    # New-round fixed-location flags are only a descriptive sensitivity check;
    # they never enter permutation tests (which must not select by group label).
    for method in ('supplied','native','old5'):
        pts=[]
        for stamp in H2:
            with np.load(OUT/f'stack_{stamp}.npz') as z:pts.append(z[method+'_points'])
        matches=cliques(pts,tuple(range(4)),6)
        flags=[np.zeros(len(p),bool) for p in pts]
        for match in matches:
            for k,j in match:flags[k][j]=True
        for stamp,flag in zip(H2,flags):
            path=OUT/f'stack_{stamp}.npz'
            with np.load(path) as z:a={k:z[k].copy() for k in z.files}
            a[method+'_recurring']=flag;np.savez_compressed(path,**a)
    provenance=dict(stacks=stacks,classification=classification,ABC=abc,
        user_design='Two wells on the same surface: well 1 labelled DNA + gold, then well 2 labelled gold only, then return to well 1. No independent preparation replication was established.',
        primary=PRIMARY,methods=METHODS,source_hashes={str(p.relative_to(REPO)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),detector_path,B/'local_spot_check.py',PACKAGE/'detector_intensity_reference.py')})
    (OUT/'computed.json').write_text(json.dumps(clean_json(provenance),indent=2),encoding='utf-8')
    return provenance


def holm(pvalues):
    values=np.asarray(pvalues,float);order=np.argsort(values);result=np.empty(len(values))
    result[order]=np.minimum(1,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return result


def exact_stack_test(left,right):
    """Equal-stack empirical-mixture W1, exact permutation of entire stack labels.

    A shape/location test; no Gaussian assumptions. Exact conditional p-values
    require exchangeability of stacks, which the two-well design does not prove.
    """
    all_stacks=left+right;n1=len(left);n=len(all_stacks)
    if any(not len(a) for a in all_stacks):return dict(available=False,reason='At least one stack has no valid candidates for this recipe')
    support=np.unique(np.concatenate([np.array([0.,90.]),*all_stacks]));widths=np.diff(support)
    cdfs=np.stack([np.searchsorted(np.sort(a),support[:-1],side='right')/len(a) for a in all_stacks])
    observed=float(np.abs(cdfs[:n1].mean(axis=0)-cdfs[n1:].mean(axis=0))@widths)
    # Verify the statistic independently through SciPy's weighted implementation.
    lv=np.concatenate(left);rv=np.concatenate(right)
    lw=np.concatenate([np.full(len(v),1/(n1*len(v))) for v in left])
    rw=np.concatenate([np.full(len(v),1/((n-n1)*len(v))) for v in right])
    np.testing.assert_allclose(observed,wasserstein_distance(lv,rv,lw,rw),rtol=1e-12,atol=1e-12)
    combs=list(itertools.combinations(range(n),n1));null=[]
    for start in range(0,len(combs),64):
        batch=combs[start:start+64];weights=np.full((len(batch),n),-1/(n-n1))
        for i,ids in enumerate(batch):weights[i,list(ids)]=1/n1
        null.extend((np.abs(weights@cdfs)@widths).tolist())
    null=np.asarray(null);p=float(np.mean(null>=observed-1e-10))
    # A location effect separate from the non-directional distribution distance.
    med1=np.array([np.median(a) for a in left]);med2=np.array([np.median(a) for a in right])
    return dict(available=True,stacks_left=n1,stacks_right=n-n1,spots_left=len(lv),spots_right=len(rv),
        distance_degrees=observed,p_exact=p,permutations=len(combs),
        mean_stack_median_difference=float(med1.mean()-med2.mean()),
        stack_medians_left=med1.tolist(),stack_medians_right=med2.tolist())


def statistics(state):
    a={}
    for s in state['stacks']:
        with np.load(OUT/f"stack_{s['stamp']}.npz",allow_pickle=False) as z:a[s['stamp']]={k:z[k].copy() for k in z.files}
    groups={'hinges1':list(H1),'gold':list(GOLD),'hinges2':list(H2),'hinges_all':list(H1)+list(H2)}
    tests=[]
    for key in METHODS:
        for left,right in COMPARISONS:
            result=exact_stack_test([all_values(a[s],key) for s in groups[left]],[all_values(a[s],key) for s in groups[right]])
            tests.append(dict(method=key,left=left,right=right,**result))
    primary=[t for t in tests if t['method']==PRIMARY and t['available']]
    for row,p in zip(primary,holm([t['p_exact'] for t in primary])):row['p_holm_primary4']=float(p)
    available=[t for t in tests if t['available']]
    for row,p in zip(available,holm([t['p_exact'] for t in available])):row['p_holm_all40']=float(p)
    group_summaries={}
    for group,ids in groups.items():
        group_summaries[group]={}
        for key in METHODS:
            vals=[all_values(a[s],key) for s in ids];med=[float(np.median(v)) if len(v) else None for v in vals]
            group_summaries[group][key]=dict(stacks=len(ids),pooled_spots=describe(np.concatenate(vals)),
                stack_medians=med,mean_stack_median=float(np.mean([v for v in med if v is not None])))
    recurrence=[]
    for s in state['stacks']:
        z=a[s['stamp']]
        for key in METHODS:
            method='old5' if key=='legacy' else ('supplied' if key.startswith('supplied') else 'native')
            v=z['theta_'+key];flag=z[method+'_recurring']
            recurrence.append(dict(stamp=s['stamp'],method=key,flagged=int(flag.sum()),valid_flagged=int((flag&np.isfinite(v)).sum()),
                unfiltered=describe(v),without_recurrence=describe(v[~flag])))
    overlaps=[]
    for x,y in itertools.combinations(H2,2):
        p=a[x]['native_points'][a[x]['local_pass']];q=a[y]['native_points'][a[y]['local_pass']]
        near_p=int(np.sum(cKDTree(q).query(p)[0]<=6)) if len(q) else 0
        near_q=int(np.sum(cKDTree(p).query(q)[0]<=6)) if len(p) else 0
        with np.load(OUT/f'field_check_{x}.npz') as z:im1=z['image']
        with np.load(OUT/f'field_check_{y}.npz') as z:im2=z['image']
        overlaps.append(dict(first=x,second=y,first_count=len(p),second_count=len(q),first_near_second=near_p,second_near_first=near_q,
            unregistered_image_correlation=float(np.corrcoef(im1.ravel(),im2.ravel())[0,1])))
    result=clean_json(dict(**state,tests=tests,group_summaries=group_summaries,recurrence_sensitivity=recurrence,
        field_overlap=overlaps,statistics='Exact whole-stack permutations of equal-stack-weighted Wasserstein-1 distance. Primary family: four comparisons, Holm corrected. All 40 recipe/comparison pairs also reported with a separate Holm correction. No spot-level independence assumption.',
        selection='Statistics use within-stack filters only, without sample-group-dependent recurrence selection. Recurrence is reported as a descriptive sensitivity analysis.'))
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('PRIMARY',json.dumps(primary),flush=True)
    return result


def render(state):
    """One compact report; all figures use cached numerical results."""
    arrays={}
    for s in state['stacks']:
        with np.load(OUT/f"stack_{s['stamp']}.npz",allow_pickle=False) as z:
            arrays[s['stamp']]={k:z[k].copy() for k in z.files}
    stacks={s['stamp']:s for s in state['stacks']}
    groups={'hinges1':H1,'gold':GOLD,'hinges2':H2}
    short={'hinges1':'Well 1, first visit','gold':'Well 2, gold label','hinges2':'Well 1, return visit','hinges_all':'Well 1, both visits'}
    titles={('hinges1','gold'):'First visit vs gold',('hinges2','gold'):'Return visit vs gold',
            ('hinges_all','gold'):'Both visits vs gold',('hinges1','hinges2'):'First vs return visit'}
    vals=lambda stamp,key:all_values(arrays[stamp],key)
    curve=lambda ids,key:np.mean([density(vals(s,key)) for s in ids],axis=0)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#28333e','text.color':'#28333e',
                         'axes.titlesize':12,'axes.titleweight':'bold','savefig.facecolor':'white'})
    def style(ax):
        ax.set_xlim(0,90);ax.set_xticks([0,15,30,45,60,75,90]);ax.set_xlabel('θ (degrees)')
        ax.set_ylabel('Density per degree');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    def save(fig,name):
        fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11.8,7.3),sharex=True,sharey=True)
    for ax,stamp in zip(axes.flat,H2):
        v=vals(stamp,PRIMARY);s=stacks[stamp];summary=s['method_summaries'][PRIMARY]
        ax.hist(v,bins=np.arange(0,91,5),density=True,color=COLORS['hinges2'],alpha=.2,edgecolor='white')
        ax.plot(GRID,density(v),color=COLORS['hinges2'],lw=2.2,label='This return stack')
        ax.plot(GRID,curve(H1,PRIMARY),color=COLORS['hinges1'],ls='--',lw=1.5,label='First visit reference')
        ax.plot(GRID,curve(GOLD,PRIMARY),color=COLORS['gold'],ls=':',lw=2,label='Gold reference')
        ax.axvline(np.median(v),color=COLORS['hinges2'],lw=1,alpha=.6)
        ax.set_title(f"{s['time']}  ·  n = {len(v)}  ·  median {summary['median']:.1f}°",loc='left')
        style(ax);ax.tick_params(labelbottom=True)
    fig.suptitle('Well 1, return visit: four wide-field θ distributions',x=.06,ha='left',fontsize=17,weight='bold',y=.98)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.94),ncol=3,frameon=False)
    fig.text(.06,.015,'Native + local contrast ≥8 · 18 × 18 pixels · 150-frame channel means · reference curves give each stack equal weight',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.11,hspace=.36,wspace=.16)
    save(fig,'return_wide_theta')

    fig,axes=plt.subplots(1,2,figsize=(12.1,4.6),gridspec_kw={'width_ratios':[1.15,1]})
    ax=axes[0]
    for group,ids in groups.items():
        d=curve(ids,PRIMARY)
        ax.plot(GRID,d,lw=2.3,color=COLORS[group],label=short[group]);ax.fill_between(GRID,d,color=COLORS[group],alpha=.055)
    style(ax);ax.set_title('Equal weight for each stack',loc='left');ax.legend(frameon=False,fontsize=9)
    ax=axes[1]
    for y,(group,ids) in enumerate(groups.items()):
        med=np.array([np.median(vals(s,PRIMARY)) for s in ids]);jitter=np.linspace(-.13,.13,len(ids))
        ax.scatter(med,y+jitter,s=48,color=COLORS[group],zorder=3)
        ax.scatter([med.mean()],[y],marker='|',s=430,lw=2.5,color='#26323d',zorder=4)
        ax.text(75,y,f'{med.mean():.1f}°',va='center',fontsize=10)
    ax.set_yticks(range(3),[short[g] for g in groups]);ax.set_xlim(20,85);ax.set_ylim(2.5,-.5)
    ax.set_xlabel('Median θ within a stack (degrees)');ax.set_title('Each dot is one recording',loc='left');ax.grid(axis='x',alpha=.15)
    fig.suptitle('The later visit shifts upward; the distributions still overlap',x=.04,ha='left',fontsize=16,weight='bold')
    fig.text(.04,.015,'Primary recipe: native + local, 18 × 18. Black marks and numbers: mean of stack medians. Two wells, no preparation replicates.',fontsize=9)
    fig.subplots_adjust(left=.065,right=.97,top=.83,bottom=.17,wspace=.49)
    save(fig,'primary_comparison')

    fig,axes=plt.subplots(3,3,figsize=(12.2,10.1),sharex=True,sharey=True)
    for row,det in enumerate(('supplied','native','local')):
        for col,width in enumerate((7,8,18)):
            ax=axes[row,col];key=f'{det}_{width}'
            for group,ids in groups.items():ax.plot(GRID,curve(ids,key),color=COLORS[group],lw=1.9,label=short[group])
            ax.set_title(METHODS[key]+(' · primary' if key==PRIMARY else ''),loc='left')
            if key==PRIMARY:ax.set_facecolor('#f6f1f8')
            style(ax);ax.tick_params(labelbottom=True)
    fig.suptitle('Change the detector or extraction window, keep the rest fixed',x=.07,ha='left',fontsize=16,weight='bold',y=.985)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.953),ncol=3,frameon=False)
    fig.text(.07,.015,'All panels: 150-frame means, same annular optical model, equal stack weights, 4° Gaussian smoothing. No group-dependent recurrence veto.',fontsize=9)
    fig.subplots_adjust(left=.07,right=.98,top=.895,bottom=.075,hspace=.45,wspace=.15)
    save(fig,'detection_sensitivity')

    fig,axes=plt.subplots(1,2,figsize=(11.8,4.3))
    for ax,key,title in zip(axes,('local_18','legacy'),('Native + local, 18 × 18','Complete old recipe: supplied + brightness gate')):
        for group,ids in groups.items():ax.plot(GRID,curve(ids,key),color=COLORS[group],lw=2,label=short[group])
        style(ax);ax.set_title(title,loc='left')
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('The old brightness-gated recipe gives a different ordering',x=.065,ha='left',fontsize=16,weight='bold')
    fig.text(.065,.014,'Old recipe: first 5 frames, 7 × 7, paired brightness >50 after corner estimate, median r of passing frames, old optical model.',fontsize=9)
    fig.subplots_adjust(left=.075,right=.98,bottom=.18,top=.8,wspace=.22)
    save(fig,'legacy_comparison')

    esc=html.escape
    def num(x,d=1):return '—' if x is None else f'{x:.{d}f}'
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{c}</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def figure(name,alt):return f'<a href="{name}.png" target="_blank"><img class="plot" src="{name}.png" alt="{alt}"></a>'
    rows=[]
    for stamp in H2:
        s=stacks[stamp];m=s['method_summaries'][PRIMARY];setting=s['settings']
        rows.append([s['time'],s['image_stats']['native_count'],s['image_stats']['local_count'],m['n'],num(m['median'])+'°',
            f"{num(m['q25'])}–{num(m['q75'])}°",'checked' if setting['sound_checkbox'] else '<strong>unchecked; conflicts with “On”</strong>'])
    newtable=table(['Wide-field stack','Native candidates','Pass local ≥8','Valid θ','Median','Middle 50%','Legacy sound checkbox'],rows)
    prows=[]
    for t in state['tests']:
        if t['method']==PRIMARY:
            prows.append([titles[t['left'],t['right']],f"{t['stacks_left']} vs {t['stacks_right']}",num(t['distance_degrees'],2)+'°',
                          f"{t['mean_stack_median_difference']:+.2f}°",num(t['p_exact'],4),num(t['p_holm_primary4'],4)])
    primarytable=table(['Comparison','Stacks','Distribution distance','Δ mean stack median¹','Raw p','Holm p (4 tests)'],prows)
    sensitivityrows=[]
    for key,label in METHODS.items():
        rows=[t for t in state['tests'] if t['method']==key]
        sensitivityrows.append([esc(label)]+[f"{t['p_exact']:.4f} <span class='muted'>/ {t['p_holm_all40']:.4f}</span>" if t['available'] else '—' for t in rows])
    sensitivitytable=table(['Recipe · raw p / Holm p (40 tests)']+[titles[c] for c in COMPARISONS],sensitivityrows)
    countrows=[]
    for stamp in (*H1,*GOLD,*H2):
        s=stacks[stamp]
        countrows.append([s['time'],short[s['group']],str(s['supplied_count']),str(s['image_stats']['native_count']),
            str(s['image_stats']['local_count']),str(s['method_summaries'][PRIMARY]['n']),num(s['image_stats']['image_mean'],2)])
    counttable=table(['Time','Group','Supplied candidates','Native candidates','Local pass','Primary valid θ','Mean image intensity'],countrows)
    cards=[]
    for s in sorted(state['classification'],key=lambda s:s['stamp']):
        v=s['image_stats'];tag='Wide · included' if s['field']=='wide' else 'Narrow · separate'
        cards.append(f'<article><a href="previews/{s["stamp"]}.png" target="_blank"><img src="previews/{s["stamp"]}.png" loading="lazy" alt="{s["time"]} {s["field"]} field"></a><h3>{s["time"]} · {tag}</h3><p>Mean {v["image_mean"]:.2f} · native {v["native_count"]} · local {v["local_count"]}</p></article>')
    settingrows=[]
    for stamp in H2:
        s=stacks[stamp];v=s['settings']
        settingrows.append([s['time'],'checked' if v['sound_checkbox'] else 'unchecked',esc(v['drive_declaration']),esc(v['note'])])
    settingtable=table(['Time','Legacy sound checkbox','Declared interference drive','Recorded note (verbatim)'],settingrows)
    recrows=[]
    for group,ids in groups.items():
        for det,key in [('Supplied','supplied_18'),('Native + local','local_18')]:
            rec=[r for r in state['recurrence_sensitivity'] if r['stamp'] in ids and r['method']==key]
            recrows.append([short[group],det,f"{min(r['valid_flagged'] for r in rec)}–{max(r['valid_flagged'] for r in rec)}",', '.join(f"{r['without_recurrence']['median']-r['unfiltered']['median']:+.1f}°" for r in rec)])
    rectable=table(['Group','Recipe (18 px)','Valid recurring candidates per stack','Median shift after removal, by stack'],recrows)
    overlaprows=[]
    for r in state['field_overlap']:
        overlaprows.append([stacks[r['first']]['time']+' / '+stacks[r['second']]['time'],
            f"{r['first_near_second']}/{r['first_count']} ; {r['second_near_first']}/{r['second_count']}",num(r['unregistered_image_correlation'],3)])
    overlaptable=table(['Return-stack pair','Local candidates within 6 px (both directions)','Image correlation, no registration'],overlaprows)
    abc=state['ABC']
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>October 9 · two-well θ comparison</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f7;color:#25313b;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:36px 32px 70px}}
h1{{font-size:32px;line-height:1.2;margin:8px 0 18px}}h2{{font-size:24px;line-height:1.3;margin:35px 0 14px}}h3{{font-size:15px;margin:8px 0 0}}p{{max-width:1050px}}a{{color:#176a93}}.eyebrow,.muted,small{{color:#596875}}.eyebrow{{font-size:13px;letter-spacing:.08em;text-transform:uppercase}}.callout{{background:#ede6f2;border-left:4px solid #8b4a9c;padding:15px 20px;margin:24px 0}}.callout p{{margin:4px 0}}
.plot{{width:100%;height:auto;background:white;border:1px solid #dce1e5;margin:12px 0 5px}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;background:white;margin:10px 0 16px}}th,td{{padding:11px 13px;border-bottom:1px solid #dce1e5;text-align:left;vertical-align:top}}th{{background:#eaf0f3;font-weight:650}}th:first-child,td:first-child{{white-space:nowrap}}
.gallery{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:20px}}article{{background:white;padding:10px;border:1px solid #dce1e5}}article img{{width:100%;display:block}}article p{{font-size:13px;margin:5px 0}}details{{border-top:1px solid #cad2da;padding:15px 0}}summary{{cursor:pointer;font-weight:650}}code{{font-size:13px;overflow-wrap:anywhere}}nav{{display:flex;gap:18px;flex-wrap:wrap;font-size:14px;margin:20px 0}}.method{{font-size:14px}}@media(max-width:750px){{main{{padding:22px 15px}}h1{{font-size:27px}}.gallery{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:450px){{.gallery{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">9 October 2026 · wide-field analysis · two wells on one surface</div>
<h1>Return to well 1: θ distributions and the prep question</h1>
<p>Your sequence was <strong>well 1 (DNA + gold label) → well 2 (gold-only label) → well 1 again</strong>. This adds the four kept wide-field stacks from the return visit. The seven narrow-field stacks remain separate.</p>
<nav><a href="#theta">New θ plots</a><a href="#compare">Compare wells</a><a href="#methods">Detector sensitivity</a><a href="#selection">Wide/narrow check</a><a href="#details">Methods and metadata</a></nav>
<div class="callout"><p><strong>The return visit gives higher θ under the native/local recipe, but there is no detector-independent separation that diagnoses a prep mix-up.</strong></p><p>The supplied detector produces smaller conditional p-values; it also separates the two visits to the same well. The full old brightness-gated recipe changes the ordering again. With only one well per label, these measurements cannot distinguish sample identity from acquisition and well effects.</p></div>
<h2 id="theta">Four new wide-field distributions</h2>
<p>The main analysis keeps the previous <strong>native camera detector + local contrast ≥8, 18 × 18 extraction</strong>. It yields <strong>417 valid θ values</strong> from 419 candidates passing the local filter; two fall outside the optical model’s invertible range. One θ value comes from each candidate’s 150-frame mean channel intensities.</p>
{figure('return_wide_theta','Four return-visit theta distributions with first-visit and gold references')}
{newtable}
<p class="muted">Bars: 5° bins. Curves: Gaussian smoothing, 4° bandwidth. The new pooled spot median is 51.0°; its smoothed pooled maximum is near 52.9°. These are descriptive summaries, not evidence of a distinct molecular state.</p>
<h2 id="compare">What separates the wells?</h2>
{figure('primary_comparison','Equal-stack distributions and individual recording medians')}
<p>The means of the individual stack medians are <strong>45.8° for well 1 first visit, 45.0° for well 2, and 51.1° for well 1 return</strong>. Combining both well-1 visits gives 48.2°. Each stack gets equal weight, so a field containing more detected objects cannot dominate the comparison.</p>
<p><strong>Exploratory test:</strong> compare the whole θ distributions, then enumerate all exchanges of <em>whole-stack</em> group labels. The distance is the average angular displacement required to match the two equal-stack distributions (Wasserstein-1). It detects changes in shape as well as location. No individual spot is treated as an independent well or preparation.</p>
{primarytable}
<p class="muted">¹ Left group minus right group. Holm correction covers these four comparisons. Exact enumerations: 252, 126, 2,002 and 126 label partitions. “Exact” describes the computation; the interpretation still requires exchangeable recordings, which this nonrandomized sequence does not establish.</p>
<p><strong>For this retained primary recipe, none of the comparisons rejects equal recording-level distributions at 0.05.</strong> This does not establish that the wells contain the same sample, nor that the preparation was mixed up. There is only one well per label; the visits are repeated measurements of well 1, not independent DNA preparations. See the <a href="https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf">ASA guidance on interpreting p-values</a>.</p>
<h2 id="methods">How much does the answer depend on spot selection?</h2>
<p>The nine panels isolate three detectors and three extraction windows. Every panel uses the same 150-frame averaging and optical coefficients. The <strong>supplied</strong> detector is the one in your ZIP; <strong>native</strong> is the camera’s offline detector; <strong>native + local</strong> rejects candidates weak relative to their neighbours.</p>
{figure('detection_sensitivity','Three detectors by three extraction windows, with all three recording groups')}
<p>The supplied detector gives raw p-values of 0.0238 for first visit vs gold, 0.0079 for return visit vs gold, and 0.0065–0.0140 with visits combined. Those differences are worth noticing. But it also gives p = 0.0079 for <strong>first vs return visit to well 1</strong>, so separation is not uniquely associated with the well label. The native/local comparisons with gold have raw p ≥0.0556 across these windows.</p>
{figure('legacy_comparison','Current recipe versus the full old brightness-gated recipe')}
<p>The full old recipe gives mean stack medians of <strong>62.7° → 46.4° → 43.9°</strong> in recording order (well 1 first → well 2 → well 1 return). The retained primary recipe gives <strong>45.8° → 45.0° → 51.1°</strong>. This reversal shows why choosing whichever pipeline gives the smallest p-value would be misleading. The old brightness gate admits far more candidates in the brighter return recordings.</p>
<details><summary>All 40 tests: raw and multiplicity-adjusted p-values</summary>
<p>Each cell shows <strong>raw p</strong> / <span class="muted">Holm p across all 40 recipe–comparison combinations</span>. None survives this conservative family correction (minimum adjusted p = 0.2597). The recipes are correlated, and there are few recordings: the smallest attainable raw p for a four-versus-five comparison is 1/126 = 0.0079. This broad sensitivity family has low resolution and low power; lack of a corrected result is not an equivalence test.</p>{sensitivitytable}</details>
<details><summary>Candidate counts and brightness for all 14 wide-field stacks</summary>{counttable}<p>Counts are objects returned by each detector, not verified individual rods. Well 1’s return has more native candidates (77–136 per stack) than its first visit (44–57) or well 2 (30–60), but the background-subtracted image mean also rises to 17.7–19.8 from 5.6–6.6. Illumination, subtraction, aggregation and detection threshold can all affect counts.</p></details>
<h2 id="selection">Wide and narrow fields separate clearly</h2>
<p>The reviewed timeline identifies <strong>17:18:48, 17:22:45, 17:34:32 and 17:41:32</strong> as wide-field recordings after the wide-field background. This is independently consistent with mean image intensity <strong>17.7–19.8</strong> and <strong>77–136 native candidates</strong>, versus <strong>1.66–1.76</strong> and <strong>0–8</strong> for the seven narrow-field recordings. Some JSON notes say “wide field” even on narrow frames; those repeated notes were not used to override the images and reviewed timeline.</p>
<p class="muted">Previews are the 150-frame mean at fixed 20× display gain (black 0, white 204.75 saved intensity units), resized for viewing. Click to enlarge. The mixed-field 16:58:14 recording and rejected 17:10:23 / 17:11:19 background mismatches are excluded.</p>
<div class="gallery">{''.join(cards)}</div>
<h2 id="details">Limits that matter for the prep question</h2>
<p><strong>Acquisition changed:</strong> first visit and gold used 72.175 fps; return wide fields used 60.010 fps. Exposure stayed 0.5975 ms and both gains stayed 1. All selected files report saved background subtraction on, but the old generic profile path does not uniquely identify the background array. Gold notes say oil was re-added and background recaptured; return notes describe a new background and less laser. The last two return stacks have an unchecked legacy sound box while the declared drive still says “On”. A well difference, time drift, background change or interference change can therefore resemble a sample difference.</p>
<details><summary>Return-visit interference metadata and original notes</summary>{settingtable}<p>The checkbox state and the declared condition are reported separately. Neither verifies the physical interference drive. No automatic on/off pairing or correction was inferred from these conflicting entries.</p></details>
<details><summary>Fixed-location candidates and repeated-field checks</summary>
<p>The same-position veto depends on the group assignment: five-stack recurrence for the first visit and gold, four-stack recurrence for the return visit. Applying it once and then shuffling group labels would bias a permutation test. The tests therefore use only within-stack filters. The primary native/local result is unchanged: <strong>zero candidates recur across every stack of any of the three groups</strong>.</p>
{rectable}
<p>These are descriptive before/after checks only. Recurrence requires a match across every stack in the group with every matched pair within 6 raw pixels. For the supplied detector, removing those candidates changes both distributions and candidate totals relative to earlier pages; it is not silently folded into these tests.</p>
{overlaptable}<p>Only 0–2 local detections align within 6 pixels between each pair of return stacks. That is consistent with different fields at the same sensor coordinates, but does not rule out translated views or overlapping fields and does not make fields independent preparations.</p></details>
<details><summary>Reconstruction, numerical checks and reproducibility</summary>
<div class="method"><p><strong>Data:</strong> first 150 frames of each 151-frame file; appended phase marker excluded. Full sensor, origin (0,0), 2,464 × 2,056 pixels. Saved arrays are used as-is; no new background is subtracted. Source file sizes, modification times and JSON hashes were checked unchanged during reading.</p>
<p><strong>Native detector:</strong> camera offline detector on the mean intensity map (each 2 × 2 polarizer cell averaged and repeated), difference of Gaussians σ = 1 / 6 raw pixels, global mean + 8 SD threshold, component area 10–250 and 10-pixel edge exclusion. Native + local adds (component DoG peak − neighbouring median) / (1.4826 MAD) ≥8, using a 12–30-pixel ring with other candidate cores masked. This score is an exploratory contrast filter, not a calibrated noise significance.</p>
<p><strong>Supplied detector:</strong> ZIP detector on the same mean, including its despiking, robust MAD threshold and broad-intensity candidate pass. The legacy panel instead uses its original first-five-frame detection, 7 × 7 extraction and paired brightness gate; reported r is the median of passing per-frame saved-channel radii, not a locally background-corrected r.</p>
<p><strong>θ:</strong> each extraction uses per-channel means according to absolute sensor parity: I₀ odd/odd, I₉₀ even/even, I₄₅ even/odd, I₁₃₅ odd/even. X = (I₀ − I₉₀)/(I₀ + I₉₀), Y = (I₄₅ − I₁₃₅)/(I₄₅ + I₁₃₅), r = √(X² + Y²). θ = asin √[Ar/(C − Br)] with A = {abc[0]:.12f}, B = {abc[1]:.12f}, C = {abc[2]:.12f}. Nonpositive pair sums or r outside the model range are excluded, not clipped. The 7-pixel window contains unequal numbers of pixels for the four channels; 8 and 18 are balanced.</p>
<p><strong>Numerical checks:</strong> direct native detector centres matched the helper reconstruction; first-visit and gold primary θ values and local pass/fail decisions matched the previous analysis. Every observed distribution distance was independently checked against <a href="https://docs.scipy.org/doc/scipy-1.5.3/reference/generated/scipy.stats.wasserstein_distance.html">SciPy’s weighted Wasserstein distance</a>. Whole-stack label partitions were exhaustively enumerated; <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html">permutation documentation</a> and <a href="https://www.statsmodels.org/stable/_modules/statsmodels/stats/multitest.html">Holm correction reference</a>.</p>
<p>Run from the repository root: <code>.venv/Scripts/python.exe analysis/analysis_2026-10-10_friday/well_comparison.py</code>. Use <code>--reuse-computed</code> to repeat statistics from cached candidates, or <code>--render-only</code> to rebuild plots. <a href="../well_comparison.py">Source script</a> · <a href="results.json">Full numerical results</a>. Generated files remain under this analysis output folder.</p></div></details>
<p><strong>What would resolve a suspected prep mix-up:</strong> an independently labelled DNA-positive/negative check, or repeated independently prepared wells measured with matched illumination, background and interference settings. These θ distributions alone do not identify DNA or its attachment to gold.</p>
</main></body></html>'''
    (OUT/'index.html').write_text(text,encoding='utf-8')
    print('Report:',OUT/'index.html',flush=True)


if __name__=='__main__':
    if '--render-only' in sys.argv:
        state=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    else:
        computed=json.loads((OUT/'computed.json').read_text(encoding='utf-8')) if '--reuse-computed' in sys.argv else compute()
        state=statistics(computed)
    render(state)
