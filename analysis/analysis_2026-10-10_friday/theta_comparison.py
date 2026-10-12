"""Matched detector/contrast/angle analysis: October hinges, gold, eight September stacks.

Run with repository .venv Python. --render-only rebuilds figures/page from results.json.
Raw data and historical reports are unchanged; results here apply the new exclusions.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from first_hinges_theta import load_module, cliques, clean_json, MODEL_PARAMETERS
from local_spot_check import filtering, local_metrics, TRIAL_CUTOFF
from spot_detection_example import display20

B=Path(__file__).resolve().parent
REPO=B.parents[1]
OUT=B/'theta_comparison'
HINGES=('154551','154730','154844','154929','155041')
GOLD=('161931','162234','162337','162429','162516')
SEPTEMBER=('145406','151106','151218','152241','152256','152740','154622','155126')
CONFIGS=[('hinges','2026-10-09',s) for s in HINGES]+[('gold','2026-10-09',s) for s in GOLD]+[('september','2026-09-18',s) for s in SEPTEMBER]
COLORS={'hinges':'#187d9a','gold':'#be811d','september':'#845b9d'}
BINS=np.arange(0,95,5)


def numeric_summary(values):
    a=np.asarray(values,float)
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,
                mean=float(np.mean(a)) if len(a) else None,
                q25=float(np.percentile(a,25)) if len(a) else None,
                q75=float(np.percentile(a,75)) if len(a) else None)


def compute():
    OUT.mkdir(exist_ok=True);(OUT/'previews').mkdir(exist_ok=True)
    detector,detector_path=load_module('native_detector','src/polarcam/live/Detection_alg_offline.py')
    channels,channels_path=load_module('native_channels','src/polarcam/live/intensity_spots.py')
    model,model_path=load_module('paper_theta_model','analysis/paper_analysis_20261008/original_scripts/hinge_analysis.py')
    A,Bc,C=model.fourkas_ABC(**MODEL_PARAMETERS)
    stacks=[]
    for group,date,stamp in CONFIGS:
        path=REPO/'data/hinges/polarcam'/date/f"frame_stack_{date.replace('-','')}-{stamp}.npy"
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8-sig'))
        initial_stat=path.stat()
        raw=np.load(path,mmap_mode='r')
        n=len(raw)-int(meta['actual']['phase_marker_appended'])
        if n!=meta['actual']['frames'] or n!=150 or raw.shape[1:]!=(2056,2464):
            raise ValueError(f'Unexpected frame count/size: {path}')
        if meta['background'].get('background_subtracted') is not True:
            raise ValueError(f'Unexpected background state: {path}')
        origin=tuple(int(meta['roi'][k]) for k in ('OffsetX','OffsetY'))
        if origin!=(0,0):raise ValueError('This comparison expects full sensor frames')
        mean=np.zeros(raw.shape[1:],float)
        for frame in raw[:n]:mean+=frame
        mean/=n
        f=filtering(detector.intensity_detection_map(mean),detector)
        pts=f['centres']
        native=np.asarray(detector.find_spot_centers_dog(f['image'][10:-10,10:-10])).reshape(-1,2)+10
        np.testing.assert_allclose(pts,native,rtol=0,atol=1e-12)
        bounds=[]
        for x,y in pts:
            left=max(0,int(round(x))-9);top=max(0,int(round(y))-9)
            bounds.append((left,left+18,top,top+18))
        if any(r>mean.shape[1] or b>mean.shape[0] for l,r,t,b in bounds):raise ValueError('Truncated aperture')
        intensities=channels.channel_means(mean,bounds,origin)
        xy=channels.anisotropy(intensities);r=np.linalg.norm(xy,axis=1)
        theta,flags=model.theta_from_r(r,A,Bc,C,return_flags=True)
        sin2=np.sin(np.radians(theta[flags['valid']]))**2
        np.testing.assert_allclose(C*sin2/(A+Bc*sin2),r[flags['valid']],atol=1e-12,rtol=1e-12)
        nearest=cKDTree(pts).query(pts,k=2)[0][:,1] if len(pts)>1 else np.full(len(pts),np.inf)
        rows=[]
        for j,((x,y),(left,right,top,bottom)) in enumerate(zip(pts,bounds)):
            try:
                score=local_metrics(f,pts,j)
                local_score=float(score['local_score'])
                score_ok=np.isfinite(local_score)
                score_issue='' if score_ok else 'nonfinite_local_score'
            except (ValueError,ZeroDivisionError,FloatingPointError) as exc:
                score={};local_score=np.nan;score_ok=False;score_issue=str(exc)
            row=dict(file=path.name,group=group,date=date,stamp=stamp,spot_id=j+1,
                x=float(x),y=float(y),roi_left=left,roi_right=right,roi_top=top,roi_bottom=bottom,
                **dict(zip(('I0','I90','I45','I135'),map(float,intensities[j]))),
                X=float(xy[j,0]),Y=float(xy[j,1]),r=float(r[j]),theta_deg=float(theta[j]),
                theta_valid=bool(flags['valid'][j]),model_over_range=bool(flags['over_range'][j]),
                local_score=local_score,local_score_valid=bool(score_ok),local_score_issue=score_issue,
                local_pass=bool(score_ok and local_score>=TRIAL_CUTOFF),
                component_area_px=int(f['stats'][f['label_ids'][j],4]),
                local_reference_samples=score.get('local_samples',0),
                global_peak_z=score.get('global_peak_z',np.nan),
                nearest_candidate_px=float(nearest[j]),neighbor_under_24px=bool(nearest[j]<24),
                aperture_saved_max=int(raw[:,top:bottom,left:right][:n].max()),
                recurring_all_five=False,included=False,exclusion_reason='')
            rows.append(row)
        time=f'{stamp[:2]}:{stamp[2:4]}:{stamp[4:]}'
        gains=meta['actual']['gains']
        stack=dict(group=group,date=date,stamp=stamp,time=time,file=path.name,rows=rows,
            image=f"previews/{path.stem}.png",source_bytes=initial_stat.st_size,source_mtime_ns=initial_stat.st_mtime_ns,
            source_json_sha256=hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest(),
            settings=dict(frames=n,width=mean.shape[1],height=mean.shape[0],origin=list(origin),
                fps=meta['actual']['fps'],exposure_us=meta['actual']['timing']['exposure_us'],
                gain_analog=gains['analog']['val'],gain_digital=gains['digital']['val'],
                sound_checkbox=meta.get('sound',{}).get('sound_on'),background_subtracted=True,
                background_profile_path=meta['background'].get('background_profile_path'),
                background_identity_known=bool(meta['background'].get('background_profile_sha256')),
                max_raw=meta['actual'].get('max_raw_value'),max_saved=meta['actual'].get('max_saved_value')),
            image_stats=dict(dog_threshold=float(f['threshold']),dog_sd=float(f['sd']),
                mean_image_median=float(np.median(f['image'])),mean_image_p99=float(np.percentile(f['image'],99))),
            note=(meta.get('conditions') or {}).get('condition_note',''))
        Image.fromarray(display20(f['image'])).save(OUT/stack['image'])
        assert path.stat().st_size==initial_stat.st_size and path.stat().st_mtime_ns==initial_stat.st_mtime_ns
        stacks.append(stack)
        print(f"{date} {time}: native {len(pts)}, local pass {sum(row['local_pass'] for row in rows)}",flush=True)
        del raw,mean,f

    matches={}
    # Recurrence is checked only inside each user-defined common-background group.
    # September remains eight separate statistics with no cross-stack veto.
    for group in ('hinges','gold'):
        selected=[s for s in stacks if s['group']==group]
        points=[np.array([[r['x'],r['y']] for r in s['rows']]).reshape(-1,2) for s in selected]
        groups=cliques(points,tuple(range(5)),6)
        matches[group]=groups
        for nodes in groups:
            for k,j in nodes:selected[k]['rows'][j]['recurring_all_five']=True
    for stack in stacks:
        for row in stack['rows']:
            reasons=[]
            if not row['local_pass']:reasons.append('low_local_contrast' if row['local_score_valid'] else 'unscorable_local_contrast')
            if row['recurring_all_five']:reasons.append('recurring_in_all_five')
            if not row['theta_valid']:reasons.append('outside_theta_model' if row['model_over_range'] else 'invalid_channel_values')
            row['included']=not reasons
            row['exclusion_reason']=';'.join(reasons)
            row['color']=to_hex(plt.cm.viridis(row['theta_deg']/90)) if row['included'] else '#f07650'
        rows=stack['rows'];kept=[r['theta_deg'] for r in rows if r['included']]
        stack['summary']=dict(file=stack['file'],group=stack['group'],time=stack['time'],
            detected=len(rows),local_excluded=sum(not r['local_pass'] for r in rows),
            recurring_after_local=sum(r['local_pass'] and r['recurring_all_five'] for r in rows),
            invalid_theta_after_filters=sum(r['local_pass'] and not r['recurring_all_five'] and not r['theta_valid'] for r in rows),
            recurring_total=sum(r['recurring_all_five'] for r in rows),**numeric_summary(kept),
            crowded_included=sum(r['included'] and r['neighbor_under_24px'] for r in rows),
            kept_saved_max_ge4080=sum(r['included'] and r['aperture_saved_max']>=4080 for r in rows),
            detections_per_mpix=len(rows)/(2056*2464/1e6),
            median_component_area=float(np.median([r['component_area_px'] for r in rows])) if rows else None)
        su=stack['summary'];assert su['detected']==su['local_excluded']+su['recurring_after_local']+su['invalid_theta_after_filters']+su['n']
    pools={}
    for group in ('hinges','gold'):
        rows=[r for s in stacks if s['group']==group for r in s['rows']]
        pools[group]=dict(detected=len(rows),local_excluded=sum(not r['local_pass'] for r in rows),
            recurring_after_local=sum(r['local_pass'] and r['recurring_all_five'] for r in rows),
            invalid_theta_after_filters=sum(r['local_pass'] and not r['recurring_all_five'] and not r['theta_valid'] for r in rows),
            **numeric_summary([r['theta_deg'] for r in rows if r['included']]))
    state=dict(stacks=stacks,pools=pools,matches=matches,
        methods=dict(detector='Native intensity DoG: k=8, sigma=1/6 raw pixels, area=10..250, 10-pixel crop',
            local_cutoff=8,local_ring=[12,30],other_candidate_mask_radius=8,
            local_statistic='Peak DoG minus annular median, divided by 1.4826*MAD; one sample per 2x2 cell',
            local_status='Exploratory spatial contrast selection, not a calibrated SNR or false-positive rate',
            aperture=18,frames=150,theta='Four channel means across the full stack, then anisotropy and model inversion',
            model='Existing manuscript_annular Fourkas water model',model_parameters=MODEL_PARAMETERS,
            ABC=[A,Bc,C],r_max=model.r_max_of(A,Bc,C),histogram_bin_degrees=5,
            recurrence='Pairwise sensor-coordinate separation <=6px in all five native lists, separately for October hinges/gold; none for September',
            grouping='October grouping is user-supplied; background profile identity cannot be recovered from generic old paths'),
        source_hashes={str(p.relative_to(REPO)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),B/'local_spot_check.py',B/'first_hinges_theta.py',detector_path,channels_path,model_path]})
    state=clean_json(state)
    (OUT/'results.json').write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8')
    for name,rows in [('spots',[r for s in stacks for r in s['rows']]),('stack_summary',[s['summary'] for s in stacks])]:
        with (OUT/f'{name}.csv').open('w',newline='',encoding='utf-8') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(pools=state['pools'],september=[s['summary'] for s in stacks if s['group']=='september'])),flush=True)
    return state


def values(stack):
    return np.array([r['theta_deg'] for r in stack['rows'] if r['included']])


def percent_hist(a):
    counts=np.histogram(a,bins=BINS)[0]
    assert counts.sum()==len(a)
    return counts*100/len(a) if len(a) else counts.astype(float)


def figures(state):
    stacks=state['stacks'];pools=state['pools']
    oct_h=np.concatenate([values(s) for s in stacks if s['group']=='hinges'])
    oct_g=np.concatenate([values(s) for s in stacks if s['group']=='gold'])
    plt.rcParams.update({'font.size':10,'figure.facecolor':'#f7f7f4','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for group,a,name in [('hinges',oct_h,'1 · October hinges'),('gold',oct_g,'2 · October gold-only')]:
        axs[0].stairs(percent_hist(a),BINS,color=COLORS[group],label=f'{name} · n={len(a)}',lw=2)
        axs[0].axvline(pools[group]['median'],color=COLORS[group],ls=':',lw=1)
    axs[0].set(xlim=(0,90),xlabel='Effective optical θ (degrees)',ylabel='Candidates (%) per 5° bin',title='Pooled October distributions')
    axs[0].legend(frameon=False,fontsize=9)
    oct_s=stacks[:10];labels=[]
    for k,s in enumerate(oct_s):
        a=values(s);su=s['summary'];labels.append(s['time'])
        axs[1].plot([k,k],[su['q25'],su['q75']],color=COLORS[s['group']],lw=5,alpha=.5)
        axs[1].plot(k,su['median'],'o',color=COLORS[s['group']],ms=5)
    axs[1].set(xticks=range(10),xticklabels=labels,ylim=(0,90),ylabel='Effective optical θ (degrees)',title='Each October stack: median and middle 50%')
    axs[1].tick_params(axis='x',rotation=60,labelsize=8);axs[1].axvline(4.5,color='#aab2b3',lw=.8)
    fig.suptitle('Same native detection and local contrast ≥8 · one θ per retained candidate',fontsize=13)
    fig.savefig(OUT/'october_theta.png',dpi=170);plt.close(fig)

    sept=[s for s in stacks if s['group']=='september']
    all_hist=[percent_hist(values(s)) for s in sept]+[percent_hist(oct_h),percent_hist(oct_g)]
    upper=max(float(a.max()) for a in all_hist)*1.18
    fig,axs=plt.subplots(2,4,figsize=(14,7),layout='constrained',sharex=True,sharey=True)
    for ax,s in zip(axs.flat,sept):
        a=values(s);su=s['summary'];hist=percent_hist(a)
        ax.bar(BINS[:-1]+2.5,hist,width=4.5,color=COLORS['september'],alpha=.65,label='This September stack')
        ax.stairs(percent_hist(oct_h),BINS,color=COLORS['hinges'],lw=1.25,label='October hinges (1)')
        ax.stairs(percent_hist(oct_g),BINS,color=COLORS['gold'],lw=1.25,label='October gold-only (2)')
        ax.set(title=f"18 Sep {s['time']} · n={len(a)}\nmedian {su['median']:.1f}°",xlim=(0,90),ylim=(0,upper),xticks=[0,30,60,90])
    axs.flat[0].legend(fontsize=7,frameon=False)
    for ax in axs[1]:ax.set_xlabel('Effective optical θ (degrees)')
    for ax in axs[:,0]:ax.set_ylabel('Candidates (%) per 5° bin')
    fig.suptitle('Eight separate September distributions · common selection and model · no September pooling',fontsize=13)
    fig.savefig(OUT/'september_theta.png',dpi=170);plt.close(fig)

    fig,axs=plt.subplots(2,1,figsize=(14,7.2),layout='constrained')
    x=np.arange(len(stacks));kept=np.array([s['summary']['n'] for s in stacks]);local=np.array([s['summary']['local_excluded'] for s in stacks]);other=np.array([s['summary']['recurring_after_local']+s['summary']['invalid_theta_after_filters'] for s in stacks])
    colors=[COLORS[s['group']] for s in stacks]
    axs[0].bar(x,kept,color=colors,label='Used for θ')
    axs[0].bar(x,local,bottom=kept,color='#ef9c77',label='Excluded: local contrast')
    axs[0].bar(x,other,bottom=kept+local,color='#707477',label='Excluded: recurrence / invalid θ')
    for k,s in enumerate(stacks):axs[0].text(k,s['summary']['detected']+1,str(s['summary']['detected']),ha='center',fontsize=8)
    axs[0].set(ylabel='Candidates per full sensor frame',title='Counts from exactly the same detector; labels show detections before exclusions')
    axs[0].legend(fontsize=8,frameon=False,ncol=3,loc='upper left')
    axs[0].set_ylim(0,max(s['summary']['detected'] for s in stacks)*1.22)
    axs[1].bar(x,[s['image_stats']['dog_threshold'] for s in stacks],color=colors)
    axs[1].set(ylabel='DoG threshold (saved intensity units)',title='k=8 is fixed; its absolute intensity threshold adapts to each image')
    for ax in axs:
        ax.set_xticks(x);ax.set_xticklabels([s['time'] for s in stacks],rotation=55,ha='right',fontsize=8)
        for sep in (4.5,9.5):ax.axvline(sep,color='#9ca7a9',lw=.8)
    fig.suptitle('October hinges (5) | October gold-only (5) | September (8) · same frame size, exposure, gains and frame count',fontsize=12)
    fig.savefig(OUT/'counts_and_thresholds.png',dpi=165);plt.close(fig)


def page(state):
    summary_rows=[]
    for s in state['stacks']:
        u=s['summary'];label={'hinges':'1 · Oct hinges','gold':'2 · Oct gold','september':'18 Sep'}[s['group']]
        summary_rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in (label,s['time'],u['detected'],u['local_excluded'],u['recurring_after_local'],u['invalid_theta_after_filters'],u['n'],f"{u['median']:.1f}°",f"{u['q25']:.1f}–{u['q75']:.1f}°"))+'</tr>')
    h=state['pools']['hinges'];g=state['pools']['gold']
    sept=[s for s in state['stacks'] if s['group']=='september']
    october=[s for s in state['stacks'] if s['group']!='september']
    s_medians=[s['summary']['median'] for s in sept]
    s_counts=[s['summary']['detected'] for s in sept]
    s_thresholds=[s['image_stats']['dog_threshold'] for s in sept]
    o_thresholds=[s['image_stats']['dog_threshold'] for s in october]
    interpretation=(f"The October hinge and gold-only distributions overlap broadly; their pooled medians differ by {h['median']-g['median']:.1f}°. "
        f"All eight September medians are higher, spanning {min(s_medians):.1f}–{max(s_medians):.1f}°. These are descriptive comparisons under the stated selection and model.")
    density_note=(f"September produces {min(s_counts)}–{max(s_counts)} native detections per stack. "
        f"Its absolute DoG thresholds are {min(s_thresholds):.1f}–{max(s_thresholds):.1f}, versus {min(o_thresholds):.1f}–{max(o_thresholds):.1f} in October. "
        "The September images therefore contain more peaks that pass the same detector even at substantially higher intensity thresholds. "
        "Exposure, camera gains, sensor size and frame count match in the metadata. The available files do not establish whether concentration, illumination, focus, motion or other acquisition differences caused the higher counts. Crowding and split detections can also inflate object counts.")
    html=HTML.replace('__DATA__',json.dumps(state['stacks'],allow_nan=False)).replace('__TABLE__',''.join(summary_rows))
    html=html.replace('__H_N__',str(h['n'])).replace('__H_MED__',f"{h['median']:.1f}").replace('__G_N__',str(g['n'])).replace('__G_MED__',f"{g['median']:.1f}")
    html=html.replace('__INTERPRETATION__',interpretation).replace('__DENSITY_NOTE__',density_note)
    (OUT/'index.html').write_text(html,encoding='utf-8')


HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>θ statistics · hinges, gold and September</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f7f7f4;color:#222e31;font:16px/1.5 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:25px}h1{font-size:29px;margin:6px 0}h2{font-size:21px;margin-top:30px}p{max-width:1100px}a{color:#176c78}.figure{width:100%;height:auto}.note{font-size:14px;color:#506368}.scroll{overflow:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:8px 11px;border-bottom:1px solid #d0dad4;text-align:left;white-space:nowrap}.controls{display:flex;gap:15px;flex-wrap:wrap;align-items:center;background:#e3ece8;padding:12px;border-radius:6px}select,button{padding:7px;font:inherit;border:1px solid #9eaea7;border-radius:4px;background:white}#viewport{max-height:74vh;overflow:auto;background:#080b0d;margin-top:12px}canvas{display:block;cursor:crosshair}.inspect{display:flex;gap:25px;flex-wrap:wrap;margin-top:16px}#crop{width:320px;max-width:100%;height:auto}.inspect p{max-width:780px;margin-top:0}#spots{max-height:320px;overflow:auto;font-size:14px}thead{background:#f7f7f4}#spots thead{position:sticky;top:0}details{margin-top:20px}</style>
<main><div class="note">UPDATED SELECTION · NATIVE k=8 + LOCAL CONTRAST ≥8</div>
<h1>θ statistics: hinges, gold-only, and eight September fields</h1>
<p><b>Statistics 1: October hinges — __H_N__ values, median __H_MED__°.</b><br><b>Statistics 2: October gold-only — __G_N__ values, median __G_MED__°.</b><br>Each September stack has its own distribution below. No September pooling or cross-stack recurrence exclusion is applied.</p>
<p>__INTERPRETATION__</p>
<p>All 18 stacks use the same detector, local contrast filter, 18 × 18 pixel aperture, 150-frame channel average and optical model. The previously questioned low-contrast candidates are now excluded from these statistics. Counts refer to candidate detections; unresolved clusters or split spots are still possible.</p>
<img class="figure" src="october_theta.png" alt="Updated October hinge and pure gold theta distributions">
<h2>September: one distribution per stack</h2><img class="figure" src="september_theta.png" alt="Eight independent September distributions compared with October references">
<p class="note">Histograms show percentage of each retained population per 5° bin, so differing counts do not change curve height by themselves. Each candidate has equal weight within its stack or October pool. θ is the effective polar angle from the optical axis.</p>
<h2>Counts and selection audit</h2><img class="figure" src="counts_and_thresholds.png" alt="Spot counts and thresholds for all 18 stacks">
<p>__DENSITY_NOTE__</p>
<div class="scroll"><table><thead><tr><th>Group</th><th>Stack</th><th>Detected</th><th>Local excluded</th><th>Repeat excluded*</th><th>Invalid θ*</th><th>θ values</th><th>Median</th><th>Middle 50%</th></tr></thead><tbody>__TABLE__</tbody></table></div>
<p class="note">*Sequential counts after local exclusion, with no double counting. October repeats require detection within 6 sensor pixels in all five stacks of that group. The CSV retains every candidate and all exclusion reasons.</p>
<h2>Inspect included and excluded spots</h2><div class="controls"><label>Stack <select id="stack"></select></label><label>Spot <select id="spot"></select></label><label><input id="marks" type="checkbox" checked> Marks</label><label><input id="labels" type="checkbox"> IDs</label><label>View <select id="zoom"><option value="fit">Fit width</option><option value="1">1:1 pixels</option><option value="2">2×</option></select></label></div>
<p class="note">Included: θ colour scale, purple 0° → green → yellow 90°. Excluded: orange ×. Click near any spot or use the selector. Images are the 150-frame polarization-averaged detection maps at the same fixed 20× display gain; bright display pixels do not alone establish acquisition saturation.</p>
<div id="viewport"><canvas id="view" aria-label="Included and excluded detected spots"></canvas></div><div class="inspect"><canvas id="crop" width="320" height="320" aria-label="Selected spot with extraction window"></canvas><div><p id="selected"></p><p id="settings" class="note"></p></div></div><div id="spots" class="scroll"></div>
<details><summary>Method and comparison limits</summary>
<p>Native intensity detector: mean of all 150 measurement frames, 2 × 2 polarization-cell average repeated to sensor resolution, Gaussian σ=1 minus σ=6 pixels, global mean + 8 SD, blob area 10–250 pixels, 10-pixel border crop. The phase-marker frame is excluded.</p>
<p>Local selection: peak filtered signal compared with the median and 1.4826 × MAD in a 12–30 pixel ring, masking 8-pixel cores around other candidates and sampling once per 2 × 2 cell. Score below 8 or an unscorable neighbourhood is excluded. This exploratory spatial-contrast criterion is not a calibrated probability/SNR. No new shape, isolation or motion rejection is silently added; neighbours within 24 pixels are flagged in the table.</p>
<p>One θ per spot: native channel extraction from an 18 × 18 raw-pixel square; average I0, I90, I45 and I135 over all frames, then X=(I0−I90)/(I0+I90), Y=(I45−I135)/(I45+I135). Existing paper-analysis Fourkas annular water model: NA=1.3, n=1.33, inner NA=0.39, r_max≈0.920825. Values outside the model are excluded, not forced to 90°. No fitted gains, extra channel-background subtraction or solid-angle correction. This gives an effective polar angle from mean intensities; motion during the stack can reduce averaged anisotropy.</p>
<p>All files record subtraction on and the same sensor dimensions, actual exposure (~0.598 ms), actual frame rate (~72.175 fps), gains (1/1), and 150 acquired frames. They do not preserve background identity hashes. October group assignments come from your acquisition account. September profiles are therefore treated independently.</p>
<p>The legacy sound checkbox is off in all September stacks and on in these October stacks; it is recorded operator state, not independent hardware verification. The first gold note also records re-added oil. Consequently, these distributions compare recordings under differing conditions, not an isolated test of hinge biology.</p>
<p>Seven September files have recorded raw maxima of 4094, near the 12-bit ceiling. Saved data have already undergone background subtraction, so they cannot reliably locate or undo raw clipping. The table flags any candidate aperture with saved maximum ≥4080, without adding an unrequested exclusion. Large blobs, very dim candidates and overlapping rods can still be selected differently despite identical parameter values; count differences are not a calibrated concentration measurement.</p>
<p>Original data and camera code are unchanged. Earlier unfiltered reports remain historical; this page is the comparison with the new local exclusions applied.</p></details>
<p><a href="spots.csv">All spot coordinates, θ and exclusion reasons</a> · <a href="stack_summary.csv">Per-stack statistics</a> · <a href="results.json">Full settings and provenance</a></p></main>
<script>const stacks=__DATA__,$=id=>document.getElementById(id),view=$('view'),ctx=view.getContext('2d'),cc=$('crop').getContext('2d');let current=0,chosen=0,scale=1,im=null,ticket=0;
const fmt=(x,d=2)=>x===null?'—':Number(x).toFixed(d);stacks.forEach((s,i)=>$('stack').add(new Option((s.group==='september'?'18 Sep':s.group==='gold'?'2 · Gold':'1 · Hinges')+' '+s.time+' · '+s.summary.n+' kept',i)));
function init(){const s=stacks[current];chosen=0;$('spot').innerHTML='';s.rows.forEach((r,i)=>$('spot').add(new Option('#'+r.spot_id+' · '+(r.included?fmt(r.theta_deg,1)+'°':'excluded'),i)));$('spots').innerHTML='<table><thead><tr><th>ID</th><th>θ</th><th>Local score</th><th>Nearest candidate</th><th>Saved max</th><th>Selection</th></tr></thead><tbody>'+s.rows.map(r=>'<tr><td>'+r.spot_id+'</td><td>'+fmt(r.theta_deg)+'°</td><td>'+fmt(r.local_score)+'</td><td>'+fmt(r.nearest_candidate_px,1)+' px'+(r.neighbor_under_24px?' · close':'')+'</td><td>'+r.aperture_saved_max+'</td><td>'+(r.included?'Included':r.exclusion_reason.replaceAll('_',' '))+'</td></tr>').join('')+'</tbody></table>';
$('settings').textContent='Exposure '+fmt(s.settings.exposure_us/1000,3)+' ms · '+fmt(s.settings.fps,3)+' fps · '+s.settings.frames+' frames · gains '+s.settings.gain_analog+'/'+s.settings.gain_digital+' · background subtraction on · sound checkbox '+(s.settings.sound_checkbox?'on':'off')+'. Global DoG threshold '+fmt(s.image_stats.dog_threshold)+'.';im=null;ctx.clearRect(0,0,view.width,view.height);$('selected').textContent='Loading image…';const id=++ticket,img=new Image();img.onload=()=>{if(id!==ticket)return;im=img;draw()};img.src=s.image}
function draw(){if(!im)return;scale=$('zoom').value==='fit'?Math.min(1,$('viewport').clientWidth/im.width):Number($('zoom').value);view.width=Math.round(im.width*scale);view.height=Math.round(im.height*scale);ctx.imageSmoothingEnabled=scale<1;ctx.drawImage(im,0,0,view.width,view.height);if($('marks').checked)stacks[current].rows.forEach((r,i)=>{const x=(r.x+.5)*scale,y=(r.y+.5)*scale,rad=Math.max(4,9*scale);ctx.strokeStyle=r.color;ctx.lineWidth=i===chosen?2:1.2;ctx.beginPath();if(r.included)ctx.arc(x,y,rad,0,2*Math.PI);else{ctx.moveTo(x-rad,y-rad);ctx.lineTo(x+rad,y+rad);ctx.moveTo(x-rad,y+rad);ctx.lineTo(x+rad,y-rad)}ctx.stroke();if($('labels').checked){ctx.fillStyle=r.color;ctx.font='12px system-ui';ctx.fillText(r.spot_id,x+rad+1,y-2)}});crop()}
function crop(){const s=stacks[current],r=s.rows[chosen];if(!r)return;const x0=Math.round(r.x)-40,y0=Math.round(r.y)-40;cc.fillStyle='black';cc.fillRect(0,0,320,320);cc.imageSmoothingEnabled=false;cc.drawImage(im,x0,y0,80,80,0,0,320,320);if($('marks').checked){cc.strokeStyle=r.color;cc.lineWidth=1.5;cc.strokeRect((r.roi_left-x0)*4,(r.roi_top-y0)*4,72,72)}$('selected').innerHTML='<b>'+s.date+' '+s.time+' · #'+r.spot_id+' · '+(r.included?'Included':'Excluded')+'</b><br>θ = '+fmt(r.theta_deg)+'°'+(r.included?'':' (not used in statistics)')+' · local score '+fmt(r.local_score)+'<br>Sensor centre ('+fmt(r.x)+', '+fmt(r.y)+') · nearest candidate '+fmt(r.nearest_candidate_px,1)+' pixels<br>'+(r.included?'Passed selection.':r.exclusion_reason.replaceAll('_',' '))+ (r.neighbor_under_24px?'<br>Close-neighbour flag: inspect possible overlap/split detection.':'')}
$('stack').onchange=()=>{current=Number($('stack').value);init()};$('spot').onchange=()=>{chosen=Number($('spot').value);draw()};for(const id of ['marks','labels','zoom'])$(id).onchange=draw;view.onclick=e=>{if(!im)return;const b=view.getBoundingClientRect(),x=(e.clientX-b.left)/scale-.5,y=(e.clientY-b.top)/scale-.5;let distance=Infinity;stacks[current].rows.forEach((r,i)=>{const d=Math.hypot(x-r.x,y-r.y);if(d<distance){distance=d;chosen=i}});$('spot').value=chosen;draw()};window.onresize=()=>{$('zoom').value==='fit'&&draw()};init();</script></html>'''

if __name__=='__main__':
    state=json.loads((OUT/'results.json').read_text(encoding='utf-8')) if '--render-only' in sys.argv else compute()
    figures(state);page(state)
