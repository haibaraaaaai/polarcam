"""Matched gold controls and conditional synthetic-rod recovery, read-only inputs.

CLI: --phase gold / simulation / render / all. No acquisition or GUI edits.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from background_percentile_spots import detect
from detector_threshold_trial import select_spots
from hinges_percentile_theta import background, stats, finite
from background_region_trial import inside, ELLIPSE
from polarcam import optics
from polarcam.live.intensity_spots import channel_means, anisotropy

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DAY=ROOT/'data/hinges/polarcam/2026-10-12'
OUT=HERE/'control_recovery_check'
HINGES=HERE/'background_overlap_check/hinges95_theta.json'
MODEL=optics.theta_model()
ABC=[MODEL[k] for k in ['A','B','C']]
SEED=2026101403
THETAS=[15,25,31,40,55,70,80]
BRIGHTNESS=[200,800,2400]
N_PHI=8
SIGMA=2.5


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def sig(p):
    s=p.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def write(name,data):
    (OUT/name).write_text(json.dumps(clean(data),ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf8')


def subtraction_function():
    p=ROOT/'src/polarcam/live/fetch_frames.py'
    tree=ast.parse(p.read_text(encoding='utf8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_crop_background_profile','_subtract_background']]
    assert len(nodes)==2
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return ns['_subtract_background'],sha(p)


def evaluate(mean_min,mean95,peak,prefix):
    """Exact same detector/extraction/exclusions as hinges_percentile_theta.py."""
    result,_=detect(mean95,peak,prefix);rows=result['spots']
    channels=channel_means(mean_min,[s['bounds'] for s in rows]);xy=anisotropy(channels)
    rr=np.linalg.norm(xy,axis=1);theta=np.degrees(optics.theta_from_radius(rr,*ABC))
    for j,s in enumerate(rows):
        x,y,w,h=s['bbox'];left,top=round(x+.5),round(y+.5)
        s['raw_peak']=max(s['raw_peak'],int(peak[top:top+h,left:left+w].max()))
        s.update(saturated=s['raw_peak']>=4094,accepted=True,channels=channels[j].tolist(),
                 xy=clean(xy[j]),r=finite(rr[j]),theta_deg=finite(theta[j]),
                 theta_status='valid' if np.isfinite(theta[j]) else 'above_model_limit' if rr[j]>MODEL['r_max'] else 'undefined_r')
    select_spots(rows)
    return dict(threshold=result['threshold'],spots=rows,summary=stats(rows))


def load_means(path,extra):
    signature=sig(path);meta_path=path.with_suffix('.json');m=json.loads(meta_path.read_text(encoding='utf8'))
    n=m['actual']['frames'];a=np.load(path,mmap_mode='r');assert n==150 and len(a)==151 and m['actual']['phase_marker_appended']
    assert m['background']['background_subtracted'] is True and a.dtype==np.uint16
    total=np.zeros(a.shape[1:],np.uint32);total95=np.zeros_like(total)
    work=np.empty(a.shape[1:],np.float32)
    for f in a[:n]:
        total+=f;np.subtract(f,extra,out=work);np.maximum(work,0,out=work);np.rint(work,out=work);total95+=work.astype('uint16')
    info=m['fetch_frames_output']['raw_peak'];p=path.parent/info['path']
    assert sha(p)==info['sha256'] and info['frames']==n
    peak=np.load(p);assert peak.shape==extra.shape and sig(path)==signature
    provenance=dict(name=path.name,source_signature=signature,sidecar_sha256=sha(meta_path),raw_peak_sha256=info['sha256'],
                    background_sha256=m['background']['background_profile_sha256'],fps=m['actual']['fps'],
                    exposure_us=m['actual']['timing']['exposure_us'],gains=m['actual']['gains'],
                    note=m['conditions'].get('condition_note',''))
    return total.astype(float)/n,total95.astype(float)/n,peak,a,provenance


def gold():
    OUT.mkdir(exist_ok=True)
    old=json.loads((HERE/'background_sensitivity/results.json').read_text(encoding='utf8'))
    inputs=[DAY/'frames'/r['name'] for r in old['records']];assert len(inputs)==14
    bghash=old['background_sha256'];extra,bgmeta=background(bghash)
    result=dict(complete=False,model=MODEL,background=bgmeta,records=[],builder_sha256=sha(Path(__file__)),
                hinge_source_sha256=sha(HINGES),pipeline='Same p95-detection/minimum-extraction pipeline as hinges95_theta.json; '
                '2 SD, >=10 component pixels without upper limit, fixed ellipse, raw >=4094 aperture OR bbox, strict 18x18 overlap suppression.',
                scope='14 gold-only wide-field survey stacks, 19:22:31-19:31:27. Earlier gold and targeted narrow-field recordings omitted.',
                limitation='Gold-only is a no-hinge control, not a rod-free blank. Notes flag illumination/oil conditions; '
                'one surface/preparation, shared backgrounds and possible repeat fields do not provide independent biological replicates.')
    for i,p in enumerate(inputs,1):
        mm,mp,peak,_,provenance=load_means(p,extra)
        assert provenance['background_sha256']==bghash
        ev=evaluate(mm,mp,peak,f'gold{i}')
        result['records'].append(dict(index=i,**provenance,**ev))
        write('gold_results.json',result)
        print(f"Gold {i}/14: {ev['summary']['selected']} kept, {ev['summary']['valid']} valid theta, {ev['summary']['raw_saturated']} saturated candidates",flush=True)
    result['complete']=True;result['pooled']=stats([s for r in result['records'] for s in r['spots']])
    write('gold_results.json',result)
    return result


def truth_catalog(base,shape,index):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,index]));h,w=shape
    tree=cKDTree([[s['x'],s['y']] for s in base['spots']]);positions=[]
    for _ in range(200000):
        p=rng.uniform([30,30],[w-30,h-30])
        if inside(*p) or tree.query(p,p=np.inf)[0]<32:continue
        if positions and np.min(np.max(np.abs(np.array(positions)-p),axis=1))<48:continue
        positions.append(p)
        if len(positions)==len(THETAS)*len(BRIGHTNESS)*N_PHI:break
    assert len(positions)==168
    # Random assignment of theta/brightness/phi to positions, not based on image intensity.
    positions=np.array(positions)[rng.permutation(len(positions))]
    catalog=[]
    for ip in range(N_PHI):
        phi=(ip+.5)*np.pi/N_PHI
        for theta in THETAS:
            for peak90 in BRIGHTNESS:
                x,y=positions[len(catalog)];r=float(optics.radius_from_theta(np.radians(theta),*ABC))
                catalog.append(dict(id=len(catalog)+1,batch=THETAS.index(theta)*len(BRIGHTNESS)+BRIGHTNESS.index(peak90),x=float(x),y=float(y),theta=theta,phi_deg=float(np.degrees(phi)),
                    peak90=peak90,mean_peak=peak90*np.sin(np.radians(theta))**2,true_r=r,
                    true_xy=[r*np.cos(2*phi),r*np.sin(2*phi)]))
    return catalog


def signal_patch(t):
    # Even origin guarantees correct analyzer phase for patch extraction.
    left=2*int(np.floor((t['x']-16)/2));top=2*int(np.floor((t['y']-16)/2))
    yy,xx=np.mgrid[top:top+34,left:left+34]
    psf=np.exp(-((xx-t['x'])**2+(yy-t['y'])**2)/(2*SIGMA**2))
    signal=np.zeros(psf.shape,np.float32)
    X,Y=t['true_xy'];amplitudes=t['mean_peak']*np.array([1+X,1-X,1+Y,1-Y])
    l,tt=round(t['x'])-9,round(t['y'])-9;in_ap=(xx>=l)&(xx<l+18)&(yy>=tt)&(yy<tt+18)
    for amp,(py,px) in zip(amplitudes,[(1,1),(0,0),(0,1),(1,0)]):
        phase=(yy%2==py)&(xx%2==px)
        # Equal integrated PSF weight in each analyzer, so supplied theta is exact before digitization.
        signal[phase]=amp*psf[phase]*(2*np.pi*SIGMA**2/4)/psf[phase&in_ap].sum()
    bounds=[l-left,l-left+18,tt-top,tt-top+18]
    channels=channel_means(signal,[bounds],origin=(left,top))[0]
    np.testing.assert_allclose(anisotropy(channels),t['true_xy'],atol=3e-8,rtol=0)
    return signal,left,top,bounds


def match_truth(ev,truth,baseline):
    kept=[s for s in ev['spots'] if s['selection']=='kept'];points=np.array([[s['x'],s['y']] for s in kept])
    candidates=np.array([[s['x'],s['y']] for s in ev['spots']]);measure=[];used=set()
    for t in truth:
        dist=np.linalg.norm(points-[t['x'],t['y']],axis=1) if len(points) else np.array([])
        matches=np.flatnonzero(dist<=6)
        assert len(matches)<=1  # strict aperture exclusion and separated injected sites
        item={**t,'recovered':bool(len(matches)),'recovered_theta':None,'recovered_r':None,'distance':None,'reason':'no_candidate_within_6px'}
        if len(matches):
            j=int(matches[0]);assert j not in used;used.add(j);s=kept[j]
            item.update(recovered_theta=s['theta_deg'],recovered_r=s['r'],distance=float(dist[j]),reason='kept',
                recovered_xy=s['xy'],intensity_mean=float(np.mean(s['channels'])),detected_id=s['id'])
        elif len(candidates):
            dc=np.linalg.norm(candidates-[t['x'],t['y']],axis=1);near=np.flatnonzero(dc<=6)
            if len(near):item['reason']=ev['spots'][int(near[np.argmin(dc[near])])]['selection']
        measure.append(item)
    return measure


def simulate():
    OUT.mkdir(exist_ok=True);subtract,subtract_sha=subtraction_function()
    hd=json.loads(HINGES.read_text(encoding='utf8'))
    result=dict(complete=False,model=MODEL,seed=SEED,theta_grid=THETAS,brightness_grid=BRIGHTNESS,phi_count=N_PHI,
        sigma_sensor_px=SIGMA,source_stack_indices=[1,6,10],backgrounds=[],fields=[],clean=[],
        source_sha256=sha(HINGES),builder_sha256=sha(Path(__file__)),subtraction_code_sha256=subtract_sha,
        assumptions=[
            'Inject isolated Gaussian spots into actual 150-frame image patches; retain existing temporal noise, fringes and residual BG. No fresh photon/read noise is simulated; no calibrated DN-per-photon conversion is available.',
            'Gaussian sigma 2.5 sensor pixels, stationary orientation and position. No orientation-dependent PSF or coherent rod-background cross term; this is a conditional pipeline test, not a complete optical forward model.',
            'Mean polarization-channel peak = P90*sin(theta)^2, P90=200/800/2400 DN. This brightness law is imposed, not fitted to the 30-degree feature.',
            'Analyzer intensities proportional to [1+X,1-X,1+Y,1-Y]; X/Y follow the existing Fresnel+BFP+annular model. Channel PSF weights normalized to give exact supplied X/Y in the true 18x18 aperture before digitization.',
            '168 sites per field, >=48 px apart, >=32 px (Chebyshev) from every existing p95 candidate and outside the prior ellipse. Sites chosen without using local intensity or measured angles. Injections are isolated and do not test crowded-spot deblending.',
            '21 sparse batches per field: 8 rods at a time with the SAME theta and P90, evenly spaced phi. This prevents adding bright test rods from raising the threshold for dim test rods. Global 2-SD threshold still recomputed each batch; changes reported, not assumed negligible.',
            'At saved positive pixels raw reconstruction is saved+Bmin. At saved zeros true raw is unknown in [0,Bmin]; lower=0, upper=Bmin. Both extremes preserve the original no-injection saved images. They bracket injected pixel values, not necessarily nonlinear angle estimates or all possible recovery rates.',
            'Camera digitization rounds and clips to 4094 DN, conservative observed raw ceiling; original verified raw-peak map is also retained. Any raw saturation in aperture OR component bbox removes the candidate.',
            'Original GUI subtraction function is called on every modified raw patch before averaging, then p95 subtraction is applied to that saved patch exactly as in the analysis. No injection changes any acquisition file or background profile.',
            'One-to-one match within 6 px of the injected position. Pre-existing or unmatched detections are not counted as recovered injections. Empty-field simulations are numerical references, not biological negative controls.'
        ])
    bg_cache={}
    # Use the same positions in both clipping scenarios, for paired comparisons.
    for index in [1,6,10]:
        rec=hd['records'][index-1];bh=rec['background_sha256']
        if bh not in bg_cache:
            extra,provenance=background(bh);bmin=np.load(DAY/'backgrounds/used'/f'{bh}.npy').astype(np.float32)
            bg_cache[bh]=(bmin,extra,provenance);result['backgrounds'].append(provenance)
        bmin,extra,_=bg_cache[bh]
        path=DAY/'frames'/rec['name'];assert sig(path)==rec['source_signature']
        mean,mean95,peak,frames,provenance=load_means(path,extra)
        base=evaluate(mean,mean95,peak,f'baseline{index}')
        # Exact reproduction of previous full detector and retained results.
        for new,old in zip(base['spots'],rec['spots']):
            assert new['component']==old['component'] and new['selection']==old['selection']
            np.testing.assert_allclose([new['x'],new['y'],new['r']],[old['x'],old['y'],old['r']],atol=1e-12,rtol=0)
        assert len(base['spots'])==len(rec['spots'])
        catalog=truth_catalog(base,mean.shape,index)
        fields=dict(index=index,background_group=rec['background_group'],provenance=provenance,
                    baseline_threshold=base['threshold'],baseline_summary=base['summary'],trials=[])
        patches={}
        for t in catalog:
            signal,l,tp,bounds=signal_patch(t);h,w=signal.shape
            saved=np.array(frames[:150,tp:tp+h,l:l+w]);bg=bmin[tp:tp+h,l:l+w];ext=extra[tp:tp+h,l:l+w]
            bgch=channel_means(bg,[bounds],origin=(l,tp))[0]
            t.update(zero_fraction=float(np.mean(saved[:,bounds[2]:bounds[3],bounds[0]:bounds[1]]==0)),
                     recorded_bg_mean=float(bgch.mean()),recorded_bg_xy=clean(anisotropy(bgch)))
            patches[t['id']]=(signal,l,tp,bounds,saved,bg,ext)
        for batch in range(len(THETAS)*len(BRIGHTNESS)):
            truth=[t for t in catalog if t['batch']==batch]
            # Clean numerical reference, same injected positions, amplitudes, phi and detector.
            clean_img=np.zeros(mean.shape,np.float64);clean_peak=np.zeros(mean.shape,np.uint16)
            for t in truth:
                signal,l,tp,bounds,*_=patches[t['id']];h,w=signal.shape
                pixel=np.rint(np.clip(signal,0,4094)).astype('uint16')
                clean_img[tp:tp+h,l:l+w]=pixel;clean_peak[tp:tp+h,l:l+w]=pixel
            clean_ev=evaluate(clean_img,clean_img,clean_peak,f'clean{index}_{batch}')
            result['clean'].append(dict(index=index,batch=batch,threshold=clean_ev['threshold'],
                measures=match_truth(clean_ev,truth,base),summary=clean_ev['summary']))
            for bound in ['lower','upper']:
                injected=mean.copy();injected95=mean95.copy();new_peak=peak.copy();truth_measure=[]
                for t in truth:
                    signal,l,tp,bounds,saved,bg,ext=patches[t['id']];h,w=signal.shape
                    raw=saved.astype(np.float32)+bg
                    if bound=='lower':raw[saved==0]=0
                    # Zero-injection reconstruction must reproduce the actual saved minimum subtraction.
                    for j in [0,74,149]:np.testing.assert_array_equal(subtract(raw[j].astype('uint16'),bg,None,None),saved[j])
                    raw_added=np.rint(np.clip(raw+signal,0,4094)).astype('uint16')
                    corrected=np.stack([subtract(f,bg,None,None) for f in raw_added])
                    corrected95=np.rint(np.maximum(corrected.astype(np.float32)-ext,0)).astype('uint16')
                    injected[tp:tp+h,l:l+w]=corrected.mean(axis=0,dtype=np.float64)
                    injected95[tp:tp+h,l:l+w]=corrected95.mean(axis=0,dtype=np.float64)
                    new_peak[tp:tp+h,l:l+w]=np.maximum(new_peak[tp:tp+h,l:l+w],raw_added.max(axis=0))
                    ch=channel_means(injected[tp:tp+h,l:l+w],[bounds],origin=(l,tp))[0]
                    rr=float(np.linalg.norm(anisotropy(ch)));angle=finite(float(np.degrees(optics.theta_from_radius(rr,*ABC))))
                    t['forced_r']=rr;t['forced_theta']=angle  # diagnostic even if detector misses
                    ap=raw_added[:,bounds[2]:bounds[3],bounds[0]:bounds[1]]
                    t['injected_saturated']=bool(ap.max()>=4094)
                ev=evaluate(injected,injected95,new_peak,f'inj{index}_{batch}_{bound}')
                measures=match_truth(ev,truth,base)
                fields['trials'].append(dict(batch=batch,bound=bound,threshold=ev['threshold'],summary=ev['summary'],measures=measures))
                if index==6 and truth[0]['theta'] in [31,55] and truth[0]['peak90']==800:
                    sample_previews(truth,patches,injected,injected95,mean,mean95,bound)
            print(f'Injection field {index}, batch {batch+1}/21: true {truth[0]["theta"]} deg, P90 {truth[0]["peak90"]}; clean + both clipping extremes complete',flush=True)
        result['fields'].append(fields)
        assert sig(path)==rec['source_signature']
        write('simulation_results.json',result)
    result['complete']=True;write('simulation_results.json',result)
    return result


def sample_previews(truth,patches,inj,inj95,base,base95,bound):
    choose=[truth[0],truth[4]]
    fig,axes=plt.subplots(2,4,figsize=(11,6),layout='constrained')
    for row,t in enumerate(choose):
        signal,l,tp,bounds,*_=patches[t['id']];h,w=signal.shape
        for col,(im,label) in enumerate([(base,'Original min-BG mean'),(inj,'Injected min-BG mean'),(base95,'Original p95 mean'),(inj95,'Injected p95 mean')]):
            ax=axes[row,col];ax.imshow(im[tp:tp+h,l:l+w],cmap='gray',vmin=0,vmax=4095/20,interpolation='nearest')
            ax.set_title(label,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
        axes[row,0].set_ylabel(f"True θ={t['theta']}°\nφ={t['phi_deg']:.1f}°; P90=800 DN")
    fig.suptitle(f'Examples in hinge stack 6 · {bound} raw-value assumption at saved zeros\n34 × 34 sensor pixels, fixed 20× display; no appearance-based selection',fontsize=11)
    fig.savefig(OUT/f'injection_examples_{bound}_{truth[0]["theta"]}.png',dpi=160);plt.close(fig)


def control_summary(records):
    rows=[s for r in records for s in r['spots'] if s['selection']=='kept']
    angles=np.array([s['theta_deg'] for s in rows if s['theta_deg'] is not None]);radius=np.array([s['r'] for s in rows])
    th=np.histogram(angles,np.arange(0,92,2))[0];rh=np.histogram(radius,np.linspace(0,np.sqrt(2),72))[0]
    per=[]
    for rec in records:
        a=np.array([s['theta_deg'] for s in rec['spots'] if s['selection']=='kept' and s['theta_deg'] is not None])
        hist=np.histogram(a,np.arange(0,92,2))[0]
        per.append(dict(index=rec['index'],name=rec['name'],n=len(a),p30=float(np.mean((a>=30)&(a<32))),
            p28=float(np.mean((a>=28)&(a<34))),hist=hist.tolist(),threshold=rec['threshold'],
            saturated=rec['summary']['raw_saturated']))
    equal=np.mean([np.array(r['hist'])/r['n'] for r in per],axis=0)
    return dict(n=len(rows),valid=len(angles),above=len(rows)-len(angles),theta_hist=th.tolist(),r_hist=rh.tolist(),
        p30=float(np.mean((angles>=30)&(angles<32))),p28=float(np.mean((angles>=28)&(angles<34))),
        equal_hist=equal.tolist(),equal_p28=float(np.mean([r['p28'] for r in per])),median=float(np.median(angles)),
        mode=[int(np.argmax(th)*2),int(np.argmax(th)*2+2)],per_stack=per,
        intensity_quantiles=np.quantile([np.mean(s['channels']) for s in rows],[.1,.5,.9]).tolist())


def simulation_summary(sim):
    out={}
    for scope in ['all','1','6','10']:
        fields=sim['source_stack_indices'] if scope=='all' else [int(scope)]
        out[scope]={}
        for case in ['clean','lower','upper']:
            if case=='clean':rows=[s for t in sim['clean'] if t['index'] in fields for s in t['measures']]
            else:rows=[s for f in sim['fields'] if f['index'] in fields for t in f['trials'] if t['bound']==case for s in t['measures']]
            cells=[]
            for peak in BRIGHTNESS:
                for theta in THETAS:
                    a=[s for s in rows if s['peak90']==peak and s['theta']==theta]
                    measured=np.array([s['recovered_theta'] for s in a if s['recovered_theta'] is not None])
                    forced=np.array([s['forced_theta'] for s in a if s.get('forced_theta') is not None])
                    cells.append(dict(peak90=peak,theta=theta,n=len(a),recovered=sum(s['recovered'] for s in a),
                        valid=len(measured),median=float(np.median(measured)) if len(measured) else None,
                        q25_q75=np.quantile(measured,[.25,.75]).tolist() if len(measured) else [],
                        forced_median=float(np.median(forced)) if len(forced) else None,
                        forced_q25_q75=np.quantile(forced,[.25,.75]).tolist() if len(forced) else [],
                        in28_34=int(np.sum((measured>=28)&(measured<34))),
                        reasons={reason:sum(s['reason']==reason for s in a) for reason in sorted(set(s['reason'] for s in a))}))
            out[scope][case]=cells
    return out


def figures(hd,gd,sim,summary):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    edges=np.arange(0,92,2);redges=np.linspace(0,np.sqrt(2),72)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for label,key,col in [('DNA / hinges','hinges','#087f8c'),('Gold-only control','gold','#bf7432')]:
        z=summary[key]
        axes[0].stairs(100*np.array(z['theta_hist'])/z['valid'],edges,label=f"{label} · n={z['valid']:,}",color=col)
        axes[1].stairs(100*np.array(z['equal_hist']),edges,label=label,color=col)
        axes[2].stairs(100*np.array(z['r_hist'])/z['n'],redges,label=label,color=col)
    for ax in axes[:2]:
        ax.axvspan(28,34,alpha=.12,color='#bfa040');ax.set(xlabel='Apparent θ (degrees)',xlim=(0,90),ylabel='Valid detections per 2° bin (%)');ax.grid(axis='y',alpha=.15)
    axes[0].set_title('Pool all detections');axes[1].set_title('Give each field equal weight')
    axes[2].set(xlabel='r (finite above-limit values included)',xlim=(0,1.05),ylabel='Retained detections per bin (%)',title='Radius before angle inversion')
    axes[2].axvline(MODEL['r_max'],color='#888',ls=':',lw=1);axes[0].legend(fontsize=8)
    fig.suptitle('Same pipeline: p95 detection at 2 SD, minimum-BG angle extraction\nOversized spots allowed; shared ellipse and raw saturation exclusions. Gold is a no-hinge control, not a rod-free blank.',fontsize=12)
    fig.savefig(OUT/'gold_vs_hinges.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(15,11),layout='constrained',sharex=True)
    for ax,rec in zip(axes.flat,summary['gold']['per_stack']):
        ax.stairs(100*np.array(rec['hist'])/rec['n'],edges,color='#bf7432')
        ax.axvspan(28,34,alpha=.14,color='#bfa040');ax.set(xlim=(0,90),xlabel='Apparent θ (degrees)',ylabel='% per 2° bin',title=f"Gold {rec['index']} · {rec['name'][-10:-4]} · n={rec['n']}")
        ax.text(.96,.92,f"28–34°: {100*rec['p28']:.1f}%",ha='right',va='top',transform=ax.transAxes,fontsize=8)
    for ax in list(axes.flat)[14:]:ax.axis('off')
    fig.suptitle('Each of the 14 gold survey stacks separately · no fields removed based on their results\nFields 10 and 12 contribute many detections; the equal-field comparison checks their influence.',fontsize=12)
    fig.savefig(OUT/'gold_each_stack.png',dpi=150);plt.close(fig)
    colors=['#b87932','#378aa1','#8c5595']
    for scope,cases in summary['simulation'].items():
        fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
        for row,bound in enumerate(['lower','upper']):
            ax,bx=axes[row]
            for peak,col in zip(BRIGHTNESS,colors):
                vals=[c for c in cases[bound] if c['peak90']==peak];clean_vals=[c for c in cases['clean'] if c['peak90']==peak]
                ax.plot(THETAS,[100*c['recovered']/c['n'] for c in vals],'-o',color=col,label=f'P90={peak} DN',ms=4)
                ax.plot(THETAS,[100*c['recovered']/c['n'] for c in clean_vals],':',color=col,alpha=.65)
                xx=[c['theta'] for c in vals if c['median'] is not None];yy=[c['median'] for c in vals if c['median'] is not None]
                low=[c['median']-c['q25_q75'][0] for c in vals if c['median'] is not None];high=[c['q25_q75'][1]-c['median'] for c in vals if c['median'] is not None]
                bx.errorbar(xx,yy,yerr=[low,high],color=col,marker='o',ms=4,capsize=2,label=f'P90={peak} DN')
            ax.set(xlim=(10,85),ylim=(-3,103),xlabel='Injected true θ (degrees)',ylabel='Detected and retained (%)',title=f'{bound.capitalize()} raw values at clipped zeros')
            bx.plot([0,90],[0,90],'--',color='#888',lw=1);bx.set(xlim=(10,85),ylim=(0,90),xlabel='Injected true θ (degrees)',ylabel='Recovered θ: median and IQR',title='Only recovered injections; bars are spread, not CI')
            ax.grid(alpha=.15);bx.grid(alpha=.15)
        axes[0,0].legend(fontsize=9)
        name='3 fields pooled' if scope=='all' else f'hinge field {scope}'
        fig.suptitle(f'Conditional recovery on {name} · same detector and extraction\nSolid: recorded backgrounds; dotted: clean numerical reference. Bright-rod losses include saturation rejection.',fontsize=12)
        fig.savefig(OUT/f'recovery_{scope}.png',dpi=160);plt.close(fig)
    # Contrast known 31-degree rods with a non-31 input, without mixing true-angle populations.
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained',sharex=True)
    for row,theta in enumerate([31,40]):
        for col,peak in enumerate(BRIGHTNESS):
            ax=axes[row,col]
            for case,color in [('lower','#397f9e'),('upper','#b66d3d')]:
                a=[s['recovered_theta'] for f in sim['fields'] for t in f['trials'] if t['bound']==case for s in t['measures'] if s['theta']==theta and s['peak90']==peak and s['recovered_theta'] is not None]
                ax.stairs(np.histogram(a,edges)[0],edges,color=color,label=f'{case}: {len(a)}/24 valid')
            ax.axvspan(28,34,color='#c4a04a',alpha=.17);ax.axvline(theta,color='#333',ls='--');ax.set(xlim=(0,65),xlabel='Recovered θ (degrees)',ylabel='Injected rods per bin',title=f'True θ={theta}° · P90={peak} DN');ax.legend(fontsize=8)
    fig.suptitle('Can a non-31° rod be reported in the near-30° band?\nEach panel has one known input angle; phi and position vary. Shading: 28–34°. Unrecovered/saturated injections absent from curves.',fontsize=12)
    fig.savefig(OUT/'near30_recovery.png',dpi=160);plt.close(fig)


def render():
    hd=json.loads(HINGES.read_text(encoding='utf8'));gd=json.loads((OUT/'gold_results.json').read_text(encoding='utf8'));sim=json.loads((OUT/'simulation_results.json').read_text(encoding='utf8'))
    assert hd['complete'] and gd['complete'] and sim['complete']
    summary=dict(hinges=control_summary(hd['records']),gold=control_summary(gd['records']),
                 gold_without_first=control_summary(gd['records'][1:]),simulation=simulation_summary(sim),
                 input_hashes={p.name:sha(p) for p in [HINGES,OUT/'gold_results.json',OUT/'simulation_results.json']},
                 rendering_code_sha256=sha(Path(__file__)),thresholds=[])
    for f in sim['fields']:
        for t in f['trials']:
            first=t['measures'][0]
            summary['thresholds'].append(dict(field=f['index'],bound=t['bound'],theta=first['theta'],peak90=first['peak90'],
                baseline=f['baseline_threshold'],injected=t['threshold'],ratio=t['threshold']/f['baseline_threshold']))
    # Trivial no-rod/no-noise numerical reference: same algorithm must retain no candidates.
    blank=np.zeros((2056,2464));ev=evaluate(blank,blank,np.zeros(blank.shape,np.uint16),'empty_reference')
    assert len(ev['spots'])==0
    summary['empty_numerical_reference']=0
    clean_errors=[s['recovered_theta']-s['theta'] for t in sim['clean'] for s in t['measures'] if s['recovered_theta'] is not None]
    summary['clean_angle_error']=dict(median_absolute=float(np.median(np.abs(clean_errors))),max_absolute=float(np.max(np.abs(clean_errors))))
    write('summary.json',summary)
    figures(hd,gd,sim,summary)
    data=dict(summary=summary,assumptions=sim['assumptions'],model=MODEL,
              gold_notes=[dict(index=r['index'],name=r['name'],note=r['note']) for r in gd['records']])
    (OUT/'index.html').write_text(HTML.replace('__DATA__',json.dumps(clean(data),ensure_ascii=True,allow_nan=False)),encoding='utf8')
    print('Report:',OUT/'index.html')


HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gold controls and known-angle rod recovery</title>
<style>:root{font:16px/1.5 system-ui,sans-serif;color:#26383d;background:#f6f7f4}body{max-width:1450px;margin:auto;padding:25px}h1{font-size:29px}h2{font-size:23px;margin-top:32px}p{max-width:1150px}a{color:#087d8b}img{width:100%;height:auto}.note{background:#fff6e5;border-left:4px solid #b77c32;padding:14px 18px}.panel{background:white;border:1px solid #d2ddd6;border-radius:6px;padding:15px;margin:15px 0;overflow:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:7px 12px;text-align:left;border-bottom:1px solid #d4ded7}thead{background:#e9efe9}.small{font-size:14px;color:#52686c}label{display:inline-block;margin:6px 15px 6px 0}select{font:inherit;padding:5px;background:white;border:1px solid #a6bab0;border-radius:4px}summary{cursor:pointer}.two{display:grid;grid-template-columns:1fr 1fr;gap:15px}@media(max-width:850px){body{padding:12px}.two{grid-template-columns:1fr}}</style>
<h1>Does the pipeline produce near-30° detections without hinges?</h1>
<p><b>Existing data are sufficient for these diagnostic checks.</b> The gold-only survey tests whether the feature is specific to the DNA sample. Known-angle injections test conditional detection and angle bias in the recorded image backgrounds. Neither replaces a verified rod-free blank or an independently prepared sample.</p>
<div class="note"><b>The near-30° feature is not specific to the hinge-labelled sample.</b> It also appears in the gold-only survey under the same pipeline. The recovery tests show that detected, unsaturated dim rods can still have biased reconstructed angles. These findings do not identify every low-angle detection as an artifact, and do not prove a preparation mix-up.</div>

<h2>1. Gold-only negative control: same processing</h2>
<p>All 14 wide-field gold survey stacks from 19:22:31 to 19:31:27, compared with all 16 hinge stacks. Their original raw BG capture and pre-subtraction raw saturation map were verified. Detection: per-frame p95 subtraction → 150-frame mean → 2×2 intensity map → DoG σ=1/6 → whole-image mean + 2 SD → components ≥10 pixels, with no upper size limit. Extraction: original minimum-BG mean intensities in an 18×18 window.</p>
<p>The same top-right ellipse, saturation cutoff (≥4094 anywhere in aperture or component bounding rectangle), and strongest-first overlap suppression remain. No survival, low-BG or additional angle filter is introduced. Finite radii above the optical limit remain in r distributions, but have no assigned θ.</p>
<img src="gold_vs_hinges.png" alt="Matched gold and hinge distributions with pooled and equal-field weighting"><div id="controls" class="panel"></div>
<p class="small">Equal-field weighting prevents fields with more detections from dominating. It does not turn fields into independent preparations. Gold notes flag illumination/oil conditions, and its BG capture differs from the two hinge BG captures. An equal pipeline is not proof that the optical conditions or surface preparation were identical.</p>
<details><summary>Each gold stack and its notes</summary><img src="gold_each_stack.png" alt="Each gold stack separately"><div id="goldNotes" class="panel"></div></details>

<h2>2. Inject rods with known angles into the recorded data</h2>
<p><b>504 injected positions:</b> three hinge fields (#1, #6, #10), seven true angles (15°, 25°, 31°, 40°, 55°, 70°, 80°), three brightness levels, and eight evenly spaced azimuths. Positions are outside the ellipse and at least 32 pixels from all existing p95 candidates. Simulated rods are isolated; this test does not measure crowded-spot performance.</p>
<p>Brightness follows the assumed sin²θ law: mean analyzer peak = <b>P90 × sin²θ</b>, with P90 = 200, 800 or 2400 raw DN. This is a specified signal strength, <b>not camera/display gain</b>. Spots have Gaussian σ=2.5 sensor pixels and channel intensities given by the existing optical model. Each batch injects eight rods with the same θ and P90, so bright test rods are not competing against dim test rods. The adaptive detector threshold is recomputed and its change is reported.</p>
<p><b>Two treatments of lost raw values:</b> positive saved pixels reconstruct as saved + Bmin. At a saved zero, the unknown original raw value lies between 0 and Bmin. “Lower” assigns 0; “upper” assigns Bmin. Both reproduce the original saved data before injection. The GUI subtraction is then applied to every modified raw patch before averaging. These two cases are sensitivity scenarios; they do not form confidence limits on nonlinear angle estimates.</p>
<label>Field <select id="field"><option value="all">All three fields</option><option value="1">Hinge 1 · BG 1</option><option value="6">Hinge 6 · BG 1</option><option value="10">Hinge 10 · BG 2</option></select></label>
<img id="recovery" src="recovery_all.png" alt="Recovery fraction and recovered angle versus known true angle">
<p class="small">Recovery means a retained detection within 6 sensor pixels of the injected position. Unmatched pre-existing detections are not successes. Dotted curves: the same rods on an otherwise empty, noiseless numerical image. Bright high-angle losses can be real saturation exclusions. Error bars show the middle half of recovered values, not confidence intervals. A missed rod has no recovered angle and is absent from the angle-bias curves.</p>
<label>Raw values at zeros <select id="case"><option value="upper">Upper: Bmin</option><option value="lower">Lower: zero</option><option value="clean">Clean numerical reference</option></select></label>
<label>P90 <select id="brightness"><option value="200">200 DN</option><option value="800" selected>800 DN</option><option value="2400">2400 DN</option></select></label>
<div id="simTable" class="panel"></div><div id="thresholds" class="panel"></div>
<p class="small">“Forced aperture” evaluates the known true position even if detection fails. It exposes selection effects; it is not a recovered detection. Fractions are conditional on these injected signals and fields. The three fields and two raw-value assumptions must not be counted as independent biological experiments.</p>

<h2>3. Is 30° a unique interpretation of a measured near-30° angle?</h2>
<img src="near30_recovery.png" alt="Known 31 and 40 degree input rods, comparing measured angle histograms">
<p>The lower row contains <b>only true 40° rods</b>. Any values in the shaded 28–34° band are therefore reconstruction error, rather than a true 30° input population. This demonstrates one route to a misleading near-30° measurement; it does not establish how much of the experimental peak follows that route.</p>
<details><summary>See injected pixels, before and after</summary><div class="two"><img src="injection_examples_upper_31.png" alt="Injected 31-degree rods in actual saved background"><img src="injection_examples_upper_55.png" alt="Injected 55-degree rods in actual saved background"></div></details>

<h2>What extra data would improve this?</h2>
<p><b>No extra files were needed to establish these two diagnostic results.</b> A verified rod-free sample under the same illumination/exposure/cancellation settings would be needed to measure an empirical false-positive rate. Raw, unsubtracted measurement frames would remove the clipped-pixel reconstruction uncertainty. Camera noise/photon calibration and a measured PSF would make the simulated recovery quantitative, while independently prepared DNA and gold samples would support biological conclusions.</p>
<p>For now, treat the 30° feature as a measured feature of this processing, not a standalone identifier for hinge attachment or a calibrated hinge angle. Keep the intensity information and background controls with the angle result.</p>
<details><summary>Simulation assumptions and limitations</summary><div id="assumptions"></div><p>Existing temporal fluctuations are retained, but the added signal has no extra shot noise. No coherent interference cross term, rod motion, orientation-dependent PSF, or re-estimated background capture is simulated. The test measures how the present processing responds to additive, stationary model rods in these recorded conditions.</p><p id="sanity"></p><p>The initial mixed-angle/brightness batches are preserved as a <a href="mixed_batch_sensitivity.json">threshold sensitivity check</a>; they are not used for the primary recovery figures because bright injections raised the global threshold for dim injections.</p></details>
<p><a href="gold_results.json">Gold detections and provenance</a> · <a href="simulation_results.json">All injection outcomes and provenance</a> · <a href="summary.json">Numerical summaries</a> · <a href="../random_roi_check/index.html">Random-ROI check</a></p>
<script>const D=__DATA__,$=id=>document.getElementById(id),S=D.summary;
const fmt=(x,n=1)=>Number.isFinite(x)?x.toFixed(n):'—',pct=x=>fmt(100*x)+'%',esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const table=(headers,rows)=>'<table><thead><tr>'+headers.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
$('controls').innerHTML=table(['Group','Retained / valid θ','30–32° (pooled)','28–34° (pooled)','28–34° (equal field weight)','Modal 2° bin'],[['Hinges · 16 fields',S.hinges],['Gold · 14 fields',S.gold],['Gold excluding first field · lighting-note sensitivity',S.gold_without_first]].map(([name,z])=>[name,z.n+' / '+z.valid,pct(z.p30),pct(z.p28),pct(z.equal_p28),z.mode.join('–')+'°']));
$('goldNotes').innerHTML=table(['Field','File','Valid θ','28–34°','Note'],D.gold_notes.map(r=>{const z=S.gold.per_stack.find(a=>a.index===r.index);return[r.index,esc(r.name),z.n,pct(z.p28),esc(r.note)];}));
function update(){const field=$('field').value,condition=$('case').value,peak=+$('brightness').value;$('recovery').src='recovery_'+field+'.png';const cells=S.simulation[field][condition].filter(c=>c.peak90===peak);$('simTable').innerHTML=table(['True θ','Recovered / inserted','Valid θ','Recovered median θ','Middle 50%','Forced-aperture median θ','Reported in 28–34°'],cells.map(c=>[c.theta+'°',c.recovered+' / '+c.n,c.valid,fmt(c.median)+'°',c.q25_q75.length?c.q25_q75.map(x=>fmt(x)).join('–')+'°':'—',fmt(c.forced_median)+'°',c.in28_34+' / '+c.n]));if(condition==='clean'){$('thresholds').textContent='The clean numerical image has its own whole-image SD threshold; it is not a sensitivity estimate for real background.';}else{const t=S.thresholds.filter(t=>(field==='all'||t.field===+field)&&t.bound===condition&&t.peak90===peak);$('thresholds').innerHTML=table(['Injected θ','Threshold after / before injection: range across shown fields'],[15,25,31,40,55,70,80].map(theta=>{const a=t.filter(t=>t.theta===theta).map(t=>t.ratio);return[theta+'°',fmt(Math.min(...a),3)+'–'+fmt(Math.max(...a),3)+'×'];}));}}
$('assumptions').innerHTML=D.assumptions.map(s=>'<p>'+esc(s)+'</p>').join('');$('sanity').textContent='Numerical checks: empty noiseless image → '+S.empty_numerical_reference+' detections. Clean recovered rods: median absolute angle error '+fmt(S.clean_angle_error.median_absolute,3)+'°, maximum '+fmt(S.clean_angle_error.max_absolute,3)+'°. Existing hinge baseline detections/radii reproduced before injection.';
$('field').onchange=$('case').onchange=$('brightness').onchange=update;update();</script></html>'''


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','gold','simulation','render'],default='all');a=p.parse_args()
    if a.phase in ['all','gold']:gold()
    if a.phase in ['all','simulation']:simulate()
    if a.phase in ['all','render']:render()
