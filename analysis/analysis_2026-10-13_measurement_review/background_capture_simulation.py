"""Known-rod injection into a fixed recorded-background template.

Uses GUI minimum projection and the actual saved-frame subtraction functions
loaded via AST without camera imports. Synthetic noise/drift are sensitivity
examples, not measured camera parameters or a fit to the experimental shoulder.
"""
from pathlib import Path
import ast,hashlib,json
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from polarcam.live.intensity_spots import channel_means,anisotropy
from polarcam.live import Detection_alg_offline as gui
from detector_grid_trial import detect,match,display,HERE,DAY

OUT=HERE/'background_capture_simulation'
REPO=HERE.parents[1]
A,B,C=.09779825985607313,.011480772916810558,.10137635007725046
RMAX=C/(A+B)
SEED=20261014

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def source_operations():
    source=REPO/'src/polarcam/live/fetch_frames.py';tree=ast.parse(source.read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_crop_background_profile','_subtract_background']]
    assert len(selected)==2;ns={'np':np}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
    gui_source=REPO/'src/polarcam/live/Spinners_gui_live.py'
    text=gui_source.read_text(encoding='utf-8')
    assert 'min_frame = np.array(arr[0], dtype=np.float32)' in text and 'np.minimum(min_frame, frame, out=min_frame)' in text
    return ns['_subtract_background'],dict(fetch_sha256=digest(source),gui_sha256=digest(gui_source))

def make_world(h,w):
    rng=np.random.default_rng(SEED);world=np.zeros((2*h,2*w),np.float32);catalog=[]
    for i in range(2000):
        x=float(rng.uniform(16,2*w-16));y=float(rng.uniform(16,2*h-16))
        theta=float(rng.uniform(5,85));phi=float(rng.uniform(0,np.pi));s2=np.sin(np.radians(theta))**2
        r=C*s2/(A+B*s2);p90=float(np.exp(rng.uniform(np.log(300),np.log(1200))))
        mean_peak=p90*s2
        ch=mean_peak*np.array([1+r*np.cos(2*phi),1-r*np.cos(2*phi),1+r*np.sin(2*phi),1-r*np.sin(2*phi)])
        np.testing.assert_allclose(np.linalg.norm(anisotropy(ch)),r,atol=1e-12)
        sigma=2.5;left=int(np.floor(x))-13;top=int(np.floor(y))-13
        yy,xx=np.mgrid[top:top+27,left:left+27];psf=np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma))
        modulation=np.empty(psf.shape)
        for value,(pr,pc) in zip(ch,[(1,1),(0,0),(0,1),(1,0)]):modulation[(yy%2==pr)&(xx%2==pc)]=value
        world[top:top+27,left:left+27]+=np.asarray(psf*modulation,np.float32)
        catalog.append(dict(id=i+1,x=x,y=y,theta=theta,phi=float(phi),r=r,peak_at_90=p90,mean_peak=mean_peak))
    return world,catalog

def camera(expectation,noise,rng):
    frame=expectation if noise==0 else expectation+rng.normal(0,noise,size=expectation.shape).astype(np.float32)
    return np.rint(np.clip(frame,0,4094)).astype(np.uint16)

def evaluate(mean,peak,truth):
    detections,_=detect(gui.intensity_detection_map(mean))
    detections=[s for s in detections if s['pass250']]
    total=len(detections)
    # Compare against the same 20-pixel interior as the truth catalogue.
    # Otherwise genuine edge rods would be falsely counted as unmatched.
    h,w=mean.shape
    detections=[s for s in detections if 20<s['x']<w-20 and 20<s['y']<h-20]
    for s in detections:
        l,t=round(s['x'])-9,round(s['y'])-9
        s['saturated']=bool(peak[t:t+18,l:l+18].max()>=4094)
    usable=[s for s in detections if not s['saturated']]
    pairs,lost,new=match(truth,usable)
    channels=channel_means(mean,[(round(s['x'])-9,round(s['x'])+9,round(s['y'])-9,round(s['y'])+9) for s in truth])
    xy=anisotropy(channels);rr=np.linalg.norm(xy,axis=1)
    measured=[]
    for j,s in enumerate(truth):
        l,t=round(s['x'])-9,round(s['y'])-9
        measured.append(dict(id=s['id'],theta=s['theta'],true_r=s['r'],measured_r=float(rr[j]),
          isolated=s['isolated'],detected=j in pairs,saturated=bool(peak[t:t+18,l:l+18].max()>=4094)))
    recovered_ids=[s['id'] for j,s in enumerate(truth) if j in pairs]
    return dict(detector_candidates_total=total,edge_excluded=total-len(detections),
        candidates=len(detections),usable_candidates=len(usable),matched_truth=len(pairs),
        unmatched=len(new),missed=len(lost),recovered_ids=recovered_ids,truth_measurements=measured)

def build():
    OUT.mkdir(exist_ok=True);subtract,code=source_operations()
    p=DAY/'backgrounds/used/80eff41d7393cdc67da2cbddefde778c3b023551a9367dc0000bbd8db1c969f1.npy'
    assert digest(p)==p.stem;bg=np.load(p).astype(np.float32);h,w=bg.shape
    world,catalog=make_world(h,w);oy,ox=h//2,w//2
    assert all(v%2==0 for v in (ox,oy,h,w))
    signal=world[oy:oy+h,ox:ox+w]
    truth=[dict(s,x=s['x']-ox,y=s['y']-oy) for s in catalog if ox+20<s['x']<ox+w-20 and oy+20<s['y']<oy+h-20]
    dist=cdist([[s['x'],s['y']] for s in truth],[[s['x'],s['y']] for s in truth]);np.fill_diagonal(dist,np.inf)
    for s,near in zip(truth,dist.min(1)):s['isolated']=bool(near>=24)
    broad=[(int(2*round(y/2)),int(2*round(x/2))) for y in np.linspace(0,h,5) for x in np.linspace(0,w,9)]
    narrow=[(oy+dy,ox+dx) for dy in [-4,-2,0,2,4] for dx in [-4,-4,-2,-2,0,2,2,4,4]]
    assert len(broad)==len(narrow)==45
    configs=[('ideal','Static BG, broad scan, no noise',broad,0.,1.),
             ('small','Static BG, only 8 px scan, no noise',narrow,0.,1.),
             ('noise','Static BG, broad scan, 4 DN noise',broad,4.,1.),
             ('drift','BG 10% brighter during measurement',broad,0.,1.1)]
    results=[];images=[];ideal_profile=None
    for ci,(key,label,path,noise,drift) in enumerate(configs):
        rng=np.random.default_rng(SEED+100+ci)
        if key=='drift':profile=ideal_profile.copy()
        else:
            profile=None
            for y,x in path:
                frame=camera(bg+world[y:y+h,x:x+w],noise,rng)
                if profile is None:profile=np.array(frame,dtype=np.float32)
                else:np.minimum(profile,frame,out=profile)
            if key=='ideal':ideal_profile=profile.copy()
        # No noise: 150 identical frames reduce to one exact evaluation. Noise:
        # average 150 independently sampled frames after actual clipped subtraction.
        n=150 if noise else 1
        total=np.zeros(bg.shape,float);oracle=np.zeros(bg.shape,float);blank=np.zeros(bg.shape,float)
        peak=np.zeros(bg.shape,np.uint16)
        for i in range(n):
            frame=camera(bg*drift+signal,noise,rng);np.maximum(peak,frame,out=peak)
            total+=subtract(frame,profile,None,None)
            oracle+=subtract(frame,bg*drift,None,None)
            # Separate background-only image; subtract the same estimated profile.
            blank+=subtract(camera(bg*drift,noise,rng),profile,None,None)
            if noise and (i+1)%50==0:print(key,'measurement frames',i+1,flush=True)
        mean=total/n;ideal=oracle/n;residual=blank/n;error=profile-bg
        if key=='ideal':
            assert np.all(error>=0)
            np.testing.assert_allclose(mean,ideal,atol=0,rtol=0)
        ev=evaluate(mean,peak,truth);ev_oracle=evaluate(ideal,peak,truth)
        residual_dets,_=detect(gui.intensity_detection_map(residual));residual_dets=[s for s in residual_dets if s['pass250']]
        record=dict(key=key,label=label,capture_noise_sd=noise,measurement_bg_factor=drift,capture_frames=45,
          measurement_frames=150,evaluated_frames=n,scan_positions=path,
          profile_error_mean=float(error.mean()),profile_error_q=np.quantile(error,[0,.01,.5,.99,1]).tolist(),
          profile_nonzero_fraction=float(np.mean(error!=0)),
          background_only_residual_mean=float(residual.mean()),background_only_candidates=len(residual_dets),
          mean_abs_difference_from_true_bg_subtraction=float(np.mean(np.abs(mean-ideal))),analysis=ev,oracle=ev_oracle)
        for suffix,image in [('mean',mean),('blank',residual)]:Image.fromarray(display(gui.intensity_detection_map(image))).save(OUT/f'{key}_{suffix}.png')
        results.append(record);images.append((error,residual))
        print(key,{k:record[k] for k in ['profile_error_mean','background_only_residual_mean','background_only_candidates']},'recovered',ev['matched_truth'],'oracle',ev_oracle['matched_truth'],flush=True)
    result=dict(seed=SEED,bg_sha256=digest(p),world_shape=list(world.shape),sensor_shape=list(bg.shape),
       rods_in_world=2000,rods_in_measurement=len(truth),isolated_in_measurement=sum(s['isolated'] for s in truth),
       catalog=catalog,truth=truth,scenarios=results,code=code,builder_sha256=digest(Path(__file__)),
       assumptions=['Recorded minimum BG reused as a fixed generative intensity template; not asserted to be the true mean optical BG.',
        'Rod sample translates laterally over a fixed sensor BG. All translations are even sensor pixels to preserve analyzer parity.',
        '2000 Gaussian point-spread spots, sigma 2.5 sensor px; theta uniform 5–85 deg, phi uniform 0–180 deg.',
        'Mean peak intensity at 90 deg log-uniform 300–1200 DN; multiplied by sin(theta)^2 as requested. This brightness law is imposed, not measured.',
        'Analyzer channel peaks = P90*sin(theta)^2 * [1+X,1-X,1+Y,1-Y], with X=r*cos(2phi), Y=r*sin(2phi) and optical-model r(theta).',
        'Independent additive Gaussian 4 DN noise is an illustrative sensitivity case, not a calibrated camera noise model; no shot noise or coherent rod–BG cross term.',
        '10% BG drift is a separate illustrative mismatch, not inferred from data.',
        'No photon calibration, orientation-dependent PSF, stage jitter model, illumination gradient, or fitting to the experimental shoulder.',
        'Detections use unchanged GUI intensity DoG k8, area10–250; saturation veto from synthetic pre-subtraction maxima at4094; one-to-one truth matches within6px. Recovery and unmatched counts use the same 20-pixel interior as the truth catalogue.'])
    (OUT/'results.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False),encoding='utf-8')
    figures(result,images)
    (OUT/'index.html').write_text(HTML.replace('__DATA__',json.dumps(clean(result)).replace('</','<\\/')),encoding='utf-8')


def figures(d,images):
    fig,axes=plt.subplots(4,2,figsize=(11,13),layout='constrained')
    for i,(s,(error,residual)) in enumerate(zip(d['scenarios'],images)):
        e=axes[i,0].imshow(error,cmap='RdBu_r',vmin=-15,vmax=15,interpolation='nearest');fig.colorbar(e,ax=axes[i,0],shrink=.65,label='Estimated BG − template (DN)')
        axes[i,0].set_title(s['label'],fontsize=10)
        e=axes[i,1].imshow(residual,cmap='magma',vmin=0,vmax=20,interpolation='nearest');fig.colorbar(e,ax=axes[i,1],shrink=.65,label='Mean saved BG-only residual (DN)')
        for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('45-frame minimum background capture with 2000 moving synthetic rods\nFixed image is the first recorded DNA background template; colour ranges are fixed',fontsize=12)
    fig.savefig(OUT/'residuals.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#167d8d','#cf7933','#7350a3','#b64b63'];edges=np.arange(0,91,10)
    for s,color in zip(d['scenarios'],colors):
        rows=[r for r in s['analysis']['truth_measurements'] if r['isolated'] and not r['saturated']]
        t=np.array([r['theta'] for r in rows]);found=np.array([r['detected'] for r in rows]);den=np.histogram(t,edges)[0];num=np.histogram(t[found],edges)[0]
        axes[0,0].plot((edges[:-1]+edges[1:])/2,100*num/np.maximum(1,den),'o-',ms=3,c=color,label=s['key'])
        vals=[r['measured_r']-r['true_r'] for r in rows if r['detected'] and r['measured_r'] is not None]
        axes[0,1].hist(vals,bins=np.linspace(-.3,.3,61),histtype='step',color=color,label=s['key'])
    axes[0,0].set(xlabel='Injected theta (deg)',ylabel='Detected isolated, unsaturated rods (%)',ylim=(0,105),title='Completeness depends on injected brightness/angle');axes[0,0].legend(frameon=False)
    axes[0,1].set(xlabel='Recovered r − injected r',ylabel='Matched rods',title='Errors at known true centres');axes[0,1].legend(frameon=False)
    for ax,key in zip(axes[1],['ideal','noise']):
        s=next(s for s in d['scenarios'] if s['key']==key);rows=s['analysis']['truth_measurements']
        rows=[r for r in rows if r['isolated'] and not r['saturated']]
        ax.hist([r['true_r'] for r in rows],bins=np.arange(0,1.02,.02),histtype='step',color='.5',label='All injected (visible, isolated)')
        ax.hist([r['measured_r'] for r in rows if r['detected'] and r['measured_r'] is not None],bins=np.arange(0,1.02,.02),histtype='step',color='#167d8d',label='Detected; r at true centre')
        ax.set(xlabel='Anisotropy radius r',ylabel='Rods',title=key+': known truth versus detected');ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Illustrative injection/recovery test — assumptions, not a fitted explanation of the observed shoulder',fontsize=13)
    fig.savefig(OUT/'recovery.png',dpi=150);plt.close(fig)


HTML=r'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Moving rods and minimum background capture</title><style>body{font:16px/1.5 system-ui;color:#253236;background:#f7f8f5;margin:0}main{max-width:1300px;margin:auto;padding:24px}h1{font-size:27px}img{max-width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ccd5d0;text-align:left}.scroll{overflow:auto}a{color:#087c89}p{max-width:1100px}.small{font-size:14px;color:#526369}</style><main>
<h1>Can laterally moving rods contaminate the minimum background?</h1>
<p>We place <b>2000 synthetic rods on a 4112×4928 sensor-pixel scene</b> and translate it over the fixed <b>2056×2464 first-DNA background</b>. A broad scan samples a 9×5 grid of 45 positions. The 8-pixel scan is an insufficient-motion stress test. All motion is lateral, with no z or orientation changes.</p>
<p><b>Exact ideal prediction:</b> B̂(p)=minₜ[B(p)+Sₜ(p)]=B(p)+minₜSₜ(p). With nonnegative rod intensity, this cannot underestimate a static noiseless background. A rod-free observation at each pixel recovers B. If rods never clear a pixel, B̂ is too high and later subtraction removes some signal.</p>
<p>The simulation uses the GUI's per-pixel minimum and executes the actual <code>_subtract_background</code> code: float32 subtraction, clip negatives to zero, round and save uint16. The synthetic camera rounds/clips to 0–4094 before subtraction. Markers are not treated as measured frames.</p>
<div class="scroll" id="table"></div><p class="small">Recovery uses the GUI detector and excludes saturation. Positional matches within 6 pixels are approximate; unresolved blends can be unmatched. The oracle subtracts the known measurement BG, with the same camera quantization, noise and zero clipping. No-noise measurements have 150 identical frames, evaluated once.</p>
<img src="residuals.png" alt="Estimated background errors and saved background-only residuals"><img src="recovery.png" alt="Known-rod detection and anisotropy recovery">
<p><b>The noise and drift rows are separate sensitivity examples.</b> The minimum of noisy measurements is biased low, even if the expected background is constant; zero clipping can also leave a positive mean after subtraction of a correct mean BG. A brighter measurement background than the captured profile can leave its spatial/polarization pattern. Neither effect has been calibrated or fitted to these data.</p>
<p><b>Brightness:</b> the injected mean channel peak is P₉₀ sin²θ. Its four channel values are proportional to [1+X,1−X,1+Y,1−Y], with r(θ) from the current Fresnel+BFP+hole model. Thus the requested total-intensity sin²θ law is imposed, while theta and phi determine polarization. This is an assumption to test angle-dependent detectability, not a brightness calibration.</p>
<details><summary>Full assumptions and scope</summary><ul id="assumptions"></ul><p>The recorded BG was itself a minimum profile; it is used here as a fixed numerical template, not claimed to be the true expected optical background. Additive B+S omits coherent interference, so this ideal result cannot rule out interference residuals in the experiment.</p></details>
<p><a href="../detector_grid_trial/index.html">Real-stack grid and size comparison, with excluded-spot cards</a> · <a href="results.json">Synthetic truth, assumptions and numerical results</a></p></main><script>const DATA=__DATA__;
document.getElementById('table').innerHTML='<table><tr><th>Case</th><th>Mean BG estimate error (DN)</th><th>Mean BG-only residual (DN)</th><th>BG-only detections</th><th>Recovered rods</th><th>Oracle recovered</th><th>Unmatched detections</th></tr>'+DATA.scenarios.map(s=>`<tr><td>${s.label}</td><td>${s.profile_error_mean.toFixed(3)}</td><td>${s.background_only_residual_mean.toFixed(3)}</td><td>${s.background_only_candidates}</td><td>${s.analysis.matched_truth}/${DATA.rods_in_measurement}</td><td>${s.oracle.matched_truth}</td><td>${s.analysis.unmatched}</td></tr>`).join('')+'</table>';
document.getElementById('assumptions').innerHTML=DATA.assumptions.map(s=>'<li>'+s+'</li>').join('');</script></html>'''

if __name__=='__main__':build()
