"""Read-only overlap audit and held-out background-reconstruction pilot.

This does not select a left-half population or change the acquisition GUI.
"""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from detector_grid_trial import HERE, DAY, NAME
from polarcam.live import Detection_alg_offline as detector

OUT = HERE/'background_overlap_check'
STAMPS = ['183650', '185032']
HASHES = ['80eff41d7393cdc67da2cbddefde778c3b023551a9367dc0000bbd8db1c969f1',
          'df4197c16247c8d73199bce95a41139c15f5c0bfbf27a404259bd30a428e41dc']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def stats(a):
    return dict(mean=float(np.mean(a)), median=float(np.median(a)),
                quantiles=np.quantile(a, [.01, .1, .5, .9, .99]).tolist())


def cell(a):
    return detector.intensity_detection_map(a)


def shared(a, b):
    al, ar, at, ab = a['bounds']; bl, br, bt, bb = b['bounds']
    return max(0, min(ar,br)-max(al,bl))*max(0,min(ab,bb)-max(at,bt))


def overlap_audit():
    d = json.loads((HERE/'detector_grid_trial/threshold_results.json').read_text())
    spots = d['levels']['2']['spots']; by_id = {s['id']: s for s in spots}
    eligible = [s for s in spots if s['accepted'] and not s['inside_ellipse']
                and not s['saturated'] and s['r'] is not None]
    comparisons = []
    for cutoff in [1, 33, 81, 162, 325]:
        kept = []
        for s in sorted(eligible, key=lambda s: (-s['dog_peak'], s['component'])):
            if not any(shared(s, t) >= cutoff for t in kept): kept.append(s)
        comparisons.append(dict(shared_pixel_cutoff=cutoff, kept=len(kept),
            pictured_rod_kept=any(s['id']=='sd2-517' for s in kept)))
    assert comparisons[0]['kept'] == d['levels']['2']['cuts']['kept']
    pairs=[]
    for akey,bkey in [('sd2-517','sd2-479'), ('sd2-807','sd2-811')]:
        a,b=by_id[akey],by_id[bkey]
        pairs.append(dict(weaker=a,stronger=b,shared_pixels=shared(a,b),
            fraction=shared(a,b)/324, distance=float(np.hypot(a['x']-b['x'],a['y']-b['y']))))
    mean=np.load(HERE/'detector_grid_trial/mean.npy'); im=cell(mean)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,pair in zip(axes,pairs):
        a,b=pair['weaker'],pair['stronger']; x,y=np.mean([[a['x'],a['y']],[b['x'],b['y']]],axis=0)
        x,y=round(x),round(y); h=44
        ax.imshow(im[y-h:y+h,x-h:x+h],cmap='gray',vmin=0,vmax=4095/10,
                  extent=(x-h-.5,x+h-.5,y+h-.5,y-h-.5),interpolation='nearest')
        for s,color,label in [(a,'#f1a641','Suppressed'),(b,'#65e2e6','Retained')]:
            l,r,t,bot=s['bounds'];ax.add_patch(Rectangle((l-.5,t-.5),r-l,bot-t,fill=False,ec=color,lw=1.8,label=label))
        ax.set_title(f"{a['id']} vs {b['id']}\n{pair['shared_pixels']}/324 shared pixels ({pair['fraction']:.1%}); centres {pair['distance']:.1f} px apart",fontsize=11)
        ax.legend(fontsize=9);ax.set_xlabel('Sensor x');ax.set_ylabel('Sensor y')
    fig.suptitle('Current any-pixel overlap rule: the first pair is unnecessarily suppressed',fontsize=13)
    fig.savefig(OUT/'overlap_examples.png',dpi=160);plt.close(fig)
    return dict(pairs=pairs,cutoff_trials=comparisons,eligible_before_overlap=len(eligible))


def background_captures():
    profiles=[];medians=[];records=[]
    for stamp,digest in zip(STAMPS,HASHES):
        path=next((DAY/'backgrounds').glob(f'background_min_20261012-{stamp}*.npy'))
        assert sha(path)==digest
        profile=np.load(path).astype(np.float32); meta=json.loads(path.with_suffix('.json').read_text())
        rawpath=DAY/'backgrounds'/meta['source_stack_file']; raw=np.load(rawpath,mmap_mode='r')
        n=meta['actual']['frames'];assert len(raw)==n+int(meta['actual']['phase_marker_appended'])
        assert meta['background']['background_subtracted'] is False
        median=np.empty(profile.shape,np.float32);temporal_sd=np.empty_like(median)
        train_min=np.empty_like(median);train_median=np.empty_like(median);test_mean=np.empty_like(median)
        minimum=np.empty_like(median)
        # Chunk along rows to avoid materialising ~1 GB of float data at once.
        for y in range(0,profile.shape[0],64):
            a=np.asarray(raw[:n,y:y+64,:],dtype=np.float32)
            minimum[y:y+64]=a.min(axis=0);median[y:y+64]=np.median(a,axis=0)
            temporal_sd[y:y+64]=a.std(axis=0)
            train_min[y:y+64]=a[::2].min(axis=0);train_median[y:y+64]=np.median(a[::2],axis=0)
            test_mean[y:y+64]=a[1::2].mean(axis=0)
        np.testing.assert_array_equal(minimum,profile)
        records.append(dict(stamp=stamp,profile=path.name,profile_sha256=digest,
            raw_stack=rawpath.name,raw_signature=dict(bytes=rawpath.stat().st_size,mtime_ns=rawpath.stat().st_mtime_ns),
            frames=n,fps=meta['actual']['fps'],exposure_us=meta['actual']['timing']['exposure_us'],
            exact_minimum_match=True,minimum=stats(profile),median=stats(median),
            median_minus_minimum=stats(median-profile),temporal_sd=stats(temporal_sd),
            alternating_held_out_signed_residual_minimum=stats(test_mean-train_min),
            alternating_held_out_signed_residual_median=stats(test_mean-train_median),
            note=meta['conditions']['condition_note']))
        profiles.append(profile);medians.append(median)
        print('Background',stamp,records[-1]['median_minus_minimum'],flush=True)
        del raw
    diff=medians[1]-medians[0]
    return profiles,medians,dict(records=records,median_change=stats(diff),
        minimum_change=stats(profiles[1]-profiles[0]),
        median_spatial_correlation=float(np.corrcoef(medians[0].ravel(),medians[1].ravel())[0,1]))


def position_means(profiles):
    records=[r for r in json.loads((HERE/'radius_detector_comparison/results.json').read_text())['records'] if r['group']=='Hinges']
    assert len(records)==16 and records[0]['name']==NAME
    means=[];info=[]
    first=np.array([[s['x'],s['y']] for s in records[0]['methods']['native']['spots']])
    for r in records:
        p=DAY/'frames'/r['name'];m=json.loads(p.with_suffix('.json').read_text());a=np.load(p,mmap_mode='r')
        n=m['actual']['frames'];assert n==150 and len(a)==151
        assert m['background']['background_subtracted'] is True
        assert m['actual']['timing']['exposure_us']==1194.2702702702702
        assert m['actual']['gains']['analog']['val']==m['actual']['gains']['digital']['val']==1
        digest=m['background']['background_profile_sha256'];assert digest==r['background_sha256']
        g=HASHES.index(digest)
        # Sum integers without loss; restoring the BG gives an upper bound on raw means
        # because original negative differences were clipped. Quantify zeros separately.
        acc=np.zeros(a.shape[1:],np.float64);zeros=0
        for frame in a[:n]:
            acc+=frame;zeros+=np.count_nonzero(frame==0)
        acc/=n
        means.append((acc+profiles[g]).astype(np.float32))
        points=np.array([[s['x'],s['y']] for s in r['methods']['native']['spots']])
        dist,_=cKDTree(points).query(first)
        info.append(dict(name=r['name'],group=g,zero_fraction=zeros/(n*acc.size),
            source_signature=dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sidecar_sha256=sha(p.with_suffix('.json'))),
            raw_proxy_mean=float(means[-1].mean()),same_position_matches_to_first=int((dist<6).sum()),
            total_first_candidates=len(first)))
        print('Position',len(means),r['name'],'zero fraction',info[-1]['zero_fraction'],flush=True)
        del a,acc
    return np.stack(means),info


def robust_positions(means,indices):
    out=np.empty(means.shape[1:],np.float32)
    for y in range(0,out.shape[0],64):out[y:y+64]=np.median(means[indices,y:y+64],axis=0)
    return out


def recorrect(index,info,profiles,candidates):
    p=DAY/'frames'/info[index]['name'];a=np.load(p,mmap_mode='r')
    old=profiles[info[index]['group']]
    # Never claim to recover pixels destroyed by clipping. Only subtract a positive
    # additional estimate; this is exactly reproducible from the saved uint16 frames.
    additions=[np.maximum(bg-old,0) for bg in candidates.values()]
    sums=[np.zeros_like(old,dtype=np.float64) for _ in range(len(additions)+1)]
    for frame in a[:150]:
        sums[0]+=frame
        for j,extra in enumerate(additions):
            corrected=np.maximum(np.asarray(frame,np.float32)-extra,0)
            sums[j+1]+=np.rint(corrected)
    outputs={'original':sums[0]/150}
    outputs.update({key:s/150 for key,s in zip(candidates,sums[1:])})
    return outputs, {key:dict(added=stats(extra),estimate_below_old_fraction=float(np.mean(bg<old)))
                    for (key,bg),extra in zip(candidates.items(),additions)}


def save_views(outputs,index):
    for key,im in outputs.items():
        image=np.rint(np.clip(cell(im)*20/4095,0,1)*255).astype(np.uint8)
        Image.fromarray(image).save(OUT/f'stack{index+1}_{key}.webp',lossless=True)


def figures(profiles,medians,targets):
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for i in range(2):
        for ax,im,title in zip(axes[i],[profiles[i],medians[i],medians[i]-profiles[i]],
                ['Recorded minimum','Median of raw capture','Median minus minimum']):
            cmap='gray' if title!='Median minus minimum' else 'magma'
            plot=ax.imshow(cell(im),cmap=cmap,vmin=0,vmax=100 if cmap=='gray' else 35)
            ax.set_title(f'Background {i+1}: {title}',fontsize=11);ax.set_xticks([]);ax.set_yticks([])
            fig.colorbar(plot,ax=ax,shrink=.7,label='Intensity (DN)')
    fig.suptitle('The stored background is the per-pixel minimum, not the typical level',fontsize=14)
    fig.savefig(OUT/'background_profiles.png',dpi=150);plt.close(fig)
    crop_figures(targets)


def crop_figures(targets, displayed=False):
    crops=[('Pictured pair sensor location',541,515,88),('Pictured background residue',1566,851,88),
           ('Large fixed feature',2060,500,440)]
    labels={'original':'Original 1x subtraction','raw_medians':'Two raw-capture medians',
            'same_group':'Other positions, same BG','all_positions':'Other 15 positions'}
    for target in targets:
        frames=target['images'];keys=list(frames)
        fig,axes=plt.subplots(3,len(keys),figsize=(3.4*len(keys),9),layout='constrained')
        for row,(label,x,y,size) in enumerate(crops):
            for col,key in enumerate(keys):
                ax=axes[row,col];im=frames[key] if displayed else cell(frames[key]);h=size//2
                ax.imshow(im[y-h:y+h,x-h:x+h],cmap='gray',vmin=0,vmax=255 if displayed else 4095/20,interpolation='nearest')
                ax.set_title(labels[key] if row==0 else '',fontsize=11);ax.set_xticks([]);ax.set_yticks([])
                if col==0:ax.set_ylabel(label,fontsize=10)
        fig.suptitle(f'Held-out stack {target["index"]+1}: same 20x display in every panel\nPilot corrections only; additional subtraction clips and rounds each frame before averaging',fontsize=13)
        filename='background_trial_crops.png' if target['index']==0 else 'background_trial_crops_stack10.png'
        fig.savefig(OUT/filename,dpi=160);plt.close(fig)


def build():
    OUT.mkdir(exist_ok=True)
    audit=overlap_audit();profiles,medians,bginfo=background_captures()
    means,info=position_means(profiles)
    targets=[];trial_profiles={}
    for idx in [0,9]:
        same=[i for i,r in enumerate(info) if r['group']==info[idx]['group'] and i!=idx]
        other=[i for i in range(len(info)) if i!=idx]
        candidates=dict(raw_medians=(medians[0]+medians[1])/2,
                        same_group=robust_positions(means,same),
                        all_positions=robust_positions(means,other))
        assert idx not in same and idx not in other
        images,correction=recorrect(idx,info,profiles,candidates)
        save_views(images,idx)
        crop_metrics={}
        for label,(x,y,h) in {'residue_patch':(1566,851,44),'large_feature':(2060,500,180)}.items():
            crop_metrics[label]={key:stats(cell(im)[y-h:y+h,x-h:x+h]) for key,im in images.items()}
        targets.append(dict(index=idx,source=info[idx]['name'],same_group_training=same,
            all_positions_training=other,correction=correction,crop_metrics=crop_metrics,images=images))
        for key,bg in candidates.items():trial_profiles[f'stack{idx+1}_{key}']=bg
        print('Pilot corrected stack',idx+1,flush=True)
    np.savez_compressed(OUT/'trial_background_estimates.npz',**trial_profiles)
    figures(profiles,medians,targets)
    for t in targets:t['images']={key:f'stack{t["index"]+1}_{key}.webp' for key in t['images']}
    result=dict(overlap=audit,background=bginfo,position_inputs=info,targets=targets,
        builder_sha256=sha(Path(__file__)),
        selection='No left-half selection; no all-stack r distributions; existing gallery and GUI unchanged.',
        methods='Raw-background proposal: average of the two per-capture pixel medians. '
                'Position proposal: per-pixel median of restored means (saved mean + recorded minimum), '
                'excluding the target stack, with no image registration; fixed background is in sensor coordinates. '
                'Test both same-background group and all other positions. Raw proxies overestimate the true '
                'mean where original subtraction clipped; zero fractions reported. Additional correction is '
                'max(estimated_BG-old_BG,0); per-frame max(saved-addition,0), rounded then averaged.',
        limits=['These are pilot background estimates, not proven rod-free truth or a revised population analysis.',
                'A stationary real object appearing in most training positions can be absorbed into the median.',
                'Sample motion, illumination changes and coherent optical cross terms need not be a fixed additive background.',
                'Correcting an additive mean does not undo coherent interference, noise or data already lost by clipping.',
                'Alternating-frame holdout assesses predictive offset but adjacent moving frames are not independent experiments.'])
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    page=HTML.replace('__DATA__',json.dumps(result).replace('</','<\\/'))
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print('Complete',OUT,flush=True)


HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Background and overlap audit</title><style>body{font:16px/1.5 system-ui;color:#263637;background:#f7f8f5;margin:0}main{max-width:1400px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:22px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:7px 16px 7px 0;border-bottom:1px solid #d5ded7;text-align:left}.scroll{overflow:auto}.small{font-size:14px;color:#526369}select{font:inherit;padding:6px}label{display:inline-block;margin-right:20px}#view{max-height:78vh;overflow:auto;background:#111}#map{display:block;width:100%;height:auto}a{color:#087985}p{max-width:1100px}</style><main>
<h1>Two checks before selecting more rods</h1>
<p><a href="percentile_comparison.html"><b>New: first stack with minimum, 25th-percentile and median backgrounds</b></a></p>
<p>This checks the pictured overlap and backgrounds. The existing gallery selections and camera GUI are unchanged. The left-half population analysis is deferred.</p>
<h2>The overlapping-window rule is too strict for the pictured pair</h2>
<p>The bright pair has centres 23.7 pixels apart, but its square extraction windows share just <b>2 of 324 pixels (0.62%)</b>. That triggered the any-pixel veto. It is a detection-selection issue, not failure to find the lower spot. Shared extraction pixels are useful as a contamination warning; they do not prove two detections are the same rod.</p>
<img src="overlap_examples.png" alt="Two pairs with extraction windows showing the overlap rule">
<div id="overlapTable" class="scroll"></div><p class="small">Threshold alternatives are sensitivity checks, not validated rod-separation criteria. For discussion: flag proximity, and suppress only near-duplicate detections or materially overlapping cores. An arbitrary window corner should not delete a visibly separate object.</p>
<h2>What the actual backgrounds contain</h2><p id="bgSummary"></p>
<img src="background_profiles.png" alt="Stored minimum profiles and medians from their original raw captures">
<p id="heldout"></p><p id="bgChange"></p>
<p>The GUI subtracts the full selected minimum profile, then clips negative results to zero; there is no fractional background scale in this path. A minimum selects downward fluctuations, so it can leave a positive residual shaped like the background. Incomplete lateral clearing instead leaves rod signal in the minimum and can over-subtract it. The earlier noiseless, fixed-background simulation isolated that special case; the real captures vary in time.</p>
<p class="small">Both background captures were recorded at about 10 fps, while survey stacks were about 72.18 fps, with the same 1.194 ms exposure and gain. That is a further difference in temporal sampling, but these recordings do not by themselves identify vibration phase or prove a cancellation fault.</p>
<h2>Can the other stage positions improve it?</h2>
<p>Three pilot estimates are compared: average the two raw-capture medians; take the median across other stack means sharing the same old background; or use all other hinge stack means. Each test stack is excluded from its own position-based estimate. Background is kept in sensor coordinates. The raw capture medians use the recorded moving-background movies; the position estimates use one mean per stage position.</p>
<img src="background_trial_crops.png" alt="Same crops before and after additional background correction">
<details><summary>Same sensor regions in the tenth stack (different rods)</summary><img src="background_trial_crops_stack10.png" alt="Tenth held-out stack before and after correction at the same sensor locations"></details>
<div id="patchTable" class="scroll"></div>
<p>Only a positive extra correction is applied, frame by frame, before averaging. This avoids claiming to recover information already clipped away. Darker images alone do not prove a better estimate: persistent rods or sample-dependent scattering can also enter a position median. These results are a visual background check, with no new all-stack r distribution.</p>
<p id="clipping"></p>
<label>Test stack <select id="stack"><option value="0">First hinge stack · 18:38:50</option><option value="1">Tenth hinge stack · 18:51:20</option></select></label><label>Image <select id="method"><option value="original">Original 1x subtraction</option><option value="raw_medians">Two raw-capture medians</option><option value="same_group">Other positions · same background</option><option value="all_positions">All other positions</option></select></label>
<p id="caption" class="small"></p><div id="view"><img id="map" alt="Selected held-out stack correction at fixed 20x"></div>
<p><a href="../detector_grid_trial/threshold_trial.html">Return to current detector gallery</a> · <a href="results.json">Numerical audit and provenance</a> · <a href="trial_background_estimates.npz">Trial background estimates (not installed in GUI)</a></p>
</main><script>const DATA=__DATA__,$=id=>document.getElementById(id);
$('overlapTable').innerHTML='<table><tr><th>Suppress when shared pixels are…</th><th>Retained at 2 SD</th><th>Pictured rod retained?</th></tr>'+DATA.overlap.cutoff_trials.map(t=>`<tr><td>${t.shared_pixel_cutoff===325?'No overlap veto':t.shared_pixel_cutoff===1?'Any shared pixel':t.shared_pixel_cutoff+' or more / 324'}</td><td>${t.kept}</td><td>${t.pictured_rod_kept?'Yes':'No'}</td></tr>`).join('')+'</table>';
$('bgSummary').textContent=DATA.background.records.map((r,i)=>`Background ${i+1}: stored profile exactly matches the minimum of its ${r.frames} raw frames. Median-minus-minimum is ${r.median_minus_minimum.median.toFixed(1)} DN at the median pixel (${r.median_minus_minimum.mean.toFixed(1)} DN on average).`).join(' ');
$('heldout').textContent='Using alternating capture frames to estimate a background and the remaining frames to check it, the median signed residual changes '+DATA.background.records.map(r=>`${r.alternating_held_out_signed_residual_minimum.median.toFixed(2)} → ${r.alternating_held_out_signed_residual_median.median.toFixed(2)} DN (minimum → median estimate)`).join(' and ')+'. This shows the minimum offset on separate frames, but does not certify the median as rod-free.';
$('bgChange').textContent=`The two temporal-median backgrounds are not identical: their spatial correlation is ${DATA.background.median_spatial_correlation.toFixed(3)}, and pixel differences span ${DATA.background.median_change.quantiles[1].toFixed(1)} to ${DATA.background.median_change.quantiles[3].toFixed(1)} DN between the 10th and 90th percentiles. A small change in image-wide mean can conceal substantial local changes.`;
$('patchTable').innerHTML='<table><tr><th>Mean intensity in the pictured residue patch (DN)</th><th>Original</th><th>Two raw-capture medians</th><th>Other positions, same BG</th><th>All other positions</th></tr>'+DATA.targets.map(t=>'<tr><td>Held-out stack '+(t.index+1)+'</td>'+['original','raw_medians','same_group','all_positions'].map(k=>'<td>'+t.crop_metrics.residue_patch[k].mean.toFixed(2)+'</td>').join('')+'</tr>').join('')+'</table>';
const z=DATA.position_inputs.map(r=>r.zero_fraction);$('clipping').textContent=`The saved survey movies have ${(100*Math.min(...z)).toFixed(1)}–${(100*Math.max(...z)).toFixed(1)}% zero-valued pixel samples. The saved mean plus old background is therefore an upper-bound raw proxy, not an exact reconstruction wherever clipping occurred. This limits quantitative use of the proposed position backgrounds.`;
function draw(){const t=DATA.targets[Number($('stack').value)],key=$('method').value;$('map').src=t.images[key];$('caption').textContent=`${t.source} · fixed 20x display. ${key==='original'?'Recorded saved data.':key==='raw_medians'?'Estimate from the two separate raw background captures.':`Estimate from ${key==='same_group'?t.same_group_training.length:t.all_positions_training.length} other stacks; test stack left out.`}`}
$('stack').onchange=draw;$('method').onchange=draw;draw();</script></html>'''

if __name__=='__main__':build()
