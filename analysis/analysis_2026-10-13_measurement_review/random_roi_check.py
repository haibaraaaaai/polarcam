"""Spatial sensitivity and random-aperture controls for the apparent 30-degree feature.

Read-only acquisitions; no detector retuning, masks or source selections changed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from polarcam.live.intensity_spots import channel_means, anisotropy
from polarcam.optics import theta_from_radius, radius_from_theta

HERE=Path(__file__).resolve().parent
DAY=HERE.parents[1]/'data/hinges/polarcam/2026-10-12'
SOURCE=HERE/'background_overlap_check/hinges95_theta.json'
OUT=HERE/'random_roi_check'
SIZES=[2,6,18,36]
FRACTIONS=[.1,.25,.5]
N_RANDOM=30000
N_CROPS=500
SEED=2026101402
R_EDGES=np.r_[np.arange(0,1.401,.02),np.sqrt(2)+1e-12]
T_EDGES=np.arange(0,92,2)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def signature(p):
    s=p.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def integral(a):
    return np.pad(np.cumsum(np.cumsum(a,axis=0,dtype=np.float64),axis=1),((1,0),(1,0)))


def rectangle_sums(ii,l,r,t,b):
    return ii[b,r]-ii[t,r]-ii[b,l]+ii[t,l]


def aperture_channels(image,centres,sizes=SIZES):
    """Channel-first box means at absolute sensor coordinates, for either parity."""
    result={s:np.empty((len(centres),4),float) for s in sizes}
    for j,(py,px) in enumerate([(1,1),(0,0),(0,1),(1,0)]):
        ii=integral(image[py::2,px::2])
        for s in sizes:
            left=centres[:,0]-s//2;top=centres[:,1]-s//2
            x=(left+((px-left)%2))//2;y=(top+((py-top)%2))//2;n=s//2
            result[s][:,j]=rectangle_sums(ii,x,x+n,y,y+n)/(n*n)
    # Compare actual pixel slicing, including odd and even extraction origins.
    for s in sizes:
        for i in [0,len(centres)//2,len(centres)-1]:
            x,y=centres[i];box=(x-s//2,x+s//2,y-s//2,y+s//2)
            np.testing.assert_allclose(result[s][i],channel_means(image,[box])[0],atol=3e-7,rtol=0)
    return result


def summary(values,model):
    values=np.asarray(values);values=values[np.isfinite(values)]
    a,b,c=[model[k] for k in ('A','B','C')]
    angles=np.degrees(theta_from_radius(values,a,b,c))
    valid=angles[np.isfinite(angles)]
    rh=np.histogram(values,R_EDGES)[0];th=np.histogram(valid,T_EDGES)[0]
    assert rh.sum()==len(values) and th.sum()==len(valid)
    lo,hi=radius_from_theta(np.radians([30,32]),a,b,c)
    return dict(n=len(values),median=float(np.median(values)) if len(values) else None,
                q05_q95=np.quantile(values,[.05,.95]).tolist() if len(values) else [],
                r_hist=rh.tolist(),theta_hist=th.tolist(),theta_valid=len(valid),
                above_limit=int(np.sum(values>model['r_max'])),
                p30_32_r=float(np.mean((values>=lo)&(values<hi))) if len(values) else None,
                theta_mode_bin=T_EDGES[np.argmax(th):np.argmax(th)+2].tolist() if len(valid) else [])


def spatial_metrics(theta):
    theta=np.asarray(theta);v=theta[np.isfinite(theta)]
    counts=np.histogram(v,T_EDGES)[0]
    n=len(v)
    p=lambda l,h:float(np.mean((v>=l)&(v<h))) if n else np.nan
    return counts,np.array([n,p(30,32),p(28,34),p(28,34)-.5*(p(22,28)+p(34,40))])


def crop_summaries(points,crops,rng):
    # points: x,y,r,theta,index,l,r,t,b. Entire 18x18 aperture must lie inside crop.
    theta=points[:,3];valid=np.isfinite(theta);all_counts,baseline=spatial_metrics(theta)
    rows=[];matched=[];hist=[]
    for l,r,t,b in crops:
        keep=(points[:,5]>=l)&(points[:,6]<=r)&(points[:,7]>=t)&(points[:,8]<=b)
        h,m=spatial_metrics(theta[keep]);rows.append(m);hist.append(h)
        count=int(m[0])
        sample=rng.choice(theta[valid],count,replace=False) if count else []
        _,mr=spatial_metrics(sample);matched.append(mr)
    rows=np.asarray(rows);matched=np.asarray(matched);hist=np.asarray(hist)
    freq=np.divide(hist,rows[:,0,None],out=np.full(hist.shape,np.nan),where=rows[:,0,None]>0)
    def intervals(a):return np.nanquantile(a,[.025,.5,.975],axis=0).tolist()
    return dict(baseline=baseline.tolist(),full_counts=all_counts.tolist(),
                metrics_025_50_975=intervals(rows),random_spots_025_50_975=intervals(matched),
                curve_025_50_975=intervals(freq),
                spatial_sd=np.nanstd(rows[:,1:],axis=0,ddof=1).tolist(),
                same_count_random_spot_sd=np.nanstd(matched[:,1:],axis=0,ddof=1).tolist(),
                fraction_broad_bump_positive=float(np.nanmean(rows[:,3]>0)),
                empty_crops=int(np.sum(rows[:,0]==0)))


def calculate():
    OUT.mkdir(exist_ok=True)
    source=json.loads(SOURCE.read_text(encoding='utf8'));model=source['model']
    assert source['complete'] and len(source['records'])==16
    H,W=2056,2464; margin=18
    rng=np.random.default_rng(SEED)
    # Common sampled centres, without replacement on a two-pixel lattice.
    xs=np.arange(margin,W-margin+1,2);ys=np.arange(margin,H-margin+1,2)
    idx=rng.choice(len(xs)*len(ys),N_RANDOM,replace=False)
    centres=np.c_[xs[idx%len(xs)],ys[idx//len(xs)]]
    e=source['ellipse'];outside=((centres[:,0]-e['cx'])/e['rx'])**2+((centres[:,1]-e['cy'])/e['ry'])**2>1
    data=dict(seed=SEED,model=model,ellipse=e,shape=[H,W],sizes=SIZES,n_random=N_RANDOM,n_crops=N_CROPS,
              source_sha256=sha(SOURCE),builder_sha256=sha(Path(__file__)),
              r_edges=R_EDGES.tolist(),theta_edges=T_EDGES.tolist(),
              target_r=radius_from_theta(np.radians([30,32]),*[model[k] for k in ('A','B','C')]).tolist(),
              backgrounds=[],stacks=[],scopes=[],crops={},
              methods=dict(apertures='30000 paired random centres, without replacement on a 2-pixel lattice. '
                  '2/6/18/36-pixel squares; average each polarizer channel before ratios. No detector or brightness cut. '
                  '18-pixel edge is shared across sizes; aperture size changes the estimator.',
                  masks='Backgrounds: both full interior and outside the prior ellipse. Stacks: outside ellipse, '
                  'exclude any raw pixel >=4094 inside the random aperture. Outside-candidate subset excludes overlap '
                  'with 18x18 apertures of ALL p95 candidates, including candidates later rejected. Undetected rods may remain.',
                  frames='Minimum-subtracted 150-frame mean, marker omitted. Saturation uses verified pre-subtraction peak files. '
                  'No subtraction or threshold is recomputed for a larger crop.',
                  backgrounds='Per-pixel minimum, median and p95 of each ORIGINAL raw background movie; linear percentiles. '
                  'These profiles are not simultaneous four-channel measurements and are not residual backgrounds.',
                  crops='500 uniform top-left placements per area size within the 10-pixel interior; original image aspect ratio. '
                  'Same sensor-coordinate crop across stacks to preserve fixed-background dependence. Entire extraction apertures '
                  'must fit. Centres near image edges have lower inclusion probability; envelopes describe this crop design, not CIs.',
                  metrics='Columns: valid-angle N, fraction 30-32, fraction 28-34, broad bump = P28-34 - (P22-28+P34-40)/2. '
                  'The bump contrast is descriptive and chosen after seeing the peak; not a significance test.',
                  resampling='Same-count random-spot subsets benchmark count-only variation under exchangeability; '
                  'not an independent-particle confidence interval. Overlapping ROIs do not create additional experiments.'))
    bg_ch={}
    for gi,bg in enumerate(source['backgrounds'],1):
        path=DAY/'backgrounds'/bg['minimum_file'];capture=DAY/'backgrounds'/bg['capture']
        assert sha(path)==bg['minimum_sha256'] and signature(capture)==bg['capture_signature']
        raw=np.load(capture,mmap_mode='r');n=bg['frames'];assert len(raw)==n+1
        profiles=np.empty((3,H,W),np.float32)
        for y in range(0,H,64):profiles[:,y:y+64]=np.percentile(raw[:n,y:y+64],[0,50,95],axis=0,method='linear')
        np.testing.assert_array_equal(profiles[0],np.load(path))
        assert np.all(profiles[0]<=profiles[1]) and np.all(profiles[1]<=profiles[2])
        assert signature(capture)==bg['capture_signature']
        ch=[aperture_channels(a,centres) for a in profiles];bg_ch[gi]=ch
        entry=dict(group=gi,provenance=bg,profiles={})
        for label,cc in zip(['minimum','median','p95'],ch):
            entry['profiles'][label]={str(s):dict(full=summary(np.linalg.norm(anisotropy(cc[s]),axis=1),model),
                masked=summary(np.linalg.norm(anisotropy(cc[s][outside]),axis=1),model)) for s in SIZES}
        data['backgrounds'].append(entry)
        # Display map for interactive spatial inspection; average 2x2 cells only for display.
        a=profiles[0];a=(a[::2,::2]+a[1::2,::2]+a[::2,1::2]+a[1::2,1::2])/4
        Image.fromarray(np.rint(np.clip(a*20/4095,0,1)*255).astype('uint8')).save(OUT/f'bg{gi}.png')
        print(f'Background {gi}: min/median/p95 complete',flush=True)
    pools={s:{k:[] for k in ['random','away']} for s in SIZES}
    for rec in source['records']:
        path=DAY/'frames'/rec['name'];assert signature(path)==rec['source_signature']
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
        assert sha(path.with_suffix('.json'))==rec['sidecar_sha256']
        assert meta['background']['background_subtracted'] and meta['background']['background_profile_sha256']==rec['background_sha256']
        frames=np.load(path,mmap_mode='r');n=meta['actual']['frames'];assert n==150 and len(frames)==151
        total=np.zeros((H,W),np.uint32)
        for f in frames[:n]:total+=f
        mean=total.astype(float)/n
        ch=aperture_channels(mean,centres)
        info=meta['fetch_frames_output']['raw_peak'];rp=path.parent/info['path'];assert sha(rp)==info['sha256']==rec['raw_peak_sha256']
        sat_int=integral(np.load(rp)>=4094)
        tree=cKDTree([[s['x'],s['y']] for s in rec['spots']])
        nearest=tree.query(centres,k=1,p=np.inf)[0]
        rr={}
        for s in SIZES:
            x,y=centres.T;hs=s//2
            sat=rectangle_sums(sat_int,x-hs,x+hs,y-hs,y+hs)>0
            mask=outside&~sat;away=mask&(nearest>=hs+9)
            radius=np.linalg.norm(anisotropy(ch[s]),axis=1)
            rr[str(s)]=dict(random=summary(radius[mask],model),away=summary(radius[away],model),
                           saturation_excluded=int(np.sum(sat&outside)),ellipse_excluded=int((~outside).sum()))
            pools[s]['random'].append(radius[mask]);pools[s]['away'].append(radius[away])
        kept=[s for s in rec['spots'] if s['selection']=='kept']
        points=[[s['x'],s['y'],s['r'],s['theta_deg'],rec['index'],*s['bounds']] for s in kept]
        # Reproduce the entire original detected set directly from the new mean, independently of sampling.
        channels=channel_means(mean,[s['bounds'] for s in kept]);radius=np.linalg.norm(anisotropy(channels),axis=1)
        np.testing.assert_allclose(radius,[s['r'] for s in kept],rtol=0,atol=1e-12)
        data['stacks'].append(dict(index=rec['index'],name=rec['name'],time=rec['time'],group=rec['background_group'],
            source_signature=rec['source_signature'],raw_peak_sha256=rec['raw_peak_sha256'],
            points=points,apertures=rr,detected=summary(radius,model)))
        assert signature(path)==rec['source_signature']
        print(f"Stack {rec['index']}/16: random apertures complete; {len(kept)} original selected radii verified",flush=True)
    # Fixed spatial crop locations used for all stacks and both backgrounds.
    for frac in FRACTIONS:
        w=round((W-20)*np.sqrt(frac));h=round((H-20)*np.sqrt(frac))
        left=rng.integers(10,W-10-w+1,N_CROPS);top=rng.integers(10,H-10-h+1,N_CROPS)
        data['crops'][str(frac)]=np.c_[left,left+w,top,top+h].tolist()
    scopes=[('all',list(range(16))),('bg1',list(range(9))),('bg2',list(range(9,16)))]+[(str(i+1),[i]) for i in range(16)]
    for sid,indices in scopes:
        selected=[data['stacks'][i] for i in indices]
        p=np.array([row for r in selected for row in r['points']],float)
        item=dict(id=sid,stack_indices=[i+1 for i in indices],detected=summary(p[:,2],model),apertures={},crops={},tiles=[])
        for s in SIZES:
            item['apertures'][str(s)]={k:summary(np.concatenate([pools[s][k][i] for i in indices]),model) for k in ['random','away']}
        for frac in FRACTIONS:
            item['crops'][str(frac)]=crop_summaries(p,np.array(data['crops'][str(frac)]),rng)
        # A fixed 4x4 tiling is a diagnostic; do not choose tiles using their outcomes.
        xedges=np.linspace(10,W-10,5);yedges=np.linspace(10,H-10,5)
        for ty in range(4):
            for tx in range(4):
                inside=(p[:,0]>=xedges[tx])&(p[:,0]<xedges[tx+1])&(p[:,1]>=yedges[ty])&(p[:,1]<yedges[ty+1])
                hc,mc=spatial_metrics(p[inside,3]);hl,ml=spatial_metrics(p[~inside,3])
                item['tiles'].append(dict(x=tx,y=ty,count=int(inside.sum()),metrics=mc.tolist(),
                    leave_out_metrics=ml.tolist(),counts=hc.tolist(),leave_out_counts=hl.tolist()))
        data['scopes'].append(item)
        print(f'Spatial crops: {sid}',flush=True)
    # No NaN in machine-readable artifact; undefined small-tile metrics become null.
    def finite_tree(x):
        if isinstance(x,dict):return {k:finite_tree(v) for k,v in x.items()}
        if isinstance(x,list):return [finite_tree(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    data=finite_tree(data)
    (OUT/'results.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf8')
    np.savez_compressed(OUT/'random_aperture_coordinates.npz',centres=centres,outside_ellipse=outside)
    return data


def plot_curve(ax,summary,label,color,kind='r',ls='-',lw=1.5):
    edges=R_EDGES if kind=='r' else T_EDGES
    h=np.array(summary['r_hist' if kind=='r' else 'theta_hist']);n=h.sum()
    ax.stairs(100*h/n if n else h,edges,label=label,color=color,ls=ls,lw=lw)


def annotate_r(ax,d):
    ax.axvspan(*d['target_r'],color='#d39b36',alpha=.18)
    ax.set(xlim=(0,1.05),xlabel='Anisotropy radius r',ylabel='Windows / detections per bin (%)')
    ax.grid(axis='y',alpha=.15)


def add_background_crops(d):
    """Random large crops of the same BG profiles, with 18x18 aperture estimates."""
    centres=np.load(OUT/'random_aperture_coordinates.npz')['centres']
    H,W=d['shape'];halfs=9
    for group in d['backgrounds']:
        bg=group['provenance'];capture=DAY/'backgrounds'/bg['capture']
        assert signature(capture)==bg['capture_signature']
        raw=np.load(capture,mmap_mode='r');profiles=np.empty((3,H,W),np.float32)
        for y in range(0,H,64):profiles[:,y:y+64]=np.percentile(raw[:bg['frames'],y:y+64],[0,50,95],axis=0,method='linear')
        group['large_crops_18']={}
        for label,profile in zip(['minimum','median','p95'],profiles):
            ch=aperture_channels(profile,centres,[18])[18];rr=np.linalg.norm(anisotropy(ch),axis=1)
            np.testing.assert_allclose(np.median(rr),group['profiles'][label]['18']['full']['median'],atol=1e-12)
            stats={};x,y=centres.T
            for frac in FRACTIONS:
                metrics=[];curves=[]
                for l,r,t,b in d['crops'][str(frac)]:
                    take=(x-halfs>=l)&(x+halfs<=r)&(y-halfs>=t)&(y+halfs<=b)&np.isfinite(rr)
                    vals=rr[take];lo,hi=d['target_r'];count=len(vals)
                    metrics.append([count,float(np.median(vals)),float(np.mean((vals>=lo)&(vals<hi)))])
                    curves.append(np.histogram(vals,R_EDGES)[0]/count)
                stats[str(frac)]=dict(metrics_025_50_975=np.quantile(metrics,[.025,.5,.975],axis=0).tolist(),
                    curve_025_50_975=np.quantile(curves,[.025,.5,.975],axis=0).tolist())
            group['large_crops_18'][label]=stats
        assert signature(capture)==bg['capture_signature']
        print(f"Background {group['group']}: large spatial crops complete",flush=True)
    d['methods']['background_large_crops']='Within each random large crop, keep the common sampled 18x18 apertures entirely inside it. '
    d['methods']['background_large_crops']+='Full BG interior, ellipse included. Same crop coordinates as the hinge analysis; no refitting. '
    d['methods']['background_large_crops']+='Distributions describe aperture r within the region; the whole region is not collapsed into one r.'
    (OUT/'results.json').write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf8')


def figures(d):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for size in SIZES:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained',sharey=True)
        for ax,g in zip(axes,d['backgrounds']):
            for key,label,col in [('minimum','Minimum','#287f93'),('median','Median','#92588c'),('p95','95th percentile','#cb743c')]:
                z=g['profiles'][key][str(size)]['full']
                plot_curve(ax,z,f"{label}: median r={z['median']:.3f}",col)
            annotate_r(ax,d);ax.set_title(f"Background {g['group']} · {size} × {size} pixel apertures");ax.legend(fontsize=9)
        fig.suptitle('30,000 paired random apertures in each BG profile; shading = r equivalent to 30–32°\nFull interior, no detector or ellipse mask; profiles are not the residual left after subtraction',fontsize=11)
        fig.savefig(OUT/f'backgrounds_{size}.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(4,4,figsize=(16,12),layout='constrained',sharex=True)
        for ax,rec in zip(axes.flat,d['stacks']):
            z=rec['apertures'][str(size)]
            for key,label,col in [('random','Random apertures','#a17c36'),('away','Outside candidate apertures','#679b85')]:
                plot_curve(ax,z[key],label,col,lw=1)
            plot_curve(ax,rec['detected'],'Detected spots (18×18)','#326fa3',lw=1.5)
            annotate_r(ax,d);ax.set_ylabel('% per bin');ax.set_title(f"#{rec['index']} · BG {rec['group']} · away n={z['away']['n']:,}",fontsize=10)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(f'Random {size} × {size} apertures in minimum-subtracted means vs retained detections\nSame ellipse and raw saturation exclusions; random windows can still contain undetected rods',fontsize=13)
        fig.savefig(OUT/f'hinges_{size}.png',dpi=145);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained',sharey=True)
        for ax,scope,title in zip(axes,d['scopes'][:3],['All 16','BG1 · stacks 1–9','BG2 · stacks 10–16']):
            for key,label,col in [('random','Random apertures','#a17c36'),('away','Outside candidate apertures','#679b85')]:
                plot_curve(ax,scope['apertures'][str(size)][key],label,col)
            plot_curve(ax,scope['detected'],'Detected spots (18×18)','#326fa3')
            annotate_r(ax,d);ax.set_title(title)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'Pooled minimum-subtracted means · random {size} × {size} apertures\nGold shading: r equivalent to 30–32°. No rod identity is assigned to random apertures.',fontsize=12)
        fig.savefig(OUT/f'pooled_{size}.png',dpi=160);plt.close(fig)
    for frac in FRACTIONS:
        fig,baxes=plt.subplots(2,3,figsize=(14,8),layout='constrained',sharex=True,sharey=True)
        for row,g in enumerate(d['backgrounds']):
            for col,label in enumerate(['minimum','median','p95']):
                ax=baxes[row,col];z=g['large_crops_18'][label][str(frac)]
                q=100*np.array(z['curve_025_50_975']);x=(R_EDGES[:-1]+R_EDGES[1:])/2
                ax.fill_between(x,q[0],q[2],color='#d2c08a',alpha=.4,label='95% location envelope')
                ax.plot(x,q[1],color='#947639',lw=1,label='Median crop')
                plot_curve(ax,g['profiles'][label]['18']['full'],'Full BG interior','#283b42')
                annotate_r(ax,d);ax.set_title(f"BG {g['group']} · {label}")
        baxes[0,0].legend(fontsize=8)
        fig.suptitle(f'Background profiles: {int(frac*100)}% area crops, r from 18×18 apertures within each crop\nFull interior including prior ellipse; shading at 30–32° equivalent r. Envelopes are not confidence intervals.',fontsize=12)
        fig.savefig(OUT/f'bg_crops_{int(frac*100)}.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained',sharey=True)
        for ax,scope,title in zip(axes,d['scopes'][:3],['All 16','BG1 · stacks 1–9','BG2 · stacks 10–16']):
            z=scope['crops'][str(frac)];q=100*np.array(z['curve_025_50_975']);x=(T_EDGES[:-1]+T_EDGES[1:])/2
            ax.fill_between(x,q[0],q[2],color='#9bc1cf',alpha=.4,label='95% of random-crop curves (pointwise)')
            ax.plot(x,q[1],color='#428399',lw=1,label='Median random crop')
            plot_curve(ax,scope['detected'],'Entire retained field','#202e33',kind='theta')
            ax.axvspan(30,32,color='#d39b36',alpha=.2);ax.set(xlim=(0,90),xlabel='Apparent θ (degrees)',ylabel='Valid-angle detections per 2° bin (%)',title=title)
            ax.grid(axis='y',alpha=.15)
        axes[0].legend(fontsize=7)
        fig.suptitle(f'{int(frac*100)}% area crops · same 500 crop locations across all selected stacks\nSpread across locations is spatial sensitivity, not a confidence interval or 500 independent experiments',fontsize=12)
        fig.savefig(OUT/f'crops_{int(frac*100)}.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    for ax,scope,title in zip(axes,d['scopes'][:3],['All 16','BG1 · stacks 1–9','BG2 · stacks 10–16']):
        vals=np.array([t['metrics'][1] if t['metrics'][1] is not None else np.nan for t in scope['tiles']]).reshape(4,4)*100
        im=ax.imshow(vals,vmin=0,vmax=25,cmap='YlOrRd',origin='upper',extent=(0,2464,2056,0))
        for t in scope['tiles']:
            value=t['metrics'][1];text='empty' if value is None else f"{100*value:.1f}%\nn={t['count']}"
            ax.text((t['x']+.5)*616,(t['y']+.5)*514,text,ha='center',va='center',fontsize=9,color='black')
        ax.set(title=title,xlabel='Sensor x',ylabel='Sensor y')
    fig.colorbar(im,ax=list(axes),shrink=.8,label='Detected spots with 30 ≤ θ < 32° (%)')
    fig.suptitle('Fixed 4 × 4 tiles: how spatially concentrated is the narrow 30–32° band?\nTiles set by coordinates; no region chosen or removed using the result.',fontsize=12)
    fig.savefig(OUT/'spatial_tiles.png',dpi=160);plt.close(fig)


def render(d):
    if 'large_crops_18' not in d['backgrounds'][0]:add_background_crops(d)
    figures(d)
    html=HTML.replace('__DATA__',json.dumps(d,ensure_ascii=True,allow_nan=False,separators=(',',':')))
    (OUT/'index.html').write_text(html,encoding='utf8')


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Random regions and the apparent 30° peak</title>
<style>:root{font:16px/1.5 system-ui,sans-serif;color:#27373c;background:#f6f7f3}body{max-width:1460px;margin:auto;padding:24px}h1{font-size:29px}h2{font-size:23px;margin:32px 0 12px}p{max-width:1160px}a{color:#087d8b}button,select{font:inherit;padding:5px 8px;border:1px solid #9aada5;border-radius:4px;background:white;color:inherit}label{display:inline-block;margin:6px 14px 6px 0}input[type=range]{width:240px;vertical-align:middle}img{width:100%;height:auto}canvas{width:100%;height:auto;background:white}table{border-collapse:collapse;font-size:14px;max-width:100%}td,th{padding:7px 12px;text-align:left;border-bottom:1px solid #cdd6ce}thead{background:#e7ece6}.note{padding:15px 18px;border-left:4px solid #be7c32;background:#fff8e8}.small{font-size:14px;color:#53676b}.panel{background:white;padding:16px;border:1px solid #d5ddd5;border-radius:6px;margin:16px 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.controls{background:#e8eee7;padding:10px 15px;border-radius:5px}summary{cursor:pointer}#map{max-width:850px}#counts{overflow:auto}@media(max-width:800px){.two{grid-template-columns:1fr}body{padding:12px}}</style>
<h1>Does the low-angle feature survive random spatial sampling?</h1>
<p>Two different tests: <b>small random apertures, with no detector</b>, and <b>large crops containing the existing detected spots</b>. The first checks what anisotropy appears without choosing a rod. The second checks whether the detected distribution depends strongly on location. Neither establishes molecular identity by itself.</p>
<div class="note"><b>The main result:</b> a low-r distribution appears even away from the detected candidate apertures. Its maximum is generally below the selected 30–32° band, so the distributions are not identical. The detected band also varies substantially with spatial location. A crop that preserves the peak is therefore insufficient to identify it as hinges; a crop that loses it is insufficient to rule hinges out.</div>
<p>Everything remains on the same 16 hinge stacks: p95 detection at 2 SD, minimum-BG intensity extraction, oversized detections allowed, prior ellipse masked, raw-saturated spots excluded. Larger crops do <b>not</b> change the detector threshold or extraction aperture. Gold is not included in this check.</p>

<h2>1. Random spot-sized apertures</h2>
<div class="controls"><label>Aperture <select id="size"><option value="2">2 × 2 sensor pixels</option><option value="6">6 × 6</option><option value="18" selected>18 × 18 — same as spots</option><option value="36">36 × 36</option></select></label></div>
<p>30,000 common randomly placed apertures in each background profile and each frame-stack mean. Each aperture yields one r, after averaging the four channel intensities separately. No intensity threshold or spot detector selects these positions. Background plots use the full interior; hinge plots retain the ellipse and raw saturation exclusions.</p>
<p><b>Gold shading marks r = <span id="target"></span>, equivalent to 30–32° under the rod model.</b> A background window is not a rod, so this mapping is only a reference. The minimum, median and p95 BG profiles are shown as stored/calculated profiles, not as residual backgrounds after subtraction.</p>
<img id="bgFig" alt="Random aperture radius distributions in minimum, median and 95th percentile background profiles"><div id="bgTable" class="panel"></div>
<img id="poolFig" alt="Detected spots versus randomly placed apertures in the hinge images"><div id="randomTable" class="panel"></div>
<p class="small">“Outside candidate apertures” means no overlap with any 18×18 aperture of any p95 candidate, including candidates subsequently rejected. Missed dim rods, fringes, and larger components can remain there. The normalization is over finite r values, including values above the model limit. These random-window and detected-spot distributions have different selection processes and must not simply be subtracted from one another.</p>
<details><summary>Show all 16 stacks separately</summary><img id="eachFig" alt="Random aperture controls and detected radius distributions for all 16 stacks"></details>
<details><summary>Why changing aperture size can change r</summary><p>Increasing an aperture changes the quantity being measured. It adds background and possibly neighbours, averages local channel fluctuations, and mixes signed X/Y contributions before taking their radius. The radius of averaged intensities is not the mean of the individual radii. A large number of overlapping pixels does not provide the same information as that many independent rods.</p></details>

<h2>2. Larger random crops: keep the estimator fixed</h2>
<div class="controls"><label>Crop area <select id="area"><option value="0.1">10% of interior area</option><option value="0.25" selected>25%</option><option value="0.5">50%</option></select></label></div>
<p>500 random crop locations at each size, preserving the image aspect ratio. A crop must fully contain each 18×18 extraction aperture. For pooled stacks, the <b>same sensor-coordinate crop is applied to every stack</b>, so shared background structure is not averaged away by choosing unrelated regions in different frames.</p>
<p>These shaded bands contain the middle 95% of crop values at each histogram bin. They are <b>location-sensitivity envelopes, not confidence intervals</b>. The crops overlap; there are still only the original images. Uniform placement also includes edge locations less often than central locations.</p>
<img id="bgCropFig" alt="How background radius distributions vary between random large crops"><div id="bgCropTable" class="panel"></div>
<img id="cropFig" alt="How detected theta distributions vary between random large crops"><div id="cropTable" class="panel"></div>
<p class="small">“Count-only spread” uses uniformly sampled, same-count subsets of the full retained spot set. It is an exchangeability benchmark for random sampling variability, not a confidence interval. A larger spatial spread suggests position matters beyond simply having fewer points. θ distributions include valid inversions; the r view below also retains the five finite above-limit values.</p>

<h2>Inspect an individual crop</h2>
<div class="controls"><label>Stack group <select id="scope"></select></label><label>Underlying BG image <select id="mapBg"><option value="1">BG 1</option><option value="2">BG 2</option></select></label><label>Crop <input id="roi" type="range" min="0" max="499" value="0"><span id="roiNumber"></span></label><button id="advance">Another pre-sampled crop</button></div>
<p class="small">Orange dots: selected spots in 30–32°. Teal: other retained detections. Rectangle: selected crop. The fixed exclusion ellipse is shown in red. For pooled groups, dots are overlaid at sensor coordinates across stacks; this is not one simultaneous field. Background display is fixed at 20×.</p>
<div class="two"><div><canvas id="map" width="1232" height="1028" aria-label="Random region on the recorded background with retained spots"></canvas></div><div><div id="roiStats" class="panel"></div><canvas id="thetaPlot" width="680" height="360" aria-label="Theta distribution for one crop and full field"></canvas><canvas id="rPlot" width="680" height="360" aria-label="Radius distribution for one crop and full field"></canvas></div></div>

<h2>3. A fixed spatial grid and what to do next</h2>
<img src="spatial_tiles.png" alt="Fraction in 30 to 32 degree band in a fixed 4 by 4 spatial grid">
<p id="leaveOut"></p>
<p><b>The usual statistical approach is spatial subsampling / block resampling.</b> Neighbouring pixels and spots may share background and other errors, so resampling individual pixels as independent observations would exaggerate precision. Blocks should reflect the spatial dependence scale; there is no universal “right” ROI size. <a href="https://arxiv.org/abs/math/0503671">Nordman &amp; Lahiri: spatial subsample size</a>; <a href="https://arxiv.org/abs/0805.2325">Loh: dependence-preserving spatial resampling</a>.</p>
<p>Here, a practical next step is to use a fixed, spatially distributed set of blocks and report how the result varies across blocks and stacks. Keep both background groups separate. The ROI analysis tests robustness of the observed pattern; it cannot remove a systematic background bias common to every block.</p>
<p><b>To establish a hinge population, the missing check is specificity:</b> apply the same detector and extraction to suitable background/negative controls, and measure detection/angle bias using simulated known-angle rods injected into realistic background. Tune thresholds on one subset and validate on held-out fields. Independent preparations are needed for a claim about the sample type. The present angle bands and masks were explored on these images, so this report does not provide a confirmatory peak p-value.</p>
<details><summary>Reproducibility and precise definitions</summary><div id="methods"></div><p>Spatial plots use 2° θ bins and 0.02 r bins (the final r bin ends at √2). A displayed “30–32°” fraction is a band count, not a test of a distinct mode. Broad bump contrast = P(28–34) − [P(22–28) + P(34–40)]/2; it is exploratory. Original background captures contain moving-sample signal, and different profile pixels may come from different frames; profile anisotropy does not directly measure the unremoved background inside any selected rod.</p></details>
<p><a href="results.json">All distributions, crop statistics, selected coordinates and provenance</a> · <a href="random_aperture_coordinates.npz">Reproducible random aperture centres</a> · <a href="../background_overlap_check/hinges95_theta.html">Original 16-stack distributions</a> · <a href="../background_overlap_check/peak_spot_examples/index.html">Example spots</a></p>
<script>
const D=__DATA__,$=id=>document.getElementById(id),images={};
const fmt=(x,n=2)=>Number.isFinite(x)?x.toFixed(n):'—',pct=(x,n=1)=>fmt(x*100,n)+'%',names={all:'All 16',bg1:'BG1 · stacks 1–9',bg2:'BG2 · stacks 10–16'};
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function table(head,rows){return '<table><thead><tr>'+head.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table>';}
function updateSize(){const s=$('size').value;$('bgFig').src='backgrounds_'+s+'.png';$('poolFig').src='pooled_'+s+'.png';$('eachFig').src='hinges_'+s+'.png';let rows=[];for(const g of D.backgrounds)for(const k of ['minimum','median','p95']){const z=g.profiles[k][s].full;rows.push(['BG '+g.group,k,fmt(z.median,3),pct(z.p30_32_r)]);}$('bgTable').innerHTML=table(['Background','Profile','Median r','Fraction in target r band'],rows);rows=[];for(const g of D.scopes.slice(0,3))for(const k of ['detected','random','away']){const z=k==='detected'?g.detected:g.apertures[s][k];rows.push([names[g.id],{detected:'Detected · 18×18',random:'Random · '+s+'×'+s,away:'Outside all candidate apertures'}[k],z.n.toLocaleString(),fmt(z.median,3),pct(z.p30_32_r),z.theta_mode_bin.join('–')+'°']);}$('randomTable').innerHTML=table(['Stack group','Positions','Finite r count','Median r','Target r band','Modal 2° bin if rod inversion applied'],rows);}
function updateArea(){const area=$('area').value,percent=Math.round(+area*100);$('bgCropFig').src='bg_crops_'+percent+'.png';$('cropFig').src='crops_'+percent+'.png';let rows=[];for(const g of D.backgrounds)for(const k of ['minimum','median','p95']){const z=g.large_crops_18[k][area].metrics_025_50_975;rows.push(['BG '+g.group,k,fmt(g.profiles[k]['18'].full.median,3),fmt(z[0][1],3)+'–'+fmt(z[2][1],3),pct(z[0][2])+'–'+pct(z[2][2])]);}$('bgCropTable').innerHTML=table(['BG','Profile','Full median r','95% location range: median r','95% location range: target band'],rows);rows=[];for(const g of D.scopes.slice(0,3)){const z=g.crops[area],q=z.metrics_025_50_975;rows.push([names[g.id],Math.round(q[0][0])+'–'+Math.round(q[2][0]),pct(z.baseline[1]),pct(q[0][1])+'–'+pct(q[2][1]),fmt(z.spatial_sd[0]/z.same_count_random_spot_sd[0],1)+'×']);}$('cropTable').innerHTML=table(['Group','95% location range: valid N','Full-field fraction 30–32°','95% location range: 30–32° fraction','SD across locations / count-only SD'],rows);drawROI();}
function hist(values,edges){const h=Array(edges.length-1).fill(0);let n=0;for(const v of values){if(!Number.isFinite(v))continue;let i=edges.findIndex((e,j)=>j<edges.length-1&&v>=e&&v<edges[j+1]);if(i===-1&&v===edges.at(-1))i=h.length-1;if(i>=0){h[i]++;n++;}}return {h,n};}
function plot(canvas,full,region,edges,label,target){const c=canvas.getContext('2d'),W=canvas.width,H=canvas.height,p={l:50,r:20,t:34,b:42},bw=W-p.l-p.r,bh=H-p.t-p.b;const f=hist(full,edges),s=hist(region,edges);const vf=f.h.map(v=>100*v/(f.n||1)),vs=s.h.map(v=>100*v/(s.n||1)),ymax=Math.max(1,Math.max(...vf,...vs)*1.15),xmax=label.startsWith('r')?1.4142135624:90;const X=v=>p.l+v/xmax*bw,Y=v=>H-p.b-v/ymax*bh;c.clearRect(0,0,W,H);c.fillStyle='white';c.fillRect(0,0,W,H);c.fillStyle='#f2d79e';c.fillRect(X(target[0]),p.t,X(target[1])-X(target[0]),bh);c.font='12px system-ui';c.textAlign='right';for(let j=0;j<=4;j++){let v=ymax*j/4;c.strokeStyle='#e4e9e3';c.beginPath();c.moveTo(p.l,Y(v));c.lineTo(W-p.r,Y(v));c.stroke();c.fillStyle='#526266';c.fillText(v.toFixed(1)+'%',p.l-5,Y(v)+4);}c.textAlign='center';for(let j=0;j<=6;j++){let v=xmax*j/6;c.fillText(v.toFixed(xmax>2?0:2),X(v),H-p.b+18);}for(const [vals,color]of [[vf,'#344c56'],[vs,'#c77730']]){c.strokeStyle=color;c.lineWidth=1.7;c.beginPath();vals.forEach((v,i)=>{if(i===0)c.moveTo(X(edges[i]),Y(v));else c.lineTo(X(edges[i]),Y(v));c.lineTo(X(edges[i+1]),Y(v));});c.stroke();}c.fillStyle='#27373c';c.fillText(label,W/2,H-4);c.textAlign='left';c.fillText('Full selected field: blue-grey · this crop: orange',p.l,17);return {full:f.n,region:s.n};}
function active(){const sc=D.scopes.find(g=>g.id===$('scope').value),rs=D.stacks.filter(r=>sc.stack_indices.includes(r.index)),pts=rs.flatMap(r=>r.points),box=D.crops[$('area').value][+$('roi').value],selected=pts.filter(p=>p[5]>=box[0]&&p[6]<=box[1]&&p[7]>=box[2]&&p[8]<=box[3]);return{sc,pts,box,selected};}
function drawROI(){const {sc,pts,box,selected}=active();$('roiNumber').textContent=(+$('roi').value+1)+' / 500';const canvas=$('map'),ctx=canvas.getContext('2d'),bg=images[$('mapBg').value];ctx.fillStyle='#111';ctx.fillRect(0,0,canvas.width,canvas.height);if(bg&&bg.complete)ctx.drawImage(bg,0,0,canvas.width,canvas.height);const sx=canvas.width/D.shape[1],sy=canvas.height/D.shape[0];for(const p of pts){ctx.fillStyle=p[3]!==null&&p[3]>=30&&p[3]<32?'#ffa846':'#5cc4d1';ctx.globalAlpha=.6;ctx.beginPath();ctx.arc(p[0]*sx,p[1]*sy,sc.stack_indices.length>1?1.8:2.5,0,2*Math.PI);ctx.fill();}ctx.globalAlpha=1;ctx.strokeStyle='#ffcb69';ctx.lineWidth=3;ctx.strokeRect(box[0]*sx,box[2]*sy,(box[1]-box[0])*sx,(box[3]-box[2])*sy);const e=D.ellipse;ctx.strokeStyle='#ff8181';ctx.lineWidth=1.5;ctx.beginPath();ctx.ellipse(e.cx*sx,e.cy*sy,e.rx*sx,e.ry*sy,0,0,2*Math.PI);ctx.stroke();const tc=plot($('thetaPlot'),pts.map(p=>p[3]),selected.map(p=>p[3]),D.theta_edges,'Apparent θ (degrees)',[30,32]);plot($('rPlot'),pts.map(p=>p[2]),selected.map(p=>p[2]),D.r_edges,'r · including finite above-model values',D.target_r);const n30=selected.filter(p=>p[3]!==null&&p[3]>=30&&p[3]<32).length;$('roiStats').innerHTML=`<b>${names[sc.id]||'Stack '+sc.id}</b><p>Crop x=${box[0]}–${box[1]}, y=${box[2]}–${box[3]}<br>${selected.length} retained detections; ${tc.region} valid θ<br>${n30} in 30–32°: <b>${tc.region?pct(n30/tc.region):'no valid angles'}</b><br>Whole selected field: ${pct(sc.crops[$('area').value].baseline[1])}</p><p class="small">All selected points remain unchanged; only spatial inclusion differs. No detector rerun.</p>`;}
$('target').textContent=D.target_r.map(v=>fmt(v,3)).join('–');$('scope').innerHTML=D.scopes.map(s=>`<option value="${s.id}">${names[s.id]||'Stack '+s.id+' · '+D.stacks[+s.id-1].time}</option>`).join('');$('scope').value='all';
for(const i of [1,2]){images[i]=new Image();images[i].onload=drawROI;images[i].src='bg'+i+'.png';}
$('size').onchange=updateSize;$('area').onchange=updateArea;$('roi').oninput=drawROI;$('mapBg').onchange=drawROI;$('scope').onchange=()=>{const sc=D.scopes.find(s=>s.id===$('scope').value);$('mapBg').value=sc.stack_indices.every(i=>i>=10)?'2':'1';drawROI();};$('advance').onclick=()=>{$('roi').value=String((+$('roi').value+73)%500);drawROI();};
const lo=D.scopes[0].tiles.map(t=>t.leave_out_metrics[1]);$('leaveOut').textContent=`Leaving out any one of the 16 fixed tiles leaves ${pct(Math.min(...lo))}–${pct(Math.max(...lo))} of the remaining valid-angle detections in 30–32° (full field: ${pct(D.scopes[0].crops['0.25'].baseline[1])}). This rules out one tile being solely responsible for the band; it does not rule out widespread background bias.`;
$('methods').innerHTML=Object.entries(D.methods).map(([k,v])=>'<p><b>'+esc(k)+':</b> '+esc(v)+'</p>').join('');updateSize();updateArea();
</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    d=json.loads((OUT/'results.json').read_text(encoding='utf8')) if args.render_only else calculate()
    if HTML:render(d)
    print(OUT/'index.html')
