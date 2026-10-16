"""Gold peak-cell brightness vs recovered orientation; no new detection or cuts.

The intensity is a spatial maximum on the 150-frame mean, not a temporal max.
Random windows use the identical peak-cell search and 18x18 angle extraction.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from polarcam.live.intensity_spots import channel_means, anisotropy
from polarcam import optics
from background_region_trial import inside, ELLIPSE
from control_recovery_check import sha, sig, clean

HERE=Path(__file__).resolve().parent
DAY=HERE.parents[1]/'data/hinges/polarcam/2026-10-12'
OUT=HERE/'gold_intensity_angle'
SOURCE=HERE/'control_recovery_check/gold_results.json'
SEED=2026101404
N_RANDOM=2000
MODEL=optics.theta_model()
ABC=[MODEL[k] for k in ['A','B','C']]


def save(name,d):
    (OUT/name).write_text(json.dumps(clean(d),ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf8')


def cells(image):
    """Non-overlapping physical four-analyzer cells, origin even/even."""
    return (image[0::2,0::2]+image[0::2,1::2]+image[1::2,0::2]+image[1::2,1::2])/4


def measure(image,boxes):
    ch=channel_means(image,boxes)
    xy=anisotropy(ch);r=np.linalg.norm(xy,axis=1)
    theta=np.degrees(optics.theta_from_radius(r,*ABC))
    cm=cells(image);out=[]
    for j,(l,rr,t,b) in enumerate(boxes):
        # Only complete globally aligned 2x2 cells inside each original aperture.
        x0,y0=(l+1)//2,(t+1)//2;x1,y1=rr//2,b//2
        cut=cm[y0:y1,x0:x1];dy,dx=np.unravel_index(np.argmax(cut),cut.shape)
        x,y=2*(x0+dx),2*(y0+dy);pk=float(cut[dy,dx])
        np.testing.assert_allclose(pk,image[y:y+2,x:x+2].mean(),rtol=0,atol=1e-10)
        c=channel_means(image,[[x,x+2,y,y+2]])[0]
        rp=float(np.linalg.norm(anisotropy(c)));tp=float(np.degrees(optics.theta_from_radius(rp,*ABC)))
        out.append(dict(bounds=list(map(int,(l,rr,t,b))),x=(l+rr)/2,y=(t+b)/2,
            peak_x=x,peak_y=y,cell_count=int(cut.size),peak_mean=pk,peak_sum=4*pk,
            aperture_mean=float(ch[j].mean()),r=float(r[j]),theta=float(theta[j]),
            sin2=float(np.sin(np.radians(theta[j]))**2),peak_r=rp,peak_theta=tp))
    return out


def random_boxes(shape,n,rng):
    h,w=shape;boxes=[]
    while len(boxes)<n:
        x,y=rng.integers([20,20],[w-20,h-20])
        if inside(x,y):continue
        boxes.append([int(x-9),int(x+9),int(y-9),int(y+9)])
    return boxes


def calculate():
    OUT.mkdir(exist_ok=True)
    source=json.loads(SOURCE.read_text(encoding='utf8'));assert source['complete']
    bginfo=source['background'];bgpath=DAY/'backgrounds'/bginfo['minimum_file']
    bmin=np.load(bgpath).astype(float);assert sha(bgpath)==bginfo['minimum_sha256']
    rawpath=DAY/'backgrounds'/bginfo['capture'];assert sig(rawpath)==bginfo['capture_signature']
    raw=np.load(rawpath,mmap_mode='r');n=bginfo['frames'];assert len(raw)==n+1
    p50=np.empty_like(bmin);p95=np.empty_like(bmin)
    for y in range(0,bmin.shape[0],64):
        block=raw[:n,y:y+64]
        np.testing.assert_array_equal(block.min(axis=0),bmin[y:y+64])
        p50[y:y+64],p95[y:y+64]=np.percentile(block,[50,95],axis=0,method='linear')
    assert hashlib.sha256(p95.astype(np.float32).tobytes()).hexdigest()==bginfo['p95_array_sha256']
    result=dict(complete=False,model=MODEL,source_sha256=sha(SOURCE),builder_sha256=sha(Path(__file__)),
        seed=SEED,background=bginfo,ellipse=ELLIPSE,records=[],background_samples={},
        methods=[
          'Original retained p95/2-SD detections and minimum-background 18x18 theta; no extra intensity, survival or size cut. Original raw saturation and ellipse exclusions retained.',
          'Brightness = maximum sum of a complete aligned 2x2 four-analyzer cell within the original 18x18 aperture, using the mean of 150 saved frames. Mean-per-pixel brightness equals sum/4. This is not the maximum over time. Odd aperture origins contain 64/72 cells; even origins contain 81.',
          'At each selected cell also measure the recorded minimum/median/p95 BG. These are profile levels, not estimates of noise SD. At saved zeros, adding minimum BG gives only an upper raw proxy.',
          '2000 uniform random 18x18 windows per field, outside the same ellipse and sensor margins. Same peak-cell search and angle extraction; raw-saturated apertures excluded. A separate nondetected subset is >=32 sensor pixels (Chebyshev) from every p95 candidate, including rejected ones.',
          'Random windows may include real unresolved rods. The nondetected subset is selected using this detector and is not an independent rod-free blank or false-positive calibration.',
          '5000 common random apertures in the recorded minimum/median/p95 profiles, with the same peak-cell search and angle formula. These are background-equivalent angles, not physical rod orientations. One shared BG capture is not 14 independent BG measurements.',
          'Comparing peak-cell intensity to an 18x18 angle mixes spatial scales intentionally to preserve the previous angle estimates. Same-cell angles are stored as a diagnostic. Both measured axes depend on image intensity; association alone does not validate a particle.',
          'For equal excitation/particle strength and an orientation-independent PSF, circular-excitation sin^2(theta) multiplied by the current collection model gives sum proportional to sin^2(theta)*(A+B*sin^2(theta)). A single-cell peak adds PSF/centering variation. No brightness law is fitted as a classifier.'
        ])
    bgboxes=random_boxes(bmin.shape,5000,np.random.default_rng([SEED,0]))
    for name,image in [('minimum',bmin),('median',p50),('p95',p95)]:
        result['background_samples'][name]=measure(image,bgboxes)
    print('Background profiles and 5000 paired reference windows ready',flush=True)
    for record in source['records']:
        i=record['index'];path=DAY/'frames'/record['name'];assert sig(path)==record['source_signature']
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
        assert sha(path.with_suffix('.json'))==record['sidecar_sha256']
        assert meta['actual']['frames']==150 and meta['actual']['phase_marker_appended'] and meta['background']['background_subtracted']
        a=np.load(path,mmap_mode='r');assert len(a)==151
        mean=np.sum(a[:150],axis=0,dtype=np.uint32).astype(float)/150
        rawpeakpath=path.parent/meta['fetch_frames_output']['raw_peak']['path']
        assert sha(rawpeakpath)==record['raw_peak_sha256'];rawpeak=np.load(rawpeakpath)
        spots=[s for s in record['spots'] if s['selection']=='kept']
        ms=measure(mean,[s['bounds'] for s in spots])
        for m,s in zip(ms,spots):
            assert not s['saturated'] and not s['inside_ellipse']
            np.testing.assert_allclose(m['r'],s['r'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(m['theta'],s['theta_deg'] if s['theta_deg'] is not None else np.nan,atol=1e-10,rtol=0,equal_nan=True)
            x,y=m['peak_x'],m['peak_y']
            bm=float(bmin[y:y+2,x:x+2].mean());bmed=float(p50[y:y+2,x:x+2].mean());bp=float(p95[y:y+2,x:x+2].mean())
            # Actual per-frame p95 correction at the SAME selected cell, not max-minus-profile.
            block=np.asarray(a[:150,y:y+2,x:x+2],float)
            after95=np.rint(np.maximum(block-(p95[y:y+2,x:x+2]-bmin[y:y+2,x:x+2]),0)).mean()
            m.update(id=s['id'],area=s['area'],raw_peak=s['raw_peak'],bg_min_mean=bm,bg_median_mean=bmed,bg_p95_mean=bp,
                signal_bg_ratio=m['peak_mean']/bm if bm>0 else np.nan,
                raw_upper_proxy_mean=m['peak_mean']+bm,peak_mean_p95_same_cell=float(after95),
                peak_fraction_zero=float(np.mean(block==0)))
        boxes=random_boxes(mean.shape,N_RANDOM,np.random.default_rng([SEED,i]))
        sample=measure(mean,boxes)
        tree=cKDTree([[s['x'],s['y']] for s in record['spots']])
        for m in sample:
            l,r,t,b=m['bounds'];distance=float(tree.query([m['x'],m['y']],p=np.inf)[0])
            m.update(raw_saturated=bool(rawpeak[t:b,l:r].max()>=4094),candidate_distance=distance,away=distance>=32)
        assert sig(path)==record['source_signature']
        result['records'].append(dict(index=i,name=record['name'],source_signature=record['source_signature'],note=record['note'],spots=ms,random=sample))
        save('results.json',result)
        print(f'Gold {i}/14: {len(ms)} retained; {sum(m["away"] and not m["raw_saturated"] for m in sample)} nondetected random windows',flush=True)
    result['complete']=True;save('results.json',result)
    return result


def values(rows,key):
    return np.array([r[key] if r[key] is not None else np.nan for r in rows],float)


def describe(rows):
    valid=[r for r in rows if r['theta'] is not None and np.isfinite(r['theta'])]
    if not valid:return dict(n=len(rows),valid=0)
    peak=values(valid,'peak_mean');ratio=values(valid,'signal_bg_ratio')
    s=values(valid,'sin2');a,b,_=ABC;shape=s*(a+b*s)/(a+b)
    i90=np.divide(peak,shape,out=np.full(len(peak),np.nan),where=shape>0)
    return dict(n=len(rows),valid=len(valid),peak_mean_median=float(np.median(peak)),peak_mean_q10_q90=np.quantile(peak,[.1,.9]).tolist(),
        model_equivalent_I90_mean_median=float(np.nanmedian(i90)),
        bg_at_peak_median=float(np.median(values(valid,'bg_min_mean'))),
        signal_bg_ratio_median=float(np.median(ratio)),fraction_signal_below_bg=float(np.mean(ratio<=1)),
        fraction_below_random95=float(np.mean([r['within_random95'] for r in valid])),
        fraction_28_34=float(np.mean([(28<=r['theta']<34) for r in valid])))


def stats(d):
    out=dict(groups={},fields=[],intensity_bands=[],ratio_bands=[],source_sha256=sha(OUT/'results.json'))
    spots=[]
    for rec in d['records']:
        random=[r for r in rec['random'] if r['away'] and not r['raw_saturated']]
        limit=float(np.quantile(values(random,'peak_mean'),.95))
        for s in rec['spots']:s.update(field=rec['index'],random95=limit,within_random95=s['peak_mean']<=limit)
        spots+=rec['spots']
        out['fields'].append(dict(index=rec['index'],random95=limit,random_n=len(random),**describe(rec['spots'])))
    for name,subset in [('all',spots),('28-34 degrees',[s for s in spots if s['theta'] is not None and 28<=s['theta']<34]),
                        ('50-60 degrees',[s for s in spots if s['theta'] is not None and 50<=s['theta']<60]),
                        ('excluding fields 10 and 12',[s for s in spots if s['field'] not in [10,12]])]:
        out['groups'][name]=describe(subset)
    for lo,hi in zip([0,25,50,100,200,400,800],[25,50,100,200,400,800,np.inf]):
        subset=[s for s in spots if lo<=s['peak_mean']<hi]
        out['intensity_bands'].append(dict(lo=lo,hi=hi,**describe(subset)))
    for lo,hi in zip([0,.5,1,2,5],[.5,1,2,5,np.inf]):
        subset=[s for s in spots if lo<=s['signal_bg_ratio']<hi]
        out['ratio_bands'].append(dict(lo=lo,hi=hi,**describe(subset)))
    save('summary.json',out);return out,spots


def heat(ax,rows,xedges,yedges,title,normalise=False):
    x=values(rows,'sin2');y=values(rows,'peak_mean');ok=np.isfinite(x)&np.isfinite(y)&(y>0)
    h=np.histogram2d(x[ok],y[ok],bins=[xedges,yedges])[0]
    assert int(h.sum())==int(ok.sum())
    if normalise:h=100*h/max(1,h.sum())
    a=np.ma.masked_equal(h.T,0)
    im=ax.pcolormesh(xedges,yedges,a,cmap='viridis',norm=LogNorm(vmin=float(a.min()),vmax=max(float(a.max()),float(a.min())*1.01)),shading='flat')
    ax.set_yscale('log');ax.set_xlim(0,1);ax.set_ylim(yedges[0],yedges[-1])
    ax.set_xlabel(r'Recovered $\sin^2\theta$ (18×18 aperture)');ax.set_ylabel('Peak 2×2 mean (DN / sensor pixel)')
    ax.set_title(f'{title}\nvalid n = {ok.sum():,}',fontsize=11)
    for ang in [30,60]:ax.axvline(np.sin(np.radians(ang))**2,color='#bc5536',ls=':',lw=.9,alpha=.8)
    return im


def figures(d,s,spots):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    random=[r for rec in d['records'] for r in rec['random'] if r['away'] and not r['raw_saturated']]
    uniform=[r for rec in d['records'] for r in rec['random'] if not r['raw_saturated']]
    groups=[('Gold detections · minimum-subtracted',spots),('Nondetected locations · minimum-subtracted',random),
        ('Uniform random locations · minimum-subtracted',uniform),('Recorded minimum BG',d['background_samples']['minimum']),
        ('Recorded median BG',d['background_samples']['median']),('Recorded p95 BG',d['background_samples']['p95'])]
    allp=np.concatenate([values(r,'peak_mean') for _,r in groups]);allp=allp[np.isfinite(allp)&(allp>0)]
    yedges=np.geomspace(10**np.floor(np.log10(allp.min())),10**np.ceil(np.log10(allp.max())),49)
    xedges=np.linspace(0,1,41)
    fig,axs=plt.subplots(2,3,figsize=(16,9.3),layout='constrained')
    for ax,(label,rows) in zip(axs.flat,groups):
        im=heat(ax,rows,xedges,yedges,label,normalise=True);fig.colorbar(im,ax=ax,label='% of valid samples / bin')
    fig.suptitle('Brightness and orientation jointly: identical peak-cell search in each 18×18 window\nSum of the four sensor pixels = 4 × plotted intensity. Dotted lines: 30° and 60°.',fontsize=15)
    fig.savefig(OUT/'joint_distribution.png',dpi=160);plt.close(fig)
    # Requested count as the third axis; log-spaced intensity bins, raw counts.
    x=values(spots,'sin2');y=values(spots,'peak_mean');ok=np.isfinite(x)&np.isfinite(y)&(y>0)
    xe=np.linspace(0,1,26);ye=np.geomspace(yedges[0],yedges[-1],31)
    h=np.histogram2d(x[ok],y[ok],bins=[xe,ye])[0];xx,yy=np.meshgrid((xe[:-1]+xe[1:])/2,(np.log10(4*ye[:-1])+np.log10(4*ye[1:]))/2,indexing='ij')
    fig=plt.figure(figsize=(11,8));ax=fig.add_subplot(projection='3d')
    surf=ax.plot_surface(xx,yy,h,cmap='viridis',linewidth=0,antialiased=True)
    ax.set_xlabel(r'Recovered $\sin^2\theta$',labelpad=12);ax.set_ylabel('log10 peak 2×2 sum (DN)',labelpad=12)
    ticks=np.arange(np.ceil(np.log10(4*ye[0])),np.floor(np.log10(4*ye[-1]))+1);ax.set_yticks(ticks)
    ax.set_zlabel('Detections / bin',labelpad=8);ax.view_init(elev=29,azim=-55)
    ax.set_title('Gold: count as the third dimension\nIntensity bins have equal log width; surface only connects binned counts',pad=15)
    fig.savefig(OUT/'count_surface.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Local BG levels and angle strata. All comparisons preserve original selections.
    valid=[z for z in spots if z['theta'] is not None]
    fig,axs=plt.subplots(1,3,figsize=(16,4.9),layout='constrained')
    sc=axs[0].scatter(values(valid,'bg_min_mean'),values(valid,'peak_mean'),c=values(valid,'theta'),s=9,alpha=.5,cmap='plasma',vmin=0,vmax=90,rasterized=True)
    axs[0].set(xscale='log',yscale='log',xlabel='Recorded minimum BG at same cell (DN / pixel)',ylabel='Saved peak-cell mean (DN / pixel)',title='Signal remaining after subtraction vs BG level')
    lo=min(values(valid,'bg_min_mean').min(),values(valid,'peak_mean').min());hi=max(values(valid,'bg_min_mean').max(),values(valid,'peak_mean').max())
    axs[0].plot([lo,hi],[lo,hi],'k--',lw=1,label='Remaining signal = BG level');axs[0].legend(fontsize=8);fig.colorbar(sc,ax=axs[0],label='Recovered θ (°)')
    axs[1].scatter(values(valid,'sin2'),values(valid,'signal_bg_ratio'),s=8,alpha=.25,color='#087e8b',rasterized=True)
    axs[1].axhline(1,color='#b75329',ls='--');axs[1].set(xlabel=r'Recovered $\sin^2\theta$',ylabel='Saved peak-cell mean / BG at that cell',yscale='log',title='A brightness ratio, not a signal-to-noise ratio')
    edges=np.arange(0,92,2)
    for label,lo,hi,col in [('Below 50 DN',0,50,'#82939b'),('50–200 DN',50,200,'#c67b12'),('At least 200 DN',200,np.inf,'#087e8b')]:
        a=np.array([z['theta'] for z in valid if lo<=z['peak_mean']<hi]);hh=np.histogram(a,edges)[0]
        axs[2].stairs(100*hh/max(1,len(a)),edges,label=f'{label} (n={len(a):,})',color=col)
    axs[2].axvspan(28,34,color='#c67b12',alpha=.1);axs[2].set(xlabel='Recovered θ (°)',ylabel='% within each brightness stratum / 2°',title='Does the near-30° feature persist in bright spots?');axs[2].legend(fontsize=8)
    fig.savefig(OUT/'brightness_context.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(4,4,figsize=(15,13),layout='constrained')
    for ax,rec in zip(axs.flat,d['records']):
        im=heat(ax,rec['spots'],xedges,yedges,f'Gold {rec["index"]} · {rec["name"].split("-")[-1][:6]}')
        fig.colorbar(im,ax=ax,label='count',shrink=.6)
    for ax in list(axs.flat)[14:]:ax.set_axis_off()
    fig.suptitle('Gold fields separately · no additional brightness cut\nIndependent colour scales per field; same position and intensity axes',fontsize=15)
    fig.savefig(OUT/'per_field.png',dpi=140);plt.close(fig)
    ss=np.linspace(0,1,501);a,b,c=ABC;rr=c*ss/(a+b*ss);intensity=ss*(a+b*ss)/(a+b)
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,xx,label in [(axs[0],ss,r'$s=\sin^2\theta$'),(axs[1],rr,'Anisotropy radius r')]:
        ax.plot(xx,intensity,color='#087e8b',lw=2.2,label='Excitation × collection')
        ax.plot(xx,ss,color='#ba7624',lw=1.2,ls='--',label='Simple sin²θ brightness only')
        for theta in [30,55,90]:
            s0=np.sin(np.radians(theta))**2;x0=s0 if ax is axs[0] else c*s0/(a+b*s0);y0=s0*(a+b*s0)/(a+b)
            ax.scatter([x0],[y0],s=25,color='#087e8b');ax.annotate(f'{theta}°',xy=(x0,y0),xytext=(-27,8) if theta==90 else (7,-8),textcoords='offset points',fontsize=10)
        ax.set(xlabel=label,ylabel=r'$I_{sum}/I_{sum}(90°)$',xlim=(0,1.03 if ax is axs[0] else 1),ylim=(0,1.1))
        ax.grid(alpha=.17);ax.legend(loc='upper left',fontsize=9)
    fig.suptitle('Expected curve for identical ideal rods: circular excitation, fixed illumination and collection\nLinear axes. A peak 2×2 block follows this only if its captured PSF fraction stays constant.',fontsize=12)
    fig.savefig(OUT/'expected_curves.png',dpi=170);plt.close(fig)
    return xedges,yedges


def render():
    d=json.loads((OUT/'results.json').read_text(encoding='utf8'));assert d['complete']
    s,spots=stats(d);xe,ye=figures(d,s,spots)
    # Compact plotting columns; numerical records retain the full provenance separately.
    cols=['field','sin2','peak_mean','theta','bg_min_mean','signal_bg_ratio','peak_theta','id','r']
    points=[[r[k] for k in cols] for r in spots]
    random=[[rec['index'],r['sin2'],r['peak_mean'],r['r']] for rec in d['records'] for r in rec['random'] if r['away'] and not r['raw_saturated']]
    bg={k:[[r['sin2'],r['peak_mean'],r['r']] for r in rows] for k,rows in d['background_samples'].items()}
    payload=clean(dict(summary=s,points=points,random=random,bg=bg,yedges=ye,model=MODEL,columns=cols))
    template=HTML_TEMPLATE
    methods=''.join('<li>'+x+'</li>' for x in d['methods'])
    html=template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':'))).replace('__METHODS__',methods)
    (OUT/'index.html').write_text(html,encoding='utf8')
    print(json.dumps(s['groups'],ensure_ascii=True,indent=2),flush=True)


HTML_TEMPLATE=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold: peak intensity and angle</title>
<style>
:root{font:16px/1.5 system-ui,sans-serif;color:#26383d;background:#f6f7f4}body{max-width:1400px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:22px;margin-top:30px}a{color:#087d8b}p{max-width:1120px}.panel{background:white;border:1px solid #d2ddd6;border-radius:6px;padding:16px;margin:18px 0;overflow:auto}.note{padding:14px 18px;background:#fff4dc;border-left:4px solid #b77c32}img{width:100%;height:auto}label{display:inline-block;margin:5px 15px 5px 0}select{font:inherit;padding:4px;background:white;border:1px solid #a9bbb1;border-radius:4px}canvas{width:100%;height:auto;display:block}table{border-collapse:collapse;font-size:14px}th,td{padding:7px 12px;text-align:left;border-bottom:1px solid #d4ded7}thead{background:#eaf0ea}.small{font-size:14px;color:#52686c}summary{cursor:pointer;font-weight:600}#hover{min-height:48px}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:750px){body{padding:12px}.two{grid-template-columns:1fr}}</style>
<h1>Gold: is the recovered angle associated with brightness or background?</h1>
<p>Fourteen gold-only survey stacks, 12 October. Original p95/2-SD detections; angles from minimum-background subtraction; oversized components allowed; saturation and the existing corner mask still excluded. <b>No new spot exclusions.</b></p>
<p class="note">A low intensity or proximity to background is a reason to inspect a detection, not proof that it is false. These axes share the same image data. The random windows and recorded BG below are reference populations, not verified rod-free controls.</p>
<h2>One point per detection: intensity versus r or sin²θ</h2>
<div class="panel">
<label>Population <select id="population"><option value="spots">Retained gold detections</option><option value="random">Random nondetected locations</option><option value="minimum">Recorded minimum BG</option><option value="median">Recorded median BG</option><option value="p95">Recorded p95 BG</option></select></label>
<label>Field <select id="field"><option value="all">All 14</option><option value="without">Exclude fields 10 and 12</option></select></label>
<label>Plot <select id="plot"><option value="scatter">Individual measurements</option><option value="heat">Count heatmap</option></select></label>
<label>Horizontal axis <select id="xaxis"><option value="r">Anisotropy radius r</option><option value="sin2">Recovered sin²θ</option></select></label>
<label>Intensity <select id="unit"><option value="sum">Sum of four pixels (DN)</option><option value="mean">Mean per sensor pixel (DN)</option></select></label>
<label>Vertical scale <select id="yscale"><option value="log">Logarithmic</option><option value="linear">Linear</option></select></label>
<label><input type="checkbox" id="guides" checked> Illustrative sin²θ + collection curves</label>
<p id="status" class="small" aria-live="polite"></p>
<canvas id="chart" width="1120" height="540" aria-label="Scatter plot of peak intensity versus anisotropy radius or recovered sin squared theta. Each point is one detection; optional curves show the ideal rod model."></canvas>
<p id="hover" class="small" aria-live="polite">Move over a bin or a point for numerical values.</p>
<p class="small">Brightness is the brightest complete aligned 2×2 block <b>within the original 18×18 extraction window</b>, on the 150-frame mean. It is not the maximum over frames. θ stays the existing 18×18 estimate. Heatmap bins have equal widths in the selected axes. The shaded strip is 28–34°. One gold detection above the model's r limit remains in the r scatter; it has no valid θ and is omitted only from the sin²θ view. Curves stop at the model limit.</p>
<p class="small">Optional curves assume equal particle strength, circular-excitation sin²θ and an orientation-independent PSF: I ∝ sin²θ(A+B sin²θ). They use arbitrary 90° mean intensities of 50, 200 and 800 DN; they are not fitted classifiers. Different rod sizes, illumination, focus, PSF sampling and residual background can all move a point away from them.</p>
</div>
<h2>What curve should an ideal rod follow?</h2>
<p>Write s = sin²θ. The collection model gives <b>I<sub>ψ</sub> = K[A + Bs + Cs cos 2(ψ−φ)]</b>, so summing the four channels cancels the φ term: <b>I<sub>sum</sub> = 4K(A+Bs)</b>. Under the assumed transverse circular excitation, K = K₀s. Therefore:</p>
<p><b>I<sub>sum</sub> = 4K₀s(A+Bs)</b>, &nbsp; <b>r = Cs/(A+Bs)</b>, &nbsp; hence &nbsp; <b>I<sub>sum</sub>(r) = 4K₀A²Cr/(C−Br)²</b>.</p>
<p>This is approximately a rising straight line, with a weak upward bend on linear axes. It is not an r² law. Here B/A ≈ 0.117, so the collection correction is modest. Without the excitation factor (K fixed), the sum would instead be 4K(A+Bs), nearly constant; that is a different assumption.</p>
<img src="expected_curves.png" alt="Normalised predicted intensity versus sine squared theta and versus anisotropy radius, including circular excitation and collection">
<p id="scaledFinding"></p>
<p>Different rod strengths produce different vertical scales K₀. For the brightest 2×2 cell, focus, subpixel position and polarization-dependent PSF shape add another scale variation. Thus a family of curves is expected across rods, not a single fitted line. Consistency with a curve is not independent validation because recovered θ uses the same intensities.</p>
<details class="panel"><summary>Count as an actual third axis</summary><img src="count_surface.png" alt="Three-dimensional count surface for gold peak intensity and sin squared theta"><p class="small">The heatmap above avoids occlusion. This surface connects the same kind of binned counts; the connecting surface does not imply an interpolated probability model.</p></details>
<h2>Are the intensities distinct from the local background?</h2>
<p id="finding"></p>
<div id="summaryTable" class="panel"></div>
<p>The local BG comparison uses the <b>same peak cell</b> chosen in the saved image. “Remaining signal / BG” compares the saved minimum-subtracted level with the recorded minimum BG level; it is <b>not SNR</b>. A genuine rod can be dimmer than the background and still be measurable if subtraction is accurate.</p>
<p>“Within random 95%” means the detection's peak intensity is below its field's 95th percentile at nondetected locations, after performing the same peak search in each random 18×18 window. This is a descriptive overlap measure, <b>not a false-positive probability</b>: those locations were selected using this detector and could contain unresolved rods.</p>
<img src="brightness_context.png" alt="Peak intensity compared with local recorded background; angle distributions split by brightness">
<details class="panel"><summary>Counts and near-30° fraction by brightness</summary><div id="bands"></div><p class="small">Intensity strata are descriptive and pool fields; different fields can have different brightness and angle distributions. They do not create newly accepted or rejected lists.</p></details>
<h2>Reference distributions with the same peak search</h2>
<img src="joint_distribution.png" alt="Matched peak intensity and angle distributions for gold detections, random image windows, and three background profiles">
<p class="small">Each panel is normalised separately; colours are percentages per bin. Recorded BG panels use 5000 identical locations in one shared capture. Their θ values only express the BG polarization ratios through the rod formula. The uniform-random panel includes candidate locations; the nondetected panel excludes positions within 32 sensor pixels of any p95 candidate. Neither panel constitutes an independent blank.</p>
<details class="panel"><summary>All fourteen gold fields separately</summary><img src="per_field.png" alt="Joint brightness and angle distributions for each gold field"><div id="fieldsTable"></div></details>
<details class="panel"><summary>Exact methods and limitations</summary><ol>__METHODS__</ol><p>Peaks favour positive noise, but all image and BG references use the same peak search. Profiles include detector noise and anything present during the moving-stage capture. BG levels are not residual-error estimates. No hypothesis test or automatic real/false label is justified by this plot alone.</p></details>
<p><a href="results.json">All measurements and provenance</a> · <a href="summary.json">Numerical summaries</a> · <a href="../control_recovery_check/index.html">Gold controls and injection recovery</a></p>
<script>
const D=__DATA__, $=id=>document.getElementById(id), fmt=(x,n=1)=>Number.isFinite(x)?x.toFixed(n):'—', pct=x=>fmt(100*x)+'%';
const table=(head,rows)=>'<table><thead><tr>'+head.map(s=>'<th>'+s+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(s=>'<td>'+s+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
for(let i=1;i<=14;i++){let o=document.createElement('option');o.value=String(i);o.textContent='Gold '+i;$('field').appendChild(o);}
$('summaryTable').innerHTML=table(['Group','Valid θ','Median peak mean, DN/pixel','Middle 80%, DN/pixel','Median same-cell BG, DN/pixel','Median remaining signal / BG','Remaining signal ≤ BG','Within random 95%'],Object.entries(D.summary.groups).map(([k,g])=>[k,g.valid,fmt(g.peak_mean_median),g.peak_mean_q10_q90.map(x=>fmt(x)).join('–'),fmt(g.bg_at_peak_median),fmt(g.signal_bg_ratio_median,2),pct(g.fraction_signal_below_bg),pct(g.fraction_below_random95)]));
const near=D.summary.groups['28-34 degrees'],high=D.summary.groups['50-60 degrees'];$('finding').textContent='The near-30° group has median peak brightness '+fmt(near.peak_mean_median)+' DN/pixel against '+fmt(near.bg_at_peak_median)+' DN/pixel of recorded BG at those cells. The 50–60° group has '+fmt(high.peak_mean_median)+' against '+fmt(high.bg_at_peak_median)+'. This identifies a dim, background-sensitive population; it does not distinguish real low-angle rods from residual structure. A brightness cut would also preferentially remove genuine low-angle rods. Fields differ, so use the field selector before interpreting the pooled relationship.';
$('scaledFinding').textContent='For otherwise identical rods, the model predicts I(55°)/I(30°) = 2.81. The measured group medians differ by '+fmt(high.peak_mean_median/near.peak_mean_median,2)+'×. Dividing each point by its own predicted angular factor gives median equivalent 90° peak means of '+fmt(near.model_equivalent_I90_mean_median)+' DN/pixel for 28–34° and '+fmt(high.model_equivalent_I90_mean_median)+' for 50–60°. Their brightness contrast is therefore compatible with the expected angle dependence; it is not by itself evidence that the low-angle population is spurious. These are pooled descriptive comparisons, not a fitted or independently validated rod classifier.';
$('bands').innerHTML=table(['Peak mean, DN/pixel','Valid θ','Fraction in 28–34°','Median signal / BG'],D.summary.intensity_bands.map(g=>[g.lo+'–'+(g.hi??'∞'),g.valid,pct(g.fraction_28_34),fmt(g.signal_bg_ratio_median,2)]));
$('fieldsTable').innerHTML=table(['Gold field','Valid θ','Median peak mean','28–34° fraction','Random peak 95th percentile','Nondetected windows'],D.summary.fields.map(g=>[g.index,g.valid,fmt(g.peak_mean_median),pct(g.fraction_28_34),fmt(g.random95),g.random_n]));
const cv=$('chart'), ctx=cv.getContext('2d'), L=92,T=28,W=900,H=420, NX=40,NY=D.yedges.length-1;
let active=[],bins=[],factor=4,pop='spots',xmode='r',xmax=1,logY=true,ylo=0,yhi=1,yedges=[];
function fieldOK(f){let v=$('field').value;return v==='all'||(v==='without'?![10,12].includes(f):f===+v);}
const xval=r=>xmode==='r'?r[pop==='spots'?8:3]:r[1];
function data(){pop=$('population').value;xmode=$('xaxis').value;$('field').disabled=!['spots','random'].includes(pop);let rows=pop==='spots'?D.points.filter(r=>fieldOK(r[0])):pop==='random'?D.random.filter(r=>fieldOK(r[0])):D.bg[pop].map(r=>[0,...r]);return rows.filter(r=>Number.isFinite(xval(r))&&Number.isFinite(r[2])&&r[2]>0);}
const rFromS=s=>D.model.C*s/(D.model.A+D.model.B*s),sFromR=r=>D.model.A*r/(D.model.C-D.model.B*r),shape=s=>s*(D.model.A+D.model.B*s)/(D.model.A+D.model.B);
const px=x=>L+W*x/xmax,ys=y=>logY?Math.log10(y):y,py=y=>T+H*(1-(ys(y)-ylo)/(yhi-ylo));
function draw(){
 active=data();factor=$('unit').value==='sum'?4:1;logY=$('yscale').value==='log';xmax=Math.max(1,...active.map(xval));
 if(logY){ylo=Math.log10(D.yedges[0]);yhi=Math.log10(D.yedges.at(-1));}else{ylo=0;const topI=Math.max(1,...active.map(r=>r[2])),step=10**Math.floor(Math.log10(topI));yhi=Math.ceil(topI/step)*step;}
 yedges=Array.from({length:NY+1},(_,j)=>{const z=ylo+(yhi-ylo)*j/NY;return logY?10**z:z;});bins=Array.from({length:NX},()=>Array(NY).fill(0));
 for(const r of active){let ix=Math.min(NX-1,Math.floor(xval(r)/xmax*NX)),iy=Math.min(NY-1,Math.floor((ys(r[2])-ylo)/(yhi-ylo)*NY));if(ix>=0&&ix<NX&&iy>=0&&iy<NY)bins[ix][iy]++;}
 const top=Math.max(1,...bins.flat());ctx.clearRect(0,0,cv.width,cv.height);ctx.fillStyle='#fff';ctx.fillRect(0,0,cv.width,cv.height);
 ctx.save();ctx.beginPath();ctx.rect(L,T,W,H);ctx.clip();
 const aa=Math.sin(28*Math.PI/180)**2,bb=Math.sin(34*Math.PI/180)**2,a=xmode==='r'?rFromS(aa):aa,b=xmode==='r'?rFromS(bb):bb;ctx.fillStyle='#fff0d8';ctx.fillRect(px(a),T,px(b)-px(a),H);
 if(xmode==='r'){ctx.fillStyle='#eceeef';ctx.fillRect(px(D.model.r_max),T,px(xmax)-px(D.model.r_max),H);}
 if($('plot').value==='heat'){
  for(let i=0;i<NX;i++)for(let j=0;j<NY;j++)if(bins[i][j]){const z=Math.log1p(bins[i][j])/Math.log1p(top);ctx.fillStyle=`hsl(${255-195*z} 65% ${36+20*z}%)`;ctx.fillRect(L+W*i/NX,T+H*(1-(j+1)/NY),W/NX+.1,H/NY+.1);}
 }else{ctx.fillStyle='rgba(8,126,139,0.4)';for(const r of active){ctx.beginPath();ctx.arc(px(xval(r)),py(r[2]),2.1,0,2*Math.PI);ctx.fill();}}
 if($('guides').checked){ctx.strokeStyle='#ac4c28';ctx.fillStyle='#81381f';ctx.lineWidth=1.3;ctx.setLineDash([5,4]);for(const P of [50,200,800]){ctx.beginPath();for(let i=1;i<=500;i++){let s=i/500,x=xmode==='r'?rFromS(s):s,y=P*shape(s);if(i===1)ctx.moveTo(px(x),py(y));else ctx.lineTo(px(x),py(y));}ctx.stroke();ctx.font='12px system-ui';ctx.fillText('I90='+fmt(P*factor,0),px(xmode==='r'?rFromS(.78):.78),py(P*shape(.78))-5);}ctx.setLineDash([]);}
 ctx.restore();ctx.strokeStyle='#82939b';ctx.lineWidth=1;ctx.strokeRect(L,T,W,H);ctx.fillStyle='#26383d';ctx.font='14px system-ui';ctx.textAlign='center';
 for(let j=0;j<=5;j++){let x=xmax*j/5;ctx.fillText(fmt(x,1),px(x),T+H+23);}ctx.fillText(xmode==='r'?'Anisotropy radius r from the 18×18 aperture':'Recovered sin²θ from the 18×18 aperture',L+W/2,T+H+51);
 ctx.textAlign='right';const ticks=logY?Array.from({length:Math.floor(yhi)-Math.ceil(ylo)+1},(_,j)=>10**(Math.ceil(ylo)+j)):Array.from({length:6},(_,j)=>yhi*j/5);for(const v of ticks)ctx.fillText(fmt(v*factor,v*factor>0&&v*factor<1?2:0),L-10,py(v)+4);
 ctx.save();ctx.translate(23,T+H/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText((factor===4?'Peak 2×2 sum (DN':'Peak 2×2 mean (DN / sensor pixel')+(logY?', logarithmic axis)':', linear axis)'),0,0);ctx.restore();
 if($('plot').value==='heat'){ctx.textAlign='left';ctx.font='12px system-ui';ctx.fillText('Count / bin',1010,42);for(let j=0;j<100;j++){let z=j/99;ctx.fillStyle=`hsl(${255-195*z} 65% ${36+20*z}%)`;ctx.fillRect(1018,65+(99-j)*2,18,2.1);}ctx.fillStyle='#26383d';ctx.fillText(String(top),1041,72);ctx.fillText('1',1041,266);}
 $('status').textContent=active.length.toLocaleString()+' measurements shown. '+(pop==='spots'?'Existing angle estimates and selections are unchanged.':pop==='random'?'Random positions away from candidates; these are not verified blanks.':'BG-equivalent angles; the field selector does not apply to this shared profile.')+(xmode==='r'?' Grey region: r above the model limit; no valid θ.':'');
 $('hover').textContent='Move over a bin or a point for numerical values.';
}
cv.onmousemove=e=>{let box=cv.getBoundingClientRect(),x=(e.clientX-box.left)*cv.width/box.width,y=(e.clientY-box.top)*cv.height/box.height;if(x<L||x>L+W||y<T||y>T+H)return;
 if($('plot').value==='heat'){let i=Math.min(NX-1,Math.floor((x-L)/W*NX)),j=Math.min(NY-1,Math.floor((1-(y-T)/H)*NY));let lo=yedges[j]*factor,hi=yedges[j+1]*factor;$('hover').textContent=`${xmode==='r'?'r':'sin²θ'} ${fmt(i/NX*xmax,3)}–${fmt((i+1)/NX*xmax,3)}; peak intensity ${fmt(lo)}–${fmt(hi)} DN: ${bins[i][j]} measurements.`;}
 else{let best=null,dist=Infinity;for(const r of active){let dd=(px(xval(r))-x)**2+(py(r[2])-y)**2;if(dd<dist){dist=dd;best=r;}}if(best&&dist<400){let r=best;$('hover').textContent=(pop==='spots'?`Gold ${r[0]}, ${r[7]} · θ ${fmt(r[3])}° · `:'')+`${xmode==='r'?'r':'sin²θ'} ${fmt(xval(r),3)} · peak ${fmt(r[2]*factor)} DN`+(pop==='spots'?` · same-cell BG ${fmt(r[4]*factor)} DN · remaining signal / BG ${fmt(r[5],2)} · same-cell θ ${fmt(r[6])}°`:'');}}
};
for(const id of ['population','field','plot','unit','guides','xaxis','yscale'])$(id).onchange=draw;draw();
</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['calculate','render','all'],default='all');args=parser.parse_args()
    if args.phase in ['calculate','all']:calculate()
    if args.phase in ['render','all']:render()
