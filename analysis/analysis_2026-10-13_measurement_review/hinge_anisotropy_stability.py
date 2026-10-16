"""Rank the existing frame-1 hinge detections by per-exposure X/Y spread."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from polarcam.live.intensity_spots import channel_means, anisotropy
from figure1_single_frame import A, B, C, R_LIMIT
from detector_grid_trial import match

HERE = Path(__file__).resolve().parent
DAY = HERE.parents[1]/'data/hinges/polarcam/2026-10-12'
OUT = HERE/'hinge_anisotropy_stability'
SOURCE = HERE/'richard_email_20261014/figure_02_results.json'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def theta(r):
    result = np.full_like(r, np.nan, dtype=float)
    good = np.isfinite(r) & (r >= 0) & (r <= R_LIMIT)
    result[good] = np.degrees(np.arcsin(np.sqrt(A*r[good]/(C-B*r[good]))))
    return result


def extract(movie, bounds):
    """Vectorized native 9x9 channel sampling, including odd aperture origins."""
    b = np.asarray(bounds)
    assert np.all(b[:,1]-b[:,0] == 18) and np.all(b[:,3]-b[:,2] == 18)
    channels = np.empty((150,len(b),4), dtype=float)
    for j,(py,px) in enumerate([(1,1),(0,0),(0,1),(1,0)]):
        yy = b[:,2,None]+((py-b[:,2])%2)[:,None]+2*np.arange(9)
        xx = b[:,0,None]+((px-b[:,0])%2)[:,None]+2*np.arange(9)
        for t in range(150):
            channels[t,:,j] = movie[t,yy[:,:,None],xx[:,None,:]].mean(axis=(1,2),dtype=float)
    for t in [0,74,149]:
        sample = np.arange(0,len(b),max(1,len(b)//12))
        np.testing.assert_array_equal(channels[t,sample],channel_means(movie[t],b[sample]))
    return channels


def summarize(ch):
    xy = anisotropy(ch)
    r = np.linalg.norm(xy,axis=-1)
    angles = theta(r)
    intensity = ch.mean(axis=-1)*255/4095
    sd = np.nanstd(xy,axis=0,ddof=1)
    spread = np.linalg.norm(sd,axis=-1)
    rmed = np.nanmedian(r,axis=0)
    ntheta=np.isfinite(angles).sum(axis=0)
    theta_median=np.full_like(rmed,np.nan);theta_sd=np.full_like(rmed,np.nan)
    theta_median[ntheta>0]=np.nanmedian(angles[:,ntheta>0],axis=0)
    theta_sd[ntheta>1]=np.nanstd(angles[:,ntheta>1],axis=0,ddof=1)
    data = dict(sd_x=sd[:,0],sd_y=sd[:,1],sd_xy=spread,sd_r=np.nanstd(r,axis=0,ddof=1),
                relative_sd=spread/rmed,median_r=rmed,median_theta=theta_median,
                theta_sd=theta_sd,mean_intensity=intensity.mean(axis=0),
                intensity_cv=intensity.std(axis=0,ddof=1)/intensity.mean(axis=0),
                valid_xy=np.isfinite(xy).all(axis=-1).sum(axis=0),valid_theta=ntheta)
    return data,xy,r,intensity


def clean(v):
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
    if isinstance(v,(np.floating,float)): return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer): return int(v)
    return v


def build():
    OUT.mkdir(exist_ok=True)
    data = json.loads(SOURCE.read_text())
    records = [r for r in data['pooled_records'] if r['group']=='DNA']
    assert len(records)==16
    sigs = {s['file']:s for s in data['pooled_cache_signature']['sources']}
    rows=[]; traces={}; sources=[]
    for rec in records:
        path = DAY/'frames'/rec['file']; meta_path=path.with_suffix('.json')
        m=json.loads(meta_path.read_text()); sig=sigs[path.name]
        assert sig['signature']==dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        assert sha(meta_path)==sig['metadata_sha256']
        assert m['actual']['frames']==150 and m['actual']['phase_marker_appended'] and m['background']['background_subtracted']
        assert m['roi']['OffsetX']==m['roi']['OffsetY']==0
        spots=[s for s in rec['thresholds']['2']['spots'] if s['selection']=='kept']
        bounds=[s['bounds'] for s in spots]
        assert all(not s['saturated'] and not s['inside_ellipse'] and s['raw_peak']<4094 for s in spots)
        rawpath=path.parent/m['fetch_frames_output']['raw_peak']['path']
        assert sha(rawpath)==sig['raw_peak_sha256']
        raw=np.load(rawpath,mmap_mode='r')
        assert all(raw[t:b,l:r].max()<4094 for l,r,t,b in bounds)
        identity=dict(source=sig,results_sha256=sha(SOURCE),bounds=bounds,algorithm='native-9x9-phase-v1')
        cache=OUT/f'channels_H{rec["index"]:02}.npz'; cachemeta=cache.with_suffix('.json')
        movie=np.load(path,mmap_mode='r'); assert len(movie)==151
        if cache.exists() and cachemeta.exists() and json.loads(cachemeta.read_text())==identity:
            with np.load(cache) as z: ch=z['channels']
        else:
            ch=extract(movie,bounds)
            np.savez_compressed(cache,channels=ch)
            cachemeta.write_text(json.dumps(identity),encoding='utf8')
        metrics,xy,r,intensity=summarize(ch)
        np.testing.assert_allclose(r[0],[s['r'] for s in spots],atol=1e-12)
        strict={}
        for k in ['4','8']:
            other=rec['thresholds'][k]['spots']; pairs,_,_=match(spots,other,radius=6)
            strict[k]={i for i,j in pairs.items() if other[j]['selection']=='kept'}
        traces[rec['index']]=(xy,r,intensity)
        for i,s in enumerate(spots):
            row=dict(id=f'H{rec["index"]:02}-SD2-C{s["component"]}',stack=rec['index'],local_index=i,
                     file=rec['file'],x=s['x'],y=s['y'],bounds=s['bounds'],raw_peak=s['raw_peak'],
                     kept4=i in strict['4'],kept8=i in strict['8'],fps=m['actual']['fps'],
                     duration=(149/m['actual']['fps']),first_theta=float(theta(np.array([r[0,i]]))[0]))
            row.update({k:float(v[i]) for k,v in metrics.items()})
            rows.append(row)
        sources.append(dict(stack=rec['index'],file=rec['file'],n=len(spots),fps=m['actual']['fps'],duration=149/m['actual']['fps'],signature=sig))
        print(f'H{rec["index"]:02}: {len(spots)} spots, all 150 exposures extracted',flush=True)
    assert len(rows)==15761
    rows.sort(key=lambda s:(s['valid_xy']!=150,s['sd_xy']))
    for rank,s in enumerate(rows,1): s['rank']=rank
    ranked=[s for s in rows if s['valid_xy']==150]

    def describe(rs):
        a=np.array([s['median_theta'] for s in rs]); finite=np.isfinite(a)
        return dict(n=len(rs),valid_median_theta=int(finite.sum()),median_theta=float(np.nanmedian(a)),
                    theta_range=[float(np.nanmin(a)),float(np.nanmax(a))],
                    below40=int(np.sum(a<40)),at_least45=int(np.sum(a>=45)),
                    median_intensity=float(np.median([s['mean_intensity'] for s in rs])),
                    sd_xy_range=[float(min(s['sd_xy'] for s in rs)),float(max(s['sd_xy'] for s in rs))],
                    stacks=len(set(s['stack'] for s in rs)))
    n1=max(1,int(np.ceil(len(ranked)*.01)))
    strict8=[s for s in ranked if s['kept8']]
    relative=sorted(ranked,key=lambda s:s['relative_sd'])
    rsorted=sorted(ranked,key=lambda s:s['sd_r'])
    summary=dict(all=describe(ranked),smallest50=describe(ranked[:50]),smallest1percent=describe(ranked[:n1]),
                 strict8_all=describe(strict8),strict8_smallest50=describe(strict8[:50]),
                 relative_smallest50=describe(relative[:50]),r_sd_smallest50=describe(rsorted[:50]),
                 incomplete_xy=len(rows)-len(ranked),sources=sources)
    summary['smallest50_also8sd']=sum(s['kept8'] for s in ranked[:50])

    # Compare the long targeted DNA recordings separately; different duration,
    # field illumination and aperture mean they are not part of the survey rank.
    recdata=json.loads((HERE/'spot_recording_gallery/results.json').read_text())
    reference=[]
    for rec in recdata['records']:
        if not rec['label'].startswith('DNA '): continue
        cache=HERE/'first_pass_review'/f'trace_{rec["stamp"]}.npz'
        assert json.loads(cache.with_suffix('.json').read_text())==rec['provenance']
        with np.load(cache) as z: ch=z['channels']
        met,*_=summarize(ch[:,None,:])
        reference.append(dict(label=rec['label'],saturated=rec['saturated'],frames=len(ch),
                              duration=(len(ch)-1)/rec['metrics']['fps'],**{k:float(v[0]) for k,v in met.items()}))
    summary['targeted_recordings']=reference
    (OUT/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False),encoding='utf8')
    (OUT/'ranking.json').write_text(json.dumps(clean(rows),separators=(',',':'),allow_nan=False),encoding='utf8')
    (OUT/'data.js').write_text('window.ROWS='+json.dumps(clean(rows),separators=(',',':'),allow_nan=False)+';',encoding='utf8')
    with (OUT/'ranking.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(clean(rows))

    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    a=np.array([s['median_theta'] for s in ranked]);spread=np.array([s['sd_xy'] for s in ranked])
    bright=np.array([s['mean_intensity'] for s in ranked])
    sc=axes[0].scatter(a,spread,c=np.log10(bright),s=3,alpha=.35,cmap='viridis',rasterized=True)
    axes[0].set(yscale='log',xlabel='Median single-exposure θ (°)',ylabel='X/Y spread: √(SD(X)² + SD(Y)²)',xlim=(0,90))
    axes[0].scatter(a[:n1],spread[:n1],s=12,facecolors='none',edgecolors='#b54b2b',lw=.6,label=f'Smallest 1% (n={n1})')
    axes[0].legend(fontsize=8)
    cb=fig.colorbar(sc,ax=axes[0]);cb.set_label('log₁₀ aperture mean intensity (0–255 scale)')
    for rs,label,col in [(ranked,'All 2 SD detections','#668797'),(ranked[:n1],'Smallest 1% X/Y spread','#b54b2b')]:
        angles=np.array([s['median_theta'] for s in rs]);angles=angles[np.isfinite(angles)]
        axes[1].hist(angles,bins=np.arange(0,95,5),weights=np.ones(len(angles))*100/len(angles),histtype='step',lw=1.8,label=label,color=col)
    axes[1].set(xlabel='Median single-exposure θ (°)',ylabel='Spots per 5° bin (%)',xlim=(0,90))
    axes[1].yaxis.set_major_formatter(PercentFormatter(100));axes[1].legend(fontsize=9)
    fig.suptitle('Hinge survey · 16 stacks · 150 individual exposures per spot',fontsize=13)
    fig.savefig(OUT/'overview.png',dpi=180);plt.close(fig)

    selected=[]
    for pool,reason,n in [(ranked,'Smallest X/Y spread overall',12),(strict8,'Smallest X/Y spread, also matched at 8 SD',6),
                          ([s for s in ranked if s['median_theta']>=45],'Smallest X/Y spread with median θ≥ 45°',6)]:
        for s in pool[:n]:
            if any(v['id']==s['id'] for v in selected): continue
            s['example_reason']=reason;selected.append(s)
    for s in selected:
        xy,r,intensity=traces[s['stack']];i=s['local_index'];movie=np.load(DAY/'frames'/s['file'],mmap_mode='r')
        x0=int(np.clip(2*round((s['x']-32)/2),0,movie.shape[2]-64));y0=int(np.clip(2*round((s['y']-32)/2),0,movie.shape[1]-64))
        crop=np.array(movie[0,y0:y0+64,x0:x0+64]);cells=crop.reshape(32,2,32,2).mean(axis=(1,3))
        fig,axes=plt.subplots(1,3,figsize=(11,2.9),gridspec_kw={'width_ratios':[1,1,1.8]},layout='constrained')
        axes[0].imshow(cells,cmap='gray',vmin=0,vmax=4095/20,interpolation='nearest',extent=(x0,x0+64,y0+64,y0))
        l,rr,t,b=s['bounds'];axes[0].add_patch(plt.Rectangle((l,t),rr-l,b-t,fill=False,edgecolor='#36c5cd',lw=.8));axes[0].set_axis_off();axes[0].set_title('Frame 1 ·20× · 64×64 pixels',fontsize=9)
        phi=np.linspace(0,2*np.pi,300);axes[1].plot(R_LIMIT*np.cos(phi),R_LIMIT*np.sin(phi),':',c='.65',lw=.8)
        axes[1].scatter(xy[:,i,0],xy[:,i,1],s=4,c='#147f9b',alpha=.65);axes[1].set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),aspect='equal',xlabel='X',ylabel='Y',title='All 150 exposures')
        axes[1].axhline(0,lw=.4,c='.8');axes[1].axvline(0,lw=.4,c='.8')
        axes[2].plot(np.arange(150)/s['fps'],r[:,i],lw=.75,c='#8255a0');axes[2].set(ylim=(0,1.05),xlabel='Time (s)',ylabel='r',title=f'θ median {s["median_theta"]:.1f}° · XY spread {s["sd_xy"]:.4f}')
        axes[2].axhline(R_LIMIT,ls=':',c='.65',lw=.8)
        fig.suptitle(s['id']+f' · rank {s["rank"]:,} ·'+('also 8 SD' if s['kept8'] else 'also 4 SD' if s['kept4'] else '2 SD only/unmatched'),fontsize=11)
        fig.savefig(OUT/(s['id']+'.png'),dpi=150);plt.close(fig)
    build_html(summary,selected)
    print(json.dumps(clean({k:v for k,v in summary.items() if k not in ['sources','targeted_recordings']}),indent=2),flush=True)


def build_html(summary,selected):
    top=summary['smallest50'];all_=summary['all'];strict=summary['strict8_smallest50']
    cards=''.join(f'<figure id="{s["id"]}"><figcaption>{s["example_reason"]} · {s["id"]}</figcaption><img loading="lazy" src="{s["id"]}.png"></figure>' for s in selected)
    example_ids={s['id'] for s in selected}
    reference_rows=''.join(f'<tr><td>{s["label"]}</td><td>{s["median_theta"]:.1f}°</td><td>{s["sd_xy"]:.5f}</td><td>{s["sd_r"]:.5f}</td><td>{s["duration"]:.1f} s</td></tr>'
                          for s in sorted(summary['targeted_recordings'],key=lambda x:x['sd_xy']) if not s['saturated'])
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hinge anisotropy stability</title>
<style>body{font:16px system-ui;background:#f4f6f5;color:#263d45;margin:24px}main{max-width:1250px;margin:auto}img{width:100%}figure{margin:22px 0;background:white;padding:12px}a{color:#087c96}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #d4dede;text-align:right}td:first-child,th:first-child{text-align:left}input,select,button{font:inherit;margin:6px;padding:5px}summary{cursor:pointer}code{font-size:14px}.scroll{overflow:auto}p{line-height:1.55}</style><main>
<h1>Which hinge detections have the smallest anisotropy cloud?</h1>
<p><a href="../spot_recording_gallery/dna_selected_rows.html">Six selected DNA recordings</a> · <a href="ranking.csv">All ranks / CSV</a> · <a href="summary.json">Numerical summary</a></p>
<p>16 wide-field hinge stacks; all 15,761 retained baseline detections. Each spot uses all 150 individual exposures in its fixed 18×18 aperture; no temporal smoothing or averaging before anisotropy reconstruction. Stack duration is about 2.06 s, much shorter than the targeted DNA 3 recording.</p>
<p><b>RESULT</b></p>
<p>Rank by <code>√(SD(X)²+SD(Y)²)</code>, the size of the X/Y cloud (sample SD). This includes both radial and sideways motion. “SD(r)” alone can miss motion along an arc. θ shown here is the median of valid single-exposure angles, not an angle calculated from averaged intensities. Frames with r above the model limit have no θ; their X/Y and r still contribute to the spread.</p>
<p>Small cloud does not by itself identify a hinge: background can be stable; brighter signals can have lower measurement noise; and the same azimuth motion makes a smaller X/Y excursion when r is small. “Spread/r” is available as a sensitivity check, not a noise correction or a full angular-motion estimate. These are descriptive ranks, not proof of immobilisation.</p>
<img src="overview.png" alt="Anisotropy spread versus angle and angle distribution of the smallest1percent">
<details><summary>Detection, background and exclusions</summary><p>Existing Figure 1 baseline: frame 1, recorded minimum background; aligned 2×2 intensity map; difference of Gaussians σ 1 and 6 sensor pixels; whole filtered mean + 2 SD; connected area ≥ 10 pixels with no upper size limit. Existing top-right ellipse and strongest-first overlapping 18×18 aperture exclusions retained. Saturating spots remain excluded: any actual raw pixel ≥ 4094 in the aperture or detection component box during any of the 150 exposures. No survival filter or low-background mask.</p><p>Extraction uses the saved minimum-subtracted pixels, unchanged for every frame. Polariser phase is preserved. Model is annular Fresnel/BFP NA 0.39–1.30, r limit ≈ 0.928. 4/8 SD labels are one-to-one matches within 6 sensor pixels to the independently detected kept lists; missing matches can also reflect shifts, merges or splits.</p></details>
<details><summary>Long DNA spot recordings, ranked separately</summary><p>Unsaturated recordings only. Different duration, illumination and 20×20 aperture; these do not enter the 16-stack ranking. DNA 4, 3, 9 and 6 have the smallest X/Y spreads in this selected recording set. DNA 7 has a small SD(r) but moves along an arc.</p><table><thead><tr><th>Recording</th><th>Median θ</th><th>X/Y spread</th><th>SD(r)</th><th>Duration</th></tr></thead><tbody>REFERENCE_ROWS</tbody></table><p><a href="../spot_recording_gallery/index.html">Open full recording gallery</a></p></details>
<h2>Sortable list</h2><p>Ranks cover all 2 SD baseline spots. The example links below show a subset; filtering the list does not change the original detection selection.</p>
<label>Sort<select id="sort"><option value="sd_xy">Smallest X/Y spread</option><option value="sd_r">Smallest SD(r)</option><option value="relative_sd">Smallest spread/r</option><option value="theta_sd">Smallest SD(θ), valid exposures</option><option value="median_theta">Smallest median θ</option><option value="mean_intensity">Lowest intensity</option></select></label>
<label>Detection<select id="detect"><option value="2">All 2 SD</option><option value="4">Also matched at 4 SD</option><option value="8">Also matched at 8 SD</option></select></label><label>Search<input id="query" placeholder="Spot ID or H01"></label><p id="count"></p>
<div class="scroll"><table><thead><tr><th>Spot / overall rank</th><th>XY spread</th><th>SD(r)</th><th>Spread/r</th><th>Median θ</th><th>SD(θ)</th><th>Mean intensity¹</th><th>Intensity CV</th><th>SD labels</th><th>Validθ /150</th></tr></thead><tbody id="rows"></tbody></table></div><button id="prev">Previous 50</button><button id="next">Next 50</button><p>¹ Mean per sensor pixel in the 18×18 aperture, on the 0–255 scale with no display gain. Full native intensities were used for X/Y.</p>
<h2>Examples</h2><p>First 12 in the overall ranking, plus the first 6 matched at 8 SD and first 6 with median θ≥ 45°, with duplicates shown once. All use the same display gain and full X/Y axes.</p>CARDS</main>
<script src="data.js"></script><script>
const examples=new Set(EXAMPLES);let page=0;const $=id=>document.getElementById(id),fmt=(x,n=3)=>x===null?'—':x.toFixed(n);
function draw(){let key=$('sort').value,q=$('query').value.trim().toLowerCase(),k=$('detect').value;let rs=ROWS.filter(s=>s.valid_xy===150&&(k==='2'||s['kept'+k])&&(!q||s.id.toLowerCase().includes(q))).sort((a,b)=>(a[key]??Infinity)-(b[key]??Infinity));let pages=Math.max(1,Math.ceil(rs.length/50));page=Math.min(page,pages-1);$('count').textContent=`${rs.length.toLocaleString()} spots · page ${page+1}/${pages}`;$('rows').innerHTML=rs.slice(page*50,page*50+50).map(s=>`<tr><td>${examples.has(s.id)?`<a href="#${s.id}">${s.id}</a>`:s.id}<br>#${s.rank}</td><td>${fmt(s.sd_xy,4)}</td><td>${fmt(s.sd_r,4)}</td><td>${fmt(s.relative_sd)}</td><td>${fmt(s.median_theta,1)}°</td><td>${fmt(s.theta_sd,2)}°</td><td>${fmt(s.mean_intensity,2)}</td><td>${fmt(s.intensity_cv)}</td><td>2${s.kept4?',4':''}${s.kept8?',8':''}</td><td>${s.valid_theta}</td></tr>`).join('');$('prev').disabled=page===0;$('next').disabled=page===pages-1;}
for(const id of ['sort','detect','query'])$(id).addEventListener('input',()=>{page=0;draw()});$('prev').onclick=()=>{page--;draw()};$('next').onclick=()=>{page++;draw()};draw();
</script>'''
    one=summary['smallest1percent'];rel=summary['relative_smallest50']
    result=(f'The {one["n"]} smallest X/Y clouds (1%) span {one["theta_range"][0]:.1f}–{one["theta_range"][1]:.1f}° '
            f'(median {one["median_theta"]:.1f}°). Of the 50 smallest clouds, {summary["smallest50_also8sd"]} also match the 8 SD detector. '
            f'Among the 50 most stable candidates also matched at 8 SD, the median is {strict["median_theta"]:.1f}°. '
            f'Ranking by spread/r instead admits {rel["at_least45"]}/50 spots at θ ≥ 45°: the definition of “stable” matters.')
    page=page.replace('RESULT',result).replace('CARDS',cards).replace('EXAMPLES',json.dumps(sorted(example_ids))).replace('REFERENCE_ROWS',reference_rows)
    (OUT/'index.html').write_text(page,encoding='utf8')


if __name__=='__main__': build()
