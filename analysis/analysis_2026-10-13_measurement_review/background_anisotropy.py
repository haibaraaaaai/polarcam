"""Anisotropy implied by the three saved minimum-projection background profiles.

No rod detection, no new rejection rule, and no angle inversion. Source files
are read only. Compare disjoint 2x2 cells with 18x18 channel-first averaging.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.ndimage import uniform_filter
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from polarcam.live.intensity_spots import channel_means,anisotropy

HERE=Path(__file__).resolve().parent
OUT=HERE/'background_anisotropy'
DAY=HERE.parents[1]/'data/hinges/polarcam/2026-10-12'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def radius(ch):
    v=anisotropy(ch)
    return v,np.linalg.norm(v,axis=-1)
def stats(ch,domain):
    v,r=radius(ch);valid=domain&np.isfinite(r)
    return dict(cells=int(valid.sum()),invalid=int(np.sum(domain&~np.isfinite(r))),
        r_quantiles=np.quantile(r[valid],[.05,.25,.5,.75,.95]).tolist(),
        mean_x=float(v[...,0][valid].mean()),mean_y=float(v[...,1][valid].mean()),
        median_x=float(np.median(v[...,0][valid])),median_y=float(np.median(v[...,1][valid])),
        above_model=float(np.mean(r[valid]>.9276834494677721)),
        mean_intensity=float(ch.mean(-1)[valid].mean()))
def write_map(a,path,kind):
    lo,hi,cmap={'intensity':(0,204.75,'gray'),'r':(0,np.sqrt(2),'viridis'),
                'x':(-1,1,'RdBu_r'),'y':(-1,1,'RdBu_r')}[kind]
    rgb=plt.get_cmap(cmap)(np.clip(np.nan_to_num((a-lo)/(hi-lo)),0,1),bytes=True)[...,:3]
    rgb[~np.isfinite(a)]=[185,185,185]
    Image.fromarray(rgb).save(path)


def build():
    OUT.mkdir(exist_ok=True)
    source=HERE/'low_background_mask/results.json';old=json.loads(source.read_text(encoding='utf-8'))
    result=dict(groups=[],source_sha256=digest(source),builder_sha256=digest(Path(__file__)),
      mapping='I0=odd row/odd col; I90=even/even; I45=even/odd; I135=odd/even, zero-based full-sensor coordinates.',
      calculation='X=(I0-I90)/(I0+I90); Y=(I45-I135)/(I45+I135); r=hypot(X,Y). No APD matrix, gain correction, extra subtraction, or rod-angle inversion.',
      averaging='2x2: one channel sample each, disjoint aligned cells. 18x18: 9x9 means within each polarization plane, then form X,Y,r. Four-cell border excluded from both distribution summaries for equal area.',
      interpretation='Saved per-pixel minimum profiles, not simultaneous polarization measurements or a pure optical-background calibration. Different channels may attain their minima on different frames. Spatial texture and pixel/channel response also contribute.',
      dimensions=dict(sensor=[2056,2464],cell_map=[1028,1232],sliding_2x2=[2055,2463]))
    maps=[];bins=np.linspace(0,np.sqrt(2),142)
    for i,g in enumerate(old['groups']):
        p=DAY/'backgrounds/used'/f'{g["sha256"]}.npy';assert digest(p)==g['sha256']
        a=np.load(p,allow_pickle=False).astype(float);h,w=a.shape
        ch=np.stack([a[1::2,1::2],a[0::2,0::2],a[0::2,1::2],a[1::2,0::2]],axis=-1)
        assert ch.shape==(h//2,w//2,4)
        domain=np.zeros(ch.shape[:2],bool);domain[4:-4,4:-4]=True
        c18=uniform_filter(ch,size=(9,9,1),mode='nearest')
        for y,x in [(4,4),(100,120),(h//4,w//4),(h//2-5,w//2-5)]:
            # An 18x18 window centred on the aligned cell; no ratio averaging.
            bounds=[(2*x-8,2*x+10,2*y-8,2*y+10)]
            np.testing.assert_allclose(c18[y,x],channel_means(a,bounds)[0],atol=1e-10,rtol=0)
            np.testing.assert_allclose(ch[y,x],channel_means(a,[(2*x,2*x+2,2*y,2*y+2)])[0],atol=0,rtol=0)
        side=DAY/'backgrounds'/Path(g['background']).with_suffix('.json')
        meta=json.loads(side.read_text(encoding='utf-8'))
        entry=dict(id=i,label=g['label'],file=g['background'],sha256=g['sha256'],
          capture_frames=meta['actual']['frames'],capture_type=meta['requested']['capture_type'],
          note=meta['conditions']['condition_note'],windows={})
        this={}
        for key,c in [('2',ch),('18',c18)]:
            v,r=radius(c);intensity=c.mean(-1)
            quantities={'intensity':intensity,'r':r,'x':v[...,0],'y':v[...,1]}
            images={}
            for kind,im in quantities.items():
                z=im.copy()
                if key=='18':z[~domain]=np.nan
                name=f'bg{i+1}_{key}_{kind}.png';write_map(z,OUT/name,kind);images[kind]=name
            entry['windows'][key]=dict(stats=stats(c,domain),images=images,
                histogram=np.histogram(r[domain&np.isfinite(r)],bins)[0].tolist())
            this[key]=quantities
        # Correctly labelled overlapping 2x2 windows. Each window still has
        # one pixel per orientation; the assignment changes with its origin.
        sliding=np.empty((h-1,w-1),float)
        for py in (0,1):
            for px in (0,1):
                sh=sliding[py::2,px::2].shape;planes=[]
                for pr,pc in ((1,1),(0,0),(0,1),(1,0)):
                    dy=(pr-py)%2;dx=(pc-px)%2
                    planes.append(a[py+dy:py+dy+2*sh[0]:2,px+dx:px+dx+2*sh[1]:2])
                sliding[py::2,px::2]=radius(np.stack(planes,axis=-1))[1]
        np.testing.assert_allclose(sliding[::2,::2],this['2']['r'],atol=0,rtol=0)
        for y,x in [(17,20),(19,21),(200,221),(201,220)]:
            np.testing.assert_allclose(sliding[y,x],np.linalg.norm(anisotropy(channel_means(a,[(x,x+2,y,y+2)])[0])),atol=1e-12)
        entry['sliding_2x2']=dict(shape=list(sliding.shape),
            r_quantiles=np.quantile(sliding[8:h-9,8:w-9],[.05,.25,.5,.75,.95]).tolist(),
            note='Overlapping cells; same sensor data reused. Aligned map equals sliding map[::2,::2] exactly. Compared over the common interior.')
        result['groups'].append(entry);maps.append(this)
    result['histogram_edges']=bins.tolist()
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    figures(result,maps)
    (OUT/'index.html').write_text(HTML.replace('__DATA__',json.dumps(result,ensure_ascii=True).replace('</','<\\/')),encoding='utf-8')
    for g in result['groups']:
        print(g['id'],g['label'].encode('ascii','replace').decode(),'median r',
              {k:round(v['stats']['r_quantiles'][2],4) for k,v in g['windows'].items()},
              'sliding',round(g['sliding_2x2']['r_quantiles'][2],4))


def figures(d,maps):
    fig,axes=plt.subplots(3,3,figsize=(14,11),layout='constrained')
    for i,(g,m) in enumerate(zip(d['groups'],maps)):
        for j,(key,kind,title) in enumerate([('2','intensity','Background intensity'),('2','r','r from each 2x2 cell'),('18','r','r after 18x18 channel means')]):
            ax=axes[i,j];im=m[key][kind].copy()
            if key=='18':im[:4]=np.nan;im[-4:]=np.nan;im[:,:4]=np.nan;im[:,-4:]=np.nan
            cm=plt.get_cmap('gray' if kind=='intensity' else 'viridis').copy();cm.set_bad('#b9b9b9')
            image=ax.imshow(im,cmap=cm,vmin=0,vmax=204.75 if kind=='intensity' else np.sqrt(2),
                extent=(-.5,2463.5,2055.5,-.5),interpolation='nearest')
            ax.set_title(title,fontsize=11);ax.set_xlabel('Sensor x');ax.set_ylabel(g['label']+'\nSensor y',fontsize=9)
            fig.colorbar(image,ax=ax,shrink=.76,label='Saved DN (20x display)' if kind=='intensity' else 'r (no model-limit clipping)')
    fig.suptitle('Saved-background anisotropy | identical colour scales across backgrounds and window sizes',fontsize=14)
    fig.savefig(OUT/'overview.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),sharex=True,sharey=True,layout='constrained')
    for ax,g in zip(axes,d['groups']):
        for key,color in [('2','#49759c'),('18','#bc4f58')]:
            s=g['windows'][key];counts=np.array(s['histogram']);ax.stairs(100*counts/counts.sum(),d['histogram_edges'],color=color,
                label=f'{key}x{key}: median {s["stats"]["r_quantiles"][2]:.3f}')
        ax.set(title=g['label'],xlabel='r of saved background',ylabel='Spatial windows per bin (%)',xlim=(0,np.sqrt(2)))
        ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Averaging channels before ratios reduces local variation; a nonzero background imbalance remains',fontsize=12)
    fig.savefig(OUT/'distributions.png',dpi=160);plt.close(fig)


HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Background anisotropy maps</title>
<style>body{margin:0;background:#f6f7f4;color:#253236;font:16px/1.5 system-ui}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px}p{max-width:1150px}select,button{font:inherit;padding:6px}label{margin-right:16px;display:inline-block;margin-bottom:10px}a{color:#087e8b}.maps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.maps figure{margin:0;min-width:0}.maps img{width:100%;image-rendering:pixelated}.maps figcaption{font-size:14px}#maps.zoom{grid-template-columns:1fr}#maps.zoom img{width:auto;max-width:none}#maps.zoom figure{overflow:auto;max-height:80vh}.small{font-size:14px;color:#536369}.stats{background:white;border:1px solid #d1d9d5;padding:14px;margin:14px 0}table{border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:8px 12px;border-bottom:1px solid #d1d9d5}.wide{max-width:100%;height:auto}.key{height:13px;max-width:300px;margin-top:6px}.rkey{background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)}.ykey{background:linear-gradient(90deg,#053061,#4393c3,#f7f7f7,#d6604d,#67001f)}@media(max-width:750px){.maps{grid-template-columns:1fr}}</style>
<main><h1>Anisotropy implied by the saved backgrounds</h1>
<p>Each 2×2 sensor cell contains one sample of each polarization channel. We calculate X=(I₀−I₉₀)/(I₀+I₉₀), Y=(I₄₅−I₁₃₅)/(I₄₅+I₁₃₅), then r=√(X²+Y²). <b>This is the saved BG itself, with no additional subtraction or spot selection.</b></p>
<p>The 18×18 option averages 9×9 samples <i>within each polarization channel first</i>, then calculates X, Y and r. It does not average cell r values. No r value is clipped at the rod model limit, and no theta is assigned to the background.</p>
<label>Background <select id="group"></select></label><label>Spatial window <select id="window"><option value="2">2×2 — one polarization cell</option><option value="18">18×18 — channel means first</option></select></label><label>Signed component <select id="component"><option value="y">Y</option><option value="x">X</option></select></label><button id="zoom">Show native cell resolution</button>
<div id="stats" class="stats"></div>
<div class="maps" id="maps"><figure><a id="intensityLink"><img id="intensity" alt="Mean intensity of background channels"></a><figcaption>Background intensity · fixed 20× display · black 0, white 204.75 saved DN.</figcaption></figure><figure><a id="rLink"><img id="r" alt="Background anisotropy radius"></a><figcaption>r · fixed scale 0 → √2 ≈ 1.414.<div class="key rkey"></div></figcaption></figure><figure><a id="signedLink"><img id="signed" alt="Signed background anisotropy component"></a><figcaption id="signedCaption"></figcaption><div class="key ykey"></div></figure></div>
<p class="small">Click any map for its full image. Each displayed pixel represents one aligned sensor cell (1232×1028 values). The 18×18 windows overlap at a two-sensor-pixel step; their outer four-cell border is grey. Grey also marks undefined ratios; summaries exclude the same border for both sizes. No intensity threshold is imposed.</p>
<p id="note" class="small"></p>
<p><b>Important limitation:</b> these are <b>per-pixel minimum projections</b> of 44–45 captured frames. The four channel values need not come from the same instant. The maps show the channel imbalance of the exact profiles used for subtraction, with spatial texture and pixel response included. They do not establish a population of background rods or prove that low-r sample detections are artifacts. Small denominators also make cell ratios sensitive to variation.</p>
<img src="distributions.png" class="wide" alt="Background radius distributions before and after spatial channel averaging">
<p>The remaining background r is mostly associated with negative Y. If residual BG contributes to a candidate, its <b>signed X/Y direction and intensity relative to the rod</b> matter; similarity in r alone is insufficient. Correct subtraction of this profile need not leave that imbalance in the measured rod.</p>
<details><summary>Does a 2n×2m image become n×m or (2n−1)×(2m−1)?</summary>
<p><b>Our GUI intensity detector uses disjoint 2×2 cells: n×m intensity values.</b> It then repeats each value into a 2×2 display/filtering block, restoring the original 2n×2m array size without adding information. DoG widths remain expressed in sensor pixels.</p>
<p>A sliding 2×2 window at one-pixel steps gives (2n−1)×(2m−1) values. Each raw pixel is reused in neighbouring windows, so the extra values are correlated. For anisotropy, the channel assignment must follow absolute pixel parity at every window origin. Our check does this; the aligned r map equals every second row and column of the sliding map exactly.</p>
<div id="comparison"></div><p>A denser sliding map can change local feature shapes and threshold-boundary candidates, but is not four times as much independent information. The table measures the BG distribution difference only; it is not a test of how many rods a different detector would recover.</p>
<p>These are separate from averaging ratios: for equal-brightness cells with X=+0.6 and −0.6, Y=0, the mean cell r is 0.6, but averaging the four channel intensities first gives r=0. This is why the 18×18 map can look very different.</p></details>
<p class="small"><a href="overview.png">All three backgrounds side by side</a> · <a href="results.json">Numbers and provenance</a> · <a href="../low_background_mask/index.html">Existing low-background spot selection</a></p></main>
<script>const DATA=__DATA__,$=id=>document.getElementById(id);
function update(){const g=DATA.groups[Number($('group').value)],w=$('window').value,z=g.windows[w],component=$('component').value,s=z.stats;
for(const [id,kind] of [['intensity','intensity'],['r','r'],['signed',component]]){$(id).src=z.images[kind];$(id+'Link').href=z.images[kind]}
$('signedCaption').textContent=component.toUpperCase()+' · blue −1, white 0, red +1.';
$('stats').textContent=`${g.label} · ${w}×${w} sensor pixels per estimate · median r=${s.r_quantiles[2].toFixed(3)} · central 90% r=${s.r_quantiles[0].toFixed(3)}–${s.r_quantiles[4].toFixed(3)} · mean X=${s.mean_x.toFixed(3)}, mean Y=${s.mean_y.toFixed(3)} · ${s.invalid} undefined interior ratios.`;
$('note').textContent=`${g.file} · minimum of ${g.capture_frames} frames. Recorded note: ${g.note}`;}
$('group').innerHTML=DATA.groups.map(g=>`<option value="${g.id}">${g.label}</option>`).join('');$('group').value='0';
$('comparison').innerHTML='<table><tr><th>Background</th><th>Aligned 2×2 median r</th><th>Sliding 2×2 median r</th><th>18×18 median r</th></tr>'+DATA.groups.map(g=>`<tr><td>${g.label}</td><td>${g.windows['2'].stats.r_quantiles[2].toFixed(4)}</td><td>${g.sliding_2x2.r_quantiles[2].toFixed(4)}</td><td>${g.windows['18'].stats.r_quantiles[2].toFixed(4)}</td></tr>`).join('')+'</table>';
$('group').onchange=$('window').onchange=$('component').onchange=update;$('zoom').onclick=()=>{const on=$('maps').classList.toggle('zoom');$('zoom').textContent=on?'Fit maps side by side':'Show native cell resolution'};update();</script></html>'''


if __name__=='__main__':build()
