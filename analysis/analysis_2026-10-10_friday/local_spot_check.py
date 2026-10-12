"""Local-contrast diagnostic for the user's suspect spots in the 15:47:30 stack.

This adds an exploratory quality flag to native k=8 candidates. It neither changes
the detector/camera code nor updates previous spot lists or theta distributions.
"""
from pathlib import Path
import csv
import importlib.util
import json
import hashlib
import numpy as np
import cv2
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from spot_detection_example import display20

B=Path(__file__).resolve().parent
REPO=B.parents[1]
OUT=B/'local_spot_check'
SOURCE=REPO/'data/hinges/polarcam/2026-10-09/frame_stack_20261009-154730.npy'
RINGS=((10,24),(12,30),(16,40))
TRIAL_CUTOFF=8.0  # Exploratory, not a calibrated false-positive or significance threshold.
SUSPECTS=(15,35,37)
EXAMPLES=(15,35,37,38,49)


def filtering(image, detector):
    e=detector.EDGE_EXCLUDE_PX
    work=image[e:-e,e:-e]
    small=cv2.GaussianBlur(work,(0,0),sigmaX=detector.DOG_SIGMA_SMALL,borderType=cv2.BORDER_REPLICATE)
    dog=small-cv2.GaussianBlur(work,(0,0),sigmaX=detector.DOG_SIGMA_LARGE,borderType=cv2.BORDER_REPLICATE)
    mu,sd=detector._finite_stats(dog)
    threshold=mu+8*sd
    _,labels,stats,centres=cv2.connectedComponentsWithStats((dog>threshold).astype(np.uint8),connectivity=8)
    keep=(stats[1:,cv2.CC_STAT_AREA]>=detector.DOG_MIN_AREA)&(stats[1:,cv2.CC_STAT_AREA]<=detector.DOG_MAX_AREA)
    label_ids=np.flatnonzero(keep)+1
    return dict(image=image,small=small,dog=dog,mu=mu,sd=sd,threshold=threshold,
                labels=labels,stats=stats,centres=centres[label_ids]+e,label_ids=label_ids,edge=e)


def local_metrics(f, points, j, ring=(12,30)):
    x,y=points[j];ix,iy=int(round(x)),int(round(y));radius=ring[1]+2;e=f['edge']
    h,w=f['image'].shape
    yy,xx=np.mgrid[max(e,iy-radius):min(h-e,iy+radius+1),max(e,ix-radius):min(w-e,ix+radius+1)]
    radial=np.hypot(xx-x,yy-y)
    mask=(radial>=ring[0])&(radial<=ring[1])
    for k,(cx,cy) in enumerate(points):
        if k!=j and np.hypot(cx-x,cy-y)<ring[1]+9:
            mask &= np.hypot(xx-cx,yy-cy)>8
    mask &= (xx%2==0)&(yy%2==0)  # One sample per repeated polarization cell.
    local_dog=f['dog'][yy-e,xx-e][mask]
    local_image=f['image'][yy,xx][mask]
    if len(local_dog)<40:raise ValueError('Too few neighbouring samples')
    median=float(np.median(local_dog));spread=float(1.4826*np.median(np.abs(local_dog-median)))
    bg=float(np.median(local_image));raw_spread=float(1.4826*np.median(np.abs(local_image-bg)))
    label_id=f['label_ids'][j]
    # Restrict the lookup to its bounding box; identical pixels and score, less
    # work when applying this diagnostic to stacks with many candidates.
    bx,by,bw,bh,_=f['stats'][label_id]
    labels=f['labels'][by:by+bh,bx:bx+bw]
    peak=float(f['dog'][by:by+bh,bx:bx+bw][labels==label_id].max())
    core=radial<=6
    smooth_peak=float(f['small'][yy-e,xx-e][core].max())
    return dict(local_score=(peak-median)/spread,local_dog_median=median,local_dog_MAD_scale=spread,
                local_samples=len(local_dog),dog_peak=peak,global_peak_z=(peak-f['mu'])/f['sd'],
                local_intensity_median=bg,local_intensity_MAD_scale=raw_spread,
                smoothed_intensity_peak=smooth_peak,raw_contrast_score=(smooth_peak-bg)/raw_spread,
                component_area_px=int(f['stats'][label_id,cv2.CC_STAT_AREA]))


def run():
    OUT.mkdir(exist_ok=True)
    detector_path=REPO/'src/polarcam/live/Detection_alg_offline.py'
    spec=importlib.util.spec_from_file_location('native_detector',detector_path)
    detector=importlib.util.module_from_spec(spec);spec.loader.exec_module(detector)
    metadata=json.loads(SOURCE.with_suffix('.json').read_text(encoding='utf-8-sig'))
    raw=np.load(SOURCE,mmap_mode='r')
    n=len(raw)-int(metadata['actual']['phase_marker_appended'])
    assert n==metadata['actual']['frames']==150
    block_means=[]
    for start in (0,50,100):
        mean=np.zeros(raw.shape[1:],dtype=float)
        for frame in raw[start:start+50]:mean+=frame
        block_means.append(mean/50)
    mean=sum(block_means)/3
    f=filtering(detector.intensity_detection_map(mean),detector)
    points=f['centres']
    native=np.array(detector.find_spot_centers_dog(f['image'][10:-10,10:-10]))+10
    np.testing.assert_allclose(points,native,rtol=0,atol=1e-12)
    old=list(csv.DictReader((B/'first_hinges_theta/spots.csv').open(encoding='utf-8')))
    old=[r for r in old if r['stack']=='154730']
    np.testing.assert_allclose(points,[[float(r['x_sensor_px']),float(r['y_sensor_px'])] for r in old],atol=1e-10)
    block_fields=[filtering(detector.intensity_detection_map(m),detector) for m in block_means]
    persistence=np.stack([cKDTree(b['centres']).query(points)[0]<=3 for b in block_fields])
    rows=[]
    for j,(x,y) in enumerate(points):
        metrics=local_metrics(f,points,j)
        scores={f'local_score_{lo}_{hi}px':local_metrics(f,points,j,(lo,hi))['local_score'] for lo,hi in RINGS}
        row=dict(spot_id=j+1,x=float(x),y=float(y),**metrics,**scores,
                 native_detected_blocks=int(persistence[:,j].sum()),
                 first50_detected=bool(persistence[0,j]),middle50_detected=bool(persistence[1,j]),last50_detected=bool(persistence[2,j]),
                 trial_local_pass=bool(metrics['local_score']>=TRIAL_CUTOFF),user_questioned=j+1 in SUSPECTS)
        rows.append(row)
    passed=sum(r['trial_local_pass'] for r in rows)
    summary=dict(source=SOURCE.name,native_count=len(rows),trial_pass=passed,trial_flag=len(rows)-passed,
        trial_flag_ids=[r['spot_id'] for r in rows if not r['trial_local_pass']],
        trial_cutoff=TRIAL_CUTOFF,ring_pixels=[12,30],other_core_mask_radius_px=8,
        statistic='(maximum DoG in native component - neighbouring median DoG) / (1.4826 * neighbouring MAD); one sample per 2x2 cell',
        interpretation='Local spatial-contrast score, not independent-noise sigma, p-value or calibrated SNR. Trial threshold only.',
        temporal_check='Native k=8 redetection in three disjoint 50-frame means; within 3 pixels of full-stack centre. Persistent background can also pass.',
        exclusions_applied=False,background='Use saved background-subtracted data; no extra image or channel subtraction',
        ring_sensitivity={f'{lo}-{hi}':sum(r[f'local_score_{lo}_{hi}px']>=TRIAL_CUTOFF for r in rows) for lo,hi in RINGS},
        examples=[r for r in rows if r['spot_id'] in EXAMPLES],
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_code_sha256=hashlib.sha256(detector_path.read_bytes()).hexdigest(),
        references=['https://photutils.readthedocs.io/en/stable/user_guide/background.html',
                    'https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html'])
    with (OUT/'scores.csv').open('w',newline='',encoding='utf-8') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    figures(f,block_fields,points,rows)
    write_html(rows,summary)
    print(json.dumps(summary))


def figures(f,blocks,points,rows):
    plt.rcParams.update({'font.size':10,'figure.facecolor':'#f7f7f4','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(3,4,figsize=(13,9.2),layout='constrained',gridspec_kw={'height_ratios':[1,1,.8]})
    choices=(15,35,38,49)
    titles=('Your first region · #15','Your crowded region · #35 / #37','Nearby stronger candidate · #38','Isolated reference · #49')
    for col,(spot_id,title) in enumerate(zip(choices,titles)):
        j=spot_id-1;x,y=points[j];cx,cy=int(round(x)),int(round(y));cy+=10 if spot_id==35 else 0;half=44
        yy,xx=np.mgrid[cy-half:cy+half,cx-half:cx+half]
        for row_idx in (0,1):
            ax=axs[row_idx,col]
            if row_idx==0:
                im=display20(f['image'][yy,xx]);ax.imshow(im,cmap='gray',vmin=0,vmax=255,extent=(cx-half-.5,cx+half-.5,cy+half-.5,cy-half-.5),interpolation='nearest')
                if col==0:ax.set_ylabel('150-frame mean · 20× display')
                ax.set_title(title,fontsize=10)
            else:
                dog=f['dog'][yy-10,xx-10];im=ax.imshow(dog,cmap='RdBu_r',vmin=-25,vmax=25,extent=(cx-half-.5,cx+half-.5,cy+half-.5,cy-half-.5),interpolation='nearest')
                if col==0:ax.set_ylabel('DoG image · fixed −25 to +25 DN')
                ax.set_title('Dashed ring: local reference (12–30 px)',fontsize=9)
                for radius in (12,30):ax.add_patch(Circle((x,y),radius,fill=False,color='#242e40',linestyle='--',lw=.8))
            for k,(px,py) in enumerate(points):
                if cx-half<=px<cx+half and cy-half<=py<cy+half:
                    color='#ef7b47' if not rows[k]['trial_local_pass'] else '#24c9df'
                    ax.add_patch(Circle((px,py),7,fill=False,color=color,lw=1))
                    ax.text(px+8,py+2,str(k+1),color=color,fontsize=9)
            ax.tick_params(labelsize=8)
        ax=axs[2,col]
        picks=[35,37] if spot_id==35 else [spot_id]
        for sid in picks:
            r=rows[sid-1]
            scores=[r[f'local_score_{lo}_{hi}px'] for lo,hi in RINGS]
            ax.plot(range(3),scores,'o-',label=f'#{sid}',lw=1.3,ms=4)
        ax.axhline(TRIAL_CUTOFF,color='#a95226',ls='--',lw=1,label='Trial cutoff 8')
        ax.set(xticks=range(3),xticklabels=['10–24','12–30','16–40'],xlabel='Neighbour-ring radii (pixels)',ylim=(0,40))
        if col==0:ax.set_ylabel('Local contrast score')
        ax.legend(fontsize=8,frameon=False)
    fig.suptitle('Local contrast check · native candidate list retained · orange = below the trial local threshold',fontsize=13)
    fig.savefig(OUT/'local_contrast.png',dpi=165);plt.close(fig)

    fig,axs=plt.subplots(3,4,figsize=(12,8),layout='constrained')
    for col,spot_id in enumerate((15,35,37,49)):
        x,y=points[spot_id-1];cx,cy=int(round(x)),int(round(y));half=20
        for k,field in enumerate(blocks):
            ax=axs[k,col]
            ax.imshow(display20(field['image'][cy-half:cy+half,cx-half:cx+half]),cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            detected=rows[spot_id-1][('first50_detected','middle50_detected','last50_detected')[k]]
            ax.add_patch(Circle((x-(cx-half),y-(cy-half)),7,fill=False,color='#24c9df' if detected else '#ef7b47',lw=1,ls='-' if detected else '--'))
            ax.set_title(f'#{spot_id} · '+('native pass' if detected else 'native fail'),fontsize=10)
            if col==0:ax.set_ylabel(f'Frames {k*50+1}–{k*50+50}')
            ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Three disjoint 50-frame means · same 20× display · persistence alone does not identify a rod',fontsize=12)
    fig.savefig(OUT/'persistence.png',dpi=150);plt.close(fig)


def write_html(rows,summary):
    selected=[rows[i-1] for i in EXAMPLES]
    table=''.join('<tr>'+''.join(f'<td>{x}</td>' for x in (r['spot_id'],f"({r['x']:.1f}, {r['y']:.1f})",r['component_area_px'],f"{r['global_peak_z']:.1f}",f"{r['local_score']:.1f}",f"{r['native_detected_blocks']}/3",'Pass' if r['trial_local_pass'] else 'Flag'))+'</tr>' for r in selected)
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Questioned spots · local contrast</title>
<style>body{{margin:0;background:#f7f7f4;color:#242e30;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1250px;margin:auto;padding:24px}}h1{{font-size:28px}}h2{{font-size:21px}}p{{max-width:1050px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}td,th{{padding:8px 12px;border-bottom:1px solid #ccd5cf;text-align:left;white-space:nowrap}}.scroll{{overflow:auto}}a{{color:#176b73}}.note{{font-size:14px;color:#52656a}}</style><main>
<h1>These are weak candidates, not established rods</h1>
<p><b>Your concern is supported by the local measurements.</b> The three questioned detections barely pass the native blob-size limit and are weaker against their own surroundings than the isolated reference. The local check below flags all three.</p>
<div class="scroll"><table><thead><tr><th>Spot ID</th><th>Sensor centre</th><th>Blob pixels</th><th>Global peak score</th><th>Local contrast score</th><th>50-frame blocks detected</th><th>Local ≥8 trial</th></tr></thead><tbody>{table}</tbody></table></div>
<p class="note">Global = peak DoG relative to the entire cropped frame's mean/SD. Local = peak DoG relative to the neighbouring median/(1.4826 × MAD). Different reference distributions: these are contrast scores, not comparable calibrated probabilities or independent-noise SNRs. Native blob area minimum is 10.</p>
<img src="local_contrast.png" alt="User-questioned regions compared with local contrast and clearer candidates">
<h2>What changes in the proposed check?</h2>
<p>The native detector already removes a broad local baseline: Gaussian blur at σ=1 pixel minus blur at σ=6 pixels. Its threshold then uses one mean and SD for the entire image. A small bump on a textured patch can therefore pass without being prominent within that patch.</p>
<p>Keep that detector as a candidate finder. For each candidate, estimate the median and spatial variation of the <i>filtered</i> image in a surrounding ring, excluding the candidate core and an 8-pixel radius around other detected centres. The default ring is 12–30 raw pixels, sampled once per 2 × 2 polarization cell. Compare the candidate's peak with this local reference.</p>
<p>A trial local-score cutoff of 8 flags <b>{summary['trial_flag']}/{summary['native_count']}</b> native candidates on this stack; <b>{summary['trial_pass']}</b> pass. This cutoff is exploratory. It is a quality flag added to the same candidate list, not a newly tuned detector or a validated classification of rods. Neighbour-ring sensitivity is shown above.</p>
<h2>Reasons to keep them available for inspection</h2>
<p>#15 passes native detection in all three 50-frame blocks. #35 and #37 pass in two of three. There is reproducible structure at these locations, so a one-frame fluctuation is not the whole explanation. A fixed background residual, interference pattern, cluster structure or faint real object could also persist. This does not establish that they are single nanorods.</p>
<img src="persistence.png" alt="Candidate persistence across three nonoverlapping frame blocks">
<p><b>Recommendation:</b> flag #15, #35 and #37 as ambiguous and withhold them from a trusted single-rod θ distribution until reviewed. Use local contrast plus isolation/shape checks for future selection. A local mean alone will not distinguish a cluster from one rod; a bright unresolved cluster can still have high contrast.</p>
<p>The previously reported 255 mathematically valid θ values were a preliminary candidate distribution. Passing the inversion's range check does not validate the physical object or the background level in its aperture. This diagnostic has not altered that distribution or the original data.</p>
<details><summary>Sources and files</summary><p>Data: frame_stack_20261009-154730.npy, first 150 measurement frames; appended marker excluded. Uses the native camera detector directly offline. Native detector unchanged. Same saved background-subtracted data, no additional subtraction applied to θ channels.</p>
<p>General methodological support: <a href="https://photutils.readthedocs.io/en/stable/user_guide/background.html">Photutils background estimation</a> discusses spatial background/noise estimation and masking sources; <a href="https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html">DAOStarFinder</a> also separates detection from sharpness and roundness cuts. Photutils is not used by this diagnostic, and these references do not validate this camera-specific cutoff.</p></details>
<p><a href="scores.csv">Scores for all 57 candidates</a> · <a href="results.json">Settings and selected measurements</a> · <a href="../spot_detection_example/index.html">Original detector example</a> · <a href="../first_hinges_theta/index.html">Unchanged preliminary θ results</a></p></main></html>'''
    (OUT/'index.html').write_text(report,encoding='utf-8')


if __name__=='__main__':
    run()
