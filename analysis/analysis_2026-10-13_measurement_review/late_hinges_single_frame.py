"""Late hinge wide-field background trial and unsmoothed spot-recording review.

Acquisition files are read only. No temporal mean is used for detection or angles.
Backgrounds use the GUI per-pixel minimum, with target stacks held out by default.
"""
from pathlib import Path
import argparse, hashlib, json, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from polarcam import optics
from polarcam.live.intensity_spots import channel_means, anisotropy
from polarcam.live.Detection_alg_offline import intensity_detection_map
from control_recovery_check import evaluate, subtraction_function, clean, sha, sig

HERE=Path(__file__).resolve().parent
OUT=HERE/'late_hinges_single_frame'
ROOT=HERE.parents[1]/'data/hinges/polarcam'
DAY=ROOT/'2026-10-12'
MODEL=optics.theta_model();ABC=[MODEL[k] for k in ['A','B','C']]
SCALE=255/4095
FRAME_IDS=[0,74,149]
VARIANTS={'saved':'Saved narrow BG only','all_min':'Minimum of all 9 stacks',
          'held_out':'Minimum of other 8 stacks','reference':'Recorded wide-field minimum'}
COL={'saved':'#909ba0','all_min':'#7a5b9a','held_out':'#176d91','reference':'#b46629'}
EMAILS=[
  {
    "date": "7 October 2026",
    "title": "Richard: Methods paper",
    "url": "https://outlook.office365.com/owa/?ItemID=AQMkAGE0MGFmNDZhLTc2N2UtNDYxOS1iNDY3LWVkODFhZDI3NTdmYwBGAAADX334HLyVcE2oj%2BJD8T7Z%2FwcAHZVNySZh8UuKJN8jBbJ8bgAAAgEMAAAAHZVNySZh8UuKJN8jBbJ8bgAGrFszHAAAAA%3D%3D&exvsurl=1&viewmodel=ReadMessageItem",
    "interpretation": "Richard explicitly calls this a fully hybridised, locked test strand. He asks both for the correct-angle fraction and to quantify how well locked selected well-assembled hinges are. Very little motion is therefore compatible with the intended sample; visible opening and closing is not a requirement for every good hinge."
  },
  {
    "date": "13 October 2026",
    "title": "Richard: for Zev conversation",
    "url": "https://outlook.office365.com/owa/?ItemID=AQMkAGE0MGFmNDZhLTc2N2UtNDYxOS1iNDY3LWVkODFhZDI3NTdmYwBGAAADX334HLyVcE2oj%2BJD8T7Z%2FwcAHZVNySZh8UuKJN8jBbJ8bgAAAgEMAAAAHZVNySZh8UuKJN8jBbJ8bgAGsCpoiAAAAA%3D%3D&exvsurl=1&viewmodel=ReadMessageItem",
    "interpretation": "In the quoted 12:25 message, Richard describes the 12 October experiment as the original theta hinge with a fully hybridised, hopefully un-nicked test strand. His statement retains uncertainty about nick status. He distinguishes this from the proposed later experiment with freely moving ssDNA test strands."
  },
  {
    "date": "25 September 2026",
    "title": "Richard: DNA hinge data",
    "url": "https://outlook.office365.com/owa/?ItemID=AQMkAGE0MGFmNDZhLTc2N2UtNDYxOS1iNDY3LWVkODFhZDI3NTdmYwBGAAADX334HLyVcE2oj%2BJD8T7Z%2FwcAHZVNySZh8UuKJN8jBbJ8bgAAAgEMAAAAHZVNySZh8UuKJN8jBbJ8bgAGpGzE2wAAAA%3D%3D&exvsurl=1&viewmodel=ReadMessageItem",
    "interpretation": "Rod orientation is not automatically the hinge opening. Richard explains an unknown rod-to-arm attachment offset and possible tilted hinge axes. The simple opening = 90 degrees minus theta requires the designed rod alignment and a correctly surface-mounted hinge. A stationary point alone provides no rotation arc with which to determine the hinge axis."
  }
]

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(name,d):(OUT/name).write_text(json.dumps(clean(d),ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf8')
def floats(rows,k):return np.array([s.get(k) if s.get(k) is not None else np.nan for s in rows],float)
def angles(ch):
    v=anisotropy(ch);r=np.linalg.norm(v,axis=-1)
    return v,r,np.degrees(optics.theta_from_radius(r,*ABC))
def entries():
    return [r for r in read(HERE/'manifest.json')['records'] if r['type']=='Frame stack' and r['stamp']>='205000' and r['first_frame_mean']>50]
def oldbg(m):
    h=m['background']['background_profile_sha256'];p=DAY/'backgrounds/used'/f'{h}.npy'
    assert sha(p)==h
    return np.load(p).astype(np.float32)
def peak(m,p):
    meta=m['fetch_frames_output']['raw_peak'];file=p.parent/meta['path']
    assert sha(file)==meta['sha256'] and meta['frames']==150
    return np.load(file)
def stats(rows):
    k=[s for s in rows if s['selection']=='kept'];th=floats(k,'theta_deg');rr=floats(k,'r');v=th[np.isfinite(th)]
    return dict(n=len(k),valid=len(v),above=int(np.sum(rr>MODEL['r_max'])),
                median_theta=float(np.median(v)) if len(v) else None,
                median_r=float(np.nanmedian(rr)) if len(rr) else None,
                near30_percent=100*float(np.mean((v>=28)&(v<34))) if len(v) else None,
                saturated_candidates=sum(s['saturated'] for s in rows),
                exclusions={k:sum(s['selection']==k for s in rows) for k in ['kept','saturated','overlap','region','invalid']})

def backgrounds():
    rows=entries();assert len(rows)==9
    mins=[];proof=[]
    for i,r in enumerate(rows):
        p=ROOT/r['path'];m=read(p.with_suffix('.json'));a=np.load(p,mmap_mode='r');b=oldbg(m)
        n=m['actual']['frames'];assert n==150 and len(a)==n+1 and m['actual']['phase_marker_appended']
        low=np.array(a[0],dtype=np.uint16)
        for f in a[1:n]:np.minimum(low,f,out=low)
        mins.append(low.astype(np.float32)+b)
        proof.append(dict(name=r['name'],stamp=r['stamp'],source_signature=sig(p),metadata_sha256=sha(p.with_suffix('.json')),
             bg_hash=r['bg_hash'],old_bg_mean=float(b.mean()),first_saved_mean=float(a[0].mean()),
             first_zero_fraction=float(np.mean(a[0]==0)),min_zero_fraction=float(np.mean(low==0)),
             exposure_us=m['actual']['timing']['exposure_us'],fps=m['actual']['fps'],gains=m['actual']['gains'],note=r['note']))
        print('Background input',i+1,r['stamp'],float(mins[-1].mean()),flush=True)
    mins=np.stack(mins)
    ref=next(r for r in read(HERE/'manifest.json')['records'] if r['stamp']=='210358')
    refpath=ROOT/ref['path'];reference=np.load(refpath).astype(np.float32);rm=read(refpath.with_suffix('.json'))
    movie=np.load(refpath.parent/rm['source_stack_file'],mmap_mode='r');n=rm['actual']['frames']
    assert np.array_equal(np.min(movie[:n],axis=0),reference)
    np.savez_compressed(OUT/'backgrounds.npz',per_stack_min=mins.astype('uint16'),reference=reference.astype('uint16'))
    all_min=mins.min(0)
    summaries=dict(all_min_mean=float(all_min.mean()),reference_mean=float(reference.mean()),
         all_minus_reference_quantiles=np.percentile(all_min-reference,[1,10,50,90,99]).tolist(),
         mean_absolute_difference=float(np.abs(all_min-reference).mean()),
         spatial_correlation=float(np.corrcoef(all_min.ravel(),reference.ravel())[0,1]))
    write('background_info.json',dict(records=proof,reference=ref['name'],reference_sha256=sha(refpath),reference_frames=n,
         summary=summaries,method='GUI per-pixel minimum after restoring each recorded narrow background. No phase marker. All 1350 frames; held-out uses 1200 frames from the other eight fields.',
         limits='Saved zeros have unknown original raw value from 0 to the old minimum. Restoration uses the upper bound. Common or stationary sample features and optical drift can remain in the estimate; a minimum over 1200 frames has a different noise bias from the 44-frame independent capture.',
         sample_identity='Notes and operator confirmation identify DNA after 20:50, despite stale Gold-only control role dropdown. Operator says mostly new sample positions for each number.'))

def fields():
    subtract,_=subtraction_function()
    b=np.load(OUT/'backgrounds.npz');mins=b['per_stack_min'].astype('float32');reference=b['reference'].astype('float32');global_min=mins.min(0)
    out=dict(complete=False,model=MODEL,records=[],frames=[1,75,150],variants=VARIANTS,
       method='Independent detection and extraction on each specified single frame; GUI repeated 2x2 map, DoG sigma1/6, whole-image mean+2SD, components >=10 with no upper size limit; original corner ellipse and strongest-first 18x18 overlap exclusion. Raw saturation >=4094 anywhere across 150 frames in aperture OR bbox still excludes the candidate. No r-limit clipping, no survival or low-BG mask.',
       sampling='Later fields were visited during targeted spot selection. Not an unbiased survey or independent preparations. Do not pool three snapshots as independent rods.',
       builder_sha256=sha(Path(__file__)))
    roi=None
    for i,r in enumerate(entries()):
        p=ROOT/r['path'];m=read(p.with_suffix('.json'));a=np.load(p,mmap_mode='r');old=oldbg(m);pk=peak(m,p)
        held=mins[np.arange(9)!=i].min(0)
        rec=dict(index=i+1,name=r['name'],stamp=r['stamp'],note=r['note'],snapshots={},source_signature=sig(p),bg_hash=r['bg_hash'],raw_peak_sha256=m['fetch_frames_output']['raw_peak']['sha256'],held_bg_mean=float(held.mean()))
        for fi in FRAME_IDS:
            raw_proxy=np.asarray(a[fi],dtype='float32')+old
            images={'saved':np.asarray(a[fi]),'all_min':subtract(raw_proxy.astype('uint16'),global_min,roi,None),
                    'held_out':subtract(raw_proxy.astype('uint16'),held,roi,None),
                    'reference':subtract(raw_proxy.astype('uint16'),reference,roi,None)}
            snap={}
            for key,im in images.items():
                ev=evaluate(im,im,pk,f'late{i+1}f{fi+1}{key}')
                ev['summary']=stats(ev['spots'])
                if fi==0:
                    filename=f'field_{i+1}_{key}.webp';display=intensity_detection_map(im)
                    Image.fromarray(np.rint(np.clip(display*5/4095,0,1)*255).astype('uint8')).save(OUT/filename,lossless=True)
                    ev['image']=filename
                # Store enough to audit detection and plot accepted/excluded overlays.
                snap[key]=ev
                print('Field',i+1,'frame',fi+1,key,ev['summary'],flush=True)
            rec['snapshots'][str(fi+1)]=snap
        out['records'].append(rec);write('fields.json',out)
    out['complete']=True;write('fields.json',out)

def survey_reference():
    """Single-frame redetection in the original 16 DNA and 14 gold survey stacks."""
    from hinges_percentile_theta import background
    sources={'Early DNA':read(HERE/'background_overlap_check/hinges95_theta.json'),
             'Gold survey':read(HERE/'control_recovery_check/gold_results.json')}
    extras={}
    out=dict(complete=False,groups={},method='Frame 1 only, independently detected. Minimum/minimum branch uses recorded minimum for both detection and extraction; p95/minimum branch preserves previous background choices but replaces the temporal mean by frame 1. Same exclusions as later wide-field trial.')
    for group,d in sources.items():
        rr=[]
        for r in d['records']:
            p=DAY/'frames'/r['name'];m=read(p.with_suffix('.json'));a=np.load(p,mmap_mode='r');frame=np.array(a[0]);pk=peak(m,p)
            h=m['background']['background_profile_sha256']
            if h not in extras:extras[h]=background(h)[0]
            im95=np.rint(np.maximum(frame.astype('float32')-extras[h],0)).astype('uint16')
            modes={}
            for key,im in [('minimum',frame),('p95',im95)]:
                ev=evaluate(frame,im,pk,f'{group}{len(rr)+1}{key}');ev['summary']=stats(ev['spots']);modes[key]=ev
            rr.append(dict(name=r['name'],index=len(rr)+1,modes=modes,source_signature=sig(p)))
            print('Single-frame survey',group,len(rr),[(k,v['summary']['n']) for k,v in modes.items()],flush=True)
        out['groups'][group]=rr;write('survey_reference.json',out)
    out['complete']=True;write('survey_reference.json',out)

def spots():
    from first_pass_review import extract
    manifest=read(HERE/'manifest.json');prev=read(HERE/'first_pass_review/results.json')
    selected=[r for r in prev['records'] if r['type']=='Spot recording']
    records=[]; traces={}
    for old in selected:
        rec=next(r for r in manifest['records'] if r['stamp']==old['stamp'])
        r,ch,_=extract(rec)  # validates cache provenance, ROI parity and raw saturation
        xy,rr,theta=angles(ch);ok=np.isfinite(theta)
        q=np.percentile(theta[ok],[2,50,98]) if ok.any() else [np.nan]*3
        p=ROOT/r['path'];meta=read(p.with_suffix('.json'));a=np.load(p,mmap_mode='r')
        x,y,size=r['bounds'];ox,oy=meta['roi']['x'],meta['roi']['y'];l,t=x-ox,y-oy
        crop=np.asarray(a[0,t:t+size,l:l+size],dtype=float)
        bg=oldbg(meta)[y:y+size,x:x+size]
        # Peak and border are descriptive on frame 1; not used to select records.
        edge=np.ones(crop.shape,bool);edge[3:-3,3:-3]=False
        peakcell=float(np.max((crop[0::2,0::2]+crop[1::2,0::2]+crop[0::2,1::2]+crop[1::2,1::2])/4))
        r.update(label=old['label'],group=old['group'],index=old['index'],single_frame=dict(
            theta_p2_p50_p98=q.tolist(),valid_fraction=float(ok.mean()),above_fraction=float(np.mean(rr>MODEL['r_max'])),
            r_median=float(np.nanmedian(rr)),xy_sd=np.nanstd(xy,axis=0).tolist(),xy_p2_p98=np.percentile(xy,[2,98],axis=0).tolist(),
            intensity_cv=float(ch.sum(1).std()/ch.sum(1).mean()),
            first_peak_8bit=peakcell*SCALE,first_border_median_8bit=float(np.median(crop[edge]))*SCALE,
            bg_aperture_mean_8bit=float(bg.mean())*SCALE,bg_aperture_max_8bit=float(bg.max())*SCALE,
            first_intensity_mean_8bit=float(crop.mean())*SCALE))
        records.append(r);traces[r['stamp']]=(xy,rr,theta,ch,crop,bg)
        print('Spot',r['label'],r['stamp'],'theta quantiles',q.tolist(),'xy SD',r['single_frame']['xy_sd'],flush=True)
        # One image per recording: pixels, per-frame density, full-duration X/Y, and theta.
        fig=plt.figure(figsize=(13.2,6.3),layout='constrained');gs=fig.add_gridspec(2,6,height_ratios=[1,1.25])
        for j,(im,title) in enumerate([(crop+bg,'Frame 1 + recorded BG'),(bg,'Recorded narrow BG'),(crop,'Frame 1 saved')]):
            ax=fig.add_subplot(gs[0,j]);ax.imshow(im*SCALE,cmap='gray',vmin=0,vmax=max(1,float((crop+bg).max())*SCALE),interpolation='nearest');ax.set_title(title,fontsize=9);ax.set_axis_off()
        ax=fig.add_subplot(gs[0,3:]);ax.hist(theta[ok],bins=np.arange(0,91,1),weights=np.full(ok.sum(),100/len(theta)),histtype='step',color='#176d91');ax.set(xlim=(0,90),xlabel='Per-frame θ (°)',ylabel='% of all frames / 1°',title=f'Median {q[1]:.1f}°; 2–98% {q[0]:.1f}–{q[2]:.1f}°')
        ax=fig.add_subplot(gs[1,:2]);ax.hexbin(xy[:,0],xy[:,1],gridsize=65,mincnt=1,bins='log',cmap='viridis');ax.set(xlabel='X',ylabel='Y',title='Every frame; density on local axes',aspect='equal')
        ax=fig.add_subplot(gs[1,2:]);time=np.arange(len(xy))/r['metrics']['fps'];ax.plot(time,xy[:,0],lw=.35,alpha=.75,label='X');ax.plot(time,xy[:,1],lw=.35,alpha=.75,label='Y');ax.set(xlabel='Time (s)',ylabel='Per-frame anisotropy',title='All frames; no temporal smoothing');ax.legend(frameon=False)
        fig.suptitle(f'{r["label"]} · {r["time"]} · {r["metrics"]["duration_s"]:.1f} s · raw peak {r["raw_peak"]*SCALE:.1f}/255'+(' · SATURATED: diagnostic only' if r['saturated'] else ''),fontsize=13)
        fig.savefig(OUT/f'spot_{r["stamp"]}.png',dpi=145);plt.close(fig)
    write('spot_results.json',dict(records=records,model=MODEL,method='Original 20x20 selected ROI and recorded narrow minimum subtraction; all single-frame channel ratios, without temporal smoothing. Entire records flagged if the selected aperture saturated at any time. Angle percentiles are descriptive fluctuations plus measurement noise, not intrinsic hinge compliance.'))

def hist(ax,values,edges,label,color,den=None,ls='-'):
    v=np.asarray(values,float);good=v[np.isfinite(v)];den=len(good) if den is None else den
    if den:ax.stairs(np.histogram(good,edges)[0]*100/den,edges,label=label,color=color,ls=ls,lw=1.4)
def kept(ev):return [s for s in ev['spots'] if s['selection']=='kept']

def quiet_apertures():
    """Small fixed-centre aperture check; three separate frames, no averaging."""
    records=read(OUT/'spot_results.json')['records'];results=[]
    fig,axes=plt.subplots(1,4,figsize=(12,3.5),layout='constrained',sharey=True)
    for ax,r in zip(axes,[r for r in records if r['stamp'] in ['210823','211300','212035','213715']]):
        p=ROOT/r['path'];m=read(p.with_suffix('.json'));a=np.load(p,mmap_mode='r');n=m['actual']['frames']
        ox,oy=m['roi']['x'],m['roi']['y'];x,y,size=r['bounds'];l,t=x-ox,y-oy
        bg=oldbg(m)[y:y+size,x:x+size];chbg=channel_means(bg,[(0,size,0,size)],origin=(x,y))[0]
        trace=np.load(HERE/'first_pass_review'/f'trace_{r["stamp"]}.npz')['channels']
        restored=angles(trace+chbg)[2];saved=angles(trace)[2]
        q=dict(label=r['label'],stamp=r['stamp'],frames=[],median_theta_saved=float(np.nanmedian(saved)),
               median_theta_add_recorded_bg=float(np.nanmedian(restored)),
               bg_fraction_of_first_saved_mean=r['single_frame']['bg_aperture_mean_8bit']/r['single_frame']['first_intensity_mean_8bit'])
        for fi,color in [(0,'#176d91'),(n//2,'#b46629'),(n-1,'#7a5b9a')]:
            crop=np.asarray(a[fi,t:t+size,l:l+size]);sizes=[8,12,16,20];vals=[]
            for s in sizes:
                o=(size-s)//2;ch=channel_means(crop,[(o,o+s,o,o+s)],origin=(x,y))
                vals.append(float(angles(ch)[2][0]))
            q['frames'].append(dict(frame=fi+1,sizes=sizes,theta_deg=vals));ax.plot(sizes,vals,'o-',label=f'Frame {fi+1}',color=color,ms=3)
        ax.set(title=r['label'],xlabel='Square aperture width (sensor px)',xticks=[8,12,16,20],ylim=(0,90));ax.legend(fontsize=7,frameon=False)
        results.append(q)
    axes[0].set_ylabel('Single-frame apparent θ (°)')
    fig.suptitle('Quiet targets: sensitivity to extraction aperture at the original centre\nRecorded narrow background only; no extra local subtraction',fontsize=12)
    fig.savefig(OUT/'quiet_apertures.png',dpi=155);plt.close(fig)
    write('quiet_apertures.json',dict(records=results,limits='Smaller apertures change PSF sampling and analyzer balance as well as background/neighbor contributions. This is sensitivity, not a calibrated correction or a rule to choose an angle.'))

def render():
    d=read(OUT/'fields.json');b=read(OUT/'background_info.json');refs=read(OUT/'survey_reference.json');sp=read(OUT/'spot_results.json')
    assert d['complete'] and refs['complete']
    quiet_apertures()
    te=np.arange(0,92,2);re=np.arange(0,1.441,.02);summaries={}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for key,label in VARIANTS.items():
        rows=sum([kept(r['snapshots']['1'][key]) for r in d['records']],[]);th=floats(rows,'theta_deg');rr=floats(rows,'r')
        hist(axes[0],th,te,f'{label}: n={np.isfinite(th).sum():,}',COL[key]);hist(axes[1],rr,re,label,COL[key])
        summaries[key]=stats(sum([r['snapshots']['1'][key]['spots'] for r in d['records']],[]))
    axes[0].set(xlim=(0,90),xlabel='Single-frame θ (°)',ylabel='% of valid angles / 2°');axes[1].set(xlim=(0,1.44),xlabel='Single-frame anisotropy radius r',ylabel='% of retained detections / 0.02')
    axes[1].axvline(MODEL['r_max'],ls='--',c='.5',lw=1);axes[1].axvspan(MODEL['r_max'],1.44,color='.94',zorder=-2)
    for ax in axes:ax.legend(fontsize=8,frameon=False)
    fig.suptitle('Later high-power DNA · frame 1 independently detected in each of nine fields\nChanging the background can change both the selected spots and their angles',fontsize=13);fig.savefig(OUT/'distributions.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,key in zip(axes,['held_out','reference']):
        for fi,color in [(1,'#176d91'),(75,'#b46629'),(150,'#7a5b9a')]:
            rows=sum([kept(r['snapshots'][str(fi)][key]) for r in d['records']],[])
            hist(ax,floats(rows,'theta_deg'),te,f'Frame {fi}; valid n={np.isfinite(floats(rows,"theta_deg")).sum():,}',color)
        ax.set(xlim=(0,90),xlabel='Single-frame θ (°)',ylabel='% of valid angles / 2°',title=VARIANTS[key]);ax.legend(frameon=False)
    fig.suptitle('Independent single-frame detection at the beginning, middle and end',fontsize=13);fig.savefig(OUT/'snapshot_comparison.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True,sharey=True,layout='constrained')
    for ax,r in zip(axes.flat,d['records']):
        for key in ['held_out','reference']:
            rows=kept(r['snapshots']['1'][key]);hist(ax,floats(rows,'theta_deg'),te,f'{"Other 8" if key=="held_out" else "Recorded"}: {np.isfinite(floats(rows,"theta_deg")).sum()}',COL[key])
        ax.set_title(f'DNA {r["index"]} · {r["stamp"]}');ax.set(xlim=(0,90),xlabel='θ (°)',ylabel='% / 2°');ax.legend(fontsize=7,frameon=False)
    fig.suptitle('Frame 1, each later field separately',fontsize=14);fig.savefig(OUT/'per_field.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for group,color in [('Early DNA','#176d91'),('Gold survey','#b46629')]:
        for ax,key in zip(axes,['minimum','p95']):
            rows=sum([kept(r['modes'][key]) for r in refs['groups'][group]],[]);hist(ax,floats(rows,'theta_deg'),te,f'{group}: valid n={np.isfinite(floats(rows,"theta_deg")).sum():,}',color)
    later=sum([kept(r['snapshots']['1']['held_out']) for r in d['records']],[])
    hist(axes[0],floats(later,'theta_deg'),te,f'Later DNA, other-8 minimum: valid n={np.isfinite(floats(later,"theta_deg")).sum():,}','#7a5b9a')
    axes[0].set_title('Minimum for detection and extraction\nLater DNA has different power and targeted field selection')
    axes[1].set_title('Original survey background choices\np95 detection / minimum extraction, now frame 1')
    for ax in axes:ax.set(xlim=(0,90),xlabel='Single-frame θ (°)',ylabel='% of valid angles / 2°');ax.legend(fontsize=8,frameon=False)
    fig.suptitle('Survey comparisons rerun with frame 1 detection and frame 1 angles',fontsize=13);fig.savefig(OUT/'survey_comparison.png',dpi=165);plt.close(fig)
    cache=np.load(OUT/'backgrounds.npz');mi=cache['per_stack_min'].min(0);ref=cache['reference']
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    for ax,im,title in zip(axes,[mi,ref,mi.astype(float)-ref],['All-stack minimum','Recorded wide-field minimum','Built − recorded background']):
        if ax is axes[2]:v=200;image=ax.imshow(im*SCALE,cmap='coolwarm',vmin=-v*SCALE,vmax=v*SCALE)
        else:image=ax.imshow(im*SCALE,cmap='gray',vmin=0,vmax=40)
        ax.set_title(title);ax.set_axis_off();fig.colorbar(image,ax=ax,shrink=.6,label='8-bit equivalent / pixel')
    fig.suptitle('Background estimates: same sensor coordinates; no image registration',fontsize=13);fig.savefig(OUT/'background_comparison.png',dpi=150);plt.close(fig)
    write('summary.json',dict(late_frame1=summaries,background=b['summary'],spot_record_count=len(sp['records'])))
    table=''
    for r in sp['records']:
        if r['group']!='Hinges':continue
        q=r['single_frame'];lo,md,hi=q['theta_p2_p50_p98']
        table+=f'<tr><td><a href="#spot{r["stamp"]}">{r["label"]}</a></td><td>{r["time"]}</td><td>{r["metrics"]["duration_s"]:.1f}</td><td>{md:.1f}</td><td>{lo:.1f}–{hi:.1f}</td><td>{q["xy_sd"][0]:.4f}, {q["xy_sd"][1]:.4f}</td><td>{"Yes — diagnostic only" if r["saturated"] else "No"}</td></tr>'
    cards=''.join(f'<details id="spot{r["stamp"]}"><summary>{r["label"]} · {r["time"]} · {r["metrics"]["duration_s"]:.1f} s'+(' · saturated' if r['saturated'] else '')+f'</summary><p>{html.escape(r["note"])}</p><img loading="lazy" src="spot_{r["stamp"]}.png"><p>Recorded background over selected aperture: mean {r["single_frame"]["bg_aperture_mean_8bit"]:.3f}, max {r["single_frame"]["bg_aperture_max_8bit"]:.2f}; first-frame saved aperture mean {r["single_frame"]["first_intensity_mean_8bit"]:.2f}. All on 0–255 equivalent scale. Pixel panels share their own common display range within this recording.</p></details>' for r in sp['records'])
    payload=[dict(index=r['index'],stamp=r['stamp'],note=r['note'],variants={k:dict(image=v['image'],summary=v['summary'],spots=[{q:s[q] for q in ['x','y','selection','saturated','r','theta_deg']} for s in v['spots']]) for k,v in r['snapshots']['1'].items()}) for r in d['records']]
    counts=''.join(f'<tr><td>{VARIANTS[k]}</td><td>{v["n"]:,}</td><td>{v["saturated_candidates"]:,}</td><td>{v["above"]:,}</td><td>{v["near30_percent"]:.1f}%</td><td>{v["median_theta"]:.1f}°</td></tr>' for k,v in summaries.items())
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Later DNA: single-frame review</title><style>
    body{font:16px/1.55 system-ui;background:#f6f7f4;color:#263943;margin:0}main{max-width:1280px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:22px;margin-top:32px}img{max-width:100%;height:auto;background:white}p{max-width:1050px}a{color:#176d91}table{border-collapse:collapse;font-size:14px}th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #ddd}details{background:white;padding:12px;margin:12px 0}summary{cursor:pointer}select{font:inherit;padding:5px}canvas{width:100%;height:auto;background:black}.note{background:#fff1cf;padding:12px}.small{font-size:14px;color:#4d616d}</style><main>
    <h1>Later high-power DNA: single-frame analysis</h1><p><a href="#distributions">Distributions</a> · <a href="#overlay">Image comparison</a> · <a href="#recordings">Quiet spot recordings</a> · <a href="#expectation">Richard’s expected behaviour</a></p>
    <p class="note"><b>Exploratory, before the email is finalised.</b> Nine wide-field stacks from 20:50:25 to 21:34:23 are separated from the interleaved narrow-field stacks. Notes and your confirmation identify these as DNA despite a stale “Gold-only control” dropdown. These later fields were visited while selecting individual targets; different power and saturation make them unsuitable as an unbiased replacement for the original survey.</p>
    <p><b>No temporal averaging for detection or angles.</b> Frame 1 is the primary population; frames 75 and 150 are independently redetected as checks and are not pooled as extra rods. Raw saturation at <i>any time</i> in the original 150 frames still excludes a candidate.</p>
    <h2>What this check establishes</h2><ul><li><b>The near-30° population feature persists in single frames.</b> The first, middle and last frames give very similar curves. Averaging 150 frames is not its main cause in these fields; each 1.194 ms exposure can still blur faster motion.</li><li><b>The background still changes the prominence of that feature.</b> The fraction at 28–34° falls from 58.4% with narrow BG only, to 44.6% with the other-eight-field minimum, to 27.3% with the recorded wide-field minimum. These are fractions of retained detections with valid angles, not estimates of hinge yield.</li><li><b>Some quiet low-angle targets are worth discussing separately.</b> DNA 3, 4, 6 and 9 show narrow per-frame angle distributions around 29–34°. Their angles remain low with smaller apertures. This supports keeping them as candidates; it does not identify their attachment.</li></ul>
    <h2>Background construction</h2><p>The GUI takes a minimum at each sensor pixel. Here it is applied after adding back each stack’s verified narrow-field background. The main estimate excludes the entire stack being analysed and takes the minimum across the other eight fields (1,200 frames). The all-nine minimum is shown as a sensitivity check, alongside the genuine wide-field background recorded at 21:03:58. You confirmed mostly new sample positions for each numbered DNA target.</p>
    <p>Saved zeros cannot be exactly restored; adding back the old background gives their upper bound. The number of frames also matters: a 1,200-frame minimum has a stronger downward noise bias than the recorded 44-frame background. Persistent sample features, registration coincidences and optical drift can still affect the constructed estimate.</p>
    <p>For scale, the all-nine minimum averages <b>4.98/255 per pixel</b>, versus <b>7.42/255</b> for the recorded wide-field minimum. The old narrow-field profiles average only about <b>0.072/255</b>. The constructed minimum is appreciably lower: both noise minima and changes between fields can contribute. Neither this constructed estimate nor the separate reference is assumed to be a perfect background for every field.</p>
    <img src="background_comparison.png"><h2 id="distributions">What happens to the distribution?</h2><img src="distributions.png"><table><tr><th>Background</th><th>Kept detections</th><th>Saturated candidates*</th><th>Above r limit</th><th>28–34° / valid θ</th><th>Median θ</th></tr>__COUNTS__</table><p class="small">*Before the sequential corner/overlap exclusions, so these are not additive to the retained count. Each background is independently detected; changes can reflect both selection and extraction. Above-limit r is retained in the radius plot and not forced into θ=90°.</p>
    <details><summary>Middle and last frames; separate fields</summary><img src="snapshot_comparison.png"><img src="per_field.png"></details>
    <h2>Inspect the first-frame detections</h2><label>Field <select id="field"></select></label> <label>Background <select id="bg"><option value="held_out">Other eight stacks</option><option value="reference">Recorded wide-field</option><option value="all_min">All nine stacks</option><option value="saved">Saved narrow BG only</option></select></label> <label><input type="checkbox" id="marks" checked>Kept spots</label> <label><input type="checkbox" id="sat">Saturated candidates</label><p id="status"></p><canvas id="map" width="2464" height="2056"></canvas><p class="small">First frame, fixed 5× display in every case. Cyan: kept; red: candidates whose original raw intensity saturated. Display gain does not affect detection.</p>
    <h2>Earlier surveys, also using single frames</h2><img src="survey_comparison.png"><p>Both survey groups were redetected on frame 1. Left: minimum background for detection and extraction, to accompany the later minimum trial. Right: the previously used p95-detection/minimum-extraction choice, with the mean replaced by frame 1. Different backgrounds, illumination and target selection still prevent interpreting the later sample as a controlled power experiment.</p>
    <h2>Long spot recordings: every frame separately</h2><p>These are selected targets, often chosen for apparent angles near 30°, so their angle counts cannot estimate the hinge population. Stationarity is compatible with a locked hinge, a stuck rod or fixed optical structure. It does not by itself establish attachment or the hinge opening angle.</p><table><tr><th>Target</th><th>Start</th><th>Duration (s)</th><th>Median θ</th><th>2–98% θ</th><th>SD X, Y</th><th>Saturated aperture?</th></tr>__SPOTTABLE__</table>
    <p>DNA 3, 4, 6 and 9 are the four quiet low-angle records. DNA 4 is a useful localized, unsaturated example; DNA 6 has a more extended/structured image, so it is a less clean single-particle example. All four stand well above the <em>recorded</em> narrow-field background. That does not establish that the recorded minimum captured every residual background contribution.</p>
    <details><summary>How sensitive are the quiet targets to aperture size?</summary><img src="quiet_apertures.png"><p>These are separate first, middle and last frames at the same fixed centre. Smaller windows change PSF sampling, analyzer balance and neighbouring/background contributions. This is a sensitivity check, not a prescription to pick whichever window gives the desired angle. <a href="quiet_apertures.json">Numbers and recorded-background add-back comparison</a>.</p></details>
    <p>The recorded BG contributes approximately 0.7%, 1.1%, 0.7% and 2.1% of their first-frame aperture means, respectively. Undoing that particular subtraction changes their median θ by only 0.13°, 0.19°, 0.08° and 0.44°. Their apparent low angles therefore do not arise from subtracting that small recorded profile. Unknown residual background and optical effects are a separate question. DNA 4 stays around 33–35° for 12–20 pixel apertures in the three tested frames, and reaches 34–38° with the smallest 8-pixel aperture.</p>
    <p class="small">The ranges include photon/read noise, residual background and real motion. They are not noise-corrected hinge fluctuations. A saturated pixel elsewhere in the camera’s 256×20 ROI does not automatically invalidate a different, unsaturated selected 20×20 aperture.</p>__CARDS__
    <h2>Exact analysis choices</h2><p class="small">__METHOD__ Angles use four analyzer means in each 18×18 aperture, then X/Y and r, then the existing water Fresnel/BFP/annular NA 0.39–1.30 model. No APD correction or new gain fit. All displayed intensities are saved units ×255/4095 before optional display gain. Only the old background profile is added back; raw saturation is checked against the original raw-peak map, never against a bright display or an over-subtracted image.</p>
    <p class="small">Histograms here show observed counts, without solid-angle weighting. Brightness selection and saturation are angle dependent, so applying a geometric sin θ correction would not remove those sampling biases. The spot-recording table uses the original 20×20 selected aperture; the 18×18 specification applies to automatic framestack detections. Historical mean-based galleries remain archived comparisons; the new survey comparisons above use single frames.</p>
    <p><a href="fields.json">Field detections and exclusions</a> · <a href="spot_results.json">Spot-recording measurements</a> · <a href="background_info.json">Background provenance</a> · <a href="survey_reference.json">Single-frame survey detections</a></p></main><script>const D=__PAYLOAD__, $=id=>document.getElementById(id);let im=new Image();$('field').innerHTML=D.map((r,i)=>`<option value="${i}">DNA ${r.index} · ${r.stamp}</option>`).join('');function draw(){const rec=D[+$('field').value],v=rec.variants[$('bg').value],c=$('map').getContext('2d');c.drawImage(im,0,0,2464,2056);for(const s of v.spots){if(s.selection==='kept'&&$('marks').checked)c.strokeStyle='#75f1f1';else if(s.saturated&&$('sat').checked)c.strokeStyle='#ff6f67';else continue;c.lineWidth=1.6;c.beginPath();c.arc(s.x,s.y,10,0,2*Math.PI);c.stroke()}$('status').textContent=`${v.summary.n} retained; ${v.summary.saturated_candidates} saturated candidates. ${rec.note}`;}function load(){im=new Image();im.onload=draw;im.src=D[+$('field').value].variants[$('bg').value].image;}$('field').onchange=load;$('bg').onchange=load;$('marks').onchange=draw;$('sat').onchange=draw;load();</script></html>'''
    source_html='<h2 id="expectation">What should a locked hinge do?</h2>'+''.join(f'<p><a href="{html.escape(e["url"],quote=True)}">{e["title"]}, {e["date"]}</a>: {e["interpretation"]}</p>' for e in EMAILS)
    source_html+='<p><b>Inference for these records:</b> a quiet localized feature is worth following up, not automatically rejected because it is near 30°. But a stuck gold rod can also be quiet. The current 1.194 ms exposure still integrates motion within each frame; using separate frames removes the additional 150-frame average, not that exposure blur. These data alone do not establish a unique hinge opening or rule out faster motion.</p>'
    page=page.replace('__COUNTS__',counts).replace('__SPOTTABLE__',table).replace('__CARDS__',source_html+cards).replace('__METHOD__',html.escape(d['method'])).replace('__PAYLOAD__',json.dumps(clean(payload),separators=(',',':')))
    page=page.replace('<h2>Inspect the first-frame detections</h2>','<h2 id="overlay">Inspect the first-frame detections</h2>').replace('<h2>Long spot recordings: every frame separately</h2>','<h2 id="recordings">Long spot recordings: every frame separately</h2>')
    (OUT/'index.html').write_text(page,encoding='utf8')
    print('SUMMARY',json.dumps(clean(summaries)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['backgrounds','fields','survey','spots','render','all']);args=parser.parse_args();OUT.mkdir(exist_ok=True)
    phases={'backgrounds':backgrounds,'fields':fields,'survey':survey_reference,'spots':spots,'render':render}
    for name,fn in phases.items():
        if args.phase in [name,'all']:fn()
