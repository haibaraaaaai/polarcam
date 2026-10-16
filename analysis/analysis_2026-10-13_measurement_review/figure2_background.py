"""Matched-background frame-1 sensitivity figure; one change per comparison."""
from pathlib import Path
import hashlib
import inspect
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import PercentFormatter, MaxNLocator
from detector_threshold_trial import select_spots
from background_region_trial import ELLIPSE, inside
from polarcam.live import Detection_alg_offline as gui
from polarcam.live.intensity_spots import channel_means, anisotropy
from figure1_single_frame import R_LIMIT, theta_from_r, percentage_curve


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def threshold_trial(frame,peak,ks=(2,4,8)):
    """Use exactly the same corrected frame for detection and extraction."""
    work=gui.intensity_detection_map(frame)[10:-10,10:-10]
    dog=cv2.GaussianBlur(work,(0,0),1,borderType=cv2.BORDER_REPLICATE)-cv2.GaussianBlur(work,(0,0),6,borderType=cv2.BORDER_REPLICATE)
    mu,sd=gui._finite_stats(dog)
    output={}
    for k in ks:
        threshold=max(0,mu+k*sd)
        _,labels,stats,centres=cv2.connectedComponentsWithStats((dog>threshold).astype('uint8'),connectivity=8)
        rows=[]
        for comp in range(1,len(stats)):
            bx,by,w,h,area=map(int,stats[comp])
            if area<10:continue
            x,y=map(float,centres[comp]+10);l,t=round(x)-9,round(y)-9
            bounds=[l,l+18,t,t+18]
            assert l>=0 and t>=0 and l+18<=frame.shape[1] and t+18<=frame.shape[0]
            raw_peak=max(int(peak[t:t+18,l:l+18].max()),int(peak[by+10:by+10+h,bx+10:bx+10+w].max()))
            xy=anisotropy(channel_means(frame,[bounds])[0]);valid=bool(np.isfinite(xy).all())
            rows.append(dict(id=f'sd{k}-{comp}',component=comp,x=x,y=y,area=area,bounds=bounds,
                accepted=True,inside_ellipse=bool(inside(x,y)),saturated=raw_peak>=4094,raw_peak=raw_peak,
                dog_peak=float(dog[by:by+h,bx:bx+w][labels[by:by+h,bx:bx+w]==comp].max()),
                r=float(np.linalg.norm(xy)) if valid else None))
        selected=select_spots(rows);values=[s['r'] for s in selected]
        output[str(k)]=dict(threshold=float(threshold),n=len(values),median_r=float(np.median(values)) if values else None,
            above_limit=sum(r>R_LIMIT for r in values),spots=rows,radii=values)
    return output


def background_extras(bgpath,capture,n):
    """Per-pixel percentiles from real background frames; omit phase marker."""
    minimum=np.load(bgpath);movie=np.load(capture,mmap_mode='r')
    assert len(movie)==n+1
    profiles=np.empty((2,*minimum.shape),dtype='float32')
    for y in range(0,minimum.shape[0],64):
        block=movie[:n,y:y+64]
        np.testing.assert_array_equal(block.min(axis=0),minimum[y:y+64])
        profiles[:,y:y+64]=np.percentile(block,[50,95],axis=0,method='linear')
    assert np.all(profiles>=minimum)
    return {'median':profiles[0]-minimum,'p95':profiles[1]-minimum}


def pool_survey(here,out,record):
    day=here.parents[1]/'data/hinges/polarcam/2026-10-12'
    survey_path=here/'late_hinges_single_frame/survey_reference.json';record(survey_path)
    survey=json.loads(survey_path.read_text());assert survey['complete']
    contexts=[];backgrounds={};signatures=[]
    for group,source_group,n in [('DNA','Early DNA',16),('Gold','Gold survey',14)]:
        records=survey['groups'][source_group];assert len(records)==n
        for rec in records:
            path=day/'frames'/rec['name'];meta_path=path.with_suffix('.json')
            meta=json.loads(meta_path.read_text());record(meta_path)
            sig=dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
            assert sig==rec['source_signature']
            assert meta['background']['background_subtracted'] and meta['actual']['frames']==150
            peakpath=path.parent/meta['fetch_frames_output']['raw_peak']['path'];record(peakpath)
            assert sha(peakpath)==meta['fetch_frames_output']['raw_peak']['sha256']
            bh=meta['background']['background_profile_sha256']
            if bh not in backgrounds:
                name=meta['background']['background_profile_source_path'].replace('\\','/').split('/')[-1]
                bgpath=day/'backgrounds'/name;record(bgpath);record(bgpath.with_suffix('.json'))
                assert sha(bgpath)==bh
                bm=json.loads(bgpath.with_suffix('.json').read_text())
                capture=day/'backgrounds'/bm['source_stack_file'];record(capture)
                assert bm['actual']['phase_marker_appended'] and not bm['background']['background_subtracted']
                backgrounds[bh]=dict(file=name,capture=capture.name,frames=bm['actual']['frames'],capture_sha256=sha(capture))
            signatures.append(dict(file=path.name,signature=sig,metadata_sha256=sha(meta_path),raw_peak_sha256=sha(peakpath)))
            contexts.append((group,rec,path,peakpath,bh))
    calculation_source='\n'.join(inspect.getsource(f) for f in [threshold_trial,background_extras,pool_survey,select_spots])
    cache_signature=dict(calculation_sha256=hashlib.sha256(calculation_source.encode()).hexdigest(),
        survey_sha256=sha(survey_path),extraction_sha256=sha(Path(inspect.getfile(channel_means))),
        detector_sha256=sha(Path(gui.__file__)),sources=signatures,backgrounds=backgrounds)
    cache=out/'figure_02_results.json'
    if cache.exists():
        prior=json.loads(cache.read_text())
        if prior.get('pooled_cache_signature')==cache_signature:return prior['pooled_records'],cache_signature
    extras={};pooled=[]
    for group,rec,path,peakpath,bh in contexts:
        if bh not in extras:
            b=backgrounds[bh]
            extras[bh]=background_extras(day/'backgrounds'/b['file'],day/'backgrounds'/b['capture'],b['frames'])
        frame=np.array(np.load(path,mmap_mode='r')[0]);peak=np.load(peakpath)
        levels=threshold_trial(frame,peak)
        selected=[s for s in levels['2']['spots'] if s['selection']=='kept']
        reference=[s for s in rec['modes']['minimum']['spots'] if s['selection']=='kept']
        # Independently reproduce Figure 1's cached baseline positions and r.
        np.testing.assert_allclose([[s['x'],s['y'],s['r']] for s in selected],
                                   [[s['x'],s['y'],s['r']] for s in reference],atol=1e-12)
        variants={}
        for key,extra in extras[bh].items():
            corrected=np.rint(np.maximum(frame.astype('float32')-extra,0)).astype('uint16')
            variants[key]=threshold_trial(corrected,peak,ks=(2,))['2']
        pooled.append(dict(group=group,file=path.name,index=rec['index'],background_sha256=bh,
                           background_variants=variants,thresholds=levels))
        print(f'Figure 2: {group} {rec["index"]}; min/median/p95='+str([levels['2']['n'],variants['median']['n'],variants['p95']['n']]),flush=True)
    return pooled,cache_signature


def build(here,out):
    inputs={}
    def record(path):
        inputs[str(path.relative_to(here)) if path.is_relative_to(here) else str(path)]=sha(path)
    pooled,signature=pool_survey(here,out,record)
    fig1path=out/'figure_01_results.json';record(fig1path)
    fig1=json.loads(fig1path.read_text());assert fig1['detection_background']==fig1['extraction_background']=='minimum'
    edges=np.arange(0,1.4401,.02)
    cdf=1-np.cos(np.deg2rad(theta_from_r(np.clip(edges,0,R_LIMIT))));cdf[edges>=R_LIMIT]=1
    isotropic=np.diff(cdf)*100;np.testing.assert_allclose(isotropic.sum(),100)
    results=dict(frame_number=1,display_gain=20,bin_width_r=.02,
        scope='A-C: first DNA stack image example. D/E: all 16 DNA stacks. F/G: all 14 gold stacks.',
        baseline='Figure 1: minimum background for BOTH detection and extraction; mean+2SD, >=10 pixels without upper size limit.',
        background_comparison_policy='D/F: independently detect at 2SD and extract r using the SAME minimum, median or p95 corrected frame. Each background is calculated from that stack’s own recorded capture. Selection can change; this is full-pipeline sensitivity, not a paired fixed-position extraction test.',
        threshold_comparison_policy='E/G: minimum background for BOTH detection and extraction; independently detect at 2,4,8SD. No maximum component size.',
        exclusions='Identical everywhere: corner ellipse, 10-pixel border, strongest-first overlapping 18x18 apertures, undefined r, and raw pixel >=4094 in aperture OR component bbox at any time during all150 frames. No finite-r cutoff, survival or low-background mask.',
        normalization='100 * count in each 0.02 r bin / all retained detections in that sample group and variant; every retained detection has equal weight.',
        subtraction='Saved minimum-subtracted frame 1 minus (selected background percentile - minimum), zero clipped and rounded. Percentiles use only real capture frames, excluding the phase marker.',
        smoothing='Gaussian sigma=0.02r. Faint steps: actual percentages. Smooth lines: guides. Isotropic dashed grey steps: exact expected bin mass, not division by isotropy.',
        annotation=ELLIPSE,pooled_records=pooled,pooled_cache_signature=signature,
        panels={'D':'DNA background comparison','E':'DNA threshold comparison','F':'Gold background comparison','G':'Gold threshold comparison'},
        background_comparison={},threshold_comparison={},baseline_matches_Figure1A={},
        bin_edges=edges.tolist(),isotropic_percent_per_bin=isotropic.tolist())
    for group in ['DNA','Gold']:
        fields=[r for r in pooled if r['group']==group]
        comparisons={'background_comparison':{},'threshold_comparison':{}}
        for kind,keys in [('background_comparison',['minimum','median','p95']),('threshold_comparison',['2','4','8'])]:
            for key in keys:
                records=[r['thresholds']['2'] if key=='minimum' else
                         r['background_variants'][key] if kind=='background_comparison' else r['thresholds'][key] for r in fields]
                radii=np.array([v for r in records for v in r['radii']])
                baseline=key=='minimum' or (kind=='threshold_comparison' and key=='2')
                curve=percentage_curve(radii,np.linspace(0,1,51) if baseline else edges,len(radii),.08)
                comparisons[kind][key]=dict(n=len(radii),median_r=float(np.median(radii)),
                    above_limit=int((radii>R_LIMIT).sum()),radii=radii.tolist(),histogram=curve,
                    per_field=[dict(file=r['file'],n=m['n'],threshold=m['threshold']) for r,m in zip(fields,records)])
                if baseline:
                    for field in ['counts','percent','curve_x','curve_percent_per_bin']:
                        np.testing.assert_allclose(curve[field],fig1['groups'][group]['r_histogram'][field],atol=1e-12)
                    results['baseline_matches_Figure1A'][group]=True
            results[kind][group]=comparisons[kind]

    day=here.parents[1]/'data/hinges/polarcam/2026-10-12'
    first=pooled[0];bgpath=day/'backgrounds'/signature['backgrounds'][first['background_sha256']]['file']
    bg=np.load(bgpath)
    images=[np.rint(np.clip(gui.intensity_detection_map(bg)*20/4095,0,1)*255).astype('uint8')]
    for key in ['minimum','p95']:
        path=here/f'background_overlap_check/percentile_spots_first_{key}.webp';record(path)
        images.append(np.array(Image.open(path).convert('L')))
    frame=np.load(day/'frames'/first['file'],mmap_mode='r')[0]
    np.testing.assert_array_equal(images[1],np.rint(np.clip(gui.intensity_detection_map(frame)*20/4095,0,1)*255).astype('uint8'))
    b=signature['backgrounds'][first['background_sha256']]
    extra=background_extras(bgpath,day/'backgrounds'/b['capture'],b['frames'])['p95']
    corrected=np.rint(np.maximum(frame.astype('float32')-extra,0)).astype('uint16')
    np.testing.assert_array_equal(images[2],np.rint(np.clip(gui.intensity_detection_map(corrected)*20/4095,0,1)*255).astype('uint8'))
    results['image_source']=first['file']
    fig=plt.figure(figsize=(18,14.6),layout='constrained')
    gs=fig.add_gridspec(3,6,height_ratios=[1.18,1,1])
    titles=['A · Recorded minimum background','B · Frame 1 − minimum background','C · Frame 1 − 95th-percentile background']
    for i,(im,title) in enumerate(zip(images,titles)):
        ax=fig.add_subplot(gs[0,2*i:2*i+2]);ax.imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        ax.add_patch(Ellipse((ELLIPSE['cx'],ELLIPSE['cy']),2*ELLIPSE['rx'],2*ELLIPSE['ry'],fill=False,edgecolor='#ff756e',lw=1.5))
        ax.set_title(title+'\n20× display',fontsize=13);ax.set_axis_off()
    ymax=max(14,2+2*np.ceil(max(max(d['histogram']['percent']) for kind in ['background_comparison','threshold_comparison']
                              for group in results[kind].values() for d in group.values())/2))
    threshold_xmax=max(1.02,.02+np.ceil(max(max(d['radii']) for g in results['threshold_comparison'].values() for d in g.values())*50)/50)
    for row,(group,n,color,letters) in enumerate([('DNA',16,'#166888','DE'),('Gold',14,'#c16b2a','FG')],1):
        for col,(kind,xmax) in enumerate([('background_comparison',1.44),('threshold_comparison',threshold_xmax)]):
            ax=fig.add_subplot(gs[row,col*3:col*3+3])
            variants=[('minimum','-','Minimum BG · 2 SD · baseline'),('median','--','Median BG · 2 SD'),('p95',':','95th-percentile BG · 2 SD')] if col==0 else [
                      ('2','-','Minimum BG · 2 SD · baseline'),('4','--','Minimum BG · 4 SD'),('8',':','Minimum BG · 8 SD')]
            for key,ls,label in variants:
                d=results[kind][group][key];c=d['histogram']
                ax.stairs(c['percent'],c['bin_edges'],color=color,lw=.65,ls=ls,alpha=.25)
                ax.plot(c['curve_x'],c['curve_percent_per_bin'],color=color,lw=1.8,ls=ls,label=f'{label} (n={d["n"]:,})')
            ax.stairs(isotropic,edges,color='.45',ls='--',lw=1.2,label='3-D isotropic')
            ax.axvspan(R_LIMIT,xmax,color='.95',zorder=-3);ax.axvline(R_LIMIT,color='.4',ls=':',lw=1)
            ax.text(R_LIMIT+.012,.48,f'r_lim = {R_LIMIT:.3f}',transform=ax.get_xaxis_transform(),rotation=90,
                    va='center',ha='left',color='.35',fontsize=10,bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
            ax.set(xlim=(0,xmax),ylim=(0,ymax),xlabel='Anisotropy radius r',ylabel='Retained detections per 0.02 bin (%)')
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=0));ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
            ax.legend(loc='upper right',fontsize=8.8,frameon=True,facecolor='white',edgecolor='white',framealpha=.94)
            ax.spines[['top','right']].set_visible(False)
            ax.set_title(f'{letters[col]} · {group}, {n} stacks · '+('Change background; keep 2 SD' if col==0 else 'Change threshold; keep minimum BG'),fontsize=12)
    fig.suptitle('Figure 2 · Background subtraction and spot selection · Frame 1',fontsize=18)
    fig.savefig(out/'figure_02_background_selection.png',dpi=300,facecolor='white');plt.close(fig)
    results['inputs_sha256']=inputs;results['builder_sha256']=sha(Path(__file__))
    (out/'figure_02_results.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf8')
    return results


if __name__=='__main__':
    here=Path(__file__).resolve().parent
    r=build(here,here/'richard_email_20261014')
    print(json.dumps({kind:{g:{k:{q:d[q] for q in ['n','median_r','above_limit']} for k,d in values.items()}
                           for g,values in r[kind].items()} for kind in ['background_comparison','threshold_comparison']},indent=2))
