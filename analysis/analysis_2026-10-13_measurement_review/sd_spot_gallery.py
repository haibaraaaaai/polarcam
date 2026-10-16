"""A stratified, reproducible 50-candidate inspection set with native-pixel movies."""
from pathlib import Path
from collections import Counter,defaultdict
import base64,gzip,hashlib,json,io
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from detector_grid_trial import match
from polarcam.live.intensity_spots import channel_means,anisotropy
from figure1_single_frame import A,B,C,R_LIMIT

HERE=Path(__file__).resolve().parent
DAY=HERE.parents[1]/'data/hinges/polarcam/2026-10-12'
OUT=HERE/'sd_spot_gallery'
SOURCE=HERE/'richard_email_20261014/figure_02_results.json'
SEED=20261016;SIZE=64


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def packed(a,dtype):
    raw=np.asarray(a,dtype=dtype).tobytes()
    return dict(dtype=dtype,shape=list(np.shape(a)),gzip=base64.b64encode(gzip.compress(raw,compresslevel=6,mtime=0)).decode())


def clean(a):
    a=np.asarray(a);return np.where(np.isfinite(a),a,None).tolist()


def traces(movie,bounds,origin):
    ch=np.array([channel_means(f,[bounds],origin=origin)[0] for f in movie])
    xy=anisotropy(ch);r=np.linalg.norm(xy,axis=1)
    theta=np.full_like(r,np.nan);valid=np.isfinite(r)&(r<=R_LIMIT)
    theta[valid]=np.degrees(np.arcsin(np.sqrt(A*r[valid]/(C-B*r[valid]))))
    return dict(channels=ch.tolist(),xy=clean(xy),r=clean(r),theta=clean(theta),
                intensity=ch.mean(axis=1).tolist(),sd_xy=clean(np.nanstd(xy,axis=0,ddof=1)),
                valid_frames=int(np.isfinite(xy).all(axis=1).sum()),above_limit_frames=int((r>R_LIMIT).sum()))


def candidates(records):
    pools=defaultdict(list)
    for rec in records:
        groups=[dict(x=s['x'],y=s['y'],members={'2':s}) for s in rec['thresholds']['2']['spots']]
        for k in ['4','8']:
            other=rec['thresholds'][k]['spots'];pairs,_,added=match(groups,other,radius=6)
            for i,j in pairs.items():groups[i]['members'][k]=other[j]
            groups.extend(dict(x=other[j]['x'],y=other[j]['y'],members={k:other[j]}) for j in added)
        for group in groups:
            members=group['members'];kept=[k for k in ['2','4','8'] if k in members and members[k]['selection']=='kept']
            if not kept:continue
            anchor_k=kept[0];anchor=members[anchor_k]
            pattern=','.join(kept)
            category={'2':'2 only','2,4':'2 and 4','2,4,8':'all three'}.get(pattern,'other pattern')
            row=dict(stack=rec['index'],file=rec['file'],group=category,pattern=pattern,anchor_sd=anchor_k,
                id=f'H{rec["index"]:02}-SD{anchor_k}-C{anchor["component"]}',anchor=anchor,members=members,
                background_sha256=rec['background_sha256'],thresholds={k:v['threshold'] for k,v in rec['thresholds'].items()})
            pools[category].append(row)
    return pools


def sample(pools):
    rng=np.random.default_rng(SEED);chosen=[];per_stack=Counter();used=set()
    targets={'2 only':18,'2 and 4':12,'all three':12,'other pattern':8}
    for category,n in targets.items():
        pool=pools[category]
        for _ in range(n):
            eligible=[s for s in pool if s['id'] not in used and not any(s['stack']==q['stack'] and
                np.hypot(s['anchor']['x']-q['anchor']['x'],s['anchor']['y']-q['anchor']['y'])<32 for q in chosen)]
            assert eligible,category
            counts=[per_stack[s['stack']] for s in eligible];lowest=min(counts)
            eligible=[s for s,c in zip(eligible,counts) if c==lowest]
            s=eligible[int(rng.integers(len(eligible)))];chosen.append(s);used.add(s['id']);per_stack[s['stack']]+=1
    assert len(chosen)==50 and len(per_stack)==16
    return sorted(chosen,key=lambda s:(s['stack'],s['anchor']['y'],s['anchor']['x']))


def build():
    OUT.mkdir(exist_ok=True);(OUT/'spots').mkdir(exist_ok=True)
    source=json.loads(SOURCE.read_text());records=[r for r in source['pooled_records'] if r['group']=='DNA']
    assert len(records)==16
    pools=candidates(records);chosen=sample(pools)
    signature=source['pooled_cache_signature'];sigs={r['file']:r for r in signature['sources']}
    manifest=dict(title='Hinge candidates across SD thresholds',seed=SEED,crop_size=SIZE,frames=150,
        model=dict(A=A,B=B,C=C,r_limit=R_LIMIT),source_sha256=sha(SOURCE),builder_sha256=sha(Path(__file__)),
        sampling='50 candidates, selected with a fixed random seed, balancing coverage across all16 stacks: 18 kept only at2SD,12 at2/4SD,12 atall3,8 with other kept patterns. At least32px between sampled anchors in a field. No selection by angle or appearance; inspection sample, not a population estimate.',
        matching='One-to-one centroid matching within6sensor pixels across complete 2/4/8SD component lists. Missing match can mean threshold loss, centroid shift, merging or splitting; it does not prove absence. Report actual exclusion reason when a match exists.',
        extraction='One fixed18x18 aperture from the lowest SD at which that candidate was kept, used for all150frames and allBG cases. Original sensor polarizer phase preserved. No time averaging or tracking. Raw, minimum, median andp95 traces use this same window.',
        raw_caveat='Raw proxy = saved minimum-subtracted pixel + recorded minimumBG. At a saved zero, raw intensity was at most the minimumBG; its exact value cannot be recovered. Raw-peak companion is an exact pre-subtraction maximum over150frames, not a raw movie.',
        detector='Frame1, minimumBG, repeated aligned2x2 cell means, DoG sigma1/6sensor pixels, whole filtered mean+kSD,8-connected components>=10pixels, no maximumsize. Existing corner and overlapping18x18 exclusions; any raw pixel>=4094 across150frames in aperture OR componentbbox excludes that detection.',
        manual_selection='Selections and notes do not change the source statistics. Export JSON to share chosen IDs/frames/notes.',
        pools={k:len(v) for k,v in pools.items()},records=[])
    bg_cache={};by_stack=defaultdict(list);thumbs=[]
    for s in chosen:by_stack[s['stack']].append(s)
    for index,spots in sorted(by_stack.items()):
        path=DAY/'frames'/spots[0]['file'];meta=json.loads(path.with_suffix('.json').read_text())
        before=dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        assert before==sigs[path.name]['signature'] and sha(path.with_suffix('.json'))==sigs[path.name]['metadata_sha256']
        movie=np.load(path,mmap_mode='r');n=meta['actual']['frames']
        assert n==150 and len(movie)==151 and meta['actual']['phase_marker_appended'] and meta['background']['background_subtracted']
        bh=meta['background']['background_profile_sha256'];binfo=signature['backgrounds'][bh]
        if bh not in bg_cache:
            bgpath=DAY/'backgrounds'/binfo['file'];cap=DAY/'backgrounds'/binfo['capture']
            assert sha(bgpath)==bh and sha(cap)==binfo['capture_sha256']
            bg_cache[bh]=(np.load(bgpath),np.load(cap,mmap_mode='r'),binfo['frames'])
        background,capture,bn=bg_cache[bh]
        rawpath=path.parent/meta['fetch_frames_output']['raw_peak']['path']
        assert sha(rawpath)==sigs[path.name]['raw_peak_sha256']
        rawpeak=np.load(rawpath,mmap_mode='r')
        for s in spots:
            a=s['anchor'];left,right,top,bottom=a['bounds']
            x0=int(np.clip(2*round((a['x']-SIZE/2)/2),0,movie.shape[2]-SIZE))
            y0=int(np.clip(2*round((a['y']-SIZE/2)/2),0,movie.shape[1]-SIZE))
            assert x0%2==0 and y0%2==0
            bounds=[left-x0,right-x0,top-y0,bottom-y0];assert min(bounds)>=0 and max(bounds)<=SIZE
            pixels=np.array(movie[:n,y0:y0+SIZE,x0:x0+SIZE]);bg=background[y0:y0+SIZE,x0:x0+SIZE]
            cap=np.asarray(capture[:bn,y0:y0+SIZE,x0:x0+SIZE]);np.testing.assert_array_equal(cap.min(axis=0),bg)
            quantiles=np.percentile(cap,[50,95],axis=0,method='linear').astype('float32')
            profiles={'minimum':bg.astype('float32'),'median':quantiles[0],'p95':quantiles[1]}
            variants={'raw':pixels.astype('float32')+bg,'minimum':pixels}
            for k in ['median','p95']:
                variants[k]=np.rint(np.maximum(pixels.astype('float32')-(profiles[k]-bg),0)).astype('uint16')
            measurements={k:traces(v,bounds,(x0,y0)) for k,v in variants.items()}
            np.testing.assert_allclose(measurements['minimum']['r'][0],a['r'],atol=1e-12)
            bgstats={}
            for k,v in profiles.items():
                ch=channel_means(v,[bounds],origin=(x0,y0))[0];xy=anisotropy(ch)
                bgstats[k]=dict(channels=ch.tolist(),xy=clean(xy),r=float(np.linalg.norm(xy)))
            ap=pixels[:,bounds[2]:bounds[3],bounds[0]:bounds[1]]
            stats=dict(raw_peak_aperture=int(rawpeak[top:bottom,left:right].max()),
                anchor_raw_peak_aperture_or_bbox=a['raw_peak'],zero_fraction=float(np.mean(ap==0)),
                first_mean_intensity=measurements['minimum']['intensity'][0],
                first_r=measurements['minimum']['r'][0],sd_xy=measurements['minimum']['sd_xy'])
            assert stats['raw_peak_aperture']<4094 and not a['saturated']
            members={}
            for k in ['2','4','8']:
                m=s['members'].get(k)
                members[k]=dict(status='unmatched',threshold=s['thresholds'][k]) if m is None else {
                    **{q:m[q] for q in ['id','x','y','bounds','area','dog_peak','r','raw_peak','suppressed_by']},
                    'status':m['selection'],'threshold':s['thresholds'][k],
                    'distance_from_anchor':float(np.hypot(m['x']-a['x'],m['y']-a['y']))}
            row=dict(id=s['id'],stack=index,file=path.name,note=meta['conditions'].get('condition_note',''),
                category=s['group'],pattern=s['pattern'],anchor_sd=s['anchor_sd'],x=a['x'],y=a['y'],
                bounds=a['bounds'],local_bounds=bounds,crop_origin=[x0,y0],size=SIZE,frames=n,fps=meta['actual']['fps'],
                background_file=binfo['file'],background_sha256=bh,background_frames=bn,
                source_signature=before,detectors=members,stats=stats,traces=measurements,background_stats=bgstats,
                pixels=packed(pixels,'<u2'),backgrounds=packed(np.array(list(profiles.values())),'<f4'),
                raw_peak=packed(rawpeak[y0:y0+SIZE,x0:x0+SIZE],'<u2'))
            fn='spots/'+s['id']+'.js'
            (OUT/fn).write_text('window.receiveSpot('+json.dumps(row,separators=(',',':'),allow_nan=False)+');',encoding='utf8')
            cells=pixels[0].reshape(SIZE//2,2,SIZE//2,2).mean(axis=(1,3))
            thumb=Image.fromarray(np.rint(np.clip(cells*20/4095,0,1)*255).astype('uint8')).resize((128,128),Image.Resampling.NEAREST).convert('RGB')
            tl,tr,tt,tb=bounds
            ImageDraw.Draw(thumb).rectangle((2*tl,2*tt,2*tr-1,2*tb-1),outline='#6be1d9',width=1)
            png=io.BytesIO();thumb.save(png,format='PNG');thumburi='data:image/png;base64,'+base64.b64encode(png.getvalue()).decode()
            compact={k:row[k] for k in ['id','stack','file','category','pattern','anchor_sd','x','y','bounds','detectors','stats']}
            compact.update(data_file=fn,data_sha256=sha(OUT/fn),thumbnail=thumburi)
            manifest['records'].append(compact);thumbs.append((row,thumb))
        assert before==dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        print(f'Stack {index}: {len(spots)} candidates, all150frames extracted',flush=True)
    manifest['gallery_id']='sd-'+hashlib.sha256((manifest['source_sha256']+','.join(r['id'] for r in manifest['records'])).encode()).hexdigest()[:16]
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf8')
    (OUT/'manifest.js').write_text('window.GALLERY='+json.dumps(manifest,separators=(',',':'),allow_nan=False)+';',encoding='utf8')
    sheet=Image.new('RGB',(10*150,5*174),'#f3f5f4');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    for i,(row,thumb) in enumerate(thumbs):
        xx=(i%10)*150+10;yy=(i//10)*174+5;sheet.paste(thumb,(xx,yy))
        draw.text((xx,yy+130),row['id'],font=font,fill='#22343a')
        draw.text((xx,yy+146),'Kept: '+row['pattern']+' SD',font=font,fill='#22343a')
    sheet.save(OUT/'overview.png')
    print(json.dumps(dict(examples=len(chosen),by_category=dict(Counter(s['group'] for s in chosen)),by_stack=dict(Counter(s['stack'] for s in chosen)),pools=manifest['pools']),indent=2))


if __name__=='__main__':build()
