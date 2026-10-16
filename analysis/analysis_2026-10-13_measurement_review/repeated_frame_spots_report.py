"""Update the existing inspector with the repeated individual-frame detector."""
import json,csv,re
import numpy as np
import matplotlib.pyplot as plt
import motion_selected_rods as original
import motion_selected_rods_report as prior
from repeated_frame_spots import OUT,read,write

ROOT=original.OUT;MAIN=prior.MAIN

def replace_once(text,old,new):
    assert old in text,old[:90]
    return text.replace(old,new,1)

def distribution(d,k='1',mode='strict',bg='minimum',pick='first'):
    return np.array([s['series'][bg]['selected'][pick]['theta'] for s in d['spots'] if s['per_k'][k][mode+'_kept'] and s['series'][bg]['selected'][pick] is not None],float)

def make_figures(data):
    for k in ['1','2','3']:
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained')
        for ax,stamp in zip(axes.ravel(),MAIN):
            d=data[stamp];before=read(ROOT/stamp/'results.json')
            a=np.array([s['series']['minimum']['selected']['first']['theta'] for s in before['spots'] if s['statistics_kept']],float);a=a[np.isfinite(a)]
            h,e=np.histogram(a,np.arange(0,92,2));ax.stairs(h*100/max(1,len(a)),e,color='#999',ls=':',label=f'Previous motion-only first · n={len(a)}')
            paired=[s for s in d['spots'] if s['per_k'][k]['strict_kept'] and all(prior.finite(s['series']['minimum']['selected'][p]['theta']) for p in ['first','dimmest'])]
            for pick,col in [('first','#176d91'),('dimmest','#c36225')]:
                t=np.array([s['series']['minimum']['selected'][pick]['theta'] for s in paired]);h,e=np.histogram(t,np.arange(0,92,2));ax.stairs(h*100/max(1,len(t)),e,color=col,label=f'New {pick} · median {np.median(t):.1f}°')
            ax.axvspan(28,34,color='#ead091',alpha=.18);ax.set(title=f'{d["label"]} · new paired n={len(paired)}',xlabel='Recovered θ (degrees)',ylabel='% per 2° bin',xlim=(0,90));ax.legend(fontsize=8)
        fig.suptitle(f'Individual-frame detection: mean + {k} SD; ≥5 frame matches\nPositive local contrast in every unmatched frame · minimum BG · saturation excluded',fontsize=12)
        fig.savefig(OUT/f'theta_k{k}.png',dpi=145);plt.close(fig)
    # Compact brightness comparison, using original numeric pixels for display.
    d=data['183650'];raw=np.load(original.DAY/'backgrounds'/d['raw_stack'],mmap_mode='r');bg=np.load(original.DAY/'backgrounds'/__import__('pathlib').Path(d['profile']).with_suffix('.npy'));pose=np.array(d['pose'])
    fig,axes=plt.subplots(1,4,figsize=(12,3.8),layout='constrained')
    for ax,(sid,gain) in zip(axes,[(1012,20),(1012,190),(249,20),(41,20)]):
        s=next(s for s in d['spots'] if s['id']==sid);fi=int(np.argmax(s['peak_cell_255']));x,y=np.array([s['x'],s['y']])+pose[fi];l=2*int(round((x-32)/2));t=2*int(round((y-32)/2));cut=original.cells(np.maximum(raw[fi,t:t+64,l:l+64].astype(float)-bg[t:t+64,l:l+64],0))*original.SCALE
        ax.imshow(cut,cmap='gray',vmin=0,vmax=255/gain,interpolation='nearest');ax.add_patch(plt.Rectangle((11.5,11.5),9,9,fill=False,color='#56d9c4',lw=.7));ax.axis('off');ax.set_title(f'#{sid} · frame {fi+1} · {gain}×\nPeak 2×2 mean: {s["peak_cell_max_255"]:.2f}/255',fontsize=10)
    fig.suptitle('First hinge capture · each candidate at its largest peak-cell intensity\nOriginal numeric data, minimum BG subtracted; same 64×64 sensor-pixel crop size',fontsize=11);fig.savefig(OUT/'brightness_comparison.png',dpi=145);plt.close(fig)

def main():
    data={stamp:read(OUT/stamp/'results.json') for stamp in original.STAMPS}
    summaries=[]
    for stamp,d in data.items():
        payload={k:d[k] for k in ['stamp','label','frames','pose','spots','model','summary','per_frame','wrong_time_controls']}
        for s in payload['spots']:
            s['score']=s['old_motion_score'] or 0;s['never_old_detected']=s['old_id'] is None;s['old_detection_frames']=0
        (OUT/stamp/'view.js').write_text('window.CAPTURES['+json.dumps(stamp)+']='+json.dumps(prior.compact(payload),ensure_ascii=False,allow_nan=False,separators=(',',':'))+';',encoding='utf8')
        out=dict(stamp=stamp,label=d['label'],counts=d['summary'],wrong_time_controls=d['wrong_time_controls'],statistics={})
        for k in ['1','2','3']:
            out['statistics'][k]={}
            for mode in ['strict','90','five_only']:
                out['statistics'][k][mode]={}
                for pick in ['first','dimmest']:
                    th=distribution(d,k,mode,pick=pick);th=th[np.isfinite(th)];out['statistics'][k][mode][pick]=dict(n=len(th),median=float(np.median(th)) if len(th) else None,near30_percent=float(100*np.mean((th>=28)&(th<34))) if len(th) else None)
        summaries.append(out)
        with (OUT/stamp/'candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
            fields=['id','x','y','repeat_candidate','old_id','raw_peak','saturated','peak_cell_max_255','peak_cell_median_255','k','detected_frames','rest_frames','rest_positive_frames','strict_accepted','strict_kept','first_theta_minimum','dimmest_theta_minimum']
            wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
            for s in d['spots']:
                for k in ['1','2','3']:
                    row={key:s[key] for key in fields if key in s};row.update({key:s['per_k'][k][key] for key in fields if key in s['per_k'][k]});row.update(k=k,first_theta_minimum=s['series']['minimum']['selected']['first']['theta'],dimmest_theta_minimum=s['series']['minimum']['selected']['dimmest']['theta']);wr.writerow(row)
    write(OUT/'summary.json',summaries);make_figures(data)
    # Preserve the prior page and all its data/JS; replace only the generated entry page.
    if not (ROOT/'motion_only.html').exists():(ROOT/'motion_only.html').write_bytes((ROOT/'index.html').read_bytes())
    text=(ROOT/'motion_only.html').read_text(encoding='utf8')
    text=replace_once(text,'<title>Motion-selected features</title>','<title>Repeated frame detections</title>')
    intro='''<h1>Spot detections that repeat along the stage path</h1>
<p><b>Default: loose 1-SD detection in each individual frame, ≥5 matched frames, and positive local contrast in every remaining frame.</b> Thresholds 2 and 3 SD, a 90%-support option and a five-detections-only option are available below. These are moving-sample background captures; each angle is still from one original frame.</p>
<p class="note"><b>#1012 is rejected.</b> It has zero individual-frame detections even at 1 SD; its centre is above its broader surroundings in only 22/45 frames. #249 is detected in 41/45 frames and has positive support in the other four. Raising display contrast does not make #1012 a convincing isolated spot.</p>
<p><a href="#inspect">Inspect spots</a> · <a href="#distributions">Distributions</a> · <a href="#brightness">#1012, #249 and brightest spots</a> · <a href="#method">Exact rule</a> · <a href="motion_only.html">Previous motion-only gallery</a></p>
'''
    start=text.index('<h1>');end=text.index('<div class="card" id="inspect">');text=text[:start]+intro+text[end:]
    text=text.replace('<h2>Inspect the motion candidates</h2>','<h2>Inspect the repeated detections</h2>')
    start=text.index('<label>Show <select id="filter">');end=text.index('</div>',start)
    text=text[:start]+'''<label>Show <select id="filter"><option value="accepted">Accepted tracks</option><option value="all">All new tracks + previous candidates</option><option value="rejected">Rejected tracks / old candidates</option><option value="kept">Eligible for statistics</option><option value="new">New tracks without an old ID</option></select></label>
<label>Threshold <select id="k"><option value="1">Mean + 1 SD (loose)</option><option value="2">Mean + 2 SD</option><option value="3">Mean + 3 SD</option></select></label>
<label>Rest of path <select id="support"><option value="strict">Positive local contrast in all remaining frames</option><option value="90">Positive in ≥90% of remaining frames</option><option value="five_only">Only require ≥5 detected frames</option></select></label>'''+text[end:]
    text=text.replace('<label>Image <select id="view"><option value="frame">Individual frame · minimum BG subtracted</option><option value="coherence">Motion score · sample coordinates</option></select></label>','<select id="view" hidden><option value="frame">Frame</option></select>')
    text=text.replace('Old k4 detections','Current per-frame detections').replace('Old k4','Previous detector')
    a=text.index('<p class="tiny legend">');b=text.index('</p>',a)+4;text=text[:a]+'''<p class="tiny legend"><span style="color:#158a77">● Accepted</span><span style="color:#bf233a">● Saturated</span><span style="color:#8752b9">● Overlap excluded</span><span style="color:#888">● Fails repeated-detection rule</span>White = selected; grey dashed = individual-frame detections.</p>'''+text[b:]
    text=text.replace('the coherence view stays in sample coordinates.','the original camera frame stays fixed.')
    text=text.replace('Minimum motion score','')
    text=text.replace('<th>Motion score</th><th>Old detections</th>','<th>Detected frames</th><th>Remaining support</th>')
    text=text.replace('<img id="detectionimg" class="plot"><img id="examplesimg" class="plot">','')
    text=text.replace('Wrong-path controls and systematic example crops','Scrambled-timing controls')
    a=text.index('<div class="card" id="distributions">');b=text.index('<div class="card" id="method">',a)
    table='<table><thead><tr><th>Capture</th><th>Accepted 1-SD tracks</th><th>After saturation / overlap</th><th>First → dimmest median θ</th></tr></thead><tbody>'
    for summary in summaries:
        if summary['stamp'] not in MAIN:continue
        z=summary['counts']['1']['strict'];st=summary['statistics']['1']['strict'];table+=f'<tr><td>{summary["label"]}</td><td>{z["accepted"]}</td><td>{z["statistics"]}</td><td>{st["first"]["median"]:.1f}° → {st["dimmest"]["median"]:.1f}°</td></tr>'
    table+='</tbody></table>'
    report='''<div class="card" id="distributions"><h2>What changes with the repeated-detection rule?</h2><p>The strong low-angle concentration from the motion-only search weakens substantially. The retained first-frame medians are about 48–50° for the hinge captures and 42° for this gold-control capture, with original minimum subtraction. <b>This is still brightness-dependent selection:</b> genuine dim, low-angle rods can fail, so the change does not by itself prove that all rejected features were background.</p>'''+table+'''
<label>Summary threshold <select id="summarychoice"><option value="repeated/theta_k1.png">1 SD</option><option value="repeated/theta_k2.png">2 SD</option><option value="repeated/theta_k3.png">3 SD</option></select></label><img id="summaryimg" src="repeated/theta_k1.png"><p class="tiny">Grey: previous motion-only selection and its original first-frame extraction. Blue/orange: new selection with the same paired finite candidates for first/dimmest. Detection method and centre localisation changed. No pooling of separate capture groups, no solid-angle correction, and no claim of independent biological replication.</p></div>
<div class="card" id="brightness"><h2>Is #1012 convincing after increasing contrast?</h2><p>It remains much less distinct than its surroundings. I would exclude it under this rule. Its largest background-subtracted aligned 2×2 mean is only <b>1.95/255</b>, versus <b>28.52/255 for #249</b>. These peak-cell values differ from the diluted mean over the 18×18 aperture plotted above.</p><img class="plot" src="repeated/brightness_comparison.png">
<p>Among accepted, nonsaturated tracks in the first hinge capture, <b>#41 has the largest observed peak-cell mean: 126.99/255</b> after minimum subtraction (median of its per-frame peaks: 67.67/255). Its largest original sensor pixel is 3046/4095, or 189.68/255. Brighter #102 reaches raw 4094 and is excluded as saturated. These maxima are across the movie and over the extraction aperture, not an average over frames or a claim of intrinsic rod brightness.</p>
<details><summary>Original-pixel contrast comparisons for #1012 and #249</summary><img class="plot" src="repeated/spot_1012_contrast.png"><img class="plot" src="repeated/spot_249_contrast.png"></details>
<p>Increasing laser power can improve photon-noise-limited precision even when rod and background both brighten: under linear scaling, signal/background stays constant but shot-noise-limited SNR grows roughly as √power. It does not remove a systematic background residual that scales with power, and some rods already saturate. This distinction follows from the <a href="https://camera.hamamatsu.com/us/en/learn/technical_information/thechnical_guide/calculating_snr.html">standard camera SNR equation</a>; the data here do not establish which noise source dominates each faint candidate.</p></div>
'''
    text=text[:a]+report+text[b:]
    a=text.index('<div class="card" id="method">');b=text.index('<script>window.CAPTURES',a)
    text=text[:a]+'''<div class="card" id="method"><h2>Exact rule and limitations</h2><ol>
<li>Subtract the recorded minimum background from each real raw frame. For intensity detection only, average aligned 2×2 cells. Apply the same half-grid DoG widths as the previous check: 1 and 6 sensor pixels.</li>
<li>Independently find per-frame local maxima above that frame’s whole DoG-image mean + k standard deviations. Require at least three above-threshold cells in the surrounding 3×3-cell neighbourhood. Local-maximum neighbourhood: 14×14 sensor pixels; no upper component-size cutoff. Default k=1; k=2 and 3 are comparisons.</li>
<li>Transform all k=1 detections into sample coordinates using the previously measured stage path. Cluster repeated positions, refine their centres from matched per-frame locations, and require detections in ≥5 distinct frames. Matching tolerance: 6 sensor pixels; suppress duplicate centres within 12 pixels. One detection may support only one track in a frame. No previous motion-score threshold is required.</li>
<li>At the fixed predicted centre in frames without a detection, require positive DoG: the narrow-blurred centre must exceed its broadly blurred surroundings. Default requires all such frames; 90% and no-between-frame-support options are explicit sensitivity checks. Comparing a signal with its own minimum would be nearly automatic, so is not the support test. This is a pragmatic screening rule, not a noise-calibrated confidence level.</li>
<li>Preserve old candidate IDs for unique counterparts within 8 pixels; otherwise assign a new ID. Old-only rejected candidates remain available for inspection and cannot pass the new independent search. Matched track centres can move slightly relative to the old catalogue.</li>
<li>For θ/r, use one original unwarped frame and an 18×18 tracked aperture, correct analyser parity, original minimum or the earlier masked-median background, and the unchanged Fresnel/BFP/annular inversion. Any original pixel ≥4094 in the aperture path at any real frame excludes that track from statistics. For overlapping apertures, more detected frames wins, then larger median peak-cell intensity. Above-limit r is retained; invalid θ is not clipped to 90°.</li>
</ol><p class="note">The masked median remains a sensitivity check: it contains unavailable regions and did not mask all newly recovered features. Dimmest-frame selection can favour a low-angle state, a negative noise excursion or a tracking error; it is not an unbiased correction. Conversely, requiring support in every frame can reject a real rod that briefly dims or moves outside the tolerance. Both limitations are visible through the alternative controls.</p>
<p>Only five captures have reliable complete stage paths. Motion can belong to sample-associated scattering or debris as well as gold. Five shuffled-time paths run the entire new detection/linking test; low control counts do not prove molecular identity or supply a calibrated false-discovery rate.</p><p><a href="repeated/summary.json">Numerical summary</a> · <a href="../repeated_frame_spots.py">Analysis source</a> · <a href="../repeated_frame_spots_report.py">Report builder</a></p></div>
'''+text[b:]
    text=text.replace('src="183650/view.js"','src="repeated/183650/view.js"').replace('src="viewer.js"','src="repeat_viewer.js"')
    (ROOT/'index.html').write_text(text,encoding='utf8')
    js=prior.JS
    a=js.index('function rows()');b=js.index('function spot()',a)
    js=js[:a]+'''function rule(s){return s.per_k[$('k').value];}
function accepted(s){return rule(s)[$('support').value+'_accepted'];}
function eligible(s){return rule(s)[$('support').value+'_kept'];}
function rows(){const f=$('filter').value;return D.spots.filter(s=>f==='all'||f==='accepted'&&accepted(s)||f==='rejected'&&!accepted(s)||f==='kept'&&eligible(s)||f==='new'&&s.old_id===null);}
function status(s){const q=rule(s);return !s.repeat_candidate?'No independent ≥5-frame track':q.detected_frames<5?'Only '+q.detected_frames+' detections':!accepted(s)?'Insufficient support between detections':s.saturated?'Raw saturated':q[$('support').value+'_overlap']?'Overlap with #'+q[$('support').value+'_overlap']:'Statistics eligible';}
function color(s){return !accepted(s)?'#999':s.saturated?'#ff4c69':!eligible(s)?'#c28dff':'#62e4cd';}
'''+js[b:]
    js=js.replace('D.old[frame]',"D.per_frame[frame][$('k').value]")
    a=js.index('function details()');b=js.index('function histData',a)
    js=js[:a]+'''function details(){const s=spot();if(!s)return;const q=series(s),v=q.selected.dimmest,xy=q.xy[frame],met=rule(s);$('spotid').value=s.id;
 $('spotinfo').innerHTML=`<b>#${s.id} · ${status(s)}</b><br>Detected ${met.detected_frames}/${D.frames} frames at ${$('k').value} SD.<br>Positive local contrast in ${met.rest_positive_frames}/${met.rest_frames} remaining frames.<br>Current frame: ${met.hit_frames.includes(frame+1)?'detected':'not detected'}, DoG ${number(s.local_contrast_dn[frame])} raw DN.<br>Original motion ID: ${s.old_id??'new track'}; old score ${number(s.old_motion_score,3)}.<br>Raw maximum: ${s.raw_peak}/4095 (${number(s.raw_peak*255/4095)}/255).<br>Current peak 2×2 mean: ${number(s.peak_cell_255[frame])}/255; whole-movie max ${number(s.peak_cell_max_255)}/255.<br>Current θ ${number(q.theta[frame],1)}°, r ${number(q.r[frame],3)}; X ${number(xy?.[0],3)}, Y ${number(xy?.[1],3)}.<br>Aperture mean: ${number(q.intensity[frame])}/255.<br>Dimmest: ${v?'frame '+v.frame+', θ '+number(v.theta,1)+'°':'unavailable'}.`;
 plot('trace',[{name:'18×18 mean intensity',color:'#176d91',points:q.intensity.map((v,i)=>[i+1,v])}],D.frames,'Frame','Intensity · 0–255 equivalent mean / sensor pixel',{vline:frame+1});
 plot('anisotropy',[{name:'X',color:'#176d91',points:q.xy.map((v,i)=>[i+1,v?.[0]])},{name:'Y',color:'#c36225',points:q.xy.map((v,i)=>[i+1,v?.[1]])}],D.frames,'Frame','Single-frame anisotropy',{lo:-1,hi:1,vline:frame+1});}
'''+js[b:]
    js=js.replace("rows().filter(s=>s.statistics_kept)","rows().filter(eligible)")
    a=js.index('function table()');b=js.index('async function loadCapture',a)
    js=js[:a]+'''function table(){const ss=rows();$('rows').innerHTML=ss.map(s=>`<tr data-id="${s.id}"><td>${s.id}</td><td>${rule(s).detected_frames}/${D.frames}</td><td>${rule(s).rest_positive_frames}/${rule(s).rest_frames}</td><td>${status(s)}</td><td>${number(sel(s,'first','theta'),1)}°</td><td>${number(sel(s,'dimmest','theta'),1)}°</td></tr>`).join('');}
function refresh(){const ss=rows();if(!ss.some(s=>s.id===selectedID)&&ss.length)selectedID=ss[0].id;const k=$('k').value,mode=$('support').value,z=D.summary[k][mode];$('counts').textContent=`${z.accepted} accepted tracks; ${z.statistics} eligible after saturation / overlap; ${z.saturated} saturated. ${ss.length} catalogue entries displayed. Default is ≥5 detections at 1 SD plus positive local contrast in every remaining frame.`;
 const key=mode==='90'?'ninety':mode,counts=D.wrong_time_controls.map(c=>c[k][key]);$('controlinfo').textContent=`Shuffled-time path accepted counts at these settings: ${counts.join(', ')}. Real path: ${z.accepted}. Controls apply the full discovery/linking/support rule, before saturation/overlap exclusions. These are diagnostic controls, not a formal false-discovery rate.`;draw();details();histograms();table();}
'''+js[b:]
    js=js.replace("s.src=stamp+'/view.js'","s.src='repeated/'+stamp+'/view.js'")
    js=replace_once(js,"$('detectionimg').src=stamp+'/detection.png';$('examplesimg').src=stamp+'/examples.png';",'')
    js=js.replace("$('jsonlink').href=stamp+'/results.json'","$('jsonlink').href='repeated/'+stamp+'/results.json'").replace("$('csvlink').href=stamp+'/candidates.csv'","$('csvlink').href='repeated/'+stamp+'/candidates.csv'")
    js=js.replace("['filter','score']","['filter','k','support']")
    js=replace_once(js,"if(D.spots.some(s=>s.id===n)){selectedID=n;draw();details();}","if(D.spots.some(s=>s.id===n)){selectedID=n;if(!rows().some(s=>s.id===n)){$('filter').value='all';refresh();}draw();details();}")
    (ROOT/'repeat_viewer.js').write_text(js,encoding='utf8')
    print('Updated',ROOT/'index.html',flush=True)

if __name__=='__main__':main()
