"""Read-only saturation sensitivity and matched-position cancellation comparison.

Run with the repository Python. --render-only reuses the numerical results.
Historical background-subtracted images cannot certify absence of raw clipping.
The September exclusions below are explicitly sensitivity screens, not raw flags.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
from html import escape
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline_audit import GRID, density, describe
from first_hinges_theta import clean_json

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
OUT = BASE / 'saturation_interference_check'
DATA = REPO / 'data/hinges/polarcam'
STAMPS = ('145406','151106','151218','152241','152256','152740','154622','155126')
PAIRS = (('172515','172833'), ('174418','174545'))
COLORS = ('#687787', '#147c91', '#d05b32')


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def frames_and_meta(path):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    m = load_json(path.with_suffix('.json'))
    n = len(a) - int(m['actual']['phase_marker_appended'])
    assert n == m['actual']['frames']
    return a[:n], m


def aperture_max(peak, points_yx, width):
    result = []
    for y, x in points_yx:
        top, left = int(round(y))-width//2, int(round(x))-width//2
        crop = peak[top:top+width, left:left+width]
        assert crop.shape == (width, width)
        result.append(crop.max())
    return np.asarray(result)


def reject_raw_saturation(raw_peak, points_yx, width, ceiling=4094):
    """Whole-spot veto: ANY ROI pixel reaches the conservative Mono12 ceiling.

    Input MUST be an unsubtracted temporal maximum, excluding marker frames.
    4094 is a one-code safety margin below nominal 4095, motivated by repeated
    4094 maxima in these camera files. Do not pass a background-subtracted peak.
    """
    return aperture_max(raw_peak, points_yx, width) >= ceiling


def summary(theta):
    theta = np.asarray(theta, float)
    theta = theta[np.isfinite(theta)]
    return describe(theta) if len(theta) else dict(n=0)


def september():
    prior = load_json(BASE/'theta_comparison/results.json')
    state, csv_rows = [], []
    for stamp in STAMPS:
        path = DATA/'2026-09-18'/f'frame_stack_20260918-{stamp}.npy'
        before = path.stat()
        a, meta = frames_and_meta(path)
        assert len(a) == 150 and meta['background']['background_subtracted']
        z = np.load(BASE/'pipeline_audit'/f'comparison_{stamp}.npz')
        previous = next(s for s in prior['stacks'] if s['stamp'] == stamp)
        peak = np.zeros(a.shape[1:], a.dtype)
        for j, frame in enumerate(a):
            np.maximum(peak, frame, out=peak)
            if j == 4: peak5 = peak.copy()
        # Plateau evidence: several repeats of exactly the same high pixel maximum.
        yy, xx = np.where(peak >= 3500)
        count = np.zeros(len(yy), np.uint16)
        for frame in a:
            count += frame[yy,xx] == peak[yy,xx]
        plateau = np.zeros(peak.shape, bool)
        plateau[yy,xx] = count >= 5
        example = None
        if len(count):
            k = int(count.argmax())
            example = dict(x=int(xx[k]),y=int(yy[k]),maximum=int(peak[yy[k],xx[k]]),
                           repeats=int(count[k]),values=a[:,yy[k],xx[k]].tolist())
        pipelines = {}
        for name, pts, width, theta, valid_before in (
            ('attached',z['old_points'],7,z['t_old'],np.isfinite(z['t_old'])),
            ('native',z['native_points'],18,
             np.array([r['theta_deg'] if r['theta_deg'] is not None else np.nan for r in previous['rows']]),
             z['native_keep'])):
            maximum = aperture_max(peak, pts, width)
            maximum5 = aperture_max(peak5, pts, width)
            plateau_spot = aperture_max(plateau, pts, width).astype(bool)
            # Respect metadata proving this entire recording stayed below the guard.
            raw_below_guard = meta['actual']['max_raw_value'] < 4094
            masks = dict(before=valid_before,
                screen4080=valid_before & (maximum < 4080),
                screen4000=valid_before & (maximum < 4000),
                screen4000_or_plateau=valid_before & (maximum < 4000) & ~plateau_spot,
                first5_screen4000=valid_before & (maximum5 < 4000))
            np.savez_compressed(OUT/f'{stamp}_{name}.npz',theta=theta,points_yx=pts,
                saved_peak150=maximum,saved_peak5=maximum5,high_plateau=plateau_spot,**masks)
            for i, ((y,x),t,mx,m5,pl) in enumerate(zip(pts,theta,maximum,maximum5,plateau_spot)):
                csv_rows.append(dict(stack=stamp,pipeline=name,spot_id=i+1,x=x,y=y,width=width,
                    theta=t,previously_included=bool(valid_before[i]),saved_peak150=int(mx),
                    saved_peak5=int(m5),high_plateau=bool(pl),retained_screen4000=bool(masks['screen4000'][i]),
                    raw_status='below_guard_entire_recording' if raw_below_guard else 'unknown_after_background_subtraction'))
            pipelines[name] = {k:summary(theta[v]) for k,v in masks.items()}
            print(stamp, name, {k:int(v.sum()) for k,v in masks.items()}, flush=True)
        state.append(dict(stamp=stamp,max_raw=meta['actual']['max_raw_value'],
            max_saved=int(peak.max()),pixels_ge4000=int((peak>=4000).sum()),
            high_plateau_pixels=int(plateau.sum()),example=example,pipelines=pipelines,
            source_size=before.st_size,source_mtime_ns=before.st_mtime_ns,
            metadata_sha256=hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest()))
        assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
    write_csv('september_spots.csv',csv_rows)
    return state


def xy(ch):
    return np.c_[(ch[:,0]-ch[:,1])/(ch[:,0]+ch[:,1]),
                 (ch[:,2]-ch[:,3])/(ch[:,2]+ch[:,3])]


def extract(a, roi, subtract=None):
    size = int(roi['win_raw'])
    ox,oy = int(roi['x']),int(roi['y'])
    left = max(0,min(a.shape[2]-size,round(roi['cx']-ox)-size//2))
    top = max(0,min(a.shape[1]-size,round(roi['cy']-oy)-size//2))
    channels = []
    for start in range(0,len(a),2048):
        crop = np.asarray(a[start:start+2048,top:top+size,left:left+size],float)
        if subtract is not None:
            crop = np.maximum(0,np.rint(crop-subtract[oy+top:oy+top+size,ox+left:ox+left+size]))
        channels.append(np.stack([crop[:,(pr-oy-top)%2::2,(pc-ox-left)%2::2].mean((1,2))
                                 for pr,pc in ((1,1),(0,0),(0,1),(1,0))],axis=-1))
    return np.concatenate(channels),[ox+left,oy+top,size]


def block_xy(ch, block):
    n = len(ch)//block*block
    return xy(ch[:n].reshape(-1,block,4).mean(1))


def metrics(ch, fps):
    result = dict(mean_channel_sum=float(ch.sum(1).mean()),frames=len(ch),duration_s=len(ch)/fps)
    for block in (1,15,153):
        v = block_xy(ch,block)
        result[f'block{block}'] = dict(dt_ms=1000*block/fps,
            std=v.std(0,ddof=1).tolist(),
            successive_difference_proxy=(np.diff(v,axis=0).std(0,ddof=1)/np.sqrt(2)).tolist(),
            central96_width=(np.percentile(v,98,axis=0)-np.percentile(v,2,axis=0)).tolist())
    freq,power = signal.welch(xy(ch),fs=fps,nperseg=16384,axis=0,detrend='constant')
    for name,lo,hi in (('slow_0p3_3Hz',.3,3),('fast_100_500Hz',100,500)):
        mask=(freq>=lo)&(freq<=hi)
        result[name+'_rms']=np.sqrt(np.sum(power[mask],axis=0)*(freq[1]-freq[0])).tolist()
    return result


def cancellation():
    pairs=[]
    for i, stamps in enumerate(PAIRS):
        records=[]
        for stamp in stamps:
            path=DATA/'2026-10-09'/f'spotrec_20261009-{stamp}.npy'
            before=path.stat()
            a,m=frames_and_meta(path)
            assert m['actual']['max_raw_value'] < 4094  # proves no clipping anywhere in capture
            ch,bounds=extract(a,m['roi'])
            cached=np.load(BASE/'traces'/f'{path.stem}.npz')['channels']
            np.testing.assert_allclose(ch,cached,rtol=0,atol=1e-12)
            extras={}
            if m['background']['background_subtracted'] is False:
                for mode,stampbg in [('narrowfield','170656'),('widefield','171808')]:
                    bgpath=DATA/'2026-10-09/backgrounds'/f'{mode}_background_min_20261009-{stampbg}_exp0.6ms_gain1.npy'
                    extras[mode],_=extract(a,m['roi'],np.load(bgpath))
            records.append(dict(stamp=stamp,cancellation=m['sound']['sound_on'],
                background_subtracted=m['background']['background_subtracted'],
                metadata=m,bounds=bounds,channels=ch,extras=extras))
            assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
        a,b=records
        for k in ('fps','gains','timing'):
            assert a['metadata']['actual'][k] == b['metadata']['actual'][k]
        assert a['bounds']==b['bounds']
        assert a['cancellation'] != b['cancellation']
        fps=a['metadata']['actual']['fps']; n=min(len(r['channels']) for r in records)
        pair=dict(id=i+1,bounds=a['bounds'],fps=fps,matched_frames=n,matched_duration_s=n/fps,
                  same_saved_background_setting=a['background_subtracted']==b['background_subtracted'],records=[])
        arrays={}
        for r in records:
            ch=r['channels'][:n]
            arrays['channels_'+r['stamp']]=ch
            data=dict(stamp=r['stamp'],cancellation=r['cancellation'],
                background_subtracted=r['background_subtracted'],max_raw=r['metadata']['actual']['max_raw_value'],
                exposure_us=r['metadata']['actual']['timing']['exposure_us'],
                total_frames=len(r['channels']),full_record_metrics=metrics(r['channels'],fps),
                metrics=metrics(ch,fps),sensitivity={})
            for mode,adjusted in r['extras'].items():
                arrays['channels_'+r['stamp']+'_'+mode]=adjusted[:n]
                data['sensitivity'][mode+'_subtracted']=metrics(adjusted[:n],fps)
            pair['records'].append(data)
        on=next(r for r in pair['records'] if r['cancellation'])['metrics']
        off=next(r for r in pair['records'] if not r['cancellation'])['metrics']
        pair['on_over_off']={}
        for block in (1,15,153):
            pair['on_over_off'][f'block{block}']={k:(np.array(on[f'block{block}'][k])/off[f'block{block}'][k]).tolist()
                for k in ('std','successive_difference_proxy')}
        print('PAIR',pair['id'],'on/off',pair['on_over_off'],flush=True)
        np.savez_compressed(OUT/f'pair_{i+1}.npz',**arrays)
        pairs.append(pair)
    return pairs


def write_csv(filename,rows):
    with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def spectral_checks(state):
    """Extend diagnostics from the validated channel caches, without rereading data."""
    for p in state['pairs']:
        z=np.load(OUT/f"pair_{p['id']}.npz")
        for r in p['records']:
            freq,power=signal.welch(xy(z['channels_'+r['stamp']]),fs=p['fps'],nperseg=16384,axis=0)
            mask=freq>20
            selected=np.flatnonzero(mask)[np.argmax(power[mask],axis=0)]
            r['spectral_checks']=dict(
                strongest_peak_above20Hz=freq[selected].tolist(),
                peak_psd=power[selected,np.arange(2)].tolist())
            for name,low,high in [('around403Hz',402,404),('around720Hz',714,724)]:
                mask=(freq>=low)&(freq<=high)
                r['spectral_checks'][name+'_rms']=np.sqrt(power[mask].sum(0)*(freq[1]-freq[0])).tolist()


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def figures(state):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    for name,title in [('attached','Attached pipeline: 7 × 7 ROI, original first-five-frame θ'),
                       ('native','Current pipeline: 18 × 18 ROI, 150-frame θ + local contrast')]:
        fig,axs=plt.subplots(2,4,figsize=(15,7),sharex=True)
        for ax,s in zip(axs.flat,state['september']):
            z=np.load(OUT/f"{s['stamp']}_{name}.npz")
            for mask,color,label in zip(('before','screen4080','screen4000'),COLORS,
                                       ('Before','Exclude ≥4080','Exclude ≥4000')):
                v=z['theta'][z[mask]]
                if len(v):ax.plot(GRID,density(v),color=color,lw=1.6,label=f'{label}: n={len(v)}')
            ax.axvspan(35,45,color='#dfac1e',alpha=.1)
            ax.set(title=s['stamp'],xlabel='θ (degrees)',ylabel='Density per degree',xlim=(0,90))
            ax.legend(fontsize=7)
        fig.suptitle(title+'\nWhole-spot screen uses any saved pixel in ALL 150 frames; clipping status remains uncertain',fontsize=13)
        fig.tight_layout();save(fig,'september_'+name)
    fig,axs=plt.subplots(2,4,figsize=(15,6))
    for ax,s in zip(axs.flat,state['september']):
        e=s['example']
        if e:
            ax.plot(np.arange(150),e['values'],lw=.8)
            ax.set(title=f"{s['stamp']} · ({e['x']}, {e['y']})\nmax {e['maximum']}, repeated {e['repeats']}/150")
        else:ax.set_title(s['stamp']+' · no saved pixel ≥3500')
        ax.set(xlabel='Measurement frame',ylabel='Saved pixel code')
    fig.suptitle('High pixel plateaus: evidence to inspect, not a recovered raw saturation mask')
    fig.tight_layout();save(fig,'plateau_examples')
    for p in state['pairs']:
        z=np.load(OUT/f"pair_{p['id']}.npz");fps=p['fps']
        fig,axs=plt.subplots(2,2,figsize=(13,7),sharex=True,sharey='col')
        fig2,ax2=plt.subplots(1,2,figsize=(12,4))
        fig3,ax3=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for row,r in enumerate(p['records']):
            ch=z['channels_'+r['stamp']];v=block_xy(ch,15);t=(np.arange(len(v))+.5)*15/fps
            color='#147c91' if r['cancellation'] else '#d05b32'
            label=f"{r['stamp']} · cancellation {'ON' if r['cancellation'] else 'OFF'}"
            for j in range(2):
                axs[row,j].plot(t,v[:,j],lw=.6,color=color)
                axs[row,j].set(title=label,xlabel='Time from recording start (s)',ylabel=('X','Y')[j])
                freq,power=signal.welch(xy(ch)[:,j],fs=fps,nperseg=16384)
                ax2[j].loglog(freq[1:],power[1:],label=label,color=color,lw=1)
                ax2[j].set(xlabel='Frequency (Hz)',ylabel=f'{("X","Y")[j]} PSD (1/Hz)',xlim=(.15,fps/2))
                small=xy(ch[:int(fps)])
                ax3[j].plot(np.arange(len(small))/fps,small[:,j]-small[:,j].mean(),lw=.45,color=color,label=label)
                ax3[j].set(xlabel='First second of recording (s)',ylabel=f'{("X","Y")[j]} minus first-second mean')
        caveat='same saved BG setting' if p['same_saved_background_setting'] else 'saved BG subtraction also changed'
        fig.suptitle(f"Pair {p['id']} · {15/fps*1000:.2f} ms channel averages · {caveat}")
        fig.tight_layout();save(fig,f"pair_{p['id']}_traces")
        for ax in ax2:ax.legend(fontsize=8)
        fig2.suptitle(f"Pair {p['id']} · native-rate spectra over equal {p['matched_duration_s']:.1f} s");fig2.tight_layout();save(fig2,f"pair_{p['id']}_psd")
        for ax in ax3:ax.legend(fontsize=8)
        fig3.suptitle(f"Pair {p['id']} · unaveraged X/Y detail (different recording times)");fig3.tight_layout();save(fig3,f"pair_{p['id']}_native")


def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def page(state):
    parts=['''<!doctype html><html><meta charset="utf-8"><title>Saturation and cancellation checks</title>
    <style>body{font:16px/1.55 system-ui;max-width:1250px;margin:35px auto;padding:0 22px;color:#233642}h1,h2{line-height:1.2}h2{margin-top:45px}img{width:100%;height:auto}td,th{padding:8px;border-bottom:1px solid #dae0e3;text-align:left}table{border-collapse:collapse;font-size:14px;width:100%}a{color:#087d99}.note{background:#f0f5f6;padding:16px}summary{cursor:pointer}</style>
    <h1>Saturation and cancellation checks</h1>
    <p>18 September 2026 frame stacks and 9 October 2026 matched-position spot recordings. Original data and earlier results are unchanged.</p>
    <p class="note"><b>Results:</b> near-ceiling rejection barely changes September’s θ distributions; the attached pipeline’s 40–43° peaks remain. Cancellation ON gives quieter ~10 ms traces in both pairs, but the fast fluctuations do not consistently improve. The cleaner late pair has 14–15% lower total X/Y SD, with fast changes of +7% (X) and −6% (Y). Both ON recordings also contain narrow high-frequency spectral peaks. These data support reduced slow variation, not a demonstrated improvement at every timescale.</p>
    <h2>1 · Neither previous pipeline excluded saturated spots</h2>
    <p>The attached pipeline flagged saved pixels ≥4080 in its first five frames, but never used that flag to reject a spot. Its hot-pixel replacement affects detection only. The current native + local-contrast pipeline also reported bright pixels without an exclusion. Neither implements the requested whole-spot veto.</p>
    <p class="note"><b>Historical limitation:</b> all eight September stacks were saved after background subtraction. Seven report a raw maximum of 4094, one code below the nominal Mono12 ceiling (4095); 15:27:40 reports 3774. No exact September background or per-pixel raw maximum is linked. A saved pixel below 4095 does not prove that the original pixel was unsaturated. The repeated 4094 maxima justify a conservative raw guard of ≥4094, but do not establish the camera’s physical clipping threshold.</p>
    <p><b>Test performed:</b> hold each pipeline’s coordinates, extraction, θ model and existing exclusions fixed. Remove a whole spot if any pixel in its extraction square reaches the saved-code screen in any of the 150 measurement frames. The synthetic final marker is excluded. Curves compare ≥4080 and a wider ≥4000 screen (97.7% of nominal full scale). These are sensitivity screens, not certified saturation-free populations. The attached pipeline’s θ values still use its original first five frames.</p>''']
    rows=[]
    for s in state['september']:
        for name in ('attached','native'):
            v=s['pipelines'][name];a=v['before'];b=v['screen4000']
            rows.append([s['stamp'],name,a['n'],a['n']-b['n'],b['n'],
                f"{a.get('median',float('nan')):.2f} → {b.get('median',float('nan')):.2f}",
                ', '.join(f'{x:.1f}' for x in a.get('modes',[]))+' → '+', '.join(f'{x:.1f}' for x in b.get('modes',[]))])
    parts.append('<p>The ≥4000 screen removes 19/11,730 previously valid attached-pipeline spots (0.16%) and 23/2,684 current-pipeline spots (0.86%). Adding the high-plateau screen removes one more spot from each. In the attached pipeline, the four low-angle peaks change from 40.1°, 43.3°, 42.3°, 40.2° to 40.1°, 43.3°, 42.4°, 40.2°. The curves nearly overlap. This argues against the identifiable near-ceiling spots explaining the earlier pipeline discrepancy.</p>')
    parts.append(table(['Stack','Pipeline','Before','Removed ≥4000','After','Median θ','KDE modes before → after'],rows))
    for name in ('attached','native'):parts.append(f'<a href="september_{name}.pdf"><img src="september_{name}.png"></a>')
    parts.append('<details><summary>Inspect high-pixel plateaus and stricter sensitivity</summary><p>The additional CSV/JSON screen rejects either ≥4000 or a pixel ≥3500 repeating its exact maximum in at least five frames. Repeated maxima can support clipping suspicion, but this heuristic is not proof. Exact raw veto code is included for recordings with an unsubtracted maximum projection; it is deliberately not misapplied to these saved images.</p><img src="plateau_examples.png"></details>')
    parts.append('''<h2>2 · Same-position cancellation ON/OFF</h2><p>The user confirmed that “sound on” means cancellation enabled. Each pair has exactly matching recorded spot centre, 20 × 20 extraction square, exposure (0.598 ms), gain (1 × 1) and frame rate (1526.403 Hz). All four raw maxima are well below the conservative saturation guard. Matching position supports a repeat-spot comparison; it cannot independently prove molecular identity.</p>
    <p>Use the same number of initial frames from each recording (42.0 s for pair 1, 46.2 s for pair 2), the same sensor parity and extraction aperture. Average the four channels before calculating X=(I0−I90)/(I0+I90), Y=(I45−I135)/(I45+I135). Main traces use 15-frame / 9.83 ms bins. Ratios below 1 mean less variation with cancellation enabled.</p>''')
    for p in state['pairs']:
        parts.append(f"<h3>Pair {p['id']}: {' → '.join(r['stamp'] for r in p['records'])}</h3>")
        if not p['same_saved_background_setting']:
            parts.append('<p class="note">Confounded comparison: cancellation ON was saved without background subtraction; cancellation OFF was saved with subtraction. The exact historical profile is unrecorded. Both preserved narrow- and wide-field profiles are tried below on the raw ON recording as sensitivity checks; neither is asserted to be the actual OFF profile.</p>')
        else:
            parts.append('<p>Both recordings were saved with background subtraction. The old JSON stores a generic profile path, so identical profile contents cannot be verified.</p>')
        rows=[]
        for r in p['records']:
            m=r['metrics'];block=m['block15'];fast=m['block1']['successive_difference_proxy']
            rows.append([r['stamp'],'ON' if r['cancellation'] else 'OFF',r['background_subtracted'],r['max_raw'],
                f"{m['mean_channel_sum']:.1f}",*[f'{x:.5f}' for x in block['std']],*[f'{x:.5f}' for x in fast]])
        parts.append(table(['Time','Cancellation','Saved BG sub.','Raw max','Mean Σchannels','SD X (9.83 ms)','SD Y (9.83 ms)','Fast proxy X','Fast proxy Y'],rows))
        rat=p['on_over_off'];parts.append('<p><b>ON/OFF ratios:</b> 9.83 ms total SD X/Y = '+
            ' / '.join(f'{v:.3f}' for v in rat['block15']['std'])+'; native successive-difference proxy X/Y = '+
            ' / '.join(f'{v:.3f}' for v in rat['block1']['successive_difference_proxy'])+'.</p>')
        if p['id']==1:
            parts.append('<p>At ~10 ms, cancellation ON has 57% lower X SD and 35% lower Y SD. At native sampling, the successive-difference proxy is 57% higher for X and 189% higher for Y. Background subtraction sensitivity checks retain this qualitative contrast. This pair therefore does not show a uniform improvement in precision.</p>')
        else:
            parts.append('<p>At ~10 ms, cancellation ON has 14% lower X SD and 15% lower Y SD. The native successive-difference proxy is 7% higher for X and 6% lower for Y; the mean channel sum is about 8% lower. The main improvement is in slower variation.</p>')
        on=next(r for r in p['records'] if r['cancellation'])
        peaks=on['spectral_checks']['strongest_peak_above20Hz']
        parts.append('<p>In the ON recording, prominent narrow peaks appear near '+
            f'{peaks[0]:.1f} Hz in X and {peaks[1]:.1f} Hz in Y. They are not comparably prominent in the OFF recording. These are frequencies in the sampled signal; the source and any aliasing are unresolved. The spectra below extend to the full Nyquist frequency.</p>')
        parts.append(f'<a href="pair_{p["id"]}_traces.pdf"><img src="pair_{p["id"]}_traces.png"></a>')
        parts.append(f'<details><summary>Native traces and frequency spectra</summary><img src="pair_{p["id"]}_native.png"><img src="pair_{p["id"]}_psd.png"></details>')
        for r in p['records']:
            if r['sensitivity']:
                rows=[]
                for name,m in r['sensitivity'].items():
                    rows.append([name,*[f'{x:.5f}' for x in m['block15']['std']],*[f'{x:.5f}' for x in m['block1']['successive_difference_proxy']]])
                parts.append('<details><summary>Pair 1 background sensitivity</summary>'+table(['Profile applied to ON','SD X','SD Y','Fast proxy X','Fast proxy Y'],rows)+'</details>')
    parts.append('''<p class="note"><b>Interpretation:</b> total SD includes real rod motion, drift and instrumental noise. The fast proxy is SD of successive native samples divided by √2; it is sensitive to fast noise but also real rapid motion and temporal correlation. Lower values demonstrate a quieter measured trace at the stated timescale, not a calibrated angular resolution or proof of cancellation causality. Background identity, illumination stability and repeat order were not fully controlled. Do not treat thousands of frames as independent replicate experiments.</p>
    <p><a href="results.json">Numerical results and full-duration checks</a> · <a href="september_spots.csv">Per-spot flags</a> · <a href="../saturation_interference_check.py">Reproduction script</a></p></html>''')
    (OUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if '--render-only' in sys.argv:state=load_json(OUT/'results.json')
    else:
        # A marker outside these peaks cannot reject a spot. A single saturated
        # edge pixel must reject it even if every other ROI pixel is dim.
        test=np.zeros((30,30),np.uint16);test[6,6]=4094
        assert reject_raw_saturation(test,np.array([[9,9],[20,20]]),7).tolist()==[True,False]
        state=dict(september=september(),pairs=cancellation())
        (OUT/'results.json').write_text(json.dumps(clean_json(state),indent=2,allow_nan=False),encoding='utf-8')
    spectral_checks(state)
    (OUT/'results.json').write_text(json.dumps(clean_json(state),indent=2,allow_nan=False),encoding='utf-8')
    figures(state);page(state)
