"""Exploratory control subtraction from saved Figure 1/2 curves; no new detection."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def build():
    out=Path(__file__).resolve().parent/'richard_email_20261014'
    source=out/'figure_01_results.json'
    saved=json.loads(source.read_text(encoding='utf8'))
    assert saved['detection_background']==saved['extraction_background']=='minimum'
    dna=saved['groups']['DNA'];gold=saved['groups']['Gold']
    assert dna['stacks']==16 and gold['stacks']==14
    d=dna['r_histogram'];g=gold['r_histogram']
    np.testing.assert_array_equal(d['bin_edges'],g['bin_edges'])
    np.testing.assert_array_equal(d['curve_x'],g['curve_x'])
    edges=np.array(d['bin_edges']);x=np.array(d['curve_x'])
    dna_rate=dna['retained']/dna['stacks'];gold_rate=gold['retained']/gold['stacks']
    net_rate=dna_rate-gold_rate;factor=dna_rate/gold_rate
    dna_counts=np.array(d['counts'])/dna['stacks']
    gold_counts=np.array(g['counts'])/gold['stacks']
    dna_smooth=np.array(d['curve_percent_per_bin'])*dna_rate/100
    gold_smooth=np.array(g['curve_percent_per_bin'])*gold_rate/100
    residual=dna_counts-gold_counts;residual_smooth=dna_smooth-gold_smooth
    requested_scaled_percent=factor*np.array(d['percent'])-np.array(g['percent'])
    np.testing.assert_allclose(residual,requested_scaled_percent*gold_rate/100,atol=1e-12)
    np.testing.assert_allclose(residual.sum(),net_rate,atol=1e-10)
    np.testing.assert_allclose(residual_smooth.sum()*(x[1]-x[0])/.02,net_rate,atol=1e-10)
    peaks,_=find_peaks(residual_smooth,prominence=1,distance=80)
    peaks=[dict(r=float(x[i]),excess_per_stack_per_bin=float(residual_smooth[i])) for i in peaks if residual_smooth[i]>0]
    result=dict(source=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        baseline='Frame 1, minimum BG for both detection and r, 2SD, oversized components allowed, original corner/overlap/raw-saturation exclusions. All finite radii retained.',
        scope='Exploratory arithmetic outside the email draft. No new spot detection, no fitted control weight, no statistical test.',
        assumptions='The detected non-hinge contribution in DNA has the same r distribution AND mean detections per field as gold control. Comparable sampled areas, acquisition and detection response are assumed. These are assumptions about detected features, not verified particle identities.',
        formula='Residual_i = DNA_count_i/16 - Gold_count_i/14. Equivalent in shape to factor * DNA_percent_i - Gold_percent_i, multiplied by Gold mean detections/stack /100.',
        dna=dict(stacks=16,detections=dna['retained'],mean_per_stack=dna_rate),
        gold=dict(stacks=14,detections=gold['retained'],mean_per_stack=gold_rate),
        scale_factor=factor,net_excess_per_stack=net_rate,
        implied_control_fraction_of_dna=gold_rate/dna_rate,implied_excess_fraction_of_dna=net_rate/dna_rate,
        positive_bin_total_per_stack=float(residual[residual>0].sum()),negative_bin_total_per_stack=float(residual[residual<0].sum()),
        bin_edges=edges.tolist(),dna_per_stack_per_bin=dna_counts.tolist(),gold_per_stack_per_bin=gold_counts.tolist(),
        signed_excess_per_stack_per_bin=residual.tolist(),requested_scaled_percent_difference=requested_scaled_percent.tolist(),
        signed_percent_of_net_excess=(residual/net_rate*100).tolist(),curve_x=x.tolist(),
        signed_excess_smooth=residual_smooth.tolist(),descriptive_r_peaks=peaks,
        interpretation='Negative bins are retained. They prevent interpreting this signed estimate directly as a nonnegative hinge-only distribution; mismatched assumptions and finite-sample fluctuations may contribute. Do not clip negatives and relabel the result as hinges.',
        smoothing='Reuse Figure 1 smoothed curves (Gaussian sigma=0.02r); subtract linearly. Peak prominence >=1 excess detection/stack/bin and minimum separation 0.08r, descriptive only.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),layout='constrained')
    for counts,smooth,color,label in [(dna_counts,dna_smooth,'#166888',f'DNA: {dna_rate:.1f} detections/stack'),
                                     (gold_counts,gold_smooth,'#c16b2a',f'Gold: {gold_rate:.1f} detections/stack')]:
        axes[0].stairs(counts,edges,color=color,alpha=.35,lw=.9)
        axes[0].plot(x,smooth,color=color,lw=2,label=label)
    axes[0].set(title='A · Put both groups on a per-stack scale',ylabel='Detections per stack per 0.02 r bin',ylim=(0,75))
    axes[0].legend(loc='upper right',frameon=False)
    axes[1].stairs(residual,edges,color='#63549a',alpha=.5,lw=.9)
    axes[1].plot(x,residual_smooth,color='#63549a',lw=2,label='DNA per stack − gold per stack')
    axes[1].fill_between(x,0,residual_smooth,where=residual_smooth>=0,color='#63549a',alpha=.13,interpolate=True)
    axes[1].fill_between(x,0,residual_smooth,where=residual_smooth<0,color='#be5964',alpha=.23,interpolate=True)
    axes[1].axhline(0,color='.4',lw=.8)
    for p in peaks:
        axes[1].plot(p['r'],p['excess_per_stack_per_bin'],'o',color='#63549a',ms=4)
        axes[1].annotate(f'r ≈ {p["r"]:.2f}',(p['r'],p['excess_per_stack_per_bin']),xytext=(0,8),
                         textcoords='offset points',ha='center',color='#63549a')
    axes[1].set(title=f'B · Subtract control: net {net_rate:.1f} detections/stack',ylabel='Signed excess per stack per 0.02 r bin',ylim=(-25,30))
    axes[1].legend(loc='upper left',frameon=False,fontsize=9)
    axes[1].text(.02,.04,'Negative bins kept; this is not a probability distribution.',transform=axes[1].transAxes,fontsize=9,color='#8b414b')
    for ax in axes:
        ax.set(xlim=(0,1),xlabel='Anisotropy radius r')
        ax.axvspan(saved['r_limit'],1,color='.95',zorder=-3)
        ax.axvline(saved['r_limit'],color='.5',ls=':',lw=1)
        ax.text(saved['r_limit']+.012,.5,f'r_lim = {saved["r_limit"]:.3f}',transform=ax.get_xaxis_transform(),rotation=90,color='.45',fontsize=9,va='center')
    fig.suptitle('Exploratory subtraction · Assume equal non-hinge detections per field\n16 DNA stacks and 14 gold stacks · Minimum BG · 2 SD · Frame 1',fontsize=14)
    fig.savefig(out/'exploratory_control_subtraction.png',dpi=240,facecolor='white');plt.close(fig)
    (out/'exploratory_control_subtraction.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:result[k] for k in ['scale_factor','net_excess_per_stack','implied_control_fraction_of_dna','positive_bin_total_per_stack','negative_bin_total_per_stack','descriptive_r_peaks']},indent=2))


def build_all():
    out=Path(__file__).resolve().parent/'richard_email_20261014'
    source=out/'figure_02_results.json'
    saved=json.loads(source.read_text(encoding='utf8'))
    assert sum(r['group']=='DNA' for r in saved['pooled_records'])==16
    assert sum(r['group']=='Gold' for r in saved['pooled_records'])==14
    settings=[('minimum','Minimum BG · 2 SD','background_comparison','minimum'),
              ('median','Median (50th-percentile) BG · 2 SD','background_comparison','median'),
              ('p95','95th-percentile BG · 2 SD','background_comparison','p95'),
              ('sd4','Minimum BG · 4 SD','threshold_comparison','4'),
              ('sd8','Minimum BG · 8 SD','threshold_comparison','8')]
    results={}
    for name,label,kind,key in settings:
        dna=saved[kind]['DNA'][key];gold=saved[kind]['Gold'][key]
        d=dna['histogram'];g=gold['histogram']
        np.testing.assert_array_equal(d['bin_edges'],g['bin_edges'])
        np.testing.assert_array_equal(d['curve_x'],g['curve_x'])
        edges=np.array(d['bin_edges']);x=np.array(d['curve_x'])
        dr=dna['n']/16;gr=gold['n']/14;factor=dr/gr;net=dr-gr
        dc=np.array(d['counts'])/16;gc=np.array(g['counts'])/14
        ds=np.array(d['curve_percent_per_bin'])*dr/100
        gs=np.array(g['curve_percent_per_bin'])*gr/100
        residual=dc-gc;smooth=ds-gs
        scaled_percent=factor*np.array(d['percent'])-np.array(g['percent'])
        np.testing.assert_allclose(residual,scaled_percent*gr/100,atol=1e-12)
        np.testing.assert_allclose(residual.sum(),net,atol=1e-10)
        np.testing.assert_allclose(smooth.sum()*(x[1]-x[0])/.02,net,atol=1e-10)
        peaks,_=find_peaks(smooth,prominence=1,distance=80)
        peaks=[dict(r=float(x[i]),excess_per_stack_per_bin=float(smooth[i])) for i in peaks if smooth[i]>0]
        results[name]=dict(label=label,source_comparison=kind,source_variant=key,
            dna_count=dna['n'],gold_count=gold['n'],dna_mean_per_stack=dr,gold_mean_per_stack=gr,
            scale_factor=factor,net_excess_per_stack=net,
            positive_bin_total_per_stack=float(residual[residual>0].sum()),
            negative_bin_total_per_stack=float(residual[residual<0].sum()),
            bin_edges=edges.tolist(),dna_per_stack_per_bin=dc.tolist(),gold_per_stack_per_bin=gc.tolist(),
            signed_excess_per_stack_per_bin=residual.tolist(),requested_scaled_percent_difference=scaled_percent.tolist(),
            signed_percent_of_net_excess=(residual/net*100).tolist(),curve_x=x.tolist(),
            signed_excess_smooth=smooth.tolist(),descriptive_r_peaks=peaks)
    previous_path=out/'exploratory_control_subtraction.json'
    if previous_path.exists():
        prior=json.loads(previous_path.read_text())
        for key in ['bin_edges','signed_excess_per_stack_per_bin','curve_x','signed_excess_smooth']:
            np.testing.assert_allclose(results['minimum'][key],prior[key],atol=1e-12)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(16,9.5),layout='constrained')
    grid=fig.add_gridspec(2,6)
    cells=[(0,0),(0,2),(0,4),(1,1),(1,3)]
    for letter,((name,_,_,_),(row,col)) in enumerate(zip(settings,cells)):
        ax=fig.add_subplot(grid[row,col:col+2]);r=results[name]
        edges=np.array(r['bin_edges']);hist=np.array(r['signed_excess_per_stack_per_bin'])
        x=np.array(r['curve_x']);y=np.array(r['signed_excess_smooth'])
        # The baseline's saved r range is 0-1 and contains every detection.
        # Extend its empty bins with zero for a common display range only.
        if edges[-1]<1.44:
            tail=np.arange(edges[-1]+.02,1.4401,.02)
            edges=np.r_[edges,tail];hist=np.r_[hist,np.zeros(len(tail))]
            x=np.r_[x,1.0005,1.44];y=np.r_[y,0,0]
        ax.stairs(hist,edges,color='#63549a',alpha=.42,lw=.8)
        ax.plot(x,y,color='#63549a',lw=1.9)
        ax.fill_between(x,0,y,where=y>=0,color='#63549a',alpha=.13,interpolate=True)
        ax.fill_between(x,0,y,where=y<0,color='#be5964',alpha=.23,interpolate=True)
        ax.axhline(0,color='.5',lw=.8)
        for p in r['descriptive_r_peaks']:
            ax.plot(p['r'],p['excess_per_stack_per_bin'],'o',color='#63549a',ms=3.5)
            ax.annotate(f'{p["r"]:.2f}',(p['r'],p['excess_per_stack_per_bin']),xytext=(0,8),
                textcoords='offset points',ha='center',fontsize=9,color='#63549a')
        ax.axvspan(.9276834494677721,1.44,color='.95',zorder=-3)
        ax.axvline(.9276834494677721,color='.5',ls=':',lw=1)
        ax.text(.944,.10,'r_lim ≈ 0.928',transform=ax.get_xaxis_transform(),rotation=90,color='.45',fontsize=8,va='bottom')
        ax.set(xlim=(0,1.44),ylim=(-25,32),xlabel='Anisotropy radius r',
               title=f'{chr(65+letter)} · {r["label"]}\n{r["dna_mean_per_stack"]:.1f} − {r["gold_mean_per_stack"]:.1f} = {r["net_excess_per_stack"]:.1f} net / stack')
        ax.title.set_fontsize(11)
        ax.set_ylabel('Signed excess / stack / 0.02 r bin',fontsize=9)
        ax.set_xticks([0,.3,.6,.9,1.2,1.4])
    fig.suptitle('Exploratory control subtraction · Five processing choices\nEach bin: DNA count ÷ 16 − gold count ÷ 14 · Negative values retained · Matched axes',fontsize=16)
    fig.savefig(out/'exploratory_control_subtraction_all.png',dpi=240,facecolor='white');plt.close(fig)
    payload=dict(source=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        variants=results,formula='DNA_count_i/16 - Gold_count_i/14; each variant uses its own counts and scale factor.',
        scope='Exploratory arithmetic only, outside the email draft. No new detection, no refitted control weight, no significance tests. The same 16 DNA and14 gold stacks enter every variant.',
        processing='Frame 1; matched background for detection and extraction in each variant; oversized components allowed; existing corner, overlap and any-frame raw saturation exclusions remain. All finite r values retained, including above r_lim.',
        assumption='Non-hinge detections in DNA have the same distribution and average number per field as the matching gold control under that processing. Actual detected populations change between variants; these are not paired spot comparisons.',
        interpretation='Signed excess, not a verified hinge-only probability distribution. Negative bins remain. Their reduction under stronger processing is not validation: background subtraction and detection selection change.',
        smoothing='Saved Figure 2 Gaussian sigma=0.02r curves subtracted linearly. Descriptive positive maxima: prominence>=1 detection/stack/bin, separation>=0.08r. No theta inversion of above-limit radii.')
    (out/'exploratory_control_subtraction_all.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:{key:v[key] for key in ['dna_mean_per_stack','gold_mean_per_stack','scale_factor','net_excess_per_stack','negative_bin_total_per_stack','descriptive_r_peaks']} for k,v in results.items()},indent=2))


if __name__=='__main__':build_all()
