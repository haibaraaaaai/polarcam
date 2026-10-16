"""Explore low-r shoulders, control subtraction, and attachment geometry.

Conditional illustrations, not a fitted population model or biological test.
Uses the existing regional trial's 1x radii without changing any selections.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE/'hinge_interpretation_check'
SOURCE = HERE/'background_region_trial/results.json'
A, B, C = .09779825985607313, .011480772916810558, .10137635007725046
COLORS = {'Hinges':'#087e8b','Gold':'#c07a11'}


def radius(theta_deg):
    s = np.sin(np.radians(theta_deg))**2
    return C*s/(A+B*s)


def theta(r):
    return np.degrees(np.arcsin(np.sqrt(A*r/(C-B*r))))


def modes(values, edges, bandwidth):
    dx = edges[1]-edges[0]
    y = gaussian_filter1d(np.histogram(values, edges)[0].astype(float), bandwidth/dx, mode='reflect')
    idx, _ = find_peaks(y, prominence=.005*y.max())
    grid = (edges[:-1]+edges[1:])/2
    return grid, y, grid[idx].tolist()


def decorate(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)


def main():
    OUT.mkdir(exist_ok=True)
    raw=SOURCE.read_bytes();d=json.loads(raw)
    methods=d['methods'];arrays={};counts={};peak_info={}
    for m in methods:
        arrays[m]={};counts[m]={};peak_info[m]={}
        for group in ('Hinges','Gold'):
            per=[np.array([s['r'] for s in r['methods'][m] if s['matched'] or not s['inside']])
                 for r in d['records'] if r['group']==group]
            a=np.concatenate(per);arrays[m][group]=a
            counts[m][group]=dict(n=len(a),stacks=len(per),per_stack=len(a)/len(per),
                low_n=int(np.sum(a<.2)),low_fraction=float(np.mean(a<.2)),
                low_per_stack=float(np.sum(a<.2)/len(per)))
        a=arrays[m]['Hinges']
        for bw in (.02,.04,.06):
            _,_,peaks=modes(a,np.arange(0,1.422,.002),bw)
            peak_info[m][str(bw)]=peaks
        valid=a<=d['r_max'];t=theta(a[valid])
        peak_info[m]['theta_excluded']=int(np.sum(~valid))
        peak_info[m]['theta_modes']={str(bw):modes(t,np.arange(0,90.1001,.1),bw)[2] for bw in (2,3,4,6)}
    report=dict(source_sha256=hashlib.sha256(raw).hexdigest(),
                builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                coefficients=dict(A=A,B=B,C=C),counts=counts,modes=peak_info,
                angle_mapping={str(r):float(theta(r)) for r in (.2,.3,.4,.7,.75,.8,.85)},
                interpretation='Descriptive detector/bandwidth sensitivity, not a significance test. One DNA well and one control well; stacks are not independent preparations.',
                selection='Existing targeted regional trial: all unsaturated finite 1x radii except 2x failures inside the fixed ellipse.',
                subtraction='Signed DNA counts/16 minus gold counts/14 per bin. Assumes equal usable field area, comparable detection, and equal nonspecific density; not established. Negative values are retained.',
                geometry='Illustration only; no fitting to the recordings. Same fixed arm tilt can generate a broad or bimodal distribution through attachment azimuth. No detection, noise, exposure averaging, or steric weighting is included.')
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    edges=np.arange(0,1.0201,.02)
    for i,m in enumerate(methods):
        hist={}
        for g in ('Hinges','Gold'):
            hist[g]=np.histogram(arrays[m][g],edges)[0]/counts[m][g]['stacks']
            axes[0,i].stairs(hist[g],edges,color=COLORS[g],lw=1.5,label=('DNA' if g=='Hinges' else 'Gold'))
        ax=axes[0,i];ax.set_title(methods[m]['label']);ax.set_ylabel('Detected spots / stack / 0.02 bin')
        ax.legend(frameon=False);decorate(ax)
        ax=axes[1,i];ax.stairs(hist['Hinges']-hist['Gold'],edges,color='#75549c',lw=1.5)
        ax.axhline(0,color='#333',lw=.8);ax.set_ylabel('DNA minus gold: spots / stack / bin');ax.set_xlabel('r from original 1x intensities');decorate(ax)
        ax.text(.02,.96,'Conditional subtraction; negative bins kept',transform=ax.transAxes,va='top',fontsize=8)
        for ax in axes[:,i]:ax.set_xlim(0,1.02);ax.axvline(d['r_max'],ls=':',color='#75838a',lw=1)
    fig.suptitle('Counts per field, not percentage: does a control-sized component account for the low-r tail?',fontsize=14)
    fig.text(.02,.013,'Same nonspecific density is an assumption. Field selection, detection completeness, and surface chemistry can differ; this is not an isolated-hinge measurement.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(OUT/'counts_and_subtraction.png',dpi=150);fig.savefig(OUT/'counts_and_subtraction.svg');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for i,m in enumerate(methods):
        a=arrays[m]['Hinges']
        for bw,color in zip((.02,.04,.06),('#929ea3','#087e8b','#703eb3')):
            grid,y,_=modes(a,np.arange(0,1.422,.002),bw)
            axes[0,i].plot(grid,y/(len(a)*.002),color=color,lw=1.5,label=f'Gaussian width {bw:.2f} r')
        ax=axes[0,i];ax.set_xlim(0,1);ax.set_title(methods[m]['label']);ax.set_ylabel('DNA density per unit r');ax.set_xlabel('r');ax.legend(fontsize=8,frameon=False);decorate(ax)
        for g in ('Hinges','Gold'):
            a=arrays[m][g];v=a<=d['r_max'];t=theta(a[v]);grid,y,_=modes(t,np.arange(0,90.1001,.1),3)
            axes[1,i].plot(grid,y/(len(t)*.1),color=COLORS[g],lw=1.6,label=('DNA' if g=='Hinges' else 'Gold'))
        ax=axes[1,i];ax.set_xlim(0,90);ax.set_ylabel('Density per degree');ax.set_xlabel('Rod polar angle from optical z / surface normal (deg)');ax.legend(frameon=False);decorate(ax)
    fig.suptitle('The low-r shoulder depends on selection and smoothing; theta density has different modes',fontsize=14)
    fig.text(.02,.01,'Top: all finite 1x radii. Bottom: transformed values, width 3 degrees; six DNA values and one gold value (supplied detector) exceed the model and are omitted.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94));fig.savefig(OUT/'peak_sensitivity.png',dpi=150);plt.close(fig)
    rng=np.random.default_rng(20261014);n=250000
    theta0=np.full(n,50.);psi=rng.uniform(0,2*np.pi,n)
    def misaligned(t0,delta):
        q=np.cos(np.radians(t0))*np.cos(np.radians(delta))+np.sin(np.radians(t0))*np.sin(np.radians(delta))*np.cos(psi)
        return np.degrees(np.arccos(np.clip(np.abs(q),0,1)))
    sims={
      'Aligned rod; arm 50 +/- 3 deg':rng.normal(50,3,n),
      'Arm 50 deg; rod offset 20 deg':misaligned(theta0,np.full(n,20.)),
      'Arm 50 +/- 3; offset 20 +/- 5 deg':misaligned(rng.normal(50,3,n),np.clip(rng.normal(20,5,n),0,90)),
    }
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    colors=('#89969c','#df7d22','#087e8b')
    for (label,t),color in zip(sims.items(),colors):
        for ax,v,e,h in ((axes[0],t,np.arange(0,90.2501,.25),1),
                         (axes[1],radius(t),np.arange(0,1.0001,.002),.015)):
            x,y,_=modes(v,e,h);ax.plot(x,y/(n*(e[1]-e[0])),color=color,lw=1.7,label=label)
    axes[0].set(xlim=(0,90),xlabel='Rod polar angle (deg)',ylabel='Probability per degree',title='One hinge geometry; different attachments')
    axes[1].set(xlim=(0,1),xlabel='Predicted r',ylabel='Probability per unit r',title='Two maxima can arise from one arm angle')
    sig=np.linspace(0,70,301);re=radius(65)*np.exp(-2*np.radians(sig)**2)
    axes[2].plot(sig,re,color='#75549c',lw=2);axes[2].axhline(.3,color='#75838a',ls=':')
    critical=float(np.degrees(np.sqrt(-.5*np.log(.3/radius(65)))))
    axes[2].plot([critical],[.3],'o',color='#75549c');axes[2].annotate(f'{critical:.1f} deg spread gives r=0.3',(critical,.3),xytext=(8,.19),textcoords='data',fontsize=9)
    axes[2].set(xlim=(0,70),ylim=(0,.9),xlabel='Within-record azimuth spread SD (deg)',ylabel='r after averaging channels',title='Another possibility: motion averaging')
    for ax in axes:decorate(ax)
    axes[0].legend(loc='upper left',frameon=False,fontsize=8)
    # Headroom for the shared legend, avoiding the simulated peaks.
    axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.8)
    fig.suptitle('Illustrations only: attachment geometry and averaging; no parameters fitted to data',fontsize=14)
    fig.text(.025,.015,'Fixed-offset model: random attachment azimuth across static rods. Motion model: fixed theta=65 deg, Gaussian azimuth fluctuations and constant brightness.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.92));fig.savefig(OUT/'geometry_simulation.png',dpi=160);fig.savefig(OUT/'geometry_simulation.svg');plt.close(fig)
    report['simulation']=dict(seed=20261014,n=n,arm_polar_deg=50,attachment_offset_deg=20,
        predicted_theta_range_deg=[30,70],predicted_r_range=[float(radius(30)),float(radius(70))],
        equation='cos(theta_rod)=cos(theta_arm)*cos(delta)+sin(theta_arm)*sin(delta)*cos(psi); psi uniform across static rods.',
        motion_equation='r_eff=r(theta)*exp(-2*sigma_phi_rad^2), fixed theta and Gaussian azimuth fluctuations with constant brightness.',
        sigma_phi_to_r03_at_theta65=critical)
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    rows=''.join(f'<tr><td>{methods[m]["label"]}</td><td>{counts[m]["Hinges"]["per_stack"]:.1f} / {counts[m]["Gold"]["per_stack"]:.1f}</td><td>{counts[m]["Hinges"]["low_per_stack"]:.2f} / {counts[m]["Gold"]["low_per_stack"]:.2f}</td><td>{100*counts[m]["Hinges"]["low_fraction"]:.1f}% / {100*counts[m]["Gold"]["low_fraction"]:.1f}%</td></tr>' for m in methods)
    (OUT/'index.html').write_text(HTML.replace('__ROWS__',rows),encoding='utf-8')
    assert SOURCE.read_bytes()==raw
    print(json.dumps(report,ensure_ascii=True))


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Interpreting the hinge distributions</title>
<style>body{margin:28px auto;padding:0 24px;max-width:1500px;font:16px/1.55 system-ui;background:#f7f8f5;color:#28383d}h1{font-size:28px}h2{font-size:21px;margin-top:30px}p{max-width:1100px}img{width:100%;background:white}a{color:#087e8b}table{border-collapse:collapse;background:white}td,th{padding:10px 16px;border-bottom:1px solid #d6dddd;text-align:left}.small{font-size:13px;color:#56676d}.note{padding:14px;border-left:4px solid #c07a11;background:#fff8e9}</style>
<h1>What can the current distributions establish?</h1>
<p>A weak low-r shoulder survives the GUI trial, but its prominence depends on the detector and smoothing. It is clearer with the supplied detector. That supports inspecting those candidates; it does not establish a distinct hinge population. No formal population confidence level is assigned.</p>
<p><a href="../background_region_trial/index.html">Regional selection and clickable spots</a> · <a href="results.json">Numbers, model and reproducible settings</a></p>
<h2>Count density changes the low-r comparison</h2>
<table><tr><th>Detector</th><th>All spots / stack<br>DNA / gold</th><th>r &lt; 0.2 spots / stack<br>DNA / gold</th><th>r &lt; 0.2 fraction<br>DNA / gold</th></tr>__ROWS__</table>
<p>DNA has about 4.5 times as many detections per field. The same absolute number of nonspecific rods would therefore make a much smaller percentage in DNA. The supplied detector finds similar absolute numbers below r=0.2 in both groups; the GUI finds fewer in DNA. A biological absence of low-r rods is therefore not robust across detection methods.</p>
<img src="counts_and_subtraction.png" alt="Counts per stack and signed control subtraction">
<p class="note">The lower panels assume that DNA and control have the same nonspecific rod density and comparable detection completeness. These are hypotheses, not measured facts. The signed residual is a diagnostic, not an identified hinge distribution. Negative bins must not be clipped away and relabelled as a pure hinge population.</p>
<p>For normalised distributions, p(DNA) = a × p(control) + (1−a) × p(hinge). The unknown fraction a is required. Equal nonspecific counts per field would give a≈0.221 with the GUI trial (32.2/146), not a=1. Neither total counts nor the normalised curves establish a. If essentially all detected rods are hinge-associated, control subtraction is unnecessary, but the control remains useful for nonspecific adsorption and measurement artefacts.</p>
<h2>r and theta peaks are different density modes</h2>
<p>Under the current static-rod optical model, r=0.3 gives theta=33.2° from the surface normal. r=0.75–0.85 maps to 62.8–72.3°. Rebuilding the theta density from individual observations gives the main GUI peak around 63° (about 27° above the surface), while the supplied detector gives around 57°. The mode of p(theta) is not obtained by simply transforming the mode of p(r), because the axis transformation changes density.</p>
<img src="peak_sensitivity.png" alt="Detector and smoothing sensitivity in r and theta">
<p class="small">Top: Gaussian widths 0.02, 0.04, 0.06 in r, applied to 0.002-bin counts. Bottom: transformed values, width 3°. Six DNA observations and one gold observation (supplied detector) outside the model's invertible range stay in r plots and are omitted only from theta plots. No spot-wise bootstrap p-value is appropriate as evidence for a preparation effect: all fields come from one DNA well and one control well.</p>
<h2>Attachment geometry can create more than one peak</h2>
<p>Illustration: give every arm the same 50° polar angle, and every rod a 20° offset from that arm. Let the direction of the offset vary uniformly across static attachments. Rod theta then spans 30–70°, with accumulation near the endpoints. Thus one arm-angle population can produce two orientation maxima. The example is not a fit or a claim about the actual preparation.</p>
<img src="geometry_simulation.png" alt="Forward simulation of rod attachment misalignment and channel averaging">
<p>The geometry is cos(theta rod) = cos(theta arm) cos(delta) + sin(theta arm) sin(delta) cos(psi). Here delta is rod–arm misalignment and psi its direction around the arm. Base tilt adds another rotation. The true hinge opening, rod polar angle, and attachment offset are different quantities.</p>
<p>The motion panel gives a separate ambiguity. With constant brightness and fixed theta, averaging four channels over azimuth fluctuations produces r effective = r(theta) exp(−2 sigma(phi)²). The current 150-frame channel average can therefore yield a low r for a moving rod even when its instantaneous tilt is large. Unresolved rods of different azimuths can also cancel in the averaged anisotropy vector. Neither effect is included in the static-attachment simulation.</p>
<h2>Useful next checks</h2>
<p>First inspect matched-brightness candidates near r=0.3 in individual frames and compare the radius of the averaged anisotropy with the distribution of per-frame radii, accounting for their noise bias. Check whether the same candidates remain rod-like across detectors. Then confirm with Guna the gold dose, washes, surface treatment, hinge identity and whether the control and DNA wells had comparable preparation. Those determine whether a scaled-control mixture is defensible.</p>
<p><a href="https://outlook.office365.com/owa/?ItemID=AQMkAGE0MGFmNDZhLTc2N2UtNDYxOS1iNDY3LWVkODFhZDI3NTdmYwBGAAADX334HLyVcE2oj%2BJD8T7Z%2FwcAHZVNySZh8UuKJN8jBbJ8bgAAAgEMAAAAHZVNySZh8UuKJN8jBbJ8bgAGpGzE2wAAAA%3D%3D&exvsurl=1&viewmodel=ReadMessageItem">Richard's 25 September geometry discussion</a> explicitly allows arbitrary rod–arm attachment and an unobservable fixed angular offset in the inferred hinge rotation. Jonathan's 27 September simulation note also cautions that the result depends on attachment to the surface. A motionless anisotropy point alone does not distinguish a locked hinge from nonspecifically stuck gold.</p>
<p class="small">Raw data and earlier selections remain unchanged. These are exploratory calculations with 1× intensities, the targeted large-feature exclusion and the existing pre-subtraction saturation veto. Counts are spot observations, not guaranteed unique rods or independent biological replicates.</p>
</html>'''

if __name__=='__main__':main()
