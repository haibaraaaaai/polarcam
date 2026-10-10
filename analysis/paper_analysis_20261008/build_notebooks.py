from pathlib import Path
import os,json
import nbformat as nbf
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent
os.environ['JUPYTER_PATH']=str(ROOT/'runtime/share/jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
def md(t):return nbf.v4.new_markdown_cell(t)
def code(t):return nbf.v4.new_code_cell(t)
def build(name,cells,execute=True):
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'polarcam-paper-trial','display_name':'Polarcam paper trial','language':'python'}})
    path=ROOT/name;nbf.write(nb,path)
    if execute:
        NotebookClient(nb,timeout=600,kernel_name='polarcam-paper-trial',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbf.write(nb,path)
    print(f'Executed and saved {path.name}',flush=True)

apd=[md('''# APD hinge publication figure trial

Existing Hugh APD records; Richard's 23–30 September display proposals. This is a reproducible figure trial, not a final inference of DNA states or absolute hinge angles. See `review.html` for the email chronology and figure requirements.

The original notebooks and Python helpers were copied into `original_scripts`; originals remain unchanged. The native 4 µs maps count **every valid sample**, not a display-thinned cloud. Circle fitting and long time traces use **100-sample (0.4 ms) means of channels**, with every complete block retained. That is a deliberate trial setting, different from fitting raw native angles; compare fit stability before publication.

## Conventions and corrections

- File channels `[90,45,135,0]` become matrix order `[0,90,45,135]`. Fit inverse relative gains, then apply Hugh's already-inverted optical matrix. No extra inversion, background fitting, centering or stretching.
- Default: manuscript annular model, NA 1.3, inner NA 0.39, water n=1.33. The source viewer instead defaults to annular+Fresnel with inner NA 0.38 and oil n=1.515. Both are compared; these are model estimates.
- Invalid angles remain NaN. No artificial mass is clipped onto 90°. Native validity fractions are reported; every displayed probability is conditional on validity.
- Optical θ is from the laboratory vertical. φ is modulo π. A low-occupancy cut chooses one representative quarter-sphere, not a unique physical trajectory.
- Stereographic radius **ρ=2 tan(θ/2)**; `(u,v)=2(x,y)/(1+z)`. Thus `dΩ=du dv/(1+ρ²/4)²`. Each square bin's solid angle is integrated by quadrature; density is `count/(N_valid ΔΩ)`. Bins crossing the branch/support boundary include the zero-density outside portion, so avoid interpreting individual edge bins as exact pointwise estimates.
- Hinge coordinates are a rigid rotation after choosing the laboratory branch. `x′ ∝ n×z` is horizontal; `y′=n×x′`. β=`atan2(p·y′,p·x′)`. A rod-to-arm offset is unmeasured, so β is not automatically the hinge opening α.

Richard's proposed four-degeneracy orbit-constrained assignment is **not** applied by default: it can compress points toward the assumed circle. A small circle residual is not independent evidence of correct reconstruction.
'''),code('''%matplotlib inline
import json
import numpy as np
import matplotlib.pyplot as plt
import paper_figures as pf
from IPython.display import display, Image

pf.MODEL = 'manuscript_annular'  # or 'notebook_fresnel'; rerun subsequent cells
FILES = ['1.tdms', '3.tdms', '34.tdms']
records = [pf.process_record(name) for name in FILES]
'''),md('''## Stereographic maps and θ profiles

Maps use a shared colour scale in sr⁻¹ in both coordinate systems. The right-hand profile is the probability in a θ shell divided by its exact solid angle `π(cos θ_low−cos θ_high)` on the folded quadrant. It is a shell average, not the ordinary probability density per degree. An isotropic folded distribution is `1/π sr⁻¹`.

The camera-like integration used for the θ profile can hide fast motion; native maps and nested windows expose that distinction. The hinge view is cropped for legibility; its included mass is in the summary, and normalization always uses all valid samples.'''),code("display(pf.overview(records)); plt.close('all')"),md('## Cut placement and raw-count audit'),code("display(pf.cut_figure(records[0])); plt.close('all')"),md('## Laboratory geometry and rotation about the fitted axis'),code("display(pf.geometry_figure(records)); plt.close('all')"),md('''## Nested native time windows

For a repeatable first look, the centre is the largest adjacent change between valid 0.4 ms θ estimates in the first five seconds. This is a display-selection rule, not a transition detector or dwell-time measurement. The shortest panel shows original 4 µs samples. Lines join consecutive valid observations; NaNs break them. The orange overlay averages **intensities before angle reconstruction**.'''),code("for r in records:\n    display(pf.zoom_figure(r)); plt.close('all')"),md('## Optical-model sensitivity'),code("display(pf.model_figure(records)); plt.close('all')"),md('''## Interactive cut and native-window inspection

The cut slider operates on the 0.4 ms display cloud only. It does not change the stored maps, fit or raw data. The native window control reloads unaveraged data from disk. The plotted circle is a model summary, not an imposed correction.'''),code('''import ipywidgets as widgets
def inspect_cut(file='1.tdms', cut_deg=0.):
    r=next(r for r in records if r['name']==file)
    ph=pf.wrap_phi(r['ph'][r['ok']],np.radians(cut_deg))
    pts=pf.stereo(pf.vectors(r['th'][r['ok']],ph))
    fig,ax=plt.subplots(figsize=(5,4));ax.scatter(*pts[::5].T,s=.25)
    ax.set(xlim=(-2.1,2.1),ylim=(-2.1,2.1),aspect='equal',xlabel='u',ylabel='v',title=f'cut={cut_deg:.1f}°; display thinned only')
    plt.show()
widgets.interact(inspect_cut,file=FILES,cut_deg=widgets.FloatSlider(value=np.degrees(records[0]['cut']),min=0,max=180,step=.5))
'''),code('''def inspect_native(file='1.tdms', start_s=.1825, width_us=600.):
    r=next(r for r in records if r['name']==file)
    t,a=pf.native_window(pf.DATA/file,r['g'],start_s,start_s+width_us/1e6)
    th,ph,ok,x,y=pf.angles(a)
    xy=pf.stereo(pf.vectors(th,pf.wrap_phi(ph,r['cut'])))
    fig,axs=plt.subplots(1,2,figsize=(9,3.4))
    axs[0].plot((t-start_s)*1e6,np.degrees(th),'.-',ms=2,lw=.6)
    axs[0].set(xlabel='time from start (µs)',ylabel='optical θ (°)')
    axs[1].plot(*xy.T,'.-',ms=2,lw=.6);axs[1].set(aspect='equal',xlabel='u',ylabel='v')
    plt.show()
widgets.interact(inspect_native,file=FILES,start_s=widgets.FloatText(value=.1825),width_us=widgets.FloatText(value=600.))
'''),code('''summary=[r['summary'] for r in records]
(pf.ROOT/'apd_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
summary
''')]
camera=[md('''# Camera population figure trial

Eight existing 18 September stacks were found locally. Each has 150 acquired frames plus a phase marker, and its metadata explicitly says vibration **off** and static background **already subtracted**. These are exploratory displays, not the final controlled hinge-yield experiment. No association with Hugh's August APD rods is inferred.

This trial retains the source's multiscale stationary-signal detector (40 evenly sampled detection frames, rather than 80, to bound memory). Detection is polarization-sensitive and brightness-gated. Large blobs and broad angular distributions are retained and flagged in the CSV, not called confirmed single rods. There is no inferred percentage of correctly assembled hinges. FOVs may overlap; summed counts are not guaranteed to be unique molecules.

## Changes from the original survey

1. Use the explicit annular optical model, with Fresnel comparison available through `paper_figures.MODEL`.
2. Reject rather than clip out-of-range radii; keep flags and counts. A near-zero radius is a noisy estimate near the pole, not mathematically undefined θ; φ is undefined at exactly zero.
3. Read parity `[90,45;135,0]` and ROI offsets from metadata. This trial rejects odd-offset stacks rather than silently shifting the convention.
4. Use equal numbers of real frames (150), excluding the appended marker by metadata. One orientation per object comes from **mean channel intensities**; the alternative time cloud gives every accepted object total weight one, even when valid frame counts differ.
5. Correct the optional background annulus to lie outside the signal crop, separately by channel. It is **off by default** because these files are already background-subtracted. Enabling a second subtraction needs a reason and a sensitivity comparison; it cannot remove coherent interference.
6. Replace ordinary per-degree histogram alone with stereographic probability per steradian and a solid-angle corrected θ profile. A common φ cut is used across all FOVs.

Read `camera_spot_summary.csv` for all bright candidates and rejection/size/angle flags. The teal reference is optical θ=40° from the emails, not a calibrated DNA opening angle.'''),code('''%matplotlib inline
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
import camera_trial as ct
import paper_figures as pf
RECOMPUTE = False  # True reprocesses all 8 existing full stacks; never edits them
ct.LOCAL_BACKGROUND=False
ct.FRAMES=150
if RECOMPUTE:
    camera_results=ct.run_camera()
print((pf.ROOT/'camera_summary.json').read_text())
'''),code("display(Image(filename=str(pf.OUT/'camera_population_stereographic.png')))\ndisplay(Image(filename=str(pf.OUT/'camera_theta_profiles.png')))"),md('''## Equal-rod-weight time-cloud alternative

Each rod contributes its per-frame orientations, normalized by its own valid-frame count. Thus a long or less rejected trajectory cannot dominate the population. This is distinct from the one-mean-channel-orientation map, because averaging and nonlinear angle inversion do not commute.'''),code('''from matplotlib.colors import LogNorm
paths=sorted(pf.ROOT.glob('frame_stack_*_population.npz'))
fig,axs=plt.subplots(2,4,figsize=(12,6),layout='constrained')
data=[dict(np.load(path)) for path in paths]
norm=LogNorm(.01,max(d['time_density'].max() for d in data))
for ax,path,d in zip(axs.flat,paths,data):
    z=d['time_density'];m=ax.pcolormesh(d['ex'],d['ey'],np.ma.masked_where(z.T<=0,z.T),cmap='magma',norm=norm,rasterized=True)
    ax.set(aspect='equal',title=path.stem[21:27],xlabel='u',ylabel='v')
fig.colorbar(m,ax=axs,label='equal-rod-weight density (sr⁻¹)',shrink=.7)
fig.suptitle('Vibration off · exploratory time-cloud distributions')
pf.save(fig,'camera_equal_rod_time_clouds');plt.show()
'''),md('''## Data needed for the final DNA figure

Hugh's original **matched Mode 1 + Mode 2 hinge recordings**, including the chosen rod coordinates/IDs, raw four-channel information, timing, background and vibration settings, are needed to reproduce Figure 5c–d and establish the screening-to-follow-up link. A static correctly oriented rod from the same preparation would supply the benchmark Richard requested. The APD files cannot reconstruct those camera panels.

For the later measurements: matched preparation comparisons, vibration-on movies and a no-DNA control; static rod examples at the target optical angle; short exposures or APDs to bound rapid motion. The September files do not supply those controls. Retain all detected objects and selection counts so gating does not masquerade as improved yield.''')]
build('APD_hinge_publication_trial.ipynb',apd)
build('Framestack_population_publication_trial.ipynb',camera)
