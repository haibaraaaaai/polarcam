"""Reproducible trial figures. No source data or acquisition code is modified."""
from pathlib import Path
import sys, json
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from nptdms import TdmsFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'original_scripts'))
import hinge_analysis as ha
DATA = (Path(__file__).resolve().parents[2] / 'data/hinges/apd_polarcam/legacy_hugh/Hinge apd recordings, 2 strand + notches')
OUT = ROOT/'figures'; OUT.mkdir(exist_ok=True)
MODELS = {'manuscript_annular':ha.fourkas_ABC(1.3,1.33,.39),
          'notebook_fresnel':ha.fourkas_ABC_fresnel(1.3,1.33,1.515,.38),
          'old_full_pupil':ha.fourkas_ABC(1.3,1.33)}
MODEL = 'manuscript_annular'
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                     'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})

def save(fig,name):
    for ext in ('png','pdf','svg'): fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
    return fig

def tdms_chunks(path,chunk=250000):
    with TdmsFile.open(path) as f:
        ch=f.groups()[0].channels()
        if len(ch)!=4 or len(set(map(len,ch)))!=1: raise ValueError('Need four equal-length APD channels')
        dt=float(ch[0].properties['wf_increment'])
        if not all(np.isclose(float(c.properties['wf_increment']),dt) for c in ch): raise ValueError('Unequal clocks')
        for start in range(0,len(ch[0]),chunk):
            # Native file channels 90,45,135,0 -> logical matrix order 0,90,45,135.
            yield start,dt,np.column_stack([ch[i][start:start+chunk] for i in (3,0,1,2)])

def fit_gains(path):
    # Streaming QR, equivalent to the notebook's native least squares without normal equations.
    wt=np.array([1,1,-1,-1])@ha.apd_tmatrix(); R=np.empty((0,4)); count=0
    for _,_,a in tdms_chunks(path):
        a=a[np.isfinite(a).all(axis=1)]; cols=a*wt
        aug=np.column_stack([cols[:,1:],-cols[:,0]])
        R=np.linalg.qr(np.vstack([R,aug]),mode='r'); count+=len(a)
    g=np.r_[1.,np.linalg.lstsq(R[:,:3],R[:,3],rcond=None)[0]]
    if not np.all(np.isfinite(g)&(g>0)): raise ValueError(f'Invalid fitted gains {g}')
    return g,float(np.linalg.cond(R[:,:3]))

def angles(a,model=None):
    model=MODEL if model is None else model
    sx=a[:,0]+a[:,1]; sy=a[:,2]+a[:,3]
    good=np.isfinite(a).all(axis=1)&(a>=0).all(axis=1)&(sx>0)&(sy>0)
    x=np.divide(a[:,0]-a[:,1],sx,out=np.full(len(a),np.nan),where=good)
    y=np.divide(a[:,2]-a[:,3],sy,out=np.full(len(a),np.nan),where=good)
    theta,flags=ha.theta_from_r(np.hypot(x,y),*MODELS[model],return_flags=True)
    valid=good&flags['valid']; theta[~valid]=np.nan
    phi=np.arctan2(y,x)/2
    return np.radians(theta),phi,valid,x,y

def branch_cut(phi,valid,bins=180):
    h,edges=np.histogram(np.mod(phi[valid],np.pi),bins=bins,range=(0,np.pi))
    smooth=gaussian_filter1d(h.astype(float),3,mode='wrap')
    return float((edges[np.argmin(smooth)]+edges[np.argmin(smooth)+1])/2)

def wrap_phi(phi,cut): return (phi-cut)%np.pi+cut

def vectors(theta,phi):
    return np.column_stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])

def stereo(v): return 2*v[...,:2]/(1+v[...,2,None])

def inverse_stereo(x,y):
    s=(x*x+y*y)/4; return np.stack([x/(1+s),y/(1+s),(1-s)/(1+s)],axis=-1)

def fit_circle(p):
    """Angular least squares, all valid 100-sample channel-average points; K=1."""
    def normal(a): return vectors(np.array([a[0]]),np.array([a[1]]))[0]
    def objective(a):
        th=np.arccos(np.clip(p@normal(a),-1,1)); mean=th.mean()
        return np.mean((th-mean)**2)*(1+np.cos(mean)**2)
    _,e=np.linalg.eigh(np.cov(p.T)); seed=e[:,0]
    starts=[[np.arccos(seed[2]),np.arctan2(seed[1],seed[0])],[np.pi/2,0],[np.pi/2,np.pi/2],[.5,1.]]
    fits=[minimize(objective,a,method='Nelder-Mead',options={'maxiter':220,'xatol':1e-7}) for a in starts]
    best=min(fits,key=lambda f:f.fun); n=normal(best.x)
    if n[2]<0: n=-n
    th=np.arccos(np.clip(p@n,-1,1)); a=th.mean()
    xp=np.cross(n,[0,0,1.]); xp/=np.linalg.norm(xp)
    yp=np.cross(n,xp); R=np.column_stack([xp,yp,n])
    return {'n':n,'R':R,'radius':a,'rms_deg':float(np.degrees(np.sqrt(np.mean((th-a)**2)))),
            'axis_theta_deg':float(np.degrees(np.arccos(n[2]))),'success':bool(best.success)}

def corrected_blocks(path,g,block=100):
    vals=[];times=[]
    for start,dt,a in tdms_chunks(path):
        n=len(a)//block*block
        vals.append(((a[:n]*g)@ha.apd_tmatrix().T).reshape(-1,block,4).mean(axis=1))
        times.append((start+np.arange(n//block)*block+(block-1)/2)*dt)
    return np.concatenate(times),np.concatenate(vals)

def density_grid(extent,bins=150):
    ex=np.linspace(extent[0],extent[1],bins+1); ey=np.linspace(extent[2],extent[3],bins+1)
    # Gauss quadrature integrates the stereographic Jacobian over each square exactly to numerical precision.
    nodes,w=np.polynomial.legendre.leggauss(4); dx=ex[1]-ex[0];dy=ey[1]-ey[0]
    xx=(ex[:-1]+ex[1:])/2; yy=(ey[:-1]+ey[1:])/2; area=np.zeros((bins,bins))
    for u,wu in zip(nodes,w):
        for v,wv in zip(nodes,w):
            r2=(xx[:,None]+u*dx/2)**2+(yy[None,:]+v*dy/2)**2
            area+=wu*wv/(1+r2/4)**2*dx*dy/4
    return ex,ey,area

def density_stream(path,g,cut,rotation=None,extent=(-2.05,2.05,-2.05,2.05),bins=150):
    ex,ey,area=density_grid(extent,bins); counts=np.zeros_like(area); valid_n=total=0
    for _,_,raw in tdms_chunks(path):
        th,ph,ok,_,_=angles((raw*g)@ha.apd_tmatrix().T)
        p=vectors(th[ok],wrap_phi(ph[ok],cut))
        if rotation is not None: p=p@rotation
        xy=stereo(p); counts+=np.histogram2d(xy[:,0],xy[:,1],bins=(ex,ey))[0]
        valid_n+=int(ok.sum()); total+=len(raw)
    return {'counts':counts,'density':counts/valid_n/area,'ex':ex,'ey':ey,'area':area,
            'valid':valid_n,'total':total,'shown':int(counts.sum())}

def draw_map(ax,d,cut=None,vmax=None,raw=False):
    z=d['counts'] if raw else d['density']; positive=z[z>0]
    norm=LogNorm(vmin=1 if raw else .01,vmax=vmax or positive.max())
    mesh=ax.pcolormesh(d['ex'],d['ey'],np.ma.masked_where(z.T<=0,z.T),cmap='magma',norm=norm,rasterized=True)
    for deg in (30,60,90):
        ph=np.linspace(0,2*np.pi,400);r=2*np.tan(np.radians(deg)/2)
        ax.plot(r*np.cos(ph),r*np.sin(ph),color='.6',lw=.5,alpha=.65)
    if cut is not None:
        rr=np.array([-2.,2.]);ax.plot(rr*np.cos(cut),rr*np.sin(cut),'--',color='.35',lw=.8)
    ax.set_aspect('equal');ax.set(xlabel='stereographic u',ylabel='stereographic v',xlim=(d['ex'][0],d['ex'][-1]),ylim=(d['ey'][0],d['ey'][-1]))
    return mesh

def native_window(path,g,lo,hi):
    with TdmsFile.open(path) as f:
        ch=f.groups()[0].channels();dt=float(ch[0].properties['wf_increment'])
        start=max(0,int(np.floor(lo/dt)));stop=min(len(ch[0]),int(np.ceil(hi/dt)))
        a=np.column_stack([ch[i][start:stop] for i in (3,0,1,2)])
    return np.arange(start,stop)*dt,(a*g)@ha.apd_tmatrix().T

def process_record(name):
    path=DATA/name; g,cond=fit_gains(path);t,a=corrected_blocks(path,g)
    th,ph,ok,x,y=angles(a);cut=branch_cut(ph,ok);p=vectors(th[ok],wrap_phi(ph[ok],cut));fit=fit_circle(p)
    lab=density_stream(path,g,cut)
    hinge_xy=stereo(p@fit['R']); mn=hinge_xy.min(axis=0)-.4;mx=hinge_xy.max(axis=0)+.4
    extent=(float(mn[0]),float(mx[0]),float(mn[1]),float(mx[1]))
    hinged=density_stream(path,g,cut,fit['R'],extent)
    summary={'file':name,'gain_order':'0,90,45,135','gains':g.tolist(),'gain_condition':cond,
             'duration_s':float(t[-1]+.0002),'model':MODEL,'cut_deg':float(np.degrees(cut)),
             'axis_theta_deg':fit['axis_theta_deg'],'cone_angle_deg':float(np.degrees(fit['radius'])),
             'angular_residual_rms_deg':fit['rms_deg'],'fit_converged':fit['success'],
             'native_valid':lab['valid'],'native_total':lab['total'],'hinge_view_mass':hinged['shown']/hinged['valid'],
             'fit_samples':len(p)}
    print(json.dumps(summary),flush=True)
    return dict(name=name,g=g,t=t,a=a,th=th,ph=ph,ok=ok,x=x,y=y,cut=cut,p=p,fit=fit,lab=lab,hinge=hinged,summary=summary)

def overview(records):
    fig,axs=plt.subplots(len(records),3,figsize=(11,3.3*len(records)),layout='constrained')
    vmax=max(max(r['lab']['density'].max(),r['hinge']['density'].max()) for r in records)
    for row,r in enumerate(records):
        f=r['fit']; ax=axs[row,0];mi=draw_map(ax,r['lab'],r['cut'],vmax)
        ax.set_title(f"{r['name']} · lab frame · native 4 µs")
        ax=axs[row,1];mj=draw_map(ax,r['hinge'],vmax=vmax)
        ax.set_title(f"Hinge frame · axis θ₀ = {f['axis_theta_deg']:.1f}°")
        phi=np.linspace(-np.pi,np.pi,1000); circle=vectors(np.full_like(phi,f['radius']),phi);xy=stereo(circle)
        ax.plot(*xy.T,color='#00a6a6',lw=1,ls='--')
        ax.text(.02,.02,f"cone {np.degrees(f['radius']):.1f}°; residual {f['rms_deg']:.1f}°",transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        ax=axs[row,2];edges=np.radians(np.arange(0,91,2));h,_=np.histogram(r['th'][r['ok']],edges)
        area=np.pi*(np.cos(edges[:-1])-np.cos(edges[1:]));rho=h/h.sum()/area
        ax.step(np.degrees((edges[:-1]+edges[1:])/2),rho,where='mid',color='#0072B2')
        ax.axhline(1/np.pi,color='.5',ls=':',label='uniform folded orientation')
        ax.set(xlabel='optical θ from vertical (°)',ylabel='shell-averaged density (sr⁻¹)',xlim=(0,90),title='θ profile · 0.4 ms channel means')
        ax.legend(fontsize=6,loc='upper right',frameon=False)
        ax.set_title(f"θ profile · 0.4 ms channel means\nnative valid: {r['lab']['valid']/r['lab']['total']:.2%}")
    fig.colorbar(mi,ax=axs[:,:2],label='conditional probability density (sr⁻¹)',shrink=.65,pad=.015)
    fig.suptitle('Existing APD hinge recordings — stereographic trial figures',fontsize=13)
    return save(fig,'apd_orientation_gallery')

def zoom_figure(r,center=None):
    t=r['t'];th=np.degrees(r['th']);valid=r['ok']
    # Reproducible display selection, not a kinetic event detector: largest adjacent 0.4 ms angular change in first 5 s.
    if center is None:
        d=np.abs(np.diff(th)); d[~(valid[:-1]&valid[1:]&(t[:-1]<5))]=-np.inf
        center=float(t[np.argmax(d)]+.0002)
    fig,axs=plt.subplots(4,1,figsize=(9.5,8),layout='constrained')
    axs[0].plot(t,th,color='#0072B2',lw=.4,rasterized=True)
    axs[0].set(xlabel='time (s)',ylabel='optical θ (°)',title=f"{r['name']} · full record · 0.4 ms means of channels")
    for k,width in enumerate((.05,.005,.0006),1):
        lo=max(0,center-width/2);hi=center+width/2;tn,an=native_window(DATA/r['name'],r['g'],lo,hi)
        theta,phi,ok,_,_=angles(an);theta=np.degrees(theta)
        scale=1000 if k<3 else 1e6;unit='ms' if k<3 else 'µs'
        axs[k].plot((tn-center)*scale,theta,color='#0072B2',lw=.45,marker='.' if k==3 else None,ms=2,rasterized=True)
        axs[k].set(xlabel=f'time relative to {center:.6f} s ({unit})',ylabel='optical θ (°)',title=f'{width*1e3:g} ms window · native 4 µs samples · {ok.sum()}/{len(ok)} valid')
        if k<3:
            nb=20;n=len(an)//nb*nb
            ta,_,_,_,_=angles(an[:n].reshape(-1,nb,4).mean(1))
            axs[k].plot((tn[:n].reshape(-1,nb).mean(1)-center)*scale,np.degrees(ta),color='#D55E00',lw=1.1,label='80 µs means of channels');axs[k].legend(fontsize=8)
    axs[0].axvspan(max(0,center-.025),center+.025,color='#D55E00',alpha=.2)
    fig.suptitle('Nested views of observed motion — no dwell-state assignment',fontsize=13)
    r['summary']['zoom_center_s']=center
    return save(fig,f"{Path(r['name']).stem}_nested_time_windows")

def model_figure(records):
    fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained');rad=np.linspace(0,1,1001)
    for name,abc in MODELS.items():
        vals=ha.theta_from_r(rad,*abc);axs[0].plot(rad,vals,label=f'{name}: rmax={ha.r_max_of(*abc):.4f}')
    axs[0].set(xlabel='measured anisotropy radius r',ylabel='estimated optical θ (°)',ylim=(0,90));axs[0].legend(fontsize=7)
    for r in records:
        rho=np.hypot(r['x'],r['y']);base=ha.theta_from_r(rho,*MODELS[MODEL]);alt=ha.theta_from_r(rho,*MODELS['notebook_fresnel']);ok=np.isfinite(base)&np.isfinite(alt)
        h,edges=np.histogram(base[ok],bins=np.arange(0,91,2));sumd,_=np.histogram(base[ok],edges,weights=alt[ok]-base[ok]);mean=np.divide(sumd,h,out=np.full_like(sumd,np.nan),where=h>0)
        axs[1].plot((edges[:-1]+edges[1:])/2,mean,label=r['name'])
    axs[1].axhline(0,color='.5',lw=.6);axs[1].set(xlabel='manuscript-model θ (°)',ylabel='Fresnel-model minus manuscript θ (°)');axs[1].legend()
    fig.suptitle('Optical-model sensitivity — model choice is explicit',fontsize=12)
    return save(fig,'optical_model_sensitivity')

def cut_figure(r):
    th=r['th'][r['ok']];ph=r['ph'][r['ok']];step=max(1,len(th)//25000)
    fig,axs=plt.subplots(1,3,figsize=(10,3.7),layout='constrained')
    for ax,cut,title in zip(axs[:2],(-np.pi/2,r['cut']),('Original −90° cut','Automatic low-occupancy cut')):
        xy=stereo(vectors(th[::step],wrap_phi(ph[::step],cut)));ax.scatter(*xy.T,s=.2,c=np.degrees(th[::step]),cmap='viridis',rasterized=True)
        for deg in (30,60,90):
            a=np.linspace(cut,cut+np.pi,200);rr=2*np.tan(np.radians(deg)/2);ax.plot(rr*np.cos(a),rr*np.sin(a),color='.6',lw=.5)
        rr=np.array([-2.,2.]);ax.plot(rr*np.cos(cut),rr*np.sin(cut),'--',color='.4',lw=.7)
        ax.set(aspect='equal',xlim=(-2.1,2.1),ylim=(-2.1,2.1),title=title,xlabel='u',ylabel='v')
    ax=axs[2];m=draw_map(ax,r['lab'],r['cut'],raw=True);ax.set_title('Native counts for audit');fig.colorbar(m,ax=ax,label='samples per projected bin',shrink=.7)
    fig.suptitle(f"{r['name']} · cut changes the branch convention, not the measured signal",fontsize=12)
    return save(fig,'cut_and_counts')

def geometry_figure(records):
    fig=plt.figure(figsize=(11,7.5),layout='constrained');gs=fig.add_gridspec(2,3,hspace=.22)
    for j,r in enumerate(records):
        ax=fig.add_subplot(gs[0,j],projection='3d');p=r['p'];stride=max(1,len(p)//10000);f=r['fit']
        ax.scatter(*p[::stride].T,s=.4,c=np.degrees(np.arccos(p[::stride,2])),cmap='viridis',vmin=0,vmax=90,rasterized=True)
        n=f['n'];base=f['R'][:,0]
        ax.quiver(0,0,0,*n,length=1.2,color='#D55E00',label='fitted axis n')
        ax.quiver(0,0,0,*base,length=1.1,color='#0072B2',label="horizontal x′")
        ph=np.linspace(0,2*np.pi,300);circle=vectors(np.full_like(ph,f['radius']),ph)@f['R'].T
        ax.plot(*circle.T,color='#009E73',lw=1)
        ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),xlabel='x',ylabel='y',zlabel='z',title=r['name']);ax.set_box_aspect((1,1,1));ax.view_init(25,45)
        ax=fig.add_subplot(gs[1,j]);q=p@f['R'];beta=np.degrees(np.arctan2(q[:,1],q[:,0]))
        ax.plot(r['t'][r['ok']],beta,lw=.4,color='#0072B2',rasterized=True)
        ax.set(xlabel='time (s)',ylabel="rotation β about fitted axis (°)",title='0.4 ms channel means; fixed branch')
    fig.suptitle('Geometry and relative rotation — β is not a calibrated hinge opening angle',fontsize=12)
    return save(fig,'hinge_geometry_and_rotation')

def run_apd():
    records=[process_record(n) for n in ('1.tdms','3.tdms','34.tdms')]
    overview(records);model_figure(records);cut_figure(records[0]);geometry_figure(records)
    for r in records:zoom_figure(r)
    (ROOT/'apd_summary.json').write_text(json.dumps([r['summary'] for r in records],indent=2),encoding='utf-8')
    plt.close('all');return records

if __name__=='__main__': run_apd()
