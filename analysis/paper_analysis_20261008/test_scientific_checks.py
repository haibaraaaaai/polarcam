import unittest
import numpy as np
import paper_figures as p
from camera_trial import spot_channels

class GeometryTests(unittest.TestCase):
    def test_projection_inverse_and_jacobian(self):
        rng=np.random.default_rng(42);v=rng.normal(size=(1000,3));v/=np.linalg.norm(v,axis=1)[:,None]
        xy=p.stereo(v);np.testing.assert_allclose(p.inverse_stereo(*xy.T),v,atol=2e-13)
        for x,y in ((0.,0.),(.4,.7),(1.5,.6)):
            eps=1e-5
            dx=(p.inverse_stereo(x+eps,y)-p.inverse_stereo(x-eps,y))/(2*eps)
            dy=(p.inverse_stereo(x,y+eps)-p.inverse_stereo(x,y-eps))/(2*eps)
            self.assertAlmostEqual(np.linalg.norm(np.cross(dx,dy)),1/(1+(x*x+y*y)/4)**2,places=8)

    def test_cut_preserves_observables(self):
        phi=np.linspace(-30,30,1000)
        for cut in (-1.,0.,.73,2.1):
            wrapped=p.wrap_phi(phi,cut)
            np.testing.assert_allclose(np.cos(2*phi),np.cos(2*wrapped),atol=2e-14)
            np.testing.assert_allclose(np.sin(2*phi),np.sin(2*wrapped),atol=2e-14)

    def test_isotropic_density_and_mass(self):
        rng=np.random.default_rng(45);n=500000
        th=np.arccos(rng.uniform(0,1,n));ph=rng.uniform(0,np.pi,n)
        xy=p.stereo(p.vectors(th,ph));ex,ey,area=p.density_grid((-2,2,0,2),60)
        h=np.histogram2d(*xy.T,bins=(ex,ey))[0];rho=h/n/area
        xx=(ex[:-1]+ex[1:])/2;yy=(ey[:-1]+ey[1:])/2
        inside=(xx[:,None]**2+yy[None,:]**2<1.8**2)&(yy[None,:]>.1)
        self.assertLess(abs(np.average(rho[inside],weights=area[inside])-1/np.pi),.003)
        self.assertAlmostEqual(np.sum(rho*area),1.,places=12)

    def test_rigid_rotation_preserves_cap_mass(self):
        rng=np.random.default_rng(0);v=rng.normal(size=(10000,3));v/=np.linalg.norm(v,axis=1)[:,None]
        R,_=np.linalg.qr(rng.normal(size=(3,3)));center=np.array([0,0,1.])
        np.testing.assert_allclose((v@R)@(center@R),v@center,atol=2e-15)
        self.assertEqual(np.count_nonzero(v@center>.8),np.count_nonzero((v@R)@(center@R)>.8))

    def test_circle_axis(self):
        n=np.array([.8,0,.6]);e=np.array([0,1.,0]);f=np.cross(n,e);a=np.linspace(0,2*np.pi,2000)
        v=np.cos(.9)*n+np.sin(.9)*(np.cos(a)[:,None]*e+np.sin(a)[:,None]*f)
        fit=p.fit_circle(v)
        self.assertGreater(abs(fit['n']@n),.99999);self.assertLess(fit['rms_deg'],.001)

    def test_out_of_model_not_clipped(self):
        A,B,C=p.MODELS[p.MODEL];th,flags=p.ha.theta_from_r(np.array([0,C/(A+B)+.02]),A,B,C,return_flags=True)
        self.assertTrue(np.isnan(th[1]));self.assertFalse(flags['valid'][1])

    def test_camera_external_annulus_and_parity(self):
        data=np.zeros((5,31,31));yy,xx=np.indices((31,31));signal=(abs(yy-15)<=3)&(abs(xx-15)<=3)
        values=[11,22,33,44]
        for value,(py,px) in zip(values,((1,1),(0,0),(0,1),(1,0))):
            mask=(yy%2==py)&(xx%2==px);data[:,mask]=7;data[:,mask&signal]+=value
        np.testing.assert_allclose(spot_channels(data,15,15,3,True),np.tile(values,(5,1)))
        np.testing.assert_allclose(spot_channels(data,15,15,3,False),np.tile(np.array(values)+7,(5,1)))

if __name__=='__main__':unittest.main(verbosity=2)
