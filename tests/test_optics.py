"""Physics invariants and calibration order, independent of a camera connection."""

import unittest

import numpy as np

from polarcam import optics
from polarcam.live import intensity_spots


class CollectionTests(unittest.TestCase):
    def test_equal_indices_recover_analytic_annular_fourkas(self):
        def cone(na, n):
            c = np.sqrt(1-(na/n)**2)
            return np.array([1/6-c/4+c**3/12, c/8-c**3/8,
                             7/48-c/16-c**2/16-c**3/48])
        for n in (1.33, 1.398, 1.47):
            for inner in (0., .39):
                expected = cone(1.3, n)-cone(inner, n)
                actual = optics.annular_coefficients(1.3, inner, n, n)
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)

    def test_quadrature_converges(self):
        for n in (1.33, 1.398, 1.47):
            np.testing.assert_allclose(
                optics.annular_coefficients(n_sample=n, quadrature_order=64),
                optics.annular_coefficients(n_sample=n, quadrature_order=256), rtol=2e-12)

    def test_abc_predict_independent_vector_pupil_integration(self):
        # Integrate both pupil dimensions and the actual s/p field vectors.
        qnodes, qw = np.polynomial.legendre.leggauss(96)
        q = (.39+(qnodes+1)*(1.3-.39)/2)[:, None]
        qw = (qw*(1.3-.39)/2)[:, None]
        az = np.arange(256)[None, :]*2*np.pi/256
        cs, sn = np.cos(az), np.sin(az)
        ns, no = 1.33, 1.515
        sa, ca = q/ns, np.sqrt(1-(q/ns)**2)
        co = np.sqrt(1-(q/no)**2)
        ts = 2*ns*ca/(ns*ca+no*co)
        tp = 2*ns*ca/(no*ca+ns*co)
        field_weight = np.sqrt(co)/ca
        abc = optics.annular_coefficients()
        for theta, phi in ((0., 0.), (.4, .3), (.9, 1.1), (np.pi/2, 2.2)):
            dx, dy, dz = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
            ep = tp*(dx*ca*cs+dy*ca*sn-dz*sa)
            es = ts*(-dx*sn+dy*cs)
            ex, ey = field_weight*(ep*cs-es*sn), field_weight*(ep*sn+es*cs)
            channels = []
            for psi in np.radians([0, 90, 45, 135]):
                power = (ex*np.cos(psi)+ey*np.sin(psi))**2
                actual = np.sum(qw*q/ns**2*power)*(2*np.pi/256)/(4*np.pi)
                a, b, c = abc
                expected = a+b*np.sin(theta)**2+c*np.sin(theta)**2*np.cos(2*(psi-phi))
                self.assertAlmostEqual(actual, expected, places=13)
                channels.append(actual)
            xy = intensity_spots.anisotropy(channels)
            self.assertAlmostEqual(np.linalg.norm(xy), float(optics.radius_from_theta(theta, *abc)), places=13)

    def test_direct_inverse_endpoints_invalids_and_roundtrip(self):
        for model in optics.camera_theta_models().values():
            abc = [model[k] for k in ("A", "B", "C")]
            theta = np.linspace(0, np.pi/2, 501)
            r = optics.radius_from_theta(theta, *abc)
            np.testing.assert_allclose(optics.theta_from_radius(r, *abc), theta, atol=3e-13)
            invalid = [-1, np.nan, np.inf, model["r_max"]+1e-10]
            self.assertTrue(np.isnan(optics.theta_from_radius(invalid, *abc)).all())
            self.assertAlmostEqual(float(optics.theta_from_radius(model["r_max"], *abc)), np.pi/2)

    def test_known_water_example_and_vector_path(self):
        model = optics.theta_model()
        self.assertAlmostEqual(model["r_max"], .9276834494677721, places=12)
        result = intensity_spots.unit_vectors(np.array([[.39473051184, 0.], [.85, 0.], [.98, 0.]]), model)
        self.assertAlmostEqual(np.degrees(np.arccos(result[0, 2])), 39.15066994496402, places=10)
        self.assertTrue(np.isfinite(result[1]).all())
        self.assertTrue(np.isnan(result[2]).all())

    def test_invalid_geometry_is_rejected(self):
        for params in ({"na_in": 1.3}, {"n_sample": 1.2}, {"na_in": -.1},
                       {"n_immersion": np.nan}, {"quadrature_order": 3}):
            with self.assertRaises(ValueError):
                optics.annular_coefficients(**params)


class APDCalibrationTests(unittest.TestCase):
    def test_supplied_inverse_and_residual_gains_are_applied_in_correct_order(self):
        ideal = np.array([[140., 60., 120., 80.], [80., 120., 180., 20.], [100., 100., 50., 150.]])
        inverse = optics.polarcam_apd_inverse()
        measured_at_reference = ideal @ np.linalg.inv(inverse).T
        np.testing.assert_allclose(optics.correct_polarcam_apd(measured_at_reference), ideal, atol=1e-12)
        g = np.array([1., .8, 1.1, 1.3])
        measured = measured_at_reference/g
        corrected = optics.correct_polarcam_apd(measured, g)
        np.testing.assert_allclose(corrected, ideal, atol=1e-12)
        np.testing.assert_allclose(optics.pair_sum_imbalance(corrected), 0., atol=1e-15)
        self.assertGreater(np.max(np.abs(optics.pair_sum_imbalance(optics.correct_polarcam_apd(measured)))), .01)

    def test_balance_does_not_prove_angle_or_calibration(self):
        # Equal pair sums can coexist with very different polarization radii.
        channels = np.array([[190., 10., 100., 100.], [110., 90., 100., 100.]])
        np.testing.assert_array_equal(optics.pair_sum_imbalance(channels), [0., 0.])
        self.assertGreater(np.ptp(np.linalg.norm(intensity_spots.anisotropy(channels), axis=1)), .7)
        self.assertTrue(np.isnan(optics.pair_sum_imbalance([0., 0., 0., 0.])))
        with self.assertRaises(ValueError):
            optics.correct_polarcam_apd(channels, [1., -1., 1., 1.])


if __name__ == "__main__":
    unittest.main()
