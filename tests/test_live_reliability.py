from collections import deque
import contextlib
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from polarcam import cli
from polarcam.live import intensity_spots
from polarcam.live.intensity_tuning import IntensityDiagnostic, IntensityTuner, compare_modulation, display_envelope, modulation_stats


LIVE_DIR = Path(__file__).resolve().parents[1] / "src" / "polarcam" / "live"
with patch.object(sys, "path", [str(LIVE_DIR), *sys.path]):
    from polarcam.live import angle_distribution_analysis, fetch_frames, recording_io
    from polarcam.live import Spinners_gui_live as live_gui
    from backend.ids_backend import IDSCamera, _StreamWorker


class GuiStartupTests(unittest.TestCase):
    def test_window_builds_and_closes_without_camera_access(self):
        self.enterContext(patch.dict(os.environ, cli._tk_environment()))
        try:
            root = live_gui.tk.Tk()
        except live_gui.tk.TclError as exc:
            if sys.platform != "win32" and ("no display name" in str(exc) or "couldn't connect to display" in str(exc)):
                self.skipTest(f"Tk display unavailable: {exc}")
            raise
        root.withdraw()
        callback_errors = []
        root.report_callback_exception = lambda *error: callback_errors.append(error)
        try:
            with tempfile.TemporaryDirectory() as directory:
                with contextlib.chdir(directory):
                    with patch.object(fetch_frames.Controller, "open", side_effect=AssertionError("Startup opened the camera")):
                        with patch.object(live_gui.messagebox, "showerror", side_effect=AssertionError("Unexpected startup error dialog")):
                            player = live_gui.BasicVideoPlayer(root)
                            root.update_idletasks()
                            root.update()
                            self.assertEqual(len(player._notebook.tabs()), 5)
                            self.assertFalse(player._live_running)
                            self.assertFalse(player._spotrec_running)
                            self.assertIsNone(player._live_controller)
                            self.assertEqual(float(player._live_gain_var.get()), 1.0)
                            for variable in (
                                player._live_exp_ms_var,
                                player._live_capture_exp_ms_var,
                                player._fetch_exp_ms_var,
                                player._spotrec_exp_ms_var,
                                player._stationary_capture_max_exp_ms_var,
                            ):
                                self.assertEqual(float(variable.get()), 1.2)
                            for variable in (
                                player._live_background_subtract_var,
                                player._live_capture_background_subtract_var,
                                player._fetch_background_subtract_var,
                                player._spotrec_background_subtract_var,
                                player._stationary_capture_background_subtract_var,
                            ):
                                self.assertFalse(variable.get())
                            self.assertFalse(player._live_background_subtract_enabled)
                            self.assertTrue((Path(directory) / "datasets").is_dir())
                            self.assertEqual(callback_errors, [])
                            player.on_close()
                            self.assertIsNone(player._ui_pump_after_id)
        finally:
            try:
                root.destroy()
            except live_gui.tk.TclError:
                pass


class IntensitySpotTests(unittest.TestCase):
    def theta_lut(self):
        theta = np.linspace(0, np.pi / 2, 1001)
        return {"r": np.sin(theta)**2, "theta_rad": theta, "r_max": 1.0}

    def test_native_channel_means_use_sensor_and_crop_parity(self):
        sensor = np.tile(np.array([[100, 700], [300, 900]], dtype=np.uint16), (10, 10))
        for phase_x in (0, 1):
            for phase_y in (0, 1):
                frame = sensor[phase_y:phase_y+16, phase_x:phase_x+16]
                for left, top in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    result = intensity_spots.channel_means(frame, [(left, left+8, top, top+8)], (phase_x, phase_y))
                    np.testing.assert_array_equal(result[0], [900, 100, 700, 300])

    def test_cartesian_average_handles_azimuth_boundary(self):
        phi = np.radians([179.0, 1.0])
        xy = .5 * np.column_stack((np.cos(2*phi), np.sin(2*phi)))
        channels = np.column_stack((100*(1+xy[:, 0]), 100*(1-xy[:, 0]), 100*(1+xy[:, 1]), 100*(1-xy[:, 1])))
        result = intensity_spots.average_directions(channels[:, None, :], self.theta_lut())
        direction = result["directions"][0, 0]
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0)
        self.assertLess(abs(direction[1]), .001)
        self.assertGreater(abs(direction[0]), .7)
        self.assertGreater(result["resultant_length"][0, 0], .99)

    def test_averaging_orders_are_distinct(self):
        channels = np.array([[[180, 20, 100, 100]], [[20, 20, 20, 20]]], dtype=float)
        results = [intensity_spots.average_directions(channels, self.theta_lut(), method)["directions"]
                   for method in intensity_spots.AVERAGING_METHODS]
        for first, second in zip(results, results[1:]):
            self.assertFalse(np.allclose(first, second))

    def test_invalid_samples_are_not_projected_to_the_pole(self):
        channels = np.array([[[0, 0, 0, 0]], [[10, 0, 10, 0]]], dtype=float)
        result = intensity_spots.average_directions(channels, self.theta_lut())
        self.assertTrue(np.isnan(result["directions"]).all())
        self.assertEqual(result["valid_counts"][0, 0], 0)

    def test_windows_and_density_are_normalized(self):
        channels = np.tile([150, 50, 100, 100], (5, 2, 1))
        result = intensity_spots.average_directions(channels, self.theta_lut(), window_frames=2)
        np.testing.assert_array_equal(result["window_stop"], [2, 4, 5])
        np.testing.assert_array_equal(result["valid_counts"][:, 0], [2, 2, 1])
        density, x_edges, y_edges = intensity_spots.projected_density(result["projected"])
        self.assertAlmostEqual(float(np.sum(density*np.diff(x_edges)[:, None]*np.diff(y_edges)[None, :])), 1.0)

    def test_detection_map_preserves_native_counts_and_ignores_polarization_split(self):
        first = np.tile(np.array([[100, 700], [300, 900]], dtype=np.uint16), (8, 8))
        second = np.full_like(first, 500)
        np.testing.assert_array_equal(live_gui.detect_spinners.intensity_detection_map(first), second)

    def test_stationary_bright_spot_is_detected(self):
        rows, columns = np.indices((64, 64))
        frame = 100 + 2000 * np.exp(-((rows - 32) ** 2 + (columns - 32) ** 2) / 18)
        detection_map = live_gui.detect_spinners.intensity_detection_map(frame)
        centers = live_gui.detect_spinners.find_spot_centers_dog(
            detection_map, sigma_small=1.0, sigma_large=3.0, k_std=1.0,
            min_area=3, max_area=None,
        )
        self.assertTrue(any(np.hypot(center[0] - 32, center[1] - 32) < 2 for center in centers))

    def test_detection_map_rejects_invalid_frames(self):
        for frame in (np.zeros((3, 4)), np.zeros((2, 2, 2)), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                live_gui.detect_spinners.intensity_detection_map(frame)


class IntensityDiagnosticTests(unittest.TestCase):
    def test_raw_channels_timing_and_clipping_are_preserved(self):
        sensor = np.tile(np.array([[10, 70], [30, 90]], dtype=np.uint16), (5, 5))
        for phase_x in (0, 1):
            for phase_y in (0, 1):
                with self.subTest(phase_x=phase_x, phase_y=phase_y):
                    diagnostic = IntensityDiagnostic()
                    diagnostic.begin(100.0, 60.0, {})
                    raw = sensor[phase_y:phase_y + 8, phase_x:phase_x + 8]
                    diagnostic.add(raw, 40.0, 101.0, 101.7, 22, (phase_x, phase_y))
                    row = dict(zip(diagnostic.COLUMNS, diagnostic.data[0]))
                    self.assertEqual(row["raw_mean_dn"], 50.0)
                    self.assertEqual(row["analysed_mean_dn"], 40.0)
                    self.assertEqual([row[key] for key in diagnostic.COLUMNS[5:9]], [90, 10, 70, 30])
                    self.assertAlmostEqual(row["processed_s"] - row["received_s"], 0.7)
                    raw = np.zeros((8, 8), dtype=np.uint16)
                    raw[0, 0] = 4095
                    diagnostic.add(raw, 0.0, 102.0, 102.0, 23, (0, 0))
                    self.assertEqual(diagnostic.count, 2)
                    self.assertEqual(diagnostic.data[1, -1], 1 / 64)
                    self.assertEqual(diagnostic.data[1, -2], 63 / 64)

    def test_capture_bounds_markers_and_export_without_reference(self):
        diagnostic = IntensityDiagnostic()
        diagnostic.begin(100.0, 60.0, {"fps": None})
        raw = np.full((4, 4), 1025, dtype=np.uint16)
        for index, timestamp in enumerate((99.0, 100.0, 159.99, 160.0)):
            diagnostic.add(raw, 1025.0, timestamp, 160.1, index, (0, 0))
        self.assertEqual(diagnostic.count, 2)
        self.assertEqual(diagnostic.metadata["pre_start_frames_excluded"], 1)
        diagnostic.mark(130.0, "Knobs unchanged", {"note": "4 Vpp dial estimate"})
        diagnostic.finish(160.2, "duration")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.npz"
            diagnostic.save(path)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["samples"], diagnostic.data[:2])
                metadata = json.loads(str(saved["metadata_json"]))
                self.assertEqual(metadata["events"][0]["label"], "Knobs unchanged")
                self.assertEqual(metadata["sample_count"], 2)
                self.assertEqual(metadata["stop_reason"], "duration")

    def test_sample_limit_stops_instead_of_dropping_silently(self):
        diagnostic = IntensityDiagnostic()
        diagnostic.MAX_SAMPLES = 2
        diagnostic.begin(0.0, 10.0, {})
        for index in range(3):
            diagnostic.add(np.ones((2, 2)), 1.0, float(index), float(index), index, (0, 0))
        self.assertFalse(diagnostic.active)
        self.assertEqual(diagnostic.count, 2)
        self.assertEqual(diagnostic.metadata["stop_reason"], "sample_limit")

    def test_invalid_duration_does_not_begin_recording(self):
        diagnostic = IntensityDiagnostic()
        for duration in (0.0, 5.0, 121.0, float("nan")):
            with self.assertRaises(ValueError):
                diagnostic.begin(0.0, duration, {})
        self.assertFalse(diagnostic.active)


class IntensityMetricTests(unittest.TestCase):
    def make_trial(self, tuner, kind, amplitude, now, *, age=0.01, clipped=False):
        tuner.begin(kind, kind, {"fps": 100.0, "exposure_us": 1200.0}, now, duration=2.0)
        for index in range(200):
            received = now + 0.5 + index / 100.0
            value = 1000.0 + amplitude * np.sin(2.0 * np.pi * index / 50.0)
            tuner.add(value, received, received + age, clipped=clipped)
        return tuner.update(now + 2.5)

    def test_trial_comparison_is_live_but_target_requires_complete_capture(self):
        tuner = IntensityTuner()
        self.make_trial(tuner, "reference", 100.0, 0.0)
        tuner.begin("test", "4 Vpp", {"fps": 100.0, "exposure_us": 1200.0}, 3.0, duration=2.0)
        for index in range(200):
            received = 3.5 + index / 100.0
            tuner.add(1000.0 + 8.0 * np.sin(2.0 * np.pi * index / 50.0), received, received)
        partial = tuner.update(4.5)
        self.assertAlmostEqual(partial["comparison"]["reduction"], 12.5)
        self.assertFalse(partial["comparison"]["target_reached"])
        complete = tuner.update(5.5)
        self.assertTrue(complete["comparison"]["target_reached"])
        self.assertEqual(len(tuner.trials), 2)
        self.assertEqual(len(complete["intensity_dn"]), 200)

    def test_old_frames_and_settling_samples_do_not_enter_trial(self):
        tuner = IntensityTuner()
        tuner.begin("reference", "ref", {"fps": 100.0}, 10.0, duration=2.0)
        for stamp in (9.0, 10.0, 10.49, 12.5):
            tuner.add(1000.0, stamp, 12.6)
        self.assertEqual(tuner.active["intensity_dn"], [])
        result = tuner.update(12.6)
        self.assertIn("Incomplete capture", result["warnings"])
        self.assertIsNone(tuner.reference)

    def test_backlog_or_saturation_cannot_qualify_a_trial(self):
        for options in ({"age": 1.0}, {"clipped": True}):
            with self.subTest(options=options):
                tuner = IntensityTuner()
                self.make_trial(tuner, "reference", 100.0, 0.0)
                result = self.make_trial(tuner, "test", 5.0, 3.0, **options)
                self.assertFalse(result["comparison"]["target_reached"])
                self.assertTrue(result["warnings"])

    def test_changed_exposure_requires_new_reference(self):
        tuner = IntensityTuner()
        self.make_trial(tuner, "reference", 100.0, 0.0)
        with self.assertRaisesRegex(ValueError, "Settings"):
            tuner.begin("test", "trial", {"fps": 100.0, "exposure_us": 600.0}, 3.0, duration=2.0)
        tuner.reset()
        self.assertIsNone(tuner.reference)
        self.assertEqual(len(tuner.trials), 1)

    def test_known_amplitude_reduction_and_target(self):
        phase = np.linspace(0.0, 8.0 * np.pi, 2000)
        reference = modulation_stats(1000.0 + 100.0 * np.sin(phase))
        current = modulation_stats(1000.0 + 8.0 * np.sin(phase))
        result = compare_modulation(reference, current)
        self.assertAlmostEqual(result["reduction"], 12.5)
        self.assertAlmostEqual(result["relative_reduction"], 12.5)
        self.assertTrue(result["target_reached"])
        smaller = modulation_stats(1000.0 + 20.0 * np.sin(phase))
        self.assertFalse(compare_modulation(reference, smaller)["target_reached"])

    def test_dimming_is_not_suppression(self):
        values = np.linspace(900.0, 1100.0, 1000)
        result = compare_modulation(modulation_stats(values), modulation_stats(values / 20.0))
        self.assertAlmostEqual(result["reduction"], 20.0)
        self.assertAlmostEqual(result["relative_reduction"], 1.0)
        self.assertFalse(result["target_reached"])
        self.assertTrue(result["warnings"])

    def test_flat_or_clipped_data_cannot_reach_target(self):
        reference = modulation_stats(np.linspace(500.0, 1500.0, 1000))
        for current in (
            modulation_stats(np.full(1000, 1000.0)),
            modulation_stats(np.linspace(995.0, 1005.0, 1000), clipped=True),
        ):
            result = compare_modulation(reference, current)
            self.assertFalse(result["target_reached"])
            self.assertTrue(result["warnings"])

    def test_invalid_samples_are_not_silently_removed(self):
        for values in (np.arange(10.0), np.full(100, np.nan), np.ones((10, 10))):
            with self.assertRaises(ValueError):
                modulation_stats(values)

    def test_display_envelope_keeps_peaks_without_changing_measurements(self):
        times = np.arange(20000) / 2000.0
        values = np.full(20000, 1000.0)
        values[123] = 1500.0
        values[3456] = 500.0
        original = values.copy()
        plotted_times, plotted = display_envelope(times, values)
        self.assertLessEqual(len(plotted), 1200)
        self.assertEqual(plotted.min(), 500.0)
        self.assertEqual(plotted.max(), 1500.0)
        self.assertTrue(np.all(np.diff(plotted_times) >= 0.0))
        np.testing.assert_array_equal(values, original)


class ChannelConventionTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.tile(np.array([[10, 70], [30, 90]], dtype=np.uint16), (4, 4))
        self.expected = (0.8, 0.4, 0.5 * np.arctan2(0.4, 0.8))
        self.player = live_gui.BasicVideoPlayer.__new__(live_gui.BasicVideoPlayer)
        self.angle_app = angle_distribution_analysis.AngleDistributionApp.__new__(
            angle_distribution_analysis.AngleDistributionApp
        )

    def test_angle_bounds_agree_with_sensor_and_live(self):
        result = self.angle_app._xy_phi_from_gray_bounds(self.frame, (0, 4, 0, 4))
        np.testing.assert_allclose(result, self.expected, atol=1e-6)
        live_result = self.player._xy_phi_stats_from_raw_window(self.frame)
        np.testing.assert_allclose(result, live_result[:3], atol=1e-6)

    def test_angle_append_uses_sensor_convention(self):
        self.angle_app._spot_bounds_int_all = [(0, 4, 0, 4)]
        self.angle_app._spot_xy_series_all = [[]]
        self.angle_app._spot_phi_series_all = [[]]
        self.angle_app._append_xy_from_frame(self.frame)
        result = (*self.angle_app._spot_xy_series_all[0][0], self.angle_app._spot_phi_series_all[0][0])
        np.testing.assert_allclose(result, self.expected, atol=1e-6)

    def test_auto_capture_uses_sensor_convention_for_all_roi_parities(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            path = out_dir / "capture.npy"
            self.player._spotrec_save_dir = Mock(return_value=out_dir)
            sensor = np.tile(np.array([[10, 70], [30, 90]], dtype=np.uint16), (5, 5))
            for offset_y in (0, 1):
                for offset_x in (0, 1):
                    with self.subTest(offset_x=offset_x, offset_y=offset_y):
                        frame = sensor[offset_y:offset_y + 8, offset_x:offset_x + 8]
                        marker = np.zeros_like(frame)
                        marker[offset_y, offset_x] = 1
                        np.save(path, np.stack([frame, marker]))
                        process = Mock(returncode=0)
                        process.communicate.return_value = (json.dumps({
                            "path": str(path), "actual_fps": 1600.0,
                            "roi": {"OffsetX": offset_x, "OffsetY": offset_y, "Width": 8, "Height": 8},
                        }), "")
                        with patch.object(live_gui.subprocess, "Popen", return_value=process):
                            xy, phi, fps, count = self.player._capture_auto_spot_series(
                                center=(4.0, 4.0), n_frames=1, roi_raw=8, exp_ms=None
                            )
                        self.assertEqual(count, 1)
                        self.assertEqual(fps, 1600.0)
                        np.testing.assert_allclose((*xy[0], phi[0]), self.expected, atol=1e-6)

    def test_playback_signed_channels_and_energy(self):
        self.player._spot_window_size = 5
        self.player.S_MAP_SMOOTH_K = 3
        with patch.object(live_gui.cv2, "boxFilter", wraps=live_gui.cv2.boxFilter) as smooth:
            with patch.object(live_gui.detect_spinners, "to_u8_preview", wraps=live_gui.detect_spinners.to_u8_preview) as preview:
                self.player._spot_playback_windows(self.frame, 4.0, 4.0)
        self.assertEqual(smooth.call_count, 2)
        np.testing.assert_array_equal(smooth.call_args_list[0].args[0], 80.0)
        np.testing.assert_array_equal(smooth.call_args_list[1].args[0], 40.0)
        np.testing.assert_array_equal(preview.call_args.args[0], 8000.0)


class PreviewBrightnessTests(unittest.TestCase):
    def test_spot_preview_stretches_nonzero_values_without_mutating_input(self):
        frame = np.array([[0, 10, 20], [30, 40, 50]], dtype=np.uint16)
        original = frame.copy()
        preview = live_gui.detect_spinners.to_u8_preview(frame, lo_pct=0.0, hi_pct=99.5)
        np.testing.assert_array_equal(preview, [[0, 0, 64], [128, 192, 255]])
        brighter_preview = live_gui.detect_spinners.to_u8_preview(frame * 4, lo_pct=0.0, hi_pct=99.5)
        np.testing.assert_array_equal(brighter_preview, preview)
        np.testing.assert_array_equal(frame, original)


class SignalStub:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        self.callbacks.remove(callback)

    def emit(self, *payload):
        for callback in list(self.callbacks):
            callback(*payload)


class IntensityLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, cli._tk_environment()))
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(contextlib.chdir(directory))
        self.enterContext(patch.object(sys, "path", [str(LIVE_DIR), *sys.path]))
        self.controllers = []
        self.events = []
        self.qt_app = SimpleNamespace(processEvents=lambda: None)
        self.enterContext(patch("PySide6.QtWidgets.QApplication", SimpleNamespace(instance=lambda: self.qt_app)))
        self.enterContext(patch(f"{fetch_frames.Controller.__module__}.Controller", side_effect=self.make_controller))
        self.enterContext(patch.object(fetch_frames.Controller, "open", side_effect=AssertionError("Real camera access")))
        self.errors = self.enterContext(patch.object(live_gui.messagebox, "showerror"))
        try:
            self.root = live_gui.tk.Tk()
        except live_gui.tk.TclError as exc:
            if sys.platform != "win32" and ("no display name" in str(exc) or "couldn't connect to display" in str(exc)):
                self.skipTest(f"Tk display unavailable: {exc}")
            raise
        self.root.withdraw()
        self.addCleanup(self.close_gui)
        self.player = live_gui.BasicVideoPlayer(self.root)
        self.root.update()
        self.player._set_live_zoom_center((32.0, 32.0))
        self.player._live_last_frame = np.full((64, 64), 1024, dtype=np.uint16)

    def close_gui(self):
        if hasattr(self, "player"):
            self.player._close_video()
            self.player._stop_live_intensity_analysis()
            self.player._stop_live_feed()
        for timer in self.root.tk.call("after", "info"):
            self.root.after_cancel(timer)
        self.root.destroy()

    def make_controller(self):
        camera = SimpleNamespace(**{name: SignalStub() for name in ("frame", "frame_timed", "roi", "timing", "gains", "error")})
        controller = Mock(cam=camera)
        index = len(self.controllers)
        for method in ("open", "start", "stop", "close"):
            getattr(controller, method).side_effect = lambda action=method: self.events.append((index, action))
        controller.set_roi.side_effect = lambda width, height, offset_x, offset_y: camera.roi.emit(
            {"Width": width, "Height": height, "OffsetX": offset_x, "OffsetY": offset_y}
        )
        controller.set_timing.side_effect = lambda fps, exposure: camera.timing.emit(
            {"resulting_fps": min(float(fps), 800.0), "exposure_us": float(exposure) * 1000.0}
        )
        controller.set_gains.side_effect = lambda analog, digital: camera.gains.emit(
            {"analog": {"val": analog}, "digital": {"val": digital}}
        )
        self.controllers.append(controller)
        return controller

    def start_analyser(self):
        self.player._live_intensity_start_btn.invoke()
        self.assertTrue(self.player._live_intensity_running)
        self.assertFalse(self.player._live_running)
        self.assertTrue(self.player._live_mag_enabled_var.get())
        return self.controllers[-1]

    def load_intensity_stack(self, *, blank_frame=False):
        rows, columns = np.indices((32, 32))
        brightness = 200 + 2000*np.exp(-((rows-16)**2+(columns-16)**2)/4.5)
        frame = np.empty((64, 64), dtype=np.uint16)
        frame[0::2, 0::2] = brightness*.8
        frame[0::2, 1::2] = brightness*1.2
        frame[1::2, 0::2] = brightness*.8
        frame[1::2, 1::2] = brightness*1.2
        stack = np.repeat(frame[None], 5, axis=0)
        if blank_frame:
            stack[2] = 0
        path = Path.cwd() / "intensity_stack.npy"
        np.save(path, stack)
        metadata = {"actual": {"fps": 80, "phase_marker_appended": False},
                    "roi": {"OffsetX": 0, "OffsetY": 0},
                    "background": {"background_subtracted": False},
                    "conditions": {"recording_role": "Sample", "interference_drive": "On"}}
        path.with_suffix(".json").write_text(json.dumps(metadata))
        self.player._detection_mode_var.set("Intensity")
        self.player._dog_k_std = 1.0
        self.player._dog_k_var.set("1")
        self.assertTrue(self.player._load_npy_source(str(path)))
        self.player.decode_thread.join(5)
        self.player.recon_thread.join(5)
        self.assertFalse(self.player.recon_thread.is_alive())
        while not self.player._ui_queue.empty():
            callback, args, kwargs = self.player._ui_queue.get_nowait()
            callback(*args, **kwargs)
        return path, stack

    def test_intensity_stack_replay_keeps_static_rods_and_native_values(self):
        path, stack = self.load_intensity_stack()
        self.assertEqual(len(self.player._spot_centers), 1)
        self.assertTrue(self.player._analysis_finished)
        self.assertTrue(self.player._abs_range_filter_enabled)
        self.assertIn("disabled", self.player._motion_range_checkbox.state())
        self.assertIn("disabled", self.player._ring_score_entry.state())
        self.player._apply_ring_filter(force=True)
        self.assertEqual(len(self.player._spot_centers), 1)
        np.testing.assert_allclose(self.player._intensity_mean_frame, stack.mean(axis=0))
        self.assertGreater(self.player._intensity_channels.max(), 255)
        self.assertEqual(self.player._intensity_channels.shape, (5, 1, 4))
        np.testing.assert_array_equal(np.load(path), stack)
        self.assertEqual(self.controllers, [])
        self.errors.assert_not_called()

    def test_intensity_distribution_export_keeps_acquisition_metadata(self):
        self.load_intensity_stack()
        result, metadata = self.player._intensity_distribution_data()
        path = Path.cwd() / "orientation.npz"
        with patch.object(live_gui.filedialog, "asksaveasfilename", return_value=str(path)):
            self.player._save_intensity_distribution(result, metadata)
        with np.load(path, allow_pickle=False) as saved:
            self.assertEqual(saved["channels"].shape, (5, 1, 4))
            self.assertEqual(saved["directions"].shape, (1, 1, 3))
            exported = json.loads(saved["metadata_json"].item())
            self.assertEqual(exported["source_metadata"][0]["conditions"]["interference_drive"], "On")
            self.assertEqual(exported["averaging"], "3D Cartesian")
            self.assertEqual(exported["origin_source"], "recording_metadata")
        self.assertFalse(path.with_suffix(".npz.tmp").exists())
        self.player._open_intensity_distribution()
        self.root.update_idletasks()
        self.assertIsNotNone(self.player._intensity_distribution_window)
        self.errors.assert_not_called()

    def test_intensity_mode_can_switch_back_to_time_variation(self):
        self.load_intensity_stack()
        self.player._detection_mode_var.set("Time variation")
        self.player._apply_spot_params()
        self.player.decode_thread.join(5)
        self.player.recon_thread.join(5)
        self.assertFalse(self.player.recon_thread.is_alive())
        while not self.player._ui_queue.empty():
            callback, args, kwargs = self.player._ui_queue.get_nowait()
            callback(*args, **kwargs)
        self.assertEqual(self.player._analysis_detection_mode, "Time variation")
        self.assertNotIn("disabled", self.player._motion_range_checkbox.state())
        self.assertEqual(len(self.player._spot_centers), 0)
        self.errors.assert_not_called()

    def test_cancelled_intensity_worker_does_not_publish_completion(self):
        self.player._analysis_detection_mode = "Intensity"
        self.player.stop_event.set()
        self.player._intensity_recon_worker((8, 8))
        self.assertFalse(self.player._analysis_finished)
        self.assertIsNone(self.player._intensity_channels)

    def test_intensity_decode_error_cancels_partial_analysis(self):
        self.player._analysis_detection_mode = "Intensity"
        self.player.npy_frames = np.zeros((1, 8, 8), dtype=np.uint16)
        self.player.npy_has_frames_dim = True
        self.player.frame_count = 1
        self.player.stop_event.clear()
        with patch.object(self.player, "_analysis_frame", side_effect=ValueError("Invalid frame format")):
            self.player._decode_worker_npy()
        self.assertTrue(self.player.stop_event.is_set())
        self.assertTrue(self.player.decode_done)
        self.root.update()
        self.errors.assert_called_once()
        self.assertFalse(self.player._analysis_finished)

    def test_intensity_origin_falls_back_to_saved_marker_not_display_coordinates(self):
        self.player._source_recording_metadata = []
        self.player._source_phase = {"phase_x": 1, "phase_y": 0}
        self.assertEqual(self.player._recorded_intensity_origin(), ((1, 0), "phase_marker"))

    def test_intensity_blank_frame_remains_invalid_without_crashing_plots(self):
        self.load_intensity_stack(blank_frame=True)
        result, _ = self.player._intensity_distribution_data()
        self.assertEqual(result["valid_counts"][0, 0], 4)
        self.assertTrue(np.isfinite(result["directions"]).all())
        self.errors.assert_not_called()

    def test_interference_unknown_is_not_inferred_from_sound_checkbox(self):
        self.player._sound_on_var.set(True)
        self.assertEqual(self.player._capture_conditions()["interference_drive"], "Unknown")
        self.player._interference_state_var.set("Off")
        self.player._capture_role_var.set("Background")
        self.assertEqual(self.player._capture_conditions()["recording_role"], "Background")
        self.assertEqual(self.player._capture_conditions()["interference_drive"], "Off")

    def test_fetch_stack_sidecar_preserves_background_and_interference_states(self):
        path = Path.cwd() / "captured.npy"
        np.save(path, np.full((2, 8, 8), 1024, dtype=np.uint16))
        output = {"path": str(path), "count": 2, "actual_fps": 80,
                  "background_subtracted": False, "phase_marker_appended": False,
                  "roi": {"OffsetX": 2, "OffsetY": 4}}
        self.player._interference_state_var.set("On")
        conditions = self.player._capture_conditions()
        with patch.object(live_gui.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(output), stderr="")):
            self.player._capture_frames_to_npy(path, 2, 80, 1.2, 1, False, conditions=conditions)
        saved = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(saved["conditions"]["interference_drive"], "On")
        self.assertFalse(saved["background"]["background_subtracted"])
        self.assertEqual(saved["actual"]["frames"], 2)

    def test_stop_button_closes_roi_camera_before_restarting_live(self):
        analyser = self.start_analyser()
        self.player._live_intensity_preview_queue.put(np.ones((14, 14), dtype=np.uint8))
        self.player._spot_centers = [(12.0, 12.0)]
        self.player._live_intensity_stop_btn.invoke()
        live = self.controllers[-1]
        self.assertIsNot(live, analyser)
        self.assertEqual(self.events[-3:], [(0, "close"), (1, "open"), (1, "start")])
        live.full_sensor.assert_called_once()
        live.set_timing.assert_called_once_with(10.0, 1.2)
        self.assertFalse(self.player._live_intensity_running)
        self.assertTrue(self.player._live_running)
        self.assertEqual(self.player._live_zoom_center, (32.0, 32.0))
        self.assertTrue(self.player._live_mag_enabled_var.get())
        self.assertTrue(self.player._live_intensity_preview_queue.empty())
        self.assertIsNone(self.player._live_intensity_after_id)
        self.assertIsNone(self.player._live_intensity_controller)
        self.assertIn("disabled", self.player._live_intensity_stop_btn.state())
        for signal in vars(analyser.cam).values():
            self.assertEqual(signal.callbacks, [])
        self.errors.assert_not_called()

    def test_returned_live_frames_update_the_magnifier(self):
        self.start_analyser()
        self.player._live_intensity_stop_btn.invoke()
        self.root.geometry("1600x1000")
        self.root.deiconify()
        self.root.update_idletasks()
        self.root.withdraw()
        live = self.controllers[-1]
        previous_image = None
        for value in (512, 1024):
            live.cam.frame.emit(np.full((64, 64), value, dtype=np.uint16))
            with patch.object(self.root, "after", return_value=None):
                self.player._live_tick()
            image = self.player._live_zoom_label.cget("image")
            self.assertTrue(image)
            self.assertNotEqual(str(image), str(self.player._live_zoom_blank_ref))
            self.assertNotEqual(str(image), previous_image)
            previous_image = str(image)
            self.assertEqual(self.player._live_mag_max_var.get(), f"Magnifier max pixel: {value >> 4}")

    def test_internal_stop_does_not_restart_camera(self):
        self.start_analyser()
        self.player._stop_live_intensity_analysis()
        self.assertEqual(len(self.controllers), 1)
        self.assertFalse(self.player._live_running)
        self.assertFalse(self.player._live_intensity_running)

    def test_repeated_cycles_release_each_controller_and_enable_magnifier(self):
        for cycle in range(3):
            analyser = self.start_analyser()
            self.player._live_intensity_stop_btn.invoke()
            self.player._on_stop_live_intensity_analysis()
            self.assertEqual(len(self.controllers), 2 * (cycle + 1))
            analyser.close.assert_called_once()
            self.assertEqual(self.player._live_zoom_center, (32.0, 32.0))
            self.assertTrue(self.player._live_mag_enabled_var.get())
            self.assertTrue(self.player._live_running)
        for controller in self.controllers[:-1]:
            controller.close.assert_called_once()

    def test_live_start_is_ignored_while_roi_analysis_owns_camera(self):
        self.start_analyser()
        self.player._start_live_feed()
        self.assertEqual(len(self.controllers), 1)
        self.assertFalse(self.player._live_running)

    def test_leaving_live_tab_stops_analysis_without_reopening_live(self):
        self.start_analyser()
        self.player._notebook.select(self.player._analysis_tab)
        self.player._on_tab_changed()
        self.assertFalse(self.player._live_intensity_running)
        self.assertFalse(self.player._live_running)
        self.assertEqual(len(self.controllers), 1)

    def test_stop_does_not_open_live_during_another_capture(self):
        self.start_analyser()
        self.player._live_capture_running = True
        self.player._live_intensity_stop_btn.invoke()
        self.assertFalse(self.player._live_running)
        self.assertEqual(len(self.controllers), 1)

    def test_start_error_cleans_up_and_restores_previous_live_view(self):
        self.player._start_live_feed()
        failed = self.make_controller()
        replacement = self.make_controller()
        failed.set_roi.side_effect = RuntimeError("ROI rejected")
        with patch(f"{fetch_frames.Controller.__module__}.Controller", side_effect=[failed, replacement]):
            self.player._live_intensity_start_btn.invoke()
        failed.stop.assert_called_once()
        failed.close.assert_called_once()
        self.assertTrue(self.player._live_running)
        self.assertFalse(self.player._live_intensity_running)
        self.assertEqual(self.player._live_zoom_center, (32.0, 32.0))
        self.errors.assert_called_once()

    def test_window_shutdown_does_not_restart_live_camera(self):
        analyser = self.start_analyser()
        with patch.object(self.root, "destroy") as destroy:
            self.player.on_close()
        destroy.assert_called_once()
        analyser.close.assert_called_once()
        self.assertEqual(len(self.controllers), 1)
        self.assertFalse(self.player._live_running)
        self.assertFalse(self.player._live_intensity_running)
        self.assertIsNone(self.player._ui_pump_after_id)

    def test_tick_does_not_reschedule_after_stop_during_qt_events(self):
        self.start_analyser()
        with patch.object(self.qt_app, "processEvents", side_effect=self.player._stop_live_intensity_analysis):
            with patch.object(self.root, "after") as schedule:
                self.player._live_intensity_tick()
        schedule.assert_not_called()
        self.assertIsNone(self.player._live_intensity_after_id)

    def test_intensity_measurements_keep_native_precision(self):
        analyser = self.start_analyser()
        for value in (1025, 1026):
            analyser.cam.frame_timed.emit(np.full((14, 14), value, dtype=np.uint16))
        self.assertEqual([sample[1] for sample in self.player._live_intensity_buffer], [1025.0, 1026.0])

    def test_absent_background_is_not_searched_again_per_frame(self):
        self.player._live_background_subtract_enabled = True
        with patch.object(self.player, "_load_background_profile", return_value=None) as load:
            analyser = self.start_analyser()
            for _ in range(10):
                analyser.cam.frame_timed.emit(np.full((14, 14), 1025, dtype=np.uint16))
        load.assert_called_once()
        self.assertEqual(self.player._live_intensity_buffer[-1][1], 1025.0)

    def test_cached_background_crop_preserves_subtraction(self):
        self.player._live_background_subtract_enabled = True
        profile = np.full((64, 64), 32.0, dtype=np.float32)
        with patch.object(self.player, "_load_background_profile", return_value=profile) as load:
            analyser = self.start_analyser()
            with patch.object(self.player, "_crop_background_profile", wraps=self.player._crop_background_profile) as crop:
                for _ in range(5):
                    analyser.cam.frame_timed.emit(np.full((14, 14), 1025, dtype=np.uint16))
        load.assert_called_once()
        self.assertEqual(self.player._live_intensity_buffer[-1][1], 993.0)
        self.assertEqual(crop.call_count, 6)
        self.player._live_background_subtract_var.set(False)
        self.player._on_live_background_toggle()
        self.assertEqual(len(self.player._live_intensity_buffer), 0)
        analyser.cam.frame_timed.emit(np.full((14, 14), 1025, dtype=np.uint16))
        self.assertEqual(self.player._live_intensity_buffer[-1][1], 1025.0)

    def test_intensity_plot_reuses_figure_and_canvas(self):
        self.player._live_intensity_buffer = deque((index / 100.0, 1000.0 + index % 10) for index in range(100))
        self.player._live_render_intensity_plot(force=True)
        figure = self.player._live_intensity_figure
        canvas = self.player._live_intensity_canvas
        self.assertIsNotNone(figure)
        self.player._live_render_intensity_plot(force=True)
        self.assertIs(self.player._live_intensity_figure, figure)
        self.assertIs(self.player._live_intensity_canvas, canvas)
        self.assertEqual(len(self.player._live_intensity_axis.lines), 1)

    def emit_tuning_trial(self, analyser, kind, amplitude, start):
        self.player._intensity_tuning_duration_var.set("2")
        self.player._intensity_tuning_label_var.set(kind)
        self.player._intensity_tuning_confirm_var.set(True)
        with patch.object(live_gui.time, "perf_counter", return_value=start):
            analyser.cam.frame_timed.emit(np.full((14, 14), 1000, dtype=np.uint16), start)
            self.player._begin_intensity_trial(kind)
        self.assertIsNotNone(self.player._intensity_tuner.active)
        for index in range(1600):
            stamp = start + 0.5 + index / 800.0
            value = int(round(1000.0 + amplitude * np.sin(2.0 * np.pi * index / 200.0)))
            with patch.object(live_gui.time, "perf_counter", return_value=stamp):
                analyser.cam.frame_timed.emit(np.full((14, 14), value, dtype=np.uint16), stamp)
        with patch.object(live_gui.time, "perf_counter", return_value=start + 2.5):
            self.player._update_intensity_tuning()

    def test_tuning_window_reports_target_and_exports_native_trials(self):
        analyser = self.start_analyser()
        self.player._open_intensity_tuning()
        self.emit_tuning_trial(analyser, "reference", 200.0, 100.0)
        self.assertIsNotNone(self.player._intensity_tuner.reference)
        self.emit_tuning_trial(analyser, "test", 10.0, 103.0)
        result = self.player._intensity_tuner.last
        self.assertTrue(result["comparison"]["target_reached"])
        self.assertIn("10x observed target", self.player._intensity_tuning_status_var.get())
        self.assertEqual(len(self.player._intensity_tuning_tree.get_children()), 2)
        path = Path.cwd() / "trial_export.json"
        with patch.object(live_gui.filedialog, "asksaveasfilename", return_value=str(path)):
            self.player._save_intensity_trials()
        payload = json.loads(path.read_text())
        self.assertEqual(len(payload["trials"]), 2)
        self.assertEqual(len(payload["trials"][1]["intensity_dn"]), 1600)
        self.assertEqual(payload["trials"][0]["context"]["exposure_us"], 1200.0)

    def test_settings_change_invalidates_tuning_reference(self):
        analyser = self.start_analyser()
        self.emit_tuning_trial(analyser, "reference", 100.0, 100.0)
        analyser.cam.timing.emit({"resulting_fps": 800.0, "exposure_us": 600.0})
        with patch.object(live_gui.time, "perf_counter", return_value=102.5):
            self.player._update_intensity_tuning()
        self.assertIsNone(self.player._intensity_tuner.reference)
        self.assertIn("changed", self.player._intensity_tuning_status_var.get())

    def test_tuning_requires_sweep_confirmation_and_background_off(self):
        self.start_analyser()
        self.player._begin_intensity_trial("reference")
        self.assertIsNone(self.player._intensity_tuner.active)
        self.assertIn("Confirm", self.player._intensity_tuning_status_var.get())
        self.player._intensity_tuning_confirm_var.set(True)
        self.player._live_background_subtract_enabled = True
        self.player._begin_intensity_trial("reference")
        self.assertIsNone(self.player._intensity_tuner.active)
        self.assertIn("subtraction off", self.player._intensity_tuning_status_var.get())

    def test_stop_clears_comparison_and_disables_tuning_capture_buttons(self):
        analyser = self.start_analyser()
        self.player._open_intensity_tuning()
        self.emit_tuning_trial(analyser, "reference", 100.0, 100.0)
        self.player._stop_live_intensity_analysis()
        self.assertIsNone(self.player._intensity_tuner.reference)
        self.assertEqual(len(self.player._intensity_tuner.trials), 1)
        self.assertIn("disabled", self.player._intensity_tuning_buttons["reference"].state())
        self.assertIn("disabled", self.player._intensity_tuning_buttons["test"].state())
        self.assertNotIn("disabled", self.player._intensity_tuning_buttons["save"].state())
        self.assertEqual(self.player._intensity_tuning_reference_var.get(), "Reference: -")

    def test_live_sidebar_can_scroll_to_analyser_controls(self):
        self.root.geometry("1600x1000")
        self.root.deiconify()
        self.root.update()
        canvas = self.player._live_controls_canvas
        canvas.yview_moveto(1.0)
        self.root.update()
        button = self.player._live_intensity_start_btn
        self.assertTrue(button.winfo_ismapped())
        self.assertGreaterEqual(button.winfo_rooty(), canvas.winfo_rooty())
        self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), canvas.winfo_rooty() + canvas.winfo_height())

    def begin_diagnostic(self, start=100.0):
        self.player._intensity_diagnostic_duration_var.set("10")
        path = Path.cwd() / "diagnostic.npz"
        with patch.object(live_gui.filedialog, "asksaveasfilename", return_value=str(path)):
            with patch.object(live_gui.time, "perf_counter", return_value=start):
                self.player._begin_intensity_diagnostic()
        self.assertTrue(self.player._intensity_diagnostic.active)
        return path

    def test_diagnostic_records_bad_timing_and_raw_corrected_values_without_reference(self):
        self.player._live_background_subtract_enabled = True
        with patch.object(self.player, "_load_background_profile", return_value=np.full((64, 64), 32.0)):
            analyser = self.start_analyser()
        self.player._open_intensity_tuning()
        path = self.begin_diagnostic()
        self.assertIsNone(self.player._intensity_tuner.reference)
        self.assertFalse(self.player._intensity_tuning_confirm_var.get())
        raw = np.tile(np.array([[100, 700], [300, 900]], dtype=np.uint16), (7, 7))
        with patch.object(live_gui.time, "perf_counter", return_value=101.8):
            analyser.cam.frame_timed.emit(raw, 101.0)
            self.player._intensity_tuning_label_var.set("Z held still")
            self.player._mark_intensity_diagnostic()
        with patch.object(live_gui.time, "perf_counter", return_value=102.0):
            self.player._finish_intensity_diagnostic()
        with np.load(path, allow_pickle=False) as saved:
            row = dict(zip(saved["columns"], saved["samples"][0]))
            self.assertEqual(row["raw_mean_dn"], 500.0)
            self.assertEqual(row["analysed_mean_dn"], 468.0)
            self.assertEqual(row["i0_mean_dn"], 900.0)
            self.assertAlmostEqual(row["processed_s"] - row["received_s"], 0.8)
            metadata = json.loads(str(saved["metadata_json"]))
            self.assertTrue(any(event["label"] == "Z held still" for event in metadata["events"]))
            self.assertTrue(metadata["initial_settings"]["background_requested"])
        self.assertTrue(self.player._live_intensity_running)

    def test_diagnostic_saves_empty_stalled_stream_and_missing_readbacks(self):
        self.start_analyser()
        self.player._live_intensity_timing_snapshot = None
        self.player._live_intensity_gains_snapshot = None
        path = self.begin_diagnostic()
        with patch.object(live_gui.time, "perf_counter", return_value=110.3):
            self.player._update_intensity_tuning()
        self.assertFalse(self.player._intensity_diagnostic.active)
        with np.load(path, allow_pickle=False) as saved:
            self.assertEqual(saved["samples"].shape, (0, 13))
            metadata = json.loads(str(saved["metadata_json"]))
            self.assertEqual(metadata["stop_reason"], "duration")
            self.assertIsNone(metadata["initial_settings"]["timing_readback"])

    def test_stopping_analyser_saves_diagnostic_before_returning_to_live(self):
        analyser = self.start_analyser()
        path = self.begin_diagnostic()
        with patch.object(live_gui.time, "perf_counter", return_value=101.0):
            analyser.cam.frame_timed.emit(np.full((14, 14), 1000, dtype=np.uint16), 101.0)
            self.player._on_stop_live_intensity_analysis()
        self.assertTrue(self.player._live_running)
        with np.load(path, allow_pickle=False) as saved:
            self.assertEqual(len(saved["samples"]), 1)
            self.assertEqual(json.loads(str(saved["metadata_json"]))["stop_reason"], "analyser_stopped")

    def test_diagnostic_save_failure_keeps_data_and_blocks_replacement(self):
        self.start_analyser()
        path = self.begin_diagnostic()
        with patch.object(self.player._intensity_diagnostic, "save", side_effect=OSError("disk full")):
            self.player._finish_intensity_diagnostic()
        self.assertFalse(self.player._intensity_diagnostic_saved)
        self.errors.assert_called_once()
        self.player._begin_intensity_diagnostic()
        self.assertFalse(self.player._intensity_diagnostic.active)
        self.assertIn("previous", self.player._intensity_diagnostic_status_var.get())
        self.player._save_intensity_diagnostic()
        self.assertTrue(path.is_file())
        self.assertTrue(self.player._intensity_diagnostic_saved)

    def test_unsaved_diagnostic_prevents_window_teardown(self):
        self.start_analyser()
        self.player._open_intensity_tuning()
        self.begin_diagnostic()
        with patch.object(self.player._intensity_diagnostic, "save", side_effect=OSError("disk full")):
            with patch.object(self.player, "_show_error"):
                with patch.object(self.player._intensity_tuning_window, "destroy") as destroy_tuning:
                    self.player._close_intensity_tuning()
                destroy_tuning.assert_not_called()
                with patch.object(self.root, "destroy") as destroy:
                    self.player.on_close()
                destroy.assert_not_called()
        self.assertIsNotNone(self.player._intensity_diagnostic.data)
        self.assertFalse(self.player._live_intensity_running)
        self.player._save_intensity_diagnostic()


class FrameTimingTests(unittest.TestCase):
    def test_worker_timestamp_survives_queued_delivery(self):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        application = QApplication.instance() or QApplication([])
        worker = _StreamWorker(None, None)
        camera = IDSCamera()
        worker.frame_timed.connect(camera._publish_frame, Qt.QueuedConnection)
        original = []
        timed = []
        camera.frame.connect(original.append)
        camera.frame_timed.connect(lambda frame, stamp: timed.append((frame, stamp, threading.get_ident())))
        frame = np.full((14, 14), 1000, dtype=np.uint16)
        producer = threading.Thread(target=lambda: worker.frame_timed.emit(frame, 123.456))
        producer.start()
        producer.join(1.0)
        self.assertFalse(producer.is_alive())
        self.assertEqual(timed, [])
        application.processEvents()
        self.assertEqual(len(timed), 1)
        self.assertIs(original[0], frame)
        self.assertIs(timed[0][0], frame)
        self.assertEqual(timed[0][1], 123.456)
        self.assertEqual(timed[0][2], threading.get_ident())

    def test_timestamp_relay_preserves_the_original_frame_signal(self):
        camera = IDSCamera()
        original = []
        timed = []
        camera.frame.connect(original.append)
        camera.frame_timed.connect(lambda frame, received_at: timed.append((frame, received_at)))
        frame = np.full((14, 14), 1000, dtype=np.uint16)
        camera._publish_frame(frame, 123.456)
        self.assertEqual(len(original), 1)
        self.assertEqual(len(timed), 1)
        self.assertIs(original[0], frame)
        self.assertIs(timed[0][0], frame)
        self.assertEqual(timed[0][1], 123.456)


class CaptureReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.now = 0.0
        self.polls = 0
        self.started = False
        self.frames = deque()
        self.on_poll = None
        self.camera = SimpleNamespace(
            **{name: SignalStub() for name in ("frame", "roi", "timing", "gains", "error")}
        )
        self.controller = Mock(cam=self.camera)
        self.controller.start.side_effect = lambda: setattr(self, "started", True)
        self.controller.stop.side_effect = lambda: setattr(self, "started", False)
        self.controller.full_sensor.side_effect = lambda: self.camera.roi.emit(
            {"OffsetX": 0, "OffsetY": 0, "Width": 8, "Height": 8}
        )
        self.controller.set_timing.side_effect = lambda *_: self.camera.timing.emit(
            {"resulting_fps": 77.0, "exposure_us": 600.0}
        )
        application = SimpleNamespace(processEvents=self.process_events)
        self.enterContext(patch.object(fetch_frames, "Controller", return_value=self.controller))
        self.enterContext(patch.object(fetch_frames, "QApplication", SimpleNamespace(instance=lambda: application)))
        self.enterContext(patch.object(fetch_frames, "monotonic", side_effect=lambda: self.now))
        self.enterContext(patch.object(fetch_frames, "sleep", return_value=None))

    def process_events(self):
        self.polls += 1
        self.assertLess(self.polls, 100, "Capture did not exit before the test watchdog.")
        self.now += 0.1
        if self.started and self.frames:
            self.camera.frame.emit(self.frames.popleft())
        if self.on_poll is not None:
            self.on_poll()

    def capture(self, **kwargs):
        options = {"n_frames": 2, "stop_after": 2, "frame_timeout_s": 0.5}
        options.update(kwargs)
        return fetch_frames.fetch_frames(Path(self.directory.name), **options)

    def assert_cleaned_up(self):
        self.controller.stop.assert_called()
        self.controller.close.assert_called_once()
        for signal in vars(self.camera).values():
            self.assertEqual(signal.callbacks, [])

    def test_open_error_fails_without_starting_acquisition(self):
        self.controller.open.side_effect = lambda: self.camera.error.emit("No device found!")
        with self.assertRaisesRegex(RuntimeError, "No device found"):
            self.capture()
        self.controller.start.assert_not_called()
        self.assert_cleaned_up()

    def test_no_frames_times_out(self):
        with self.assertRaisesRegex(RuntimeError, "No camera frames"):
            self.capture()
        self.assertLess(self.now, 1.0)
        self.assert_cleaned_up()

    def test_stalled_stream_times_out_without_reporting_success(self):
        self.frames.extend(np.full((8, 8), value, dtype=np.uint16) for value in (100, 101))
        with self.assertRaisesRegex(RuntimeError, "No camera frames"):
            self.capture()
        self.assertEqual(list(Path(self.directory.name).glob("*.npy")), [])
        self.assert_cleaned_up()

    def test_camera_error_during_acquisition_is_reported(self):
        self.on_poll = lambda: self.camera.error.emit("Stream disconnected") if self.started else None
        with self.assertRaisesRegex(RuntimeError, "Stream disconnected"):
            self.capture()
        self.assert_cleaned_up()

    def test_success_preserves_depth_marker_and_actual_fps(self):
        self.frames.extend(np.full((8, 8), value, dtype=np.uint16) for value in (999, 1000, 1001))
        preview_path = Path(self.directory.name) / "preview.npy"
        path, actual_fps, count, metadata = self.capture(fps=1600.0, preview_path=preview_path, preview_every=1)
        stack = np.load(path, allow_pickle=False)
        preview = np.load(preview_path, allow_pickle=False)
        self.assertEqual(preview.dtype, np.uint16)
        np.testing.assert_array_equal(preview, np.full((8, 8), 1001, dtype=np.uint16))
        self.assertEqual(stack.dtype, np.uint16)
        self.assertEqual(stack.shape, (3, 8, 8))
        np.testing.assert_array_equal(stack[:, 0, 0], [1000, 1001, 1])
        self.assertEqual(int(stack[-1].sum()), 1)
        self.assertEqual(count, 2)
        self.assertEqual(actual_fps, 77.0)
        self.assertTrue(metadata["phase_marker_appended"])
        self.assert_cleaned_up()

    def test_timeout_is_inactivity_not_total_recording_duration(self):
        self.frames.extend(np.full((8, 8), value, dtype=np.uint16) for value in range(8))
        _, _, count, _ = self.capture(n_frames=7, stop_after=7, frame_timeout_s=0.25)
        self.assertEqual(count, 7)
        self.assertGreater(self.now, 0.25)
        self.assert_cleaned_up()

    def test_stop_flag_finishes_a_partial_recording(self):
        self.frames.extend(np.full((8, 8), value, dtype=np.uint16) for value in (100, 101))
        stop_flag = Mock()
        stop_flag.exists.side_effect = lambda: self.started and not self.frames
        _, _, count, _ = self.capture(stop_flag=stop_flag)
        self.assertEqual(count, 1)
        self.assert_cleaned_up()

    def test_invalid_timeout_is_rejected_before_opening_camera(self):
        for timeout in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "Frame timeout"):
                    self.capture(frame_timeout_s=timeout)
        self.controller.open.assert_not_called()


class RecordingReloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "capture.npy"
        self.frame = np.tile(np.array([[10, 70], [30, 90]], dtype=np.uint16), (4, 4))
        self.marker = np.zeros_like(self.frame)
        self.marker[0, 1] = 1
        self.stack = np.stack([self.frame, self.frame, self.marker])
        np.save(self.path, self.stack)
        self.player = live_gui.BasicVideoPlayer.__new__(live_gui.BasicVideoPlayer)
        self.player.root = Mock()
        self.player.npy_frames = None
        self.player._start_after_load = Mock()
        self.angle_app = angle_distribution_analysis.AngleDistributionApp.__new__(
            angle_distribution_analysis.AngleDistributionApp
        )
        self.angle_app._fps_mode_var = SimpleNamespace(get=lambda: "Auto")
        self.loaded_arrays = []
        self.addCleanup(self.close_mappings)
        self.prompt = self.enterContext(patch.object(live_gui.simpledialog, "askfloat", return_value=77.0))
        self.errors = self.enterContext(patch.object(live_gui.messagebox, "showerror"))

    def close_mappings(self):
        for array in [self.player.npy_frames, *self.loaded_arrays]:
            while array is not None:
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
                    break
                array = getattr(array, "base", None)
        self.player.npy_frames = None

    def write_metadata(self, payload):
        self.path.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")

    def test_live_reload_uses_actual_fps_and_excludes_marker(self):
        self.write_metadata({"requested": {"fps": 2000.0}, "actual": {"fps": 1640.0, "phase_marker_appended": True}})
        self.assertTrue(self.player._load_npy_source(str(self.path)))
        self.assertEqual(self.player.source_fps, 1640.0)
        self.assertEqual(self.player.frame_count, 2)
        np.testing.assert_array_equal(self.player.npy_frames, self.stack[:2])
        self.prompt.assert_not_called()
        self.errors.assert_not_called()

    def test_angle_reload_excludes_marker_and_preserves_native_values(self):
        self.write_metadata({"actual": {"fps": 1640.0, "phase_marker_appended": True}})
        frames, has_frames, count, _ = self.angle_app._load_npy(self.path)
        self.loaded_arrays.append(frames)
        self.assertTrue(has_frames)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(frames, self.stack[:2])
        self.assertEqual(self.angle_app._resolve_fps(self.path, self.frame.shape, False), 1640.0)

    def test_old_stack_uses_explicit_fps_and_legacy_marker_detection(self):
        self.assertTrue(self.player._load_npy_source(str(self.path)))
        self.assertEqual(self.player.frame_count, 2)
        self.assertEqual(self.player.source_fps, 77.0)
        self.prompt.assert_called_once()

    def test_requested_fps_is_not_treated_as_measured_fps(self):
        self.write_metadata({"requested": {"fps": 2000.0}})
        self.assertTrue(self.player._load_npy_source(str(self.path)))
        self.prompt.assert_called_once()
        self.assertEqual(self.player.source_fps, 77.0)

    def test_cancel_fps_prompt_does_not_start_analysis(self):
        self.prompt.return_value = None
        self.assertFalse(self.player._load_npy_source(str(self.path)))
        self.player._start_after_load.assert_not_called()

    def test_angle_auto_mode_does_not_guess_missing_fps(self):
        with self.assertRaisesRegex(RuntimeError, "No recorded FPS"):
            self.angle_app._resolve_fps(self.path, (14, 256), True)
        self.angle_app._fps_mode_var = SimpleNamespace(get=lambda: "Manual")
        self.angle_app._fps_manual_var = SimpleNamespace(get=lambda: "78")
        self.assertEqual(self.angle_app._resolve_fps(self.path, self.frame.shape, False), 78.0)

    def test_explicit_no_marker_preserves_one_hot_measurement(self):
        self.write_metadata({"actual": {"fps": 77.0, "phase_marker_appended": False}})
        self.assertTrue(self.player._load_npy_source(str(self.path)))
        self.assertEqual(self.player.frame_count, 3)

    def test_declared_marker_must_match_the_last_frame(self):
        with self.assertRaisesRegex(ValueError, "last frame"):
            recording_io.strip_phase_marker(self.stack[:2], marker_appended=True)

    def test_marker_removal_is_a_view_and_preserves_phase(self):
        roi = {}
        frames = recording_io.strip_phase_marker(self.stack, roi_meta=roi)
        self.assertTrue(np.shares_memory(frames, self.stack))
        self.assertEqual(roi, {"phase_x": 1, "phase_y": 0})

    def test_fps_metadata_variants_and_invalid_values(self):
        for payload in (
            {"actual_fps": 123.0},
            {"actual": {"timing_snapshot": {"resulting_fps": 123.0}}},
            {"fetch_frames_output": {"actual_fps": 123.0}},
            {"timing": {"fps": 123.0}},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(recording_io.recording_fps([payload]), 123.0)
        for value in (None, True, 0, -1, "nan", "inf"):
            with self.subTest(value=value):
                self.assertIsNone(recording_io.recording_fps([{"actual": {"fps": value}}]))


class WorkerShutdownTests(unittest.TestCase):
    def setUp(self):
        self.player = live_gui.BasicVideoPlayer.__new__(live_gui.BasicVideoPlayer)
        self.player.stop_event = threading.Event()
        self.player.frame_q = queue.Queue()
        self.player.bottom_var = Mock()
        self.player.root = Mock()
        self.player.decode_thread = None
        self.player.recon_thread = None

    def test_unfinished_worker_retains_stop_signal_resources_and_reference(self):
        worker = Mock()
        worker.is_alive.return_value = True
        self.player.decode_thread = worker
        frame = object()
        self.player.frame_q.put(frame)
        self.assertFalse(self.player._stop_workers())
        self.assertTrue(self.player.stop_event.is_set())
        self.assertIs(self.player.decode_thread, worker)
        self.assertIs(self.player.frame_q.get_nowait(), frame)
        worker.is_alive.return_value = False
        self.assertTrue(self.player._stop_workers())
        self.assertIsNone(self.player.decode_thread)

    def test_cooperative_workers_exit_and_queue_is_drained(self):
        workers = [threading.Thread(target=self.player.stop_event.wait, args=(1.0,)) for _ in range(2)]
        for worker in workers:
            worker.start()
            self.addCleanup(worker.join, 1.0)
        self.addCleanup(self.player.stop_event.set)
        self.player.decode_thread, self.player.recon_thread = workers
        self.player.frame_q.put(object())
        self.assertTrue(self.player._stop_workers())
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertTrue(self.player.stop_event.is_set())
        self.assertTrue(self.player.frame_q.empty())
        self.assertIsNone(self.player.decode_thread)
        self.assertIsNone(self.player.recon_thread)

    def test_close_does_not_reset_or_release_a_source_still_in_use(self):
        self.player._stop_workers = Mock(return_value=False)
        self.player._clear_all_caches = Mock()
        self.player.cap = Mock()
        frames = np.ones((2, 8, 8), dtype=np.uint16)
        self.player.npy_frames = frames
        self.assertFalse(self.player._close_video())
        self.player._clear_all_caches.assert_not_called()
        self.player.cap.release.assert_not_called()
        self.assertIs(self.player.npy_frames, frames)

    def test_open_does_not_replace_source_until_workers_stop(self):
        self.player._close_video = Mock(return_value=False)
        self.player._load_source = Mock(return_value=True)
        with patch.object(live_gui.filedialog, "askopenfilename", return_value="next.npy"):
            self.player.open_video()
            self.player._load_source.assert_not_called()
            self.player._close_video.return_value = True
            self.player.open_video()
        self.player._load_source.assert_called_once_with("next.npy")

    def test_window_stays_alive_if_analysis_has_not_stopped(self):
        self.player._stop_live_intensity_analysis = Mock()
        self.player._stop_live_feed = Mock()
        self.player._stop_spotrec = Mock()
        self.player._stop_spotrec_preview_loop = Mock()
        self.player._close_video = Mock(return_value=False)
        self.player.on_close()
        self.player.root.destroy.assert_not_called()
        self.player._close_video.return_value = True
        self.player.on_close()
        self.player.root.destroy.assert_called_once()

    def test_cancelled_analysis_does_not_publish_completion(self):
        self.player.stop_event.set()
        self.player._ui_call = Mock()
        self.player._init_spot_analysis = Mock()
        self.player._recon_worker((8, 8))
        self.player._ui_call.assert_not_called()
        self.player._init_spot_analysis.assert_not_called()


if __name__ == "__main__":
    unittest.main()