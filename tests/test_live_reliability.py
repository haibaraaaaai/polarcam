from collections import deque
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


LIVE_DIR = Path(__file__).resolve().parents[1] / "src" / "polarcam" / "live"
with patch.object(sys, "path", [str(LIVE_DIR), *sys.path]):
    from polarcam.live import angle_distribution_analysis, fetch_frames, recording_io
    from polarcam.live import Spinners_gui_live as live_gui


class SignalStub:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        self.callbacks.remove(callback)

    def emit(self, payload):
        for callback in list(self.callbacks):
            callback(payload)


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
        path, actual_fps, count, metadata = self.capture(fps=1600.0)
        stack = np.load(path, allow_pickle=False)
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