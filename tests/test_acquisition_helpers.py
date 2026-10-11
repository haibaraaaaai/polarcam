from pathlib import Path
import tempfile
import unittest

import numpy as np

from polarcam.live.acquisition_helpers import (
    brightness_source, capture_path, filename_label, preview_u8, save_raw_peak, spot_brightness,
)


class AcquisitionHelpersTests(unittest.TestCase):
    def test_dim_mono12_does_not_turn_into_bright_eight_bit_image(self):
        low = np.array([[0, 16, 255, 256, 4095]], dtype=np.uint16)
        np.testing.assert_array_equal(preview_u8(low), [[0, 1, 16, 16, 255]])
        np.testing.assert_array_equal(preview_u8(low[:, :3]), [[0, 1, 16]])
        np.testing.assert_array_equal(preview_u8(np.array([[255]], dtype=np.uint8)), [[255]])

    def test_labels_cannot_escape_the_directory_and_collisions_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = filename_label('../gold: well 2\\wide?')
            first = capture_path(root, 'frame_stack', label, timestamp='20261012-120000')
            first.write_bytes(b'original')
            second = capture_path(root, 'frame_stack', label, timestamp='20261012-120000')
            self.assertEqual(first.parent, root)
            self.assertEqual(second.parent, root)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b'original')
            self.assertIn('gold_well_2_wide', first.name)

    def test_each_spot_counts_once_and_final_bin_includes_full_scale(self):
        peak = np.array([[0, 4014, 4015, 4095]], dtype=np.uint16)
        result = spot_brightness(peak, [(i, i+1, 0, 1) for i in range(4)])
        self.assertEqual(result['total'], 4)
        self.assertEqual(sum(result['counts']), 4)
        self.assertEqual(result['near'], 2)
        self.assertEqual(result['clipped'], 1)
        self.assertEqual(result['counts'][-1], 2)

    def test_background_subtraction_does_not_hide_raw_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stack.npy'
            raw = np.array([[4095, 4020], [1000, 2000]], dtype=np.uint16)
            saved = raw - 1000
            diagnostic = save_raw_peak(path, raw, 150)
            metadata = [{'fetch_frames_output': {'raw_peak': diagnostic},
                         'background': {'background_subtracted': True}}]
            peak, scale, reliable, source = brightness_source(path, metadata, saved, 150)
            self.assertTrue(reliable)
            result = spot_brightness(peak, [(0, 2, 0, 2)], scale)
            self.assertEqual(result['clipped'], 1)
            (path.parent / diagnostic['path']).write_bytes(b'corrupt')
            peak, _, reliable, source = brightness_source(path, metadata, saved, 150)
            self.assertFalse(reliable)
            self.assertIn('missing/invalid', source)

    def test_legacy_subtracted_or_unknown_sources_do_not_claim_sensor_clipping(self):
        peak = np.array([[4095]], dtype=np.uint16)
        for metadata in ({}, {'background': {'background_subtracted': True}}):
            self.assertFalse(brightness_source(Path('stack.npy'), [metadata], peak, 150)[2])
        self.assertTrue(brightness_source(Path('stack.npy'), [{'background': {'background_subtracted': False}}], peak, 150)[2])


if __name__ == '__main__':
    unittest.main()
