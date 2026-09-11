import importlib.util
import unittest

import numpy as np

from mmdii.data.time_frequency import (
    RepresentationNormalizer,
    cwt_representation,
    dwt_representation,
    stft_representation,
    transform_representation,
)


PYWT_AVAILABLE = importlib.util.find_spec("pywt") is not None


class TimeFrequencyTests(unittest.TestCase):
    def setUp(self) -> None:
        time = np.arange(2048, dtype=np.float64) / 5400.0
        self.signal = np.vstack(
            [np.sin(2 * np.pi * 120 * time), np.cos(2 * np.pi * 300 * time), time]
        )

    def test_stft_variants_have_expected_frequency_bins(self) -> None:
        original = self.signal.copy()
        small, mask_small = stft_representation(
            self.signal, n_fft=256, hop_length=64, output_time_bins=32
        )
        large, mask_large = stft_representation(
            self.signal, n_fft=512, hop_length=128, output_time_bins=32
        )
        self.assertEqual(small.shape, (3, 129, 32))
        self.assertEqual(large.shape, (3, 257, 32))
        self.assertEqual(mask_small.shape, (32,))
        self.assertEqual(mask_large.shape, (32,))
        self.assertTrue(np.isfinite(small).all())
        np.testing.assert_array_equal(self.signal, original)

    def test_short_signal_is_padded_and_masked(self) -> None:
        transformed, mask = stft_representation(self.signal[:, :32], output_time_bins=16)
        self.assertEqual(transformed.shape, (3, 129, 16))
        self.assertEqual(mask.shape, (16,))
        self.assertTrue(mask.any())

    @unittest.skipUnless(PYWT_AVAILABLE, "PyWavelets is required for CWT/DWT tests")
    def test_cwt_and_dwt_have_fixed_time_axis(self) -> None:
        cwt, cwt_mask = cwt_representation(self.signal, output_time_bins=64)
        dwt, dwt_mask = dwt_representation(self.signal, output_time_bins=64)
        self.assertEqual(cwt.shape, (3, 48, 64))
        self.assertEqual(dwt.shape, (3, 6, 64))
        self.assertEqual(cwt_mask.shape, (64,))
        self.assertEqual(dwt_mask.shape, (64,))
        self.assertTrue(np.isfinite(cwt).all())
        self.assertTrue(np.isfinite(dwt).all())

    @unittest.skipUnless(PYWT_AVAILABLE, "PyWavelets is required for CWT/DWT tests")
    def test_dispatch_is_deterministic_and_propagates_mask(self) -> None:
        sample_mask = np.zeros(self.signal.shape[1], dtype=bool)
        sample_mask[:1024] = True
        first, first_mask = transform_representation(
            self.signal, "cwt_morl", sample_mask=sample_mask, output_time_bins=64
        )
        second, second_mask = transform_representation(
            self.signal, "cwt_morl", sample_mask=sample_mask, output_time_bins=64
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_mask, second_mask)
        self.assertTrue(first_mask.any())
        self.assertFalse(first_mask.all())

    def test_normalizer_uses_training_arrays_only(self) -> None:
        first = np.zeros((3, 4, 5), dtype=np.float64)
        second = np.ones((2, 3, 4, 5), dtype=np.float64) * 2.0
        normalizer = RepresentationNormalizer.fit((first, second))
        self.assertEqual(normalizer.means.shape, (3,))
        transformed = normalizer.transform(np.ones((3, 4, 5), dtype=np.float64))
        self.assertEqual(transformed.shape, (3, 4, 5))
        self.assertTrue(np.isfinite(transformed).all())

    def test_rejects_nonfinite_signal_and_unknown_representation(self) -> None:
        invalid = self.signal.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            stft_representation(invalid)
        with self.assertRaises(ValueError):
            transform_representation(self.signal, "unknown")
        with self.assertRaises(ValueError):
            stft_representation(np.empty((3, 0)))


if __name__ == "__main__":
    unittest.main()
