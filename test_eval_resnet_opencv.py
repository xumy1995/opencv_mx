#!/usr/bin/env python3
"""Fast unit checks for the backend-independent ResNet preprocessing helpers."""

import unittest

import numpy as np
import torch

from eval_resnet_opencv import (
    IMAGE_SIZE,
    RESIZE_SHORT_EDGE,
    center_crop,
    normalize_to_nchw,
    resize_output_size,
)


class ResNetPreprocessTest(unittest.TestCase):
    def test_resize_dimensions_match_short_edge_rule(self):
        self.assertEqual(resize_output_size(400, 300), (256, 341))
        self.assertEqual(resize_output_size(300, 400), (341, 256))
        self.assertEqual(resize_output_size(224, 224), (256, 256))

    def test_center_crop_uses_floor_offset(self):
        image = np.arange(257 * 258 * 3, dtype=np.uint32).reshape(257, 258, 3)
        cropped = center_crop(image)
        self.assertEqual(cropped.shape, (IMAGE_SIZE, IMAGE_SIZE, 3))
        self.assertTrue(np.array_equal(cropped, image[16:240, 17:241]))

    def test_normalize_converts_to_nchw_float32(self):
        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        tensor = normalize_to_nchw(image)
        self.assertEqual(tuple(tensor.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(np.isclose(tensor[0, 0, 0].item(), -0.485 / 0.229))

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            resize_output_size(0, 100)
        with self.assertRaises(ValueError):
            center_crop(np.zeros((IMAGE_SIZE - 1, IMAGE_SIZE, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
