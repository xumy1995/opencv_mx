#!/usr/bin/env python3
"""Fast unit checks for the backend-independent ResNet preprocessing helpers."""

import unittest
from unittest.mock import Mock

import numpy as np
import torch

from eval_resnet_opencv import (
    IMAGE_SIZE,
    RESIZE_SHORT_EDGE,
    TimedDataset,
    center_crop,
    make_backend,
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

    def test_timed_dataset_tracks_item_access(self):
        dataset = TimedDataset([{"label": 3}])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["label"], 3)
        self.assertGreaterEqual(dataset.seconds, 0.0)

    def test_hf_collate_reports_breakdown(self):
        image = Mock()
        image.convert.return_value = "RGB"
        processor = Mock(return_value={
            "pixel_values": torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        })
        pixels, labels, details = make_backend("hf", processor, None).collate(
            [{"image": image, "label": 3}]
        )
        image.convert.assert_called_once_with("RGB")
        processor.assert_called_once_with(images=["RGB"], return_tensors="pt")
        self.assertEqual(tuple(pixels.shape), (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(labels.tolist(), [3])
        self.assertEqual(set(details), {"collate", "rgb_convert", "image_processor"})
        self.assertTrue(all(elapsed >= 0 for elapsed in details.values()))


if __name__ == "__main__":
    unittest.main()
