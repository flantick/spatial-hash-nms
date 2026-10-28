import random

import pytest

from NMS import iou, classic_nms, soft_nms, adaptive_nms, hashing_nms, hashing_classic_nms, inverted_nms
from ModifiedNMS import modified_hashing_nms


def random_detections(rnd, n):
    """Рамки сильно различающихся размеров и пропорций, как в Ablation.random_exactness."""
    detections = []
    for _ in range(n):
        w = rnd.uniform(2, 400) * rnd.choice([1, 1, 0.2, 5])
        h = rnd.uniform(2, 400) * rnd.choice([1, 1, 0.2, 5])
        x, y = rnd.uniform(0, 600), rnd.uniform(0, 600)
        detections.append((x, y, x + w, y + h, rnd.random()))
    return detections


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_classic_nms_suppresses_duplicates():
    detections = [(0, 0, 10, 10, 0.9), (1, 1, 11, 11, 0.8), (50, 50, 60, 60, 0.7)]
    assert classic_nms(detections, 0.5) == [(0, 0, 10, 10, 0.9), (50, 50, 60, 60, 0.7)]


@pytest.mark.parametrize("algorithm", [classic_nms, soft_nms, adaptive_nms, hashing_nms,
                                       hashing_classic_nms, inverted_nms, modified_hashing_nms])
def test_empty_input(algorithm):
    assert algorithm([], 0.5) == []


@pytest.mark.parametrize("iou_threshold", [0.2, 0.3, 0.5, 0.7, 0.8])
@pytest.mark.parametrize("alpha", [None, 10.0, 30.0, 100.0])
def test_modified_hashing_nms_matches_classic(iou_threshold, alpha):
    rnd = random.Random(f"{iou_threshold}-{alpha}")
    for _ in range(25):
        detections = random_detections(rnd, rnd.randint(2, 120))
        expected = classic_nms(detections, iou_threshold)
        assert modified_hashing_nms(detections, iou_threshold, alpha=alpha) == expected
