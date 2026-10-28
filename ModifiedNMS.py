import math
from collections import defaultdict
from NMS import iou


def modified_hashing_nms(detections, iou_threshold=0.5, alpha=None):
    """
    Модифицированный Hashing-based NMS.

    Хеш-таблица используется как пространственный индекс: рамки хешируются
    в сетку по центру, и кандидат сравнивается по IoU только с уже сохраненными
    рамками из ячеек, отстоящих от его ячейки не более чем на (rx, ry).
    Радиус выбирается по размеру кандидата так, чтобы в окрестность гарантированно
    попадали все рамки с IoU >= iou_threshold, поэтому результат совпадает
    с результатом классического NMS.

    Args:
        detections (list of tuples): Каждая запись в формате (x1, y1, x2, y2, score).
        iou_threshold (float): Порог IoU для подавления.
        alpha (float): Размер ячейки сетки. Если None, берется половина медианного размера рамки.

    Returns:
        list: Отфильтрованные детекции.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x[4], reverse=True)

    if alpha is None:
        sizes = sorted(max(det[2] - det[0], det[3] - det[1]) for det in detections)
        alpha = max(1.0, 0.5 * sizes[len(sizes) // 2])

    # максимальный сдвиг центров рамок с IoU >= T в долях размера кандидата
    t = iou_threshold
    shift = max(1 - t, (1 - t) / (2 * t))

    grid = defaultdict(list)
    final_detections = []

    for det in detections:
        u = int((det[0] + det[2]) / 2 // alpha)
        v = int((det[1] + det[3]) / 2 // alpha)
        rx = max(1, math.ceil(shift * (det[2] - det[0]) / alpha))
        ry = max(1, math.ceil(shift * (det[3] - det[1]) / alpha))

        suppressed = False
        for du in range(-rx, rx + 1):
            for dv in range(-ry, ry + 1):
                for kept in grid.get((u + du, v + dv), ()):
                    if iou(det, kept) >= iou_threshold:
                        suppressed = True
                        break
                if suppressed:
                    break
            if suppressed:
                break

        if not suppressed:
            grid[(u, v)].append(det)
            final_detections.append(det)

    return final_detections
