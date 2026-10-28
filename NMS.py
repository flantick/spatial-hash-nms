import math


def iou(box1, box2):
    """
    Расчет Intersection over Union (IoU).

    Args:
        box1 (tuple): Координаты первого прямоугольника (x1, y1, x2, y2).
        box2 (tuple): Координаты второго прямоугольника (x1, y1, x2, y2).

    Returns:
        float: Значение IoU.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def classic_nms(detections, iou_threshold):
    """
    Классический Non-Maximum Suppression.

    Args:
        detections (list of tuples): Каждая запись в формате (x1, y1, x2, y2, score).
        iou_threshold (float): Порог IoU для подавления.

    Returns:
        list: Отфильтрованные детекции.
    """
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    final_detections = []

    while detections:
        best = detections.pop(0)
        final_detections.append(best)
        detections = [det for det in detections if iou(best, det) < iou_threshold]

    return final_detections


def soft_nms(detections, iou_threshold, sigma=0.5, score_threshold=0.001):
    """
    Soft Non-Maximum Suppression (гауссова функция штрафа).

    Args:
        detections (list of tuples): Каждая запись в формате (x1, y1, x2, y2, score).
        iou_threshold (float): Не используется, оставлен для единого интерфейса.
        sigma (float): Параметр гауссовского подавления.
        score_threshold (float): Детекции с оценкой ниже порога отбрасываются.

    Returns:
        list: Отфильтрованные детекции с пересчитанными оценками.
    """
    detections = [tuple(det) for det in detections]
    final_detections = []

    while detections:
        m = max(range(len(detections)), key=lambda i: detections[i][4])
        best = detections.pop(m)
        final_detections.append(best)

        updated = []
        for det in detections:
            score = det[4] * math.exp(-(iou(best, det) ** 2) / sigma)
            if score >= score_threshold:
                updated.append((det[0], det[1], det[2], det[3], score))
        detections = updated

    return final_detections


def adaptive_nms(detections, iou_threshold, max_threshold=0.7):
    """
    Adaptive Non-Maximum Suppression.

    Детектор не содержит ветви предсказания плотности, поэтому плотность объекта
    оценивается по самим предсказаниям как максимальный IoU рамки M с оставшимися
    кандидатами. Адаптивный порог ограничен сверху значением max_threshold.

    Args:
        detections (list of tuples): Каждая запись в формате (x1, y1, x2, y2, score).
        iou_threshold (float): Базовый порог IoU (Nt).
        max_threshold (float): Верхняя граница адаптивного порога.

    Returns:
        list: Отфильтрованные детекции.
    """
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    final_detections = []

    while detections:
        current = detections.pop(0)
        final_detections.append(current)

        density = max((iou(current, det) for det in detections), default=0.0)
        threshold = min(max(iou_threshold, density), max_threshold)

        detections = [det for det in detections if iou(current, det) < threshold]

    return final_detections


def iou_hash(det, alpha, W0, H0, bx, by):
    """Хеш-код рамки (IoUHash, Wang et al., 2020): уровни ширины и высоты и положение центра."""
    x1, y1, x2, y2 = det[:4]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    k = (1 - alpha) / (1 + alpha)

    i = math.floor((math.log(W0) - math.log(w)) / math.log(alpha) + 0.5)
    j = math.floor((math.log(H0) - math.log(h)) / math.log(alpha) + 0.5)
    di = W0 / (alpha ** i) * k
    dj = H0 / (alpha ** j) * k

    return i, j, math.floor(cx / di - bx + 0.5), math.floor(cy / dj - by + 0.5)


def hashing_nms(detections, iou_threshold, alpha=0.7, K=1):
    """
    Hashing-based Non-Maximum Suppression (Wang et al., 2020).

    В каждой ячейке хеш-таблицы остается рамка с наибольшей уверенностью.
    При K > 1 процедура повторяется с K сдвинутыми хеш-функциями.

    Args:
        detections (list of tuples): Каждая запись в формате (x1, y1, x2, y2, score).
        iou_threshold (float): Не используется, оставлен для единого интерфейса.
        alpha (float): Параметр хеширования, 0 < alpha < 1.
        K (int): Число хеш-функций.

    Returns:
        list: Отфильтрованные детекции.
    """
    for k in range(K):
        W0 = H0 = math.exp(-math.log(alpha) * k / K)
        bx = by = k / K
        grid = {}
        for det in detections:
            key = iou_hash(det, alpha, W0, H0, bx, by)
            if key not in grid or grid[key][4] < det[4]:
                grid[key] = det
        detections = list(grid.values())

    return detections


def hashing_classic_nms(detections, iou_threshold, alpha=0.7, K=1):
    """
    Hashing-based NMS как предварительный фильтр перед классическим NMS
    (схема, предложенная авторами для одностадийных детекторов).
    """
    return classic_nms(hashing_nms(detections, iou_threshold, alpha, K), iou_threshold)


def inverted_nms(detections, iou_threshold):
    """
    Inverted Non-Maximum Suppression.

    Args:
        detections (list of tuples): Каждая запись в формате (x1, y1, x2, y2, score).
        iou_threshold (float): Порог IoU.

    Returns:
        list: Отфильтрованные детекции.
    """
    detections = sorted(detections, key=lambda x: x[4])
    discarded = set()

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if iou(detections[i], detections[j]) >= iou_threshold:
                discarded.add(i)
                break

    return [det for i, det in enumerate(detections) if i not in discarded]
