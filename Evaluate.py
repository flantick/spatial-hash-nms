import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NMS import classic_nms, soft_nms, adaptive_nms, hashing_nms, hashing_classic_nms, inverted_nms, iou
from ModifiedNMS import modified_hashing_nms

iou_threshold = 0.5         # порог IoU для NMS
match_threshold = 0.5       # порог IoU для сопоставления с разметкой
raw_confidence = 0.05       # порог уверенности для сырых предсказаний детектора
score_threshold = 0.25      # рабочая точка для Precision / Recall / F1

results_dir = "results"
predictions_dir = "predictions"

datasets = {
    "CrowdHuman": {"images": "data/CrowdHuman/test/images", "labels": "data/CrowdHuman/test/labels",
                   "model": "yolov8n.pt", "classes": [0]},
    "BloodCell": {"images": "data/BloodCell/test/images", "labels": "data/BloodCell/test/labels",
                  "model": "weights/bloodcell_yolov8n.pt", "classes": None},
}

algorithms = {
    "Classic NMS": classic_nms,
    "Soft NMS": soft_nms,
    "Adaptive NMS": adaptive_nms,
    "Inverted NMS": inverted_nms,
    "Hashing-based NMS": hashing_nms,
    "HNMS + NMS": hashing_classic_nms,
    "Modified Hashing-based NMS": modified_hashing_nms,
}


def yolo_to_xyxy(label, img_width, img_height):
    """Преобразование YOLO разметки в формат x1, y1, x2, y2."""
    cls, x_center, y_center, width, height = label[:5]
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    return int(cls), (x1, y1, x2, y2)


def load_ground_truth(label_path, img_width, img_height):
    """Считывание ground truth боксов, сгруппированных по классам."""
    gt = {}
    if not os.path.exists(label_path):
        return gt
    with open(label_path, "r") as f:
        for line in f:
            if line.strip():
                cls, box = yolo_to_xyxy(list(map(float, line.split())), img_width, img_height)
                gt.setdefault(cls, []).append(box)
    return gt


def collect_predictions(cfg):
    """Сырые предсказания детектора без встроенного NMS (iou=1.0)."""
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(cfg["model"])
    samples = []
    for name in sorted(os.listdir(cfg["images"])):
        if not name.endswith((".jpg", ".png")):
            continue
        img_path = os.path.join(cfg["images"], name)
        img_width, img_height = Image.open(img_path).size
        label_path = os.path.join(cfg["labels"], os.path.splitext(name)[0] + ".txt")
        gt = load_ground_truth(label_path, img_width, img_height)

        result = model.predict(img_path, conf=raw_confidence, iou=1.0, max_det=20000,
                               classes=cfg["classes"], verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)

        preds = {}
        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, labels):
            cls = 0 if cfg["classes"] == [0] else int(cls)
            preds.setdefault(cls, []).append((float(x1), float(y1), float(x2), float(y2), float(score)))
        samples.append((preds, gt))
    return samples


def load_samples(name):
    cache = os.path.join(predictions_dir, f"raw_predictions_{name}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    samples = collect_predictions(datasets[name])
    os.makedirs(predictions_dir, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(samples, f)
    return samples


def match(detections, ground_truths):
    """Жадное сопоставление один к одному: каждая истинная рамка используется не более одного раза."""
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    used = [False] * len(ground_truths)
    flags = []
    for det in detections:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(ground_truths):
            if not used[j]:
                value = iou(det, gt)
                if value > best_iou:
                    best_iou, best_j = value, j
        if best_iou >= match_threshold:
            used[best_j] = True
            flags.append((det[4], 1))
        else:
            flags.append((det[4], 0))
    return flags


def run_algorithm(samples, algorithm, **kwargs):
    """Применение NMS к каждому изображению: списки сопоставлений, число объектов и время."""
    per_image, elapsed = [], 0.0
    for preds, gt in samples:
        flags, n_gt = [], sum(len(v) for v in gt.values())
        for cls in set(preds) | set(gt):
            detections = preds.get(cls, [])
            start = time.perf_counter()
            kept = algorithm(detections, iou_threshold, **kwargs) if detections else []
            elapsed += time.perf_counter() - start
            flags.extend(match(kept, gt.get(cls, [])))
        per_image.append((flags, n_gt))
    return per_image, 1000 * elapsed / len(samples)


def average_precision(recall, precision):
    """AP как площадь под интерполированной PR-кривой (all-point)."""
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def operating_point(per_image):
    """TP, FP и число объектов при рабочем пороге уверенности."""
    tp = sum(f[1] for flags, _ in per_image for f in flags if f[0] >= score_threshold)
    fp = sum(1 - f[1] for flags, _ in per_image for f in flags if f[0] >= score_threshold)
    n_gt = sum(n for _, n in per_image)
    return tp, fp, n_gt


def f1_score(tp, fp, n_gt):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def metrics(per_image):
    flags = sorted((f for flags, _ in per_image for f in flags), key=lambda x: x[0], reverse=True)
    total_gt = sum(n for _, n in per_image)
    tp = np.cumsum([f[1] for f in flags])
    fp = np.cumsum([1 - f[1] for f in flags])
    recall_curve = tp / max(total_gt, 1)
    precision_curve = tp / np.maximum(tp + fp, 1)

    precision, recall, f1 = f1_score(*operating_point(per_image))
    return {
        "Precision": precision, "Recall": recall, "F1-Score": f1,
        "AP50": average_precision(recall_curve, precision_curve),
        "curve": (recall_curve, precision_curve),
    }


def run(name):
    samples = load_samples(name)
    n_boxes = sum(len(v) for preds, _ in samples for v in preds.values())
    n_gt = sum(len(v) for _, gt in samples for v in gt.values())
    print(f"{name}: {len(samples)} изображений, {n_gt} объектов, {n_boxes / len(samples):.1f} рамок на изображение до NMS")

    rows, curves = [], {}
    for alg_name, algorithm in algorithms.items():
        per_image, ms = run_algorithm(samples, algorithm)
        res = metrics(per_image)
        curves[alg_name] = res.pop("curve")
        rows.append({"Algorithm": alg_name, **res, "Time, ms/img": ms})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, f"{name}_nms_evaluation_results.csv"), index=False)
    print(df.round(3).to_string(index=False))

    max_recall = max(curves[a][0][-1] for a in algorithms)
    baselines = ["Classic NMS", "Soft NMS", "Adaptive NMS", "Inverted NMS", "Hashing-based NMS"]
    hashing = ["Classic NMS", "Hashing-based NMS", "HNMS + NMS", "Modified Hashing-based NMS"]
    for subset, suffix in [(baselines, ""), (hashing, "_hashing")]:
        plt.figure(figsize=(8, 5.5))
        for alg_name in subset:
            r, p = curves[alg_name]
            style = "--" if alg_name == "Modified Hashing-based NMS" else "-"
            plt.plot(r, p, style, label=alg_name, linewidth=1.8)
        plt.title(f"Precision-Recall Curves ({name})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0, min(1.0, max_recall + 0.05))
        plt.ylim(0, 1.02)
        plt.grid(True)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{name}_precision_recall_curve{suffix}.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)
    for name in (sys.argv[1:] or list(datasets)):
        run(name)
