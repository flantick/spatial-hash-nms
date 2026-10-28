import os
import numpy as np
import pandas as pd
import random
from NMS import classic_nms, soft_nms, adaptive_nms, hashing_nms, hashing_classic_nms, inverted_nms
from ModifiedNMS import modified_hashing_nms
from Evaluate import datasets, load_samples, run_algorithm, metrics, operating_point, f1_score, results_dir

rng = np.random.default_rng(0)


def bootstrap_f1_difference(per_image_a, per_image_b, n_boot=2000):
    """95%-й доверительный интервал разности F1 (a - b), бутстрэп по изображениям."""
    stats_a = np.array([operating_point([x]) for x in per_image_a], dtype=float)
    stats_b = np.array([operating_point([x]) for x in per_image_b], dtype=float)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(stats_a), len(stats_a))
        diffs.append(f1_score(*stats_a[idx].sum(0))[2] - f1_score(*stats_b[idx].sum(0))[2])
    return np.percentile(diffs, [2.5, 97.5])


def row(name, per_image, ms, **extra):
    res = metrics(per_image)
    res.pop("curve")
    return {"Dataset": name, **extra, **{k: round(v, 3) for k, v in res.items()}, "Time, ms/img": round(ms, 3)}


def hashing_sensitivity(name, samples):
    """Чувствительность Hashing-based NMS к параметрам alpha и K."""
    rows = []
    for alpha in (0.5, 0.6, 0.7, 0.8):
        for K in (1, 2, 4):
            per_image, ms = run_algorithm(samples, hashing_nms, alpha=alpha, K=K)
            rows.append(row(name, per_image, ms, alpha=alpha, K=K))
    return rows


def adaptive_control(name, samples):
    """Сравнение Adaptive NMS с классическим NMS при повышенном пороге."""
    rows = []
    for label, algorithm, kwargs in [("Adaptive NMS", adaptive_nms, {}),
                                     ("Classic NMS, T = 0.5", classic_nms, {}),
                                     ("Classic NMS, T = 0.7", lambda d, t: classic_nms(d, 0.7), {})]:
        per_image, ms = run_algorithm(samples, algorithm, **kwargs)
        rows.append(row(name, per_image, ms, Algorithm=label))
    return rows


def f1_intervals(name, samples):
    """Доверительные интервалы разности F1 относительно классического NMS."""
    reference, _ = run_algorithm(samples, classic_nms)
    rows = []
    for label, algorithm in [("Soft NMS", soft_nms), ("Adaptive NMS", adaptive_nms),
                             ("Inverted NMS", inverted_nms), ("Hashing-based NMS", hashing_nms),
                             ("HNMS + NMS", hashing_classic_nms)]:
        per_image, _ = run_algorithm(samples, algorithm)
        delta = metrics(per_image)["F1-Score"] - metrics(reference)["F1-Score"]
        lo, hi = bootstrap_f1_difference(per_image, reference)
        rows.append({"Dataset": name, "Algorithm": label, "dF1": round(delta, 3),
                     "CI low": round(lo, 3), "CI high": round(hi, 3)})
    return rows


def cell_size(name, samples, repeats=5):
    """Время работы модифицированного алгоритма при разном размере ячейки (медиана по повторам)."""
    rows = []
    for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
        def algorithm(detections, t, factor=factor):
            sizes = sorted(max(d[2] - d[0], d[3] - d[1]) for d in detections)
            return modified_hashing_nms(detections, t, alpha=max(1.0, factor * sizes[len(sizes) // 2]))
        runs = [run_algorithm(samples, algorithm) for _ in range(repeats)]
        per_image = runs[0][0]
        ms = float(np.median([r[1] for r in runs]))
        rows.append(row(name, per_image, ms, factor=factor))
    return rows


def exactness(name, samples):
    """Число рамок, по которым результат алгоритма расходится с эталоном."""
    rows = []
    for label, algorithm, reference in [
            ("Modified Hashing-based NMS", modified_hashing_nms, lambda d, t: classic_nms(d, t)),
            ("HNMS + NMS", hashing_classic_nms, lambda d, t: classic_nms(d, t)),
            ("Adaptive NMS vs Classic NMS, T = 0.7", adaptive_nms, lambda d, t: classic_nms(d, 0.7))]:
        diff = kept = 0
        for preds, _ in samples:
            for detections in preds.values():
                ref = set(reference(detections, 0.5))
                diff += len(ref ^ set(algorithm(detections, 0.5)))
                kept += len(ref)
        rows.append({"Dataset": name, "Algorithm": label, "Reference boxes": kept, "Different boxes": diff})
    return rows


def random_exactness(trials=3000, seed=1):
    """Сравнение с классическим NMS на случайных наборах рамок сильно различающихся размеров и пропорций."""
    rnd = random.Random(seed)
    mismatches = 0
    for _ in range(trials):
        t = rnd.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        detections = []
        for _ in range(rnd.randint(2, 120)):
            w = rnd.uniform(2, 400) * rnd.choice([1, 1, 0.2, 5])
            h = rnd.uniform(2, 400) * rnd.choice([1, 1, 0.2, 5])
            x, y = rnd.uniform(0, 600), rnd.uniform(0, 600)
            detections.append((x, y, x + w, y + h, rnd.random()))
        alpha = rnd.choice([None, 3.0, 20.0, 100.0])
        mismatches += len(set(classic_nms(detections, t)) ^ set(modified_hashing_nms(detections, t, alpha=alpha)))
    return [{"Trials": trials, "Different boxes": mismatches}]


if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)
    random_rows = pd.DataFrame(random_exactness())
    random_rows.to_csv(os.path.join(results_dir, "ablation_random_exactness.csv"), index=False)
    print(random_rows.to_string(index=False))

    results = {"hashing_sensitivity": [], "adaptive_control": [], "f1_intervals": [], "cell_size": [],
               "exactness": []}
    for name in datasets:
        samples = load_samples(name)
        for key in results:
            results[key].extend(globals()[key](name, samples))
    for key, rows in results.items():
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(results_dir, f"ablation_{key}.csv"), index=False)
        print(key)
        print(df.to_string(index=False))
