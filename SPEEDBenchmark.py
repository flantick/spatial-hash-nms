import os
import time
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from NMS import *
from ModifiedNMS import modified_hashing_nms

random.seed(42)

iou_threshold = 0.5
object_area = 150 * 150     # средняя площадь изображения, приходящаяся на один объект
dense_side = 1500           # сторона изображения в режиме растущей плотности
results_dir = "results"

algorithms = {
    "Classic NMS": classic_nms,
    "Soft NMS": soft_nms,
    "Adaptive NMS": adaptive_nms,
    "Inverted NMS": inverted_nms,
    "Hashing-based NMS": hashing_nms,
    "HNMS + NMS": hashing_classic_nms,
    "Modified Hashing-based NMS": modified_hashing_nms,
}
baselines = ["Classic NMS", "Soft NMS", "Adaptive NMS", "Inverted NMS", "Hashing-based NMS"]


def generate_fake_detections(num_objects, side=None):
    """
    Имитация выхода детектора до NMS: вокруг каждого объекта генерируется
    кластер из 3-10 смещенных рамок с разной уверенностью.
    Если side не задан, размер изображения растет вместе с числом объектов
    и плотность сцены постоянна; иначе плотность растет на кадре фиксированного размера.
    """
    if side is None:
        side = math.sqrt(num_objects * object_area)
    detections = []
    for _ in range(num_objects):
        w, h = random.uniform(30, 120), random.uniform(30, 120)
        cx, cy = random.uniform(0, side), random.uniform(0, side)
        base_score = random.uniform(0.5, 1.0)
        for _ in range(random.randint(3, 10)):
            dw, dh = w * random.uniform(0.9, 1.1), h * random.uniform(0.9, 1.1)
            dx, dy = cx + w * random.uniform(-0.1, 0.1), cy + h * random.uniform(-0.1, 0.1)
            score = base_score * random.uniform(0.6, 1.0)
            detections.append((dx - dw / 2, dy - dh / 2, dx + dw / 2, dy + dh / 2, score))
    return detections


def measure(algorithm, images):
    start = time.perf_counter()
    for detections in images:
        algorithm(detections, iou_threshold)
    return time.perf_counter() - start


def test_algorithms(count_fake_picture=10000):
    """Суммарное время обработки набора изображений с 5-50 объектами."""
    images = [generate_fake_detections(random.randint(5, 50)) for _ in range(count_fake_picture)]
    mean_boxes = sum(len(d) for d in images) / len(images)
    print(f"{count_fake_picture} изображений, в среднем {mean_boxes:.1f} рамок на изображение")

    results = []
    for name, algorithm in algorithms.items():
        execution_time = measure(algorithm, images)
        print(f"Execution Time for {name}: {execution_time:.4f} seconds")
        results.append({"Algorithm": name, "Time, s": execution_time})
    return pd.DataFrame(results)


def test_scaling(side=None, object_counts=(10, 25, 50, 100, 200, 400, 800), repeats=5):
    """Зависимость времени обработки одного изображения от числа рамок N."""
    rows = []
    for num_objects in object_counts:
        images = [generate_fake_detections(num_objects, side) for _ in range(repeats)]
        n = sum(len(d) for d in images) / repeats
        for name, algorithm in algorithms.items():
            t = measure(algorithm, images) / repeats
            rows.append({"Algorithm": name, "N": n, "Time, ms": 1000 * t})
        print(f"N = {n:.0f}: done")
    return pd.DataFrame(rows)


def plot_scaling(df, path, title):
    plt.figure(figsize=(7, 5))
    for name in algorithms:
        part = df[df["Algorithm"] == name]
        style = "--" if name == "HNMS + NMS" else "-"
        plt.plot(part["N"], part["Time, ms"], style, marker="o", label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Число рамок на изображении, N")
    plt.ylabel("Время обработки, мс")
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)
    total = test_algorithms()
    total.to_csv(os.path.join(results_dir, "nms_speed_results.csv"), index=False)

    part = total[total["Algorithm"].isin(baselines)].sort_values("Time, s")
    plt.figure(figsize=(10, 5.5))
    plt.barh(part["Algorithm"], part["Time, s"], color=["blue", "green", "orange", "purple", "red"])
    plt.title("Время выполнения алгоритмов NMS")
    plt.xlabel("Время выполнения (секунды)")
    plt.ylabel("Алгоритмы")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "nms_speed.png"), dpi=150)
    plt.close()

    for side, suffix, title in [(None, "scaling", "Постоянная плотность сцены"),
                                (dense_side, "density", f"Растущая плотность, кадр {dense_side}×{dense_side}")]:
        df = test_scaling(side)
        df.to_csv(os.path.join(results_dir, f"nms_time_{suffix}.csv"), index=False)
        plot_scaling(df, os.path.join(results_dir, f"nms_time_{suffix}.png"), title)
        print(df.pivot(index="N", columns="Algorithm", values="Time, ms").round(2).to_string())
