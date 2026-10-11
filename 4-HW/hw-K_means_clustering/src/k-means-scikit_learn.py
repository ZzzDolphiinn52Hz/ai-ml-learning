"""Compare the K-means exercise with scikit-learn on exactly the same data."""

import argparse
import io
import runpy
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        type=Path,
        help="Save the comparison chart to a PNG file instead of opening a window.",
    )
    args = parser.parse_args()

    # The exercise script runs immediately when loaded and calls plt.show() twice.
    # Reuse its actual X, labels and centers while suppressing those two windows.
    source = Path(__file__).with_name("k-means-ex1-103.py")
    with redirect_stdout(io.StringIO()), patch.object(plt, "show"):
        exercise = runpy.run_path(str(source))
    plt.close("all")

    X = exercise["X"]
    custom_centers = exercise["centroids"][-1]
    custom_labels = exercise["labels"][-1]
    n_clusters = exercise["K"]

    # Explicit n_init avoids a version-dependent default. tol=0 asks for
    # convergence without the library's usual early-stopping tolerance.
    # k-means++ initialization can still give a different iteration count.
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10, tol=0.0)
    sklearn_labels = model.fit_predict(X)

    # Cluster numbers are arbitrary. Match centers before comparing labels.
    distances = cdist(custom_centers, model.cluster_centers_)
    custom_ids, sklearn_ids = linear_sum_assignment(distances)
    to_custom_id = np.empty(n_clusters, dtype=int)
    to_custom_id[sklearn_ids] = custom_ids
    aligned_labels = to_custom_id[sklearn_labels]
    aligned_centers = np.empty_like(custom_centers)
    aligned_centers[custom_ids] = model.cluster_centers_[sklearn_ids]

    np.set_printoptions(precision=6, suppress=True)
    print("Centers found by our algorithm:\n", custom_centers)
    print("Centers found by scikit-learn (matched order):\n", aligned_centers)
    print("Distance between corresponding centers:", np.linalg.norm(custom_centers - aligned_centers, axis=1))
    print("Our algorithm's updates:", exercise["it"])
    print("scikit-learn iterations:", model.n_iter_)
    custom_inertia = np.sum((X - custom_centers[custom_labels]) ** 2)
    print("Our algorithm's inertia:", custom_inertia)
    print("scikit-learn inertia:", model.inertia_)
    print("Matching labels:", np.count_nonzero(custom_labels == aligned_labels), "/", len(X))
    print("Adjusted Rand index:", adjusted_rand_score(custom_labels, sklearn_labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    colors = plt.cm.tab10(np.arange(n_clusters))
    for ax, labels, centers, title in zip(
        axes,
        (custom_labels, aligned_labels),
        (custom_centers, aligned_centers),
        ("Our K-means", "scikit-learn KMeans"),
    ):
        for cluster_id in range(n_clusters):
            points = X[labels == cluster_id]
            ax.scatter(points[:, 0], points[:, 1], s=10, color=colors[cluster_id])
        ax.scatter(centers[:, 0], centers[:, 1], s=150, c="black", marker="X")
        ax.set(title=title, xlabel="x1", ylabel="x2")
        ax.grid(alpha=0.2)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=160)
        print("Chart saved to:", args.save)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
