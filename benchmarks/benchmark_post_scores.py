"""Measure optional post-fit structure scoring independently from DAG search.

This benchmark intentionally uses a production-shaped star search space (many
candidate features pointing to one outcome). It does not replace the real-data
Colab benchmark; it is a small, deterministic regression benchmark for the
``structure_score_methods`` integration.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import bnlearn as bn
import numpy as np
import pandas as pd


def make_frame(rows: int, features: int, seed: int) -> pd.DataFrame:
    """Create a deterministic discrete frame with a correlated binary outcome."""
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 4, size=(rows, features), dtype=np.int16)
    logits = (
        1.2 * (values[:, 0] == 3)
        + 0.9 * (values[:, 1] >= 2)
        - 0.8 * (values[:, 2] == 0)
        + 0.5 * (values[:, 3] == values[:, 4])
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    outcome = (rng.random(rows) < probabilities).astype(np.int8)
    columns = [f"F{i:02d}" for i in range(features)]
    frame = pd.DataFrame(values, columns=columns)
    frame["Outcome"] = outcome
    return frame


def summarize(samples: list[float]) -> dict[str, float]:
    """Return robust timing statistics for one mode."""
    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    return {
        "median_seconds": median,
        "mad_seconds": float(np.median(np.abs(values - median))),
        "p10_seconds": float(np.percentile(values, 10)),
        "p90_seconds": float(np.percentile(values, 90)),
        "mean_seconds": float(values.mean()),
        "stdev_seconds": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
    }


def fit_once(
    frame: pd.DataFrame,
    mode: str,
    backend: str,
    max_iter: int,
) -> tuple[float, set[tuple[str, str]], set[str]]:
    """Fit once and return elapsed time plus equivalence observables."""
    features = [column for column in frame if column != "Outcome"]
    kwargs = {}
    if mode == "selected":
        kwargs["structure_score_methods"] = "selected"
    elif mode == "off":
        kwargs["structure_score_methods"] = []

    start = time.perf_counter()
    result = bn.structure_learning.fit(
        frame,
        methodtype="hc",
        scoretype="bic",
        white_list=[(feature, "Outcome") for feature in features],
        bw_list_method="edges",
        max_indegree=7,
        max_iter=max_iter,
        n_jobs=1,
        compute_backend=backend,
        verbose=0,
        **kwargs,
    )
    elapsed = time.perf_counter() - start
    return elapsed, set(result["model_edges"]), set(result["structure_scores"])


def run(args: argparse.Namespace) -> dict:
    """Run interleaved modes and verify that reporting does not change the DAG."""
    frame = make_frame(args.rows, args.features, args.seed)
    modes = ["historical", "selected", "off"]

    for mode in modes:
        fit_once(frame, mode, args.backend, args.max_iter)

    timings = {mode: [] for mode in modes}
    reference_edges = None
    score_keys = {}
    for repeat_index in range(args.repeats):
        offset = repeat_index % len(modes)
        interleaved_modes = modes[offset:] + modes[:offset]
        for mode in interleaved_modes:
            elapsed, edges, keys = fit_once(frame, mode, args.backend, args.max_iter)
            timings[mode].append(elapsed)
            score_keys[mode] = sorted(keys)
            if reference_edges is None:
                reference_edges = edges
            elif edges != reference_edges:
                raise AssertionError(f"learned graph changed in mode={mode}")

    summaries = {mode: summarize(samples) for mode, samples in timings.items()}
    historical = summaries["historical"]["median_seconds"]
    for mode in ("selected", "off"):
        summaries[mode]["speedup_vs_historical"] = (
            historical / summaries[mode]["median_seconds"]
        )

    return {
        "schema_version": 1,
        "rows": args.rows,
        "features": args.features,
        "backend": args.backend,
        "max_iter": args.max_iter,
        "repeats": args.repeats,
        "score_keys": score_keys,
        "edges": sorted([list(edge) for edge in reference_edges or set()]),
        "timings": timings,
        "summary": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=34)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backend", choices=("numpy", "cupy", "auto"), default="numpy")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    payload = run(parsed)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if parsed.output is not None:
        parsed.output.parent.mkdir(parents=True, exist_ok=True)
        parsed.output.write_text(rendered + "\n", encoding="utf-8")
