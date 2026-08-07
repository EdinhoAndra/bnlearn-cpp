"""Benchmark pgmpy 0.1.25 scores against bnlearn's vectorized backends."""

import argparse
import json
from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd
from pgmpy.estimators import BicScore

from bnlearn.accelerated_scores import get_accelerated_score


def make_queries(columns, max_parents):
    queries = []
    for variable in columns:
        candidates = [column for column in columns if column != variable]
        for parent_count in range(min(max_parents, len(candidates)) + 1):
            queries.extend((variable, parents) for parents in combinations(candidates, parent_count))
    return queries


def benchmark(data, queries, backend, repeats):
    init_start = perf_counter()
    if backend == "legacy":
        score = BicScore(data)
        resolved_backend = "pandas"
    else:
        score = get_accelerated_score(data, "bic", compute_backend=backend)
        resolved_backend = score.resolved_backend_
    initialization_seconds = perf_counter() - init_start

    durations = []
    values = None
    for _ in range(repeats):
        start = perf_counter()
        values = [score.local_score(variable, parents) for variable, parents in queries]
        durations.append(perf_counter() - start)

    return {
        "requested_backend": backend,
        "resolved_backend": resolved_backend,
        "initialization_seconds": initialization_seconds,
        "best_score_batch_seconds": min(durations),
        "scores": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--variables", type=int, default=8)
    parser.add_argument("--cardinality", type=int, default=4)
    parser.add_argument("--max-parents", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--backends", nargs="+", default=["legacy", "numpy", "cupy"])
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    columns = [f"X{index}" for index in range(args.variables)]
    data = pd.DataFrame(
        rng.integers(0, args.cardinality, size=(args.rows, args.variables)),
        columns=columns,
    )
    queries = make_queries(columns, args.max_parents)
    results = [benchmark(data, queries, backend, args.repeats) for backend in args.backends]

    reference = np.asarray(results[0].pop("scores"))
    for result in results[1:]:
        scores = np.asarray(result.pop("scores"))
        result["max_abs_error_vs_legacy"] = float(np.max(np.abs(scores - reference)))
        result["speedup_vs_legacy"] = results[0]["best_score_batch_seconds"] / result[
            "best_score_batch_seconds"
        ]
    results[0].pop("scores", None)

    print(
        json.dumps(
            {
                "rows": args.rows,
                "variables": args.variables,
                "queries": len(queries),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
