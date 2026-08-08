"""Benchmark the discrete backends supplied by the custom pgmpy-cpp fork."""

import argparse
import json
from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd
from pgmpy.structure_score import BIC


def make_queries(columns, max_parents):
    queries = []
    for variable in columns:
        candidates = [column for column in columns if column != variable]
        for parent_count in range(min(max_parents, len(candidates)) + 1):
            queries.extend(
                (variable, parents)
                for parents in combinations(candidates, parent_count)
            )
    return queries


def benchmark(data, queries, backend, repeats):
    init_start = perf_counter()
    score = BIC(data, compute_backend=backend)
    resolved_backend = score.resolved_backend_
    initialization_seconds = perf_counter() - init_start

    durations = []
    values = None
    for _ in range(repeats):
        start = perf_counter()
        values = []
        for variable in data.columns:
            parent_sets = [
                parents
                for query_variable, parents in queries
                if query_variable == variable
            ]
            values.extend(score.batch_local_scores(variable, parent_sets))
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
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("numpy", "cupy", "cuda_fused", "cpp", "auto"),
        default=["numpy", "cpp"],
    )
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    columns = [f"X{index}" for index in range(args.variables)]
    data = pd.DataFrame(
        rng.integers(0, args.cardinality, size=(args.rows, args.variables)),
        columns=columns,
    )
    queries = make_queries(columns, args.max_parents)
    results = [
        benchmark(data, queries, backend, args.repeats) for backend in args.backends
    ]

    numpy_result = next(
        (result for result in results if result["requested_backend"] == "numpy"),
        None,
    )
    if numpy_result is not None:
        reference = np.asarray(numpy_result["scores"])
        for result in results:
            scores = np.asarray(result["scores"])
            result["max_abs_error_vs_numpy"] = float(np.max(np.abs(scores - reference)))
            result["speedup_vs_numpy"] = (
                numpy_result["best_score_batch_seconds"]
                / result["best_score_batch_seconds"]
            )
    for result in results:
        result.pop("scores", None)

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
