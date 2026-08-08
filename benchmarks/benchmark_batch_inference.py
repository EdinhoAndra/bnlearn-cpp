"""Focused benchmark for bnlearn's compiled batch-inference API."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def build_model():
    edges = [('P0', 'Y'), ('P1', 'Y'), ('P2', 'Y')]
    state_names = {name: [0, 1] for name in ['P0', 'P1', 'P2', 'Y']}
    cpds = [
        TabularCPD(name, 2, [[0.5], [0.5]], state_names={name: [0, 1]})
        for name in ['P0', 'P1', 'P2']
    ]
    probabilities = np.array(
        [0.05, 0.15, 0.25, 0.40, 0.60, 0.75, 0.85, 0.95]
    )
    cpds.append(
        TabularCPD(
            'Y',
            2,
            [1.0 - probabilities, probabilities],
            evidence=['P0', 'P1', 'P2'],
            evidence_card=[2, 2, 2],
            state_names=state_names,
        )
    )
    return bn.make_DAG(edges, CPD=cpds)


def measure(callable_, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        samples.append(time.perf_counter() - start)
    return result, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=5_000)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()

    model = build_model()
    evidences = [
        {'P0': index & 1, 'P1': (index >> 1) & 1, 'P2': (index >> 2) & 1}
        for index in range(args.rows)
    ]
    ve = VariableElimination(model['model'])
    engine = bn.inference.compile(model)

    ve_loop = lambda: [
        ve.query(['Y'], evidence=evidence, show_progress=False)
        for evidence in evidences
    ]
    direct_batch = lambda: engine.query_many(['Y'], evidences)

    # Warm both paths outside the measured region.
    ve_loop()
    direct_batch()
    expected, ve_times = measure(ve_loop, args.repeats)
    actual, batch_times = measure(direct_batch, args.repeats)
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result.values, reference.values, rtol=0, atol=1e-12)

    ve_median = statistics.median(ve_times)
    batch_median = statistics.median(batch_times)
    print(f'rows={args.rows} repeats={args.repeats}')
    print(f'reused_ve_loop_median_s={ve_median:.6f}')
    print(f'direct_cpd_batch_median_s={batch_median:.6f}')
    print(f'speedup={ve_median / batch_median:.2f}x')


if __name__ == '__main__':
    main()
