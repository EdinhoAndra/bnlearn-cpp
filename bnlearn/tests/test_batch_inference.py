import numpy as np
import pandas as pd
import pytest

import bnlearn as bn
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def _model_with_named_states():
    edges = [('G', 'A'), ('A', 'T'), ('B', 'T'), ('T', 'C')]
    cpds = [
        TabularCPD('G', 2, [[0.6], [0.4]], state_names={'G': ['g0', 'g1']}),
        TabularCPD(
            'A',
            2,
            [[0.7, 0.2], [0.3, 0.8]],
            evidence=['G'],
            evidence_card=[2],
            state_names={'A': ['a0', 'a1'], 'G': ['g0', 'g1']},
        ),
        TabularCPD('B', 2, [[0.55], [0.45]], state_names={'B': ['b0', 'b1']}),
        TabularCPD(
            'T',
            3,
            [
                [0.70, 0.20, 0.10, 0.05],
                [0.20, 0.50, 0.30, 0.15],
                [0.10, 0.30, 0.60, 0.80],
            ],
            evidence=['A', 'B'],
            evidence_card=[2, 2],
            state_names={
                'T': ['low', 'mid', 'high'],
                'A': ['a0', 'a1'],
                'B': ['b0', 'b1'],
            },
        ),
        TabularCPD(
            'C',
            2,
            [[0.9, 0.6, 0.1], [0.1, 0.4, 0.9]],
            evidence=['T'],
            evidence_card=[3],
            state_names={'C': ['c0', 'c1'], 'T': ['low', 'mid', 'high']},
        ),
    ]
    return bn.make_DAG(edges, CPD=cpds)


def _assert_same_factor(actual, expected):
    assert actual.variables == expected.variables
    assert actual.state_names == expected.state_names
    np.testing.assert_allclose(actual.values, expected.values, rtol=0, atol=1e-12)


def test_query_many_direct_lookup_matches_ve_and_preserves_order():
    model = _model_with_named_states()
    evidences = [
        {'A': 'a1', 'B': 'b0'},
        {'A': 'a0', 'B': 'b1', 'G': 'g1'},
        {'A': 'a0', 'B': 'b0'},
    ]
    engine = bn.inference.compile(model)

    actual = engine.query_many(['T'], evidences)
    ve = VariableElimination(model['model'])
    expected = [ve.query(['T'], evidence=evidence, show_progress=False) for evidence in evidences]

    assert [factor.state_names['T'] for factor in actual] == [
        ['low', 'mid', 'high'],
        ['low', 'mid', 'high'],
        ['low', 'mid', 'high'],
    ]
    for result, reference in zip(actual, expected):
        _assert_same_factor(result, reference)


def test_query_many_reuses_ve_and_falls_back_when_lookup_is_not_exact(monkeypatch):
    model = _model_with_named_states()
    engine = bn.inference.compile(model)
    calls = []
    original_query = engine._ve.query

    def recording_query(*args, **kwargs):
        calls.append(kwargs['evidence'])
        return original_query(*args, **kwargs)

    monkeypatch.setattr(engine._ve, 'query', recording_query)
    evidences = [
        {'A': 'a1', 'B': 'b0'},  # direct CPD lookup
        {'A': 'a1'},  # missing parent: VE
        {'A': 'a1', 'B': 'b0', 'C': 'c1'},  # descendant evidence: VE
    ]

    actual = engine.query_many(['T'], evidences)

    assert calls == [evidences[1], evidences[2]]
    reference_ve = VariableElimination(model['model'])
    for result, evidence in zip(actual, evidences):
        reference = reference_ve.query(['T'], evidence=evidence, show_progress=False)
        _assert_same_factor(result, reference)


def test_query_many_wrapper_accepts_compiled_engine_and_builds_dataframes():
    model = _model_with_named_states()
    engine = bn.inference.compile(model)
    evidences = [{'A': 'a1', 'B': 'b1'}, {'A': 'a0'}]

    results = bn.inference.query_many(
        engine, ['T'], evidences, to_df=True, verbose=0
    )

    assert len(results) == 2
    for result in results:
        assert list(result.df.columns) == ['T', 'p']
        assert result.df['T'].tolist() == ['low', 'mid', 'high']
        assert result.df['p'].sum() == pytest.approx(1.0)


def test_query_many_joint_false_keeps_pgmpy_result_shape():
    model = _model_with_named_states()

    results = bn.inference.query_many(
        model,
        ['T'],
        [{'A': 'a0', 'B': 'b1'}],
        joint=False,
    )

    assert list(results[0]) == ['T']
    assert results[0]['T'].state_names['T'] == ['low', 'mid', 'high']


def test_query_many_multiple_targets_falls_back_and_matches_ve():
    model = _model_with_named_states()
    evidence = {'A': 'a1', 'B': 'b0'}

    actual = bn.inference.query_many(model, ['T', 'C'], [evidence])[0]
    expected = VariableElimination(model['model']).query(
        ['T', 'C'], evidence=evidence, show_progress=False
    )

    _assert_same_factor(actual, expected)


def test_query_many_rejects_dataframe_conversion_for_non_joint_results():
    model = _model_with_named_states()

    with pytest.raises(ValueError, match='to_df=True requires joint=True'):
        bn.inference.query_many(
            model,
            ['T'],
            [{'A': 'a0', 'B': 'b1'}],
            joint=False,
            to_df=True,
        )


def test_predict_uses_batch_inference_without_changing_results():
    model = _model_with_named_states()
    data = pd.DataFrame(
        [
            {'A': 'a0', 'B': 'b0'},
            {'A': 'a1', 'B': 'b0'},
            {'A': 'a1', 'B': 'b1'},
            {'A': 'a0', 'B': 'b0'},
        ]
    )

    result = bn.predict(model, data, variables='T', verbose=0)

    assert result['T'].tolist() == ['low', 'high', 'high', 'low']
    np.testing.assert_allclose(result['p'], [0.70, 0.60, 0.80, 0.70])
