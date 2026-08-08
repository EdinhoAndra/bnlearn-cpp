import numpy as np
import pandas as pd
import pytest

import bnlearn as bn
import bnlearn.structure_learning as structure_learning
from pgmpy.base import DAG


@pytest.fixture
def discrete_data():
    rng = np.random.default_rng(2026)
    a = rng.integers(0, 3, size=2_000)
    b = (a + rng.integers(0, 2, size=2_000)) % 3
    return pd.DataFrame(
        {
            'A': a,
            'B': b,
            'C': rng.integers(0, 2, size=2_000),
        }
    )


def test_cpp_score_path_matches_numpy(discrete_data):
    pytest.importorskip('pgmpy._native_discrete')
    numpy_score = structure_learning._SetScoringType(
        discrete_data, 'bic', compute_backend='numpy', verbose=0
    )
    cpp_score = structure_learning._SetScoringType(
        discrete_data, 'bic', compute_backend='cpp', verbose=0
    )

    assert cpp_score.compute_backend == 'cpp'
    assert cpp_score.resolved_backend_ == 'cpp'
    assert cpp_score.local_score('C', ('A', 'B')) == pytest.approx(
        numpy_score.local_score('C', ('A', 'B')), rel=1e-12, abs=1e-12
    )


def test_structure_scores_accepts_cpp_and_matches_numpy(discrete_data):
    pytest.importorskip('pgmpy._native_discrete')
    model = bn.make_DAG([('A', 'B'), ('B', 'C')], verbose=0)

    numpy_scores = bn.structure_scores(
        model,
        discrete_data,
        scoring_method=['bic', 'k2'],
        compute_backend='numpy',
        verbose=0,
    )
    cpp_scores = bn.structure_scores(
        model,
        discrete_data,
        scoring_method=['bic', 'k2'],
        compute_backend='cpp',
        verbose=0,
    )

    assert cpp_scores == pytest.approx(numpy_scores, rel=1e-12, abs=1e-12)


def test_structure_learning_fit_accepts_cpp(discrete_data):
    pytest.importorskip('pgmpy._native_discrete')

    result = bn.structure_learning.fit(
        discrete_data,
        methodtype='hc',
        scoretype='bic',
        compute_backend='cpp',
        n_jobs=-1,
        max_iter=10,
        structure_score_methods='selected',
        verbose=0,
    )

    assert result['config']['compute_backend'] == 'cpp'
    assert result['config']['resolved_compute_backend'] == 'cpp'
    assert set(result['model'].nodes()) == set(discrete_data.columns)
    assert set(result['structure_scores']) == {'bic'}


def test_cpp_hill_climb_forces_one_host_worker(discrete_data, monkeypatch):
    pytest.importorskip('pgmpy._native_discrete')
    captured = {}

    class RecordingHillClimbSearch:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, data):
            self.causal_graph_ = DAG()
            self.causal_graph_.add_nodes_from(data.columns)
            return self

    monkeypatch.setattr(
        structure_learning, 'HillClimbSearch', RecordingHillClimbSearch
    )
    result = structure_learning._hillclimbsearch(
        discrete_data,
        scoretype='bic',
        n_jobs=-1,
        compute_backend='cpp',
        max_iter=2,
        verbose=0,
    )

    assert captured['n_jobs'] == 1
    assert set(result['model'].nodes()) == set(discrete_data.columns)


def test_missing_cpp_extension_warns_and_keeps_numpy_parallelism(
    discrete_data, monkeypatch
):
    import pgmpy.utils.native_discrete as native_discrete

    captured = {}

    class RecordingHillClimbSearch:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, data):
            self.causal_graph_ = DAG()
            self.causal_graph_.add_nodes_from(data.columns)
            return self

    monkeypatch.setattr(native_discrete, '_native_discrete', None)
    monkeypatch.setattr(
        structure_learning, 'HillClimbSearch', RecordingHillClimbSearch
    )
    with pytest.warns(RuntimeWarning, match='C\\+\\+ extension'):
        result = structure_learning._hillclimbsearch(
            discrete_data,
            scoretype='bic',
            n_jobs=4,
            compute_backend='cpp',
            max_iter=2,
            verbose=0,
        )

    assert captured['scoring_method'].resolved_backend_ == 'numpy'
    assert captured['n_jobs'] == 4
    assert result['_resolved_compute_backend'] == 'numpy'


@pytest.mark.parametrize('entrypoint', ['fit', 'structure_scores'])
def test_invalid_backend_is_rejected_at_bnlearn_boundary(
    entrypoint, discrete_data
):
    if entrypoint == 'fit':
        call = lambda: bn.structure_learning.fit(
            discrete_data,
            methodtype='hc',
            compute_backend='cuda',
            verbose=0,
        )
    else:
        model = bn.make_DAG([('A', 'B'), ('B', 'C')], verbose=0)
        call = lambda: bn.structure_scores(
            model,
            discrete_data,
            scoring_method='bic',
            compute_backend='cuda',
            verbose=0,
        )

    with pytest.raises(ValueError, match='numpy.*cupy.*cpp.*auto'):
        call()
