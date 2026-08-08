import numpy as np
import pandas as pd
import pytest
from pgmpy.base import DAG

import bnlearn as bn
import bnlearn.structure_learning as structure_learning


@pytest.fixture
def discrete_data():
    rng = np.random.default_rng(2026)
    a = rng.integers(0, 3, size=2_000)
    return pd.DataFrame(
        {
            "A": a,
            "B": (a + rng.integers(0, 2, size=2_000)) % 3,
            "C": rng.integers(0, 2, size=2_000),
        }
    )


@pytest.mark.parametrize(
    "scoretype", ["k2", "bdeu", "bds", "loglik-g", "aic-g", "bic-g"]
)
def test_cuda_fused_rejects_unsupported_search_scores(scoretype, discrete_data):
    with pytest.raises(ValueError, match="cuda_fused.*bic.*aic"):
        bn.structure_learning.fit(
            discrete_data,
            methodtype="hc",
            scoretype=scoretype,
            compute_backend="cuda_fused",
            verbose=0,
        )


def test_cuda_fused_hill_climb_forces_one_host_worker(discrete_data, monkeypatch):
    captured = {}

    class FakeScore:
        resolved_backend_ = "cuda_fused"

    class RecordingHillClimbSearch:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, data):
            self.causal_graph_ = DAG()
            self.causal_graph_.add_nodes_from(data.columns)
            return self

    monkeypatch.setattr(
        structure_learning, "_SetScoringType", lambda *args, **kwargs: FakeScore()
    )
    monkeypatch.setattr(structure_learning, "HillClimbSearch", RecordingHillClimbSearch)
    result = structure_learning._hillclimbsearch(
        discrete_data,
        scoretype="bic",
        n_jobs=-1,
        compute_backend="cuda_fused",
        max_iter=2,
        verbose=0,
    )

    assert captured["n_jobs"] == 1
    assert result["_resolved_compute_backend"] == "cuda_fused"


def test_cuda_fused_post_scores_route_unsupported_formulas_to_cupy(
    discrete_data, monkeypatch
):
    requested_backends = {}

    class FakeScore:
        def __init__(self, score_name):
            self.score_name = score_name

        def score(self, model):
            return float(len(model.nodes()))

    def record_backend(df, score_name, *, compute_backend, **kwargs):
        requested_backends[score_name] = compute_backend
        return FakeScore(score_name)

    monkeypatch.setattr(structure_learning, "_SetScoringType", record_backend)
    model = bn.make_DAG([("A", "B"), ("B", "C")], verbose=0)
    scores = bn.structure_scores(
        model,
        discrete_data,
        scoring_method=["k2", "bic", "bdeu", "bds", "aic"],
        compute_backend="cuda_fused",
        verbose=0,
    )

    assert set(scores) == {"k2", "bic", "bdeu", "bds", "aic"}
    assert requested_backends == {
        "k2": "cupy",
        "bic": "cuda_fused",
        "bdeu": "cupy",
        "bds": "cupy",
        "aic": "cuda_fused",
    }


@pytest.mark.parametrize("score_name", ["bic", "aic"])
def test_cuda_fused_scores_match_numpy_through_bnlearn(score_name, discrete_data):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device is available")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("No usable CUDA device is available")

    cpu_score = structure_learning._SetScoringType(
        discrete_data,
        score_name,
        compute_backend="numpy",
        verbose=0,
    )
    fused_score = structure_learning._SetScoringType(
        discrete_data,
        score_name,
        compute_backend="cuda_fused",
        verbose=0,
    )

    assert fused_score.resolved_backend_ == "cuda_fused"
    assert fused_score.local_score("C", ("A", "B")) == pytest.approx(
        cpu_score.local_score("C", ("A", "B")), rel=1e-12, abs=1e-8
    )
