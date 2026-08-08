import numpy as np
import pandas as pd
import pytest

import bnlearn as bn
import bnlearn.structure_learning as structure_learning
from pgmpy.causal_discovery import ExpertKnowledge, HillClimbSearch
from pgmpy.structure_score import AIC, BDeu, BDs, BIC, K2


REFERENCE_SCORES = {
    "bic": BIC,
    "k2": K2,
    "bdeu": BDeu,
    "bds": BDs,
    "aic": AIC,
}


@pytest.fixture
def discrete_data():
    rng = np.random.default_rng(42)
    a = rng.integers(0, 3, size=2_000)
    return pd.DataFrame(
        {
            "A": a,
            "B": a,
            "C": rng.integers(0, 2, size=2_000),
        }
    )


@pytest.mark.parametrize("score_name, score_class", REFERENCE_SCORES.items())
def test_set_scoring_type_uses_native_pgmpy_score(score_name, score_class, discrete_data):
    score = structure_learning._SetScoringType(
        discrete_data,
        score_name,
        compute_backend="numpy",
        verbose=0,
    )

    assert type(score) is score_class
    assert score.resolved_backend_ == "numpy"


def test_auto_backend_configuration_is_forwarded_to_pgmpy(discrete_data):
    score = structure_learning._SetScoringType(
        discrete_data,
        "bic",
        compute_backend="auto",
        min_gpu_rows=len(discrete_data) + 1,
        verbose=0,
    )

    assert score.compute_backend == "auto"
    assert score.min_gpu_rows == len(discrete_data) + 1
    assert score.resolved_backend_ == "numpy"


def test_hill_climb_imports_canonical_pgmpy_api():
    assert structure_learning.HillClimbSearch is HillClimbSearch
    assert structure_learning.ExpertKnowledge is ExpertKnowledge
    assert HillClimbSearch.__module__ == "pgmpy.causal_discovery.HillClimbSearch"
    assert ExpertKnowledge.__module__ == "pgmpy.causal_discovery.ExpertKnowledge"


def test_edge_constraints_use_native_expert_knowledge(discrete_data):
    result = bn.structure_learning.fit(
        discrete_data,
        methodtype="hc",
        scoretype="bic",
        white_list=[("A", "B"), ("B", "C")],
        black_list=[("B", "C")],
        fixed_edges=[("A", "B")],
        bw_list_method="edges",
        max_iter=20,
        n_jobs=1,
        compute_backend="numpy",
        verbose=0,
    )

    assert set(result["model"].nodes()) == set(discrete_data.columns)
    assert set(result["model"].edges()) == {("A", "B")}


@pytest.mark.parametrize(
    "black_list, white_list, message",
    [
        ([("A", "B")], None, "fixed_edges.*black_list"),
        (None, [("B", "C")], "fixed_edges.*white_list"),
    ],
)
def test_conflicting_fixed_edges_are_rejected(
    discrete_data,
    black_list,
    white_list,
    message,
):
    with pytest.raises(ValueError, match=message):
        bn.structure_learning.fit(
            discrete_data,
            methodtype="hc",
            scoretype="bic",
            black_list=black_list,
            white_list=white_list,
            fixed_edges=[("A", "B")],
            bw_list_method="edges",
            max_iter=1,
            verbose=0,
        )


@pytest.mark.parametrize("score_name", REFERENCE_SCORES)
def test_cupy_scores_match_numpy_through_bnlearn(score_name, discrete_data):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device is available")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("No usable CUDA device is available")

    kwargs = {"equivalent_sample_size": 5} if score_name in {"bdeu", "bds"} else {}
    cpu_score = structure_learning._SetScoringType(
        discrete_data,
        score_name,
        compute_backend="numpy",
        verbose=0,
        **kwargs,
    )
    gpu_score = structure_learning._SetScoringType(
        discrete_data,
        score_name,
        compute_backend="cupy",
        verbose=0,
        **kwargs,
    )

    assert gpu_score.local_score("C", ("A", "B")) == pytest.approx(
        cpu_score.local_score("C", ("A", "B")),
        rel=1e-10,
        abs=1e-10,
    )


def test_structure_score_methods_selected_only_scores_search_metric(discrete_data):
    result = bn.structure_learning.fit(
        discrete_data,
        methodtype="hc",
        scoretype="bic",
        max_iter=5,
        n_jobs=1,
        compute_backend="numpy",
        structure_score_methods="selected",
        verbose=0,
    )

    assert set(result["structure_scores"]) == {"bic"}
    assert result["config"]["structure_score_methods"] == ["bic"]


def test_empty_structure_score_methods_skips_post_fit_scoring(discrete_data, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("post-fit structure scoring must be skipped")

    monkeypatch.setattr(bn, "structure_scores", fail_if_called)
    result = bn.structure_learning.fit(
        discrete_data,
        methodtype="hc",
        scoretype="bic",
        max_iter=5,
        n_jobs=1,
        compute_backend="numpy",
        structure_score_methods=[],
        verbose=0,
    )

    assert result["structure_scores"] == {}
    assert result["config"]["structure_score_methods"] == []


def test_unknown_structure_score_method_is_rejected(discrete_data):
    with pytest.raises(ValueError, match="structure_score_methods"):
        bn.structure_learning.fit(
            discrete_data,
            methodtype="hc",
            structure_score_methods=["not-a-score"],
            verbose=0,
        )
