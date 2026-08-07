import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators import AICScore, BDeuScore, BDsScore, BicScore, HillClimbSearch, K2Score
from bnlearn.accelerated_scores import get_accelerated_score
from bnlearn.parallel_hill_climb import ParallelHillClimbSearch

REFERENCE_SCORES = {
    "bic": BicScore,
    "k2": K2Score,
    "bdeu": BDeuScore,
    "bds": BDsScore,
    "aic": AICScore,
}


@pytest.fixture
def discrete_data():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "A": rng.integers(0, 3, size=2_000),
            "B": rng.integers(0, 4, size=2_000),
            "C": rng.integers(0, 2, size=2_000),
        }
    )
    data.loc[[3, 101], "B"] = np.nan
    return data


@pytest.mark.parametrize("score_name", REFERENCE_SCORES)
def test_numpy_scores_match_pgmpy_legacy(score_name, discrete_data):
    kwargs = {"equivalent_sample_size": 5} if score_name in {"bdeu", "bds"} else {}
    expected = REFERENCE_SCORES[score_name](discrete_data, **kwargs)
    actual = get_accelerated_score(
        discrete_data,
        score_name,
        compute_backend="numpy",
        equivalent_sample_size=5,
    )

    assert actual.resolved_backend_ == "numpy"
    assert isinstance(actual.local_score("C", ("A", "B")), np.float64)
    assert actual.local_score("C", ("A", "B")) == pytest.approx(
        expected.local_score("C", ("A", "B")), rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("score_name", REFERENCE_SCORES)
def test_numpy_scores_match_sparse_parent_configurations(score_name):
    data = pd.DataFrame(
        {
            "A": [0, 0, 0, 1, 1, 1, np.nan],
            "B": [0, 0, 0, 1, 1, 1, 0],
            "C": [0, 1, 0, 1, 0, 1, 1],
        }
    )
    state_names = {"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1]}
    kwargs = {"equivalent_sample_size": 5} if score_name in {"bdeu", "bds"} else {}
    expected = REFERENCE_SCORES[score_name](data, state_names=state_names, **kwargs)
    actual = get_accelerated_score(
        data,
        score_name,
        compute_backend="numpy",
        equivalent_sample_size=5,
        state_names=state_names,
    )

    assert actual.local_score("C", ("A", "B")) == pytest.approx(
        expected.local_score("C", ("A", "B")), rel=1e-12, abs=1e-12
    )


def test_auto_backend_stays_on_cpu_below_threshold(discrete_data):
    score = get_accelerated_score(
        discrete_data,
        "bic",
        compute_backend="auto",
        min_gpu_rows=len(discrete_data) + 1,
    )

    assert score.resolved_backend_ == "numpy"


def test_parallel_hill_climb_matches_serial(discrete_data):
    clean_data = discrete_data.dropna()
    score = get_accelerated_score(clean_data, "bic", compute_backend="numpy")
    common = {
        "scoring_method": score,
        "max_indegree": 2,
        "max_iter": 20,
        "show_progress": False,
    }

    serial = ParallelHillClimbSearch(clean_data, n_jobs=1).estimate(**common)
    parallel = ParallelHillClimbSearch(clean_data, n_jobs=2).estimate(**common)

    assert set(serial.edges()) == set(parallel.edges())


def test_accelerated_hill_climb_matches_pgmpy_legacy(discrete_data):
    clean_data = discrete_data.dropna()
    common = {
        "max_indegree": 2,
        "max_iter": 20,
        "show_progress": False,
    }
    legacy = HillClimbSearch(clean_data).estimate(scoring_method=BicScore(clean_data), **common)
    accelerated = ParallelHillClimbSearch(clean_data, n_jobs=1).estimate(
        scoring_method=get_accelerated_score(clean_data, "bic", compute_backend="numpy"),
        **common,
    )

    assert set(legacy.edges()) == set(accelerated.edges())


@pytest.mark.parametrize("compute_backend", ["jax", "cuda", "gpu"])
def test_invalid_compute_backend(compute_backend, discrete_data):
    with pytest.raises(ValueError, match="compute_backend"):
        get_accelerated_score(discrete_data, "bic", compute_backend=compute_backend)


@pytest.mark.parametrize("score_name", REFERENCE_SCORES)
def test_cupy_scores_match_numpy(score_name, discrete_data):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device is available")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("No usable CUDA device is available")

    cpu_score = get_accelerated_score(
        discrete_data,
        score_name,
        compute_backend="numpy",
        equivalent_sample_size=5,
    )
    gpu_score = get_accelerated_score(
        discrete_data,
        score_name,
        compute_backend="cupy",
        equivalent_sample_size=5,
    )

    assert gpu_score.resolved_backend_ == "cupy"
    assert gpu_score.local_score("C", ("A", "B")) == pytest.approx(
        cpu_score.local_score("C", ("A", "B")), rel=1e-10, abs=1e-10
    )


def test_cupy_hill_climb_matches_numpy(discrete_data):
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device is available")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("No usable CUDA device is available")

    clean_data = discrete_data.dropna()

    def estimate(backend):
        score = get_accelerated_score(clean_data, "bic", compute_backend=backend)
        return set(
            ParallelHillClimbSearch(clean_data, n_jobs=1)
            .estimate(
                scoring_method=score,
                max_indegree=2,
                max_iter=20,
                show_progress=False,
            )
            .edges()
        )

    assert estimate("cupy") == estimate("numpy")
