"""Vectorized NumPy and CuPy structure scores for pgmpy 0.1.25."""

from math import lgamma, log

import numpy as np
import pandas as pd
from scipy.special import gammaln as numpy_gammaln

try:
    from pgmpy.estimators import StructureScore
except ImportError:
    from pgmpy.estimators.StructureScore import StructureScore


def _resolve_backend(compute_backend, n_rows, min_gpu_rows):
    if compute_backend not in {"numpy", "cupy", "auto"}:
        raise ValueError(
            "compute_backend must be one of: 'numpy', 'cupy', or 'auto'. "
            f"Got: {compute_backend!r}"
        )
    if not isinstance(min_gpu_rows, int) or min_gpu_rows < 0:
        raise ValueError(f"min_gpu_rows must be a non-negative integer. Got: {min_gpu_rows!r}")

    if compute_backend == "numpy" or (compute_backend == "auto" and n_rows < min_gpu_rows):
        return "numpy", np, numpy_gammaln

    try:
        import cupy as cp
        from cupyx.scipy.special import gammaln as cupy_gammaln

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device is available.")
    except (ImportError, RuntimeError, OSError) as error:
        if compute_backend == "auto":
            return "numpy", np, numpy_gammaln
        raise ImportError(
            "The CuPy backend requires a working CUDA device and a matching CuPy "
            "package, for example `pip install 'bnlearn[gpu-cu12]'`."
        ) from error

    return "cupy", cp, cupy_gammaln


class _AcceleratedDiscreteScore(StructureScore):
    """Base class that encodes data once and keeps arrays on the selected device."""

    def __init__(self, data, compute_backend="numpy", min_gpu_rows=50_000, **kwargs):
        super().__init__(data, **kwargs)
        self.compute_backend = compute_backend
        self.min_gpu_rows = min_gpu_rows
        self.resolved_backend_, self._xp, self._gammaln = _resolve_backend(
            compute_backend,
            len(self.data),
            min_gpu_rows,
        )

        self._cardinalities = {column: len(states) for column, states in self.state_names.items()}
        self._codes = {}
        for column in self.data.columns:
            categorical = pd.Categorical(self.data[column], categories=self.state_names[column])
            self._codes[column] = self._xp.asarray(np.asarray(categorical.codes, dtype=np.int64))

    def _state_counts_array(self, variable, parents):
        xp = self._xp
        parents = tuple(parents)
        variable_codes = self._codes[variable]
        variable_cardinality = self._cardinalities[variable]

        valid = variable_codes >= 0
        for parent in parents:
            valid &= self._codes[parent] >= 0

        if bool(valid.all().item()):
            variable_values = variable_codes
            parent_values = [self._codes[parent] for parent in parents]
        else:
            variable_values = variable_codes[valid]
            parent_values = [self._codes[parent][valid] for parent in parents]

        if not parents:
            counts = xp.bincount(variable_values, minlength=variable_cardinality).astype(xp.float64, copy=False)
            return counts.reshape(variable_cardinality, 1)

        parent_index = xp.zeros(len(variable_values), dtype=xp.int64)
        num_parent_states = 1
        for values, parent in zip(parent_values, parents):
            parent_index = parent_index * self._cardinalities[parent] + values
            num_parent_states *= self._cardinalities[parent]

        flat_index = parent_index * variable_cardinality + variable_values
        counts = xp.bincount(
            flat_index,
            minlength=num_parent_states * variable_cardinality,
        ).astype(xp.float64, copy=False)
        return counts.reshape(num_parent_states, variable_cardinality).T

    @staticmethod
    def _as_float(value):
        return np.float64(value.item())


class AcceleratedK2Score(_AcceleratedDiscreteScore):
    """K2 score using a vectorized NumPy or CuPy contingency-table kernel."""

    def local_score(self, variable, parents):
        counts = self._state_counts_array(variable, parents)
        variable_cardinality = self._cardinalities[variable]
        num_parent_states = counts.shape[1]
        log_gamma_counts = self._gammaln(counts + 1)
        log_gamma_conditions = self._gammaln(
            self._xp.sum(counts, axis=0, dtype=self._xp.float64) + variable_cardinality
        )
        score = (
            self._xp.sum(log_gamma_counts)
            - self._xp.sum(log_gamma_conditions)
            + num_parent_states * lgamma(variable_cardinality)
        )
        return self._as_float(score)


class AcceleratedBDeuScore(_AcceleratedDiscreteScore):
    """BDeu score using a vectorized NumPy or CuPy contingency-table kernel."""

    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        self.equivalent_sample_size = equivalent_sample_size
        super().__init__(data, **kwargs)

    def local_score(self, variable, parents):
        counts = self._state_counts_array(variable, parents)
        num_parent_states = counts.shape[1]
        variable_cardinality = self._cardinalities[variable]
        counts_size = num_parent_states * variable_cardinality
        alpha = self.equivalent_sample_size / num_parent_states
        beta = self.equivalent_sample_size / counts_size

        log_gamma_counts = self._gammaln(counts + beta)
        log_gamma_conditions = self._gammaln(self._xp.sum(counts, axis=0, dtype=self._xp.float64) + alpha)
        score = (
            self._xp.sum(log_gamma_counts)
            - self._xp.sum(log_gamma_conditions)
            + num_parent_states * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return self._as_float(score)


class AcceleratedBDsScore(AcceleratedBDeuScore):
    """Sparse BDeu score using a vectorized NumPy or CuPy kernel."""

    def structure_prior_ratio(self, operation):
        if operation == "+":
            return -log(2.0)
        if operation == "-":
            return log(2.0)
        return 0

    def structure_prior(self, model):
        num_edges = float(len(model.edges()))
        num_nodes = float(len(model.nodes()))
        possible_edges = num_nodes * (num_nodes - 1) / 2.0
        return -(num_edges + possible_edges) * log(2.0)

    def local_score(self, variable, parents):
        counts = self._state_counts_array(variable, parents)
        num_parent_states = counts.shape[1]
        variable_cardinality = self._cardinalities[variable]
        counts_size = num_parent_states * variable_cardinality
        condition_counts = self._xp.sum(counts, axis=0, dtype=self._xp.float64)
        num_observed_value = self._xp.count_nonzero(condition_counts)
        num_observed = int(
            num_observed_value.item()
            if hasattr(num_observed_value, "item")
            else num_observed_value
        )
        alpha = self.equivalent_sample_size / num_observed
        beta = self.equivalent_sample_size / counts_size

        score = (
            self._xp.sum(self._gammaln(counts + beta))
            - self._xp.sum(self._gammaln(condition_counts + alpha))
            + num_observed * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return self._as_float(score)


class _AcceleratedLogLikelihoodScore(_AcceleratedDiscreteScore):
    def _log_likelihood(self, variable, parents):
        counts = self._state_counts_array(variable, parents)
        positive_counts = counts > 0
        log_likelihoods = self._xp.log(self._xp.where(positive_counts, counts, 1))
        conditions = self._xp.sum(counts, axis=0, dtype=self._xp.float64)
        log_conditions = self._xp.log(self._xp.where(conditions > 0, conditions, 1))
        log_likelihoods -= log_conditions
        log_likelihoods *= counts
        return self._as_float(self._xp.sum(log_likelihoods)), counts.shape[1], self._cardinalities[variable]


class AcceleratedBICScore(_AcceleratedLogLikelihoodScore):
    """BIC score using a vectorized NumPy or CuPy contingency-table kernel."""

    def local_score(self, variable, parents):
        likelihood, num_parent_states, variable_cardinality = self._log_likelihood(variable, parents)
        return likelihood - 0.5 * log(len(self.data)) * num_parent_states * (variable_cardinality - 1)


class AcceleratedAICScore(_AcceleratedLogLikelihoodScore):
    """AIC score using a vectorized NumPy or CuPy contingency-table kernel."""

    def local_score(self, variable, parents):
        likelihood, num_parent_states, variable_cardinality = self._log_likelihood(variable, parents)
        return likelihood - num_parent_states * (variable_cardinality - 1)


def get_accelerated_score(
    data,
    scoretype,
    compute_backend="numpy",
    min_gpu_rows=50_000,
    equivalent_sample_size=5,
    **kwargs,
):
    """Create a vectorized discrete score compatible with pgmpy 0.1.25."""
    score_classes = {
        "bic": AcceleratedBICScore,
        "k2": AcceleratedK2Score,
        "bdeu": AcceleratedBDeuScore,
        "bds": AcceleratedBDsScore,
        "aic": AcceleratedAICScore,
    }
    if scoretype not in score_classes:
        raise ValueError(f"Unknown accelerated scoretype: {scoretype}")

    if scoretype in {"bdeu", "bds"}:
        kwargs["equivalent_sample_size"] = equivalent_sample_size
    return score_classes[scoretype](
        data,
        compute_backend=compute_backend,
        min_gpu_rows=min_gpu_rows,
        **kwargs,
    )
