"""Thread-parallel Hill Climb candidate scoring for pgmpy 0.1.25."""

from itertools import permutations

import networkx as nx
from joblib import Parallel, delayed
from pgmpy.estimators import HillClimbSearch


class ParallelHillClimbSearch(HillClimbSearch):
    """HillClimbSearch that scores legal operations with a shared thread pool."""

    def __init__(self, data, n_jobs=1, **kwargs):
        self.n_jobs = n_jobs
        super().__init__(data, **kwargs)

    def _legal_operations(
        self,
        model,
        score,
        structure_score,
        tabu_list,
        max_indegree,
        black_list,
        white_list,
        fixed_edges,
    ):
        tabu_list = set(tabu_list)
        edges = list(model.edges())
        edge_set = set(edges)
        reverse_edge_set = {(target, source) for source, target in edges}
        candidates = []

        potential_new_edges = set(permutations(self.variables, 2)) - edge_set - reverse_edge_set
        for source, target in potential_new_edges:
            if nx.has_path(model, target, source):
                continue
            operation = ("+", (source, target))
            if operation not in tabu_list and (source, target) not in black_list and (source, target) in white_list:
                old_parents = tuple(model.get_parents(target))
                new_parents = old_parents + (source,)
                if len(new_parents) <= max_indegree:
                    candidates.append(
                        (operation, ((target, new_parents),), ((target, old_parents),), structure_score("+"))
                    )

        for source, target in edges:
            operation = ("-", (source, target))
            if operation not in tabu_list and (source, target) not in fixed_edges:
                old_parents = tuple(model.get_parents(target))
                new_parents = tuple(parent for parent in old_parents if parent != source)
                candidates.append(
                    (operation, ((target, new_parents),), ((target, old_parents),), structure_score("-"))
                )

        for source, target in edges:
            if any(len(path) > 2 for path in nx.all_simple_paths(model, source, target)):
                continue
            operation = ("flip", (source, target))
            if (
                operation not in tabu_list
                and ("flip", (target, source)) not in tabu_list
                and (source, target) not in fixed_edges
                and (target, source) not in black_list
                and (target, source) in white_list
            ):
                old_source_parents = tuple(model.get_parents(source))
                old_target_parents = tuple(model.get_parents(target))
                new_source_parents = old_source_parents + (target,)
                new_target_parents = tuple(parent for parent in old_target_parents if parent != source)
                if len(new_source_parents) <= max_indegree:
                    candidates.append(
                        (
                            operation,
                            ((source, new_source_parents), (target, new_target_parents)),
                            ((source, old_source_parents), (target, old_target_parents)),
                            structure_score("flip"),
                        )
                    )

        def score_candidate(candidate):
            operation, new_families, old_families, prior = candidate
            new_score = sum(score(variable, list(parents)) for variable, parents in new_families)
            old_score = sum(score(variable, list(parents)) for variable, parents in old_families)
            return operation, new_score - old_score + prior

        if self.n_jobs == 1:
            scored_candidates = map(score_candidate, candidates)
        else:
            scored_candidates = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(score_candidate)(candidate) for candidate in candidates
            )
        yield from scored_candidates
