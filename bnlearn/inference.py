"""Inference is same as asking conditional probability questions to the models.

# ------------------------------------
# Name        : inference.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Licence     : See licences
# ------------------------------------

"""
# %% Libraries
from collections.abc import Mapping

import matplotlib.pyplot as plt
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
from pgmpy.utils import compat_fns
import numpy as np
import bnlearn
import warnings
warnings.filterwarnings("ignore")


class CompiledInference:
    """Reusable exact-inference engine for a fitted discrete Bayesian network.

    The pgmpy :class:`VariableElimination` object is built once and reused for
    every query. For a single target whose direct parents are all observed, a
    batch can be answered directly from its CPD. The direct lookup remains
    exact in the presence of additional non-descendant evidence; rows with
    missing parents or descendant evidence fall back to variable elimination.

    Recompile the engine after mutating the graph or any of its CPDs.
    """

    def __init__(self, model):
        if not isinstance(model, dict):
            raise Exception('[bnlearn] >Error: Input requires a object that contains the key: model.')
        if 'model' not in model:
            raise Exception('[bnlearn] >Error: Input requires a object that contains the key: model.')

        self.model_dict = model
        self.model = model['model']
        if 'BayesianNetwork' not in str(type(self.model)):
            raise TypeError(
                '[bnlearn] >Error: Inference requires BayesianNetwork. '
                'hint: try: parameter_learning.fit(DAG, df, methodtype="bayes")'
            )
        try:
            self._ve = VariableElimination(self.model)
        except ValueError as exc:
            raise Exception(f'[bnlearn] >Error: {exc}') from exc

        self._nodes = set(self.model.nodes())
        self._direct_lookups = {}

    def _validate_request(self, variables, evidences):
        if isinstance(variables, str):
            variables = [variables]
        elif variables is None:
            raise ValueError('[bnlearn] >Error: [variables] must contain at least one node.')
        else:
            variables = list(variables)

        if not variables:
            raise ValueError('[bnlearn] >Error: [variables] must contain at least one node.')
        unknown_variables = set(variables) - self._nodes
        if unknown_variables:
            raise ValueError(
                f'[bnlearn] >Error: [variables] contains nodes not present in the model: '
                f'{sorted(unknown_variables, key=str)}'
            )

        evidences = list(evidences)
        for index, evidence in enumerate(evidences):
            if not isinstance(evidence, Mapping):
                raise TypeError(f'[bnlearn] >Error: evidence at index {index} must be a mapping.')
            unknown_evidence = set(evidence) - self._nodes
            if unknown_evidence:
                raise ValueError(
                    f'[bnlearn] >Error: evidence at index {index} contains nodes not present '
                    f'in the model: {sorted(unknown_evidence, key=str)}'
                )
        return variables, evidences

    def _get_direct_lookup(self, target):
        lookup = self._direct_lookups.get(target)
        if lookup is not None:
            return lookup

        cpd = self.model.get_cpds(target)
        if cpd is None:
            raise ValueError(f'[bnlearn] >Error: No CPD is associated with target {target!r}.')

        parents = tuple(cpd.variables[1:])
        parent_cards = tuple(int(card) for card in cpd.cardinality[1:])
        descendants = set()
        frontier = list(self.model.successors(target))
        while frontier:
            node = frontier.pop()
            if node not in descendants:
                descendants.add(node)
                frontier.extend(self.model.successors(node))

        lookup = {
            'cpd': cpd,
            'parents': parents,
            'parent_cards': parent_cards,
            'descendants': descendants,
            'values': compat_fns.to_numpy(cpd.get_values()),
        }
        self._direct_lookups[target] = lookup
        return lookup

    @staticmethod
    def _factor_from_probabilities(target, cpd, probabilities):
        return DiscreteFactor(
            variables=[target],
            cardinality=[int(cpd.variable_card)],
            values=probabilities,
            state_names={target: list(cpd.state_names[target])},
        )

    def _query_direct_rows(self, target, evidences, result, joint):
        lookup = self._get_direct_lookup(target)
        parents = lookup['parents']
        descendants = lookup['descendants']

        eligible_indexes = [
            index
            for index, evidence in enumerate(evidences)
            if target not in evidence
            and all(parent in evidence for parent in parents)
            and descendants.isdisjoint(evidence)
        ]
        if not eligible_indexes:
            return set()

        cpd = lookup['cpd']
        if parents:
            parent_codes = np.empty((len(eligible_indexes), len(parents)), dtype=np.intp)
            for row, index in enumerate(eligible_indexes):
                evidence = evidences[index]
                for column, parent in enumerate(parents):
                    parent_codes[row, column] = cpd.get_state_no(parent, evidence[parent])
            column_indexes = np.ravel_multi_index(
                parent_codes.T,
                lookup['parent_cards'],
            )
        else:
            column_indexes = np.zeros(len(eligible_indexes), dtype=np.intp)

        probabilities = lookup['values'][:, column_indexes]
        probabilities = probabilities / probabilities.sum(axis=0, keepdims=True)
        for column, index in enumerate(eligible_indexes):
            factor = self._factor_from_probabilities(target, cpd, probabilities[:, column])
            result[index] = factor if joint else {target: factor}
        return set(eligible_indexes)

    def query_many(
        self,
        variables,
        evidences,
        elimination_order='greedy',
        joint=True,
        to_df=False,
        groupby=None,
        show_progress=False,
        verbose=0,
    ):
        """Evaluate several evidence configurations in input order.

        Parameters mirror :func:`fit`, except ``evidences`` is a sequence of
        mappings and plotting is deliberately omitted for batch workloads.
        Results are pgmpy factors (or dictionaries when ``joint=False``), in
        the same order as the supplied evidence rows.
        """
        variables, evidences = self._validate_request(variables, evidences)
        if to_df and not joint:
            raise ValueError('[bnlearn] >Error: to_df=True requires joint=True in query_many().')

        results = [None] * len(evidences)
        direct_indexes = set()
        if len(variables) == 1:
            direct_indexes = self._query_direct_rows(
                variables[0], evidences, results, joint=joint
            )

        for index, evidence in enumerate(evidences):
            if index in direct_indexes:
                continue
            results[index] = self._ve.query(
                variables=variables,
                evidence=dict(evidence),
                elimination_order=elimination_order,
                joint=joint,
                show_progress=show_progress,
            )

        if to_df:
            for evidence, query in zip(evidences, results):
                query.df = bnlearn.query2df(
                    query, variables=variables.copy(), groupby=groupby, verbose=verbose
                )
                query.text = summarize_inference(
                    variables, evidence, query, plot=False, verbose=verbose
                )
        return results


def compile(model):
    """Compile a reusable :class:`CompiledInference` engine."""
    return CompiledInference(model)


def query_many(model, variables, evidences, **kwargs):
    """Batch inference convenience wrapper.

    ``model`` can be either a bnlearn model dictionary or an already compiled
    :class:`CompiledInference` instance. Passing a compiled instance avoids
    rebuilding pgmpy's variable-elimination engine across calls.
    """
    engine = model if isinstance(model, CompiledInference) else CompiledInference(model)
    return engine.query_many(variables=variables, evidences=evidences, **kwargs)


# %% Exact inference using Variable Elimination
def fit(model,
        variables=None,
        evidence=None,
        to_df=True,
        elimination_order='greedy',
        joint=True,
        groupby=None,
        plot=False,
        verbose=3,
        ):
    """Inference using using Variable Elimination.

    The basic concept of variable elimination is same as doing marginalization over Joint Distribution.
    But variable elimination avoids computing the Joint Distribution by doing marginalization over much smaller factors.
    So basically if we want to eliminate X from our distribution, then we compute the product of all the factors
    involving X and marginalize over them, thus allowing us to work on much smaller factors.

    Parameters
    ----------
    model : dict
        Contains model.
    variables : List, optional
        For exact inference, P(variables | evidence). The default is None.
            * ['Name_of_node_1']
            * ['Name_of_node_1', 'Name_of_node_2']
    evidence : dict, optional
        For exact inference, P(variables | evidence). The default is None.
            * {'Rain':1}
            * {'Rain':1, 'Sprinkler':0, 'Cloudy':1}
    to_df : Bool, (default is True)
        The output is converted in the dataframe [query.df]. Enabling this function may impact the processing speed.
    elimination_order: str or list (default='greedy')
        Order in which to eliminate the variables in the algorithm. If list is provided,
        should contain all variables in the model except the ones in `variables`. str options
        are: `greedy`, `WeightedMinFill`, `MinNeighbors`, `MinWeight`, `MinFill`. Please
        refer https://pgmpy.org/exact_infer/ve.html#module-pgmpy.inference.EliminationOrder
        for details.
    joint: boolean (default: True)
        If True, returns a Joint Distribution over `variables`.
        If False, returns a dict of distributions over each of the `variables`.
    groupby: list of strings (default: None)
        The query is grouped on the variable name by taking the maximum P value for each catagory.
    plot : bool, optional
        If True, display a bar plot.
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    query inference object.

    Examples
    --------
    >>> import bnlearn as bn
    >>>
    >>> # Load example data
    >>> model = bn.import_DAG('sprinkler')
    >>> bn.plot(model)
    >>>
    >>> # Do the inference
    >>> query = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})
    >>> print(query)
    >>> query.df
    >>>
    >>> query = bn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})
    >>> print(query)
    >>> query.df
    >>>

    """
    if not isinstance(model, dict): raise Exception('[bnlearn] >Error: Input requires a object that contains the key: model.')
    adjmat = model['adjmat']
    if not np.all(np.isin(variables, adjmat.columns)):
        raise Exception('[bnlearn] >Error: [variables] should match names in the model (Case sensitive!)')
    if not np.all(np.isin([*evidence.keys()], adjmat.columns)):
        raise Exception('[bnlearn] >Error: [evidence] should match names in the model (Case sensitive!)')
    if verbose>=3: print('[bnlearn] >Variable Elimination.')

    # Extract model
    if isinstance(model, dict):
        model = model['model']

    # Check BayesianNetwork
    if 'BayesianNetwork' not in str(type(model)):
        if verbose>=1: print('[bnlearn] >Warning: Inference requires BayesianNetwork. hint: try: parameter_learning.fit(DAG, df, methodtype="bayes") <return>')
        return None

    # Convert to BayesianNetwork
    if 'BayesianNetwork' not in str(type(model)):
        model = bnlearn.to_bayesiannetwork(adjmat, verbose=verbose)

    try:
        model_infer = VariableElimination(model)
    except ValueError as e:
        raise Exception(f'[bnlearn] >Error: {e}')
        # Input model does not contain learned CPDs. hint: did you run parameter_learning.fit()?

    # Computing the probability P(class | evidence)
    query = model_infer.query(variables=variables, evidence=evidence, elimination_order=elimination_order, joint=joint, show_progress=(verbose>=3))

    # Store dataframe in query
    if to_df or plot:
        # Convert to Dataframe
        query.df = bnlearn.query2df(query, variables=variables, groupby=groupby, verbose=verbose)
        # Make readable text
        query.text = summarize_inference(variables, evidence, query, plot=plot, verbose=verbose)
        if verbose>=3 and query.text is not None: print(query.text)
    else:
        query.df = None
        query.text = None

    # Return
    return query

#%%
def summarize_inference(variables, evidence, query, plot=False, verbose=3):
    """
    Summarize inference results based on a Bayesian Network inference output.

    Parameters
    ----------
    variables : list of str
        Variables being queried (e.g., ['Machine failure'] or multiple).
    evidence : dict
        Evidence variables and their fixed values (e.g., {'Torque [Nm]_category': 'high'}).
    query : Object from inference.fit()
        Inference output containing the queried variables and probability 'p' in a Dataframe (query.df)
    plot : bool, optional
        If True, display a bar plot.
    verbose : int, optional
        Print progress to screen. The default is 3.
        0: None, 1: ERROR, 2: WARN, 3: INFO (default), 4: DEBUG, 5: TRACE

    Returns
    -------
    str
        A textual summary.

    """
    df = query.df

    def is_binary(series):
        return sorted(series.dropna().unique()) in [[0, 1], [1, 0]]

    lines = []
    lines.append(f"\nSummary for variables: {variables}")
    evidence_txt = f"{', '.join([f'{k}={v}' for k, v in evidence.items()])}"
    lines.append(f"Given evidence: {evidence_txt}")

    for var in variables:
        lines.append(f"\n{var} outcomes:")
        grouped = df.groupby(var)['p'].sum()
        total = grouped.sum()
        for val, prob in grouped.items():
            description = f"{var}: {val}"
            lines.append(f"- {description} ({prob/total:.1%})")

    if plot:
        # Plot dominant probabilities
        for var in variables:
            grouped = df.groupby(var)['p'].sum()
            total = grouped.sum()
            percentages = (grouped / total) * 100

            plt.figure(figsize=(8, 4))
            labels = [f'state_{x}' for x in percentages.index]
            bars = plt.barh(labels, percentages.values, color='#4a90e2', edgecolor='black')
            plt.xlabel('Percentage (%)', fontsize=12)
            plt.title(f'Inference Summary: {var}\n{evidence_txt}', fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.gca().invert_yaxis()

            # Add percentages at end of bars
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 1.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontsize=10)

            plt.xlim(0, max(percentages.values)*1.1)  # Make 10% larger
            plt.tight_layout()
            plt.show()

    return "\n".join(lines)
