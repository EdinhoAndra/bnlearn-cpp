Black and white lists
========================

Nodes or directed edges can be black- or white-listed. With
``bw_list_method='nodes'``, node names filter the dataframe before structure
learning. With ``bw_list_method='edges'``, each list item must be a directed
``(source, target)`` pair. A white edge list defines the complete search space;
a black edge list identifies forbidden edges.

**White list example**

.. code-block:: python

    import bnlearn
    # Load example mixed dataset
    df_raw = bnlearn.import_example(data='titanic')

    # Convert to onehot
    dfhot, dfnum = bnlearn.df2onehot(df_raw)

    # Structure learning by including only 'Survived','Pclass','Sex','Embarked','Parch'.
    DAG_nodes = bnlearn.structure_learning.fit(dfnum, methodtype='hc', bw_list_method='nodes', white_list=['Survived','Pclass','Sex','Embarked','Parch'])

    # Limit the search to these directed edges.
    allowed_edges = [
        ('Pclass', 'Survived'),
        ('Sex', 'Survived'),
        ('Embarked', 'Survived'),
        ('Parch', 'Survived'),
    ]
    DAG_edges = bnlearn.structure_learning.fit(
        dfnum,
        methodtype='hc',
        bw_list_method='edges',
        white_list=allowed_edges,
    )

    # Plot
    Gf = bnlearn.plot(DAG_nodes)
    Ge = bnlearn.plot(DAG_edges)



**Black list example**

.. code-block:: python

    import bnlearn
    # Load example mixed dataset
    df_raw = bnlearn.import_example(data='titanic')

    # Convert to onehot
    dfhot, dfnum = bnlearn.df2onehot(df_raw)

    # Structure learning after removing 'Survived','Pclass','Sex','Embarked','Parch'.
    DAG_nodes = bnlearn.structure_learning.fit(dfnum, methodtype='hc', bw_list_method='nodes', black_list=['Survived','Pclass','Sex','Embarked','Parch'])

    # Forbid these directed edges while leaving the remaining search space open.
    forbidden_edges = [
        ('Pclass', 'Survived'),
        ('Sex', 'Survived'),
        ('Embarked', 'Survived'),
        ('Parch', 'Survived'),
    ]
    DAG_edges = bnlearn.structure_learning.fit(
        dfnum,
        methodtype='hc',
        bw_list_method='edges',
        black_list=forbidden_edges,
    )

    # Plot
    Gf = bnlearn.plot(DAG_nodes)
    Ge = bnlearn.plot(DAG_edges)





.. include:: add_bottom.add
